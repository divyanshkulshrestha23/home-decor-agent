import json

import pytest

from evals import analytics_eval as ae


@pytest.mark.parametrize("expected,got,ok", [
    (1000.0, 1009.0, True), (1000.0, 1020.0, False),
    (0.0743, 7.43, True), (7.43, 0.0743, True), (0.0743, 0.09, False),
])
def test_number_matches(expected, got, ok):
    assert ae.number_matches(expected, got) is ok


def test_value_found_numbers_strings_and_months():
    results = [{"rows": [{"category": "outdoor_decor", "revenue": "1,23,456"}]},
               {"rows": [{"month": "2026-10-01", "n": 5}]}]
    assert ae.value_found(123456, results) is True  # Indian digit grouping is accepted
    assert ae.value_found(5, results) is True
    assert ae.value_found("decor", results) is False
    assert ae.value_found("outdoor_decor", results) is True
    assert ae.value_found("2026-10", results) is True


def test_grade_needs_every_expected_value():
    calls = [
        {"name": "run_sql", "response": {"error": "BigQuery error: x"}},
        {"name": "run_sql", "response": {"rows": [{"a": 100, "b": 50}]}},
        {"name": "describe_store_tables", "response": {"result": "text"}},
    ]
    assert ae.grade({"v2025": 100, "v2026": 50}, calls) == {
        "correct": True, "found": {"v2025": True, "v2026": True}, "queries": 2, "failed_queries": 1}
    assert ae.grade({"v2025": 100, "v2026": 70}, calls)["correct"] is False


class FakeRunner:
    def run(self, sql):
        if "broken" in sql:
            return {"error": "bad"}
        return {"rows": [{"v": 42}], "row_count": 1}


def test_reference_answers_fill_table_prefix():
    cases = [{"id": "a", "sql": "SELECT 42 FROM `{t}.orders`"}, {"id": "b", "sql": "broken"}]
    answers = ae.reference_answers(FakeRunner(), cases, "p.d")
    assert answers == {"a": {"v": 42}, "b": {"error": "bad"}}


class FakeSession:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

    async def new_session(self):
        return "s"

    async def send(self, session_id, message):
        return {"reply": "done", "tool_calls": self.tool_calls}


@pytest.mark.asyncio
async def test_run_analyst_and_routing_summary():
    cases = [{"id": "a", "question": "q"}, {"id": "b", "question": "q"}, {"id": "c", "question": "q"}]
    answers = {"a": {"v": 42}, "b": {"v": 7}, "c": {"error": "bad"}}
    good = [{"name": "run_sql", "response": {"rows": [{"v": 42}]}}]
    analyst = await ae.run_analyst(lambda: FakeSession(good), cases, answers)
    routing = await ae.run_routing(
        lambda: FakeSession([{"name": "transfer_to_agent", "args": {"agent_name": "store_analyst"}}]),
        cases=[("revenue?", "store_analyst"), ("a rug", "find_similar")])
    summary = ae.summarize(analyst, routing)
    assert summary["analyst_questions"] == 2 and summary["analyst_correct_pct"] == 50.0
    assert list(summary["analyst_failures"]) == ["b"]
    assert summary["reference_errors"] == {"c": "bad"}
    assert summary["routing_correct_pct"] == 50.0
    assert summary["routing_failures"] == [{"message": "a rug", "expected": "find_similar", "got": "store_analyst"}]


def test_cases_file_is_valid():
    cases = json.loads(ae.CASES_FILE.read_text(encoding="utf-8"))
    assert len(cases) == 20 and len({c["id"] for c in cases}) == 20
    for case in cases:
        assert "{t}" in case["sql"] and case["question"].strip()
        assert ";" not in case["sql"]
