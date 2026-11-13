"""Evaluation C: does the analyst compute the right answers, and does routing work?

Analyst: each case in evals/data/analytics_cases.json has a business question
and a reference SQL query. The reference answer comes from running that SQL;
the analyst passes if any of its own query results contains the same value
(numbers within 1%; a share may be given as 0.07 or 7%), and for questions
with several values (desk_trend) all of them.

Routing: short shopper and manager messages, checked for the agent the root
transfers to first.

    python3 -m evals.analytics_eval --project YOUR_PROJECT_ID
    python3 -m evals.analytics_eval --project YOUR_PROJECT_ID --only revenue_2025 --skip-routing
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

CASES_FILE = Path("evals/data/analytics_cases.json")
RESULTS_DIR = Path("evals/results")
REL_TOLERANCE = 0.01

ROUTING_CASES = [
    ("What was our revenue last month?", "store_analyst"),
    ("Which city orders the most rugs?", "store_analyst"),
    ("Why did returns go up in June?", "store_analyst"),
    ("How many orders came through the app this year?", "store_analyst"),
    ("Is the festive season sale working compared to last year?", "store_analyst"),
    ("I'm looking for a jute rug under 5000", "find_similar"),
    ("Find me a sofa like this but in grey", "find_similar"),
    ("Do you have a brass table lamp?", "find_similar"),
    ("Mujhe ek sasta stool chahiye", "find_similar"),
    ("Which desk would fit a 100 cm wall?", "find_similar"),
]


def _numbers(value):
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            return [float(value.replace(",", ""))]
        except ValueError:
            return []
    return []


def number_matches(expected, got):
    """True if got equals expected within 1%, allowing a share as a fraction or a percent."""
    for candidate in (got, got / 100, got * 100):
        if abs(candidate - expected) <= max(abs(expected) * REL_TOLERANCE, 1e-9):
            return True
    return False


def value_found(expected, results):
    """Is the expected value present in any cell of any query result?"""
    cells = [cell for result in results for row in result.get("rows", []) for cell in row.values()]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return any(number_matches(float(expected), n) for cell in cells for n in _numbers(cell))
    wanted = str(expected).strip().lower()

    def same(cell):
        text = str(cell).strip().lower()
        # Exact match, so "decor" does not match "outdoor_decor"; a month like
        # "2026-10" may come back as the date "2026-10-01".
        return text == wanted or (re.fullmatch(r"\d{4}-\d{2}", wanted) is not None and text.startswith(wanted))

    return any(same(cell) for cell in cells if cell is not None)


def reference_answers(runner, cases, table_prefix):
    answers = {}
    for case in cases:
        result = runner.run(case["sql"].replace("{t}", table_prefix))
        if "error" in result or not result["rows"]:
            answers[case["id"]] = {"error": result.get("error", "no rows")}
        else:
            answers[case["id"]] = result["rows"][0]
    return answers


def grade(expected_row, tool_calls):
    results = [c["response"] for c in tool_calls if c["name"] == "run_sql" and isinstance(c.get("response"), dict)]
    ok_results = [r for r in results if "rows" in r]
    found = {col: value_found(value, ok_results) for col, value in expected_row.items()}
    return {"correct": all(found.values()), "found": found, "queries": len(results),
            "failed_queries": len(results) - len(ok_results)}


async def run_analyst(make_session, cases, answers, concurrency=3):
    from agent.run import build_message

    semaphore = asyncio.Semaphore(concurrency)

    async def one(case):
        expected = answers[case["id"]]
        if "error" in expected:
            return case["id"], {"correct": None, "reference_error": expected["error"]}, ""
        async with semaphore:
            session = make_session()
            try:
                result = await session.send(await session.new_session(), build_message(text=case["question"]))
            except Exception as exc:  # noqa: BLE001
                result = {"reply": f"error: {exc}", "tool_calls": []}
        return case["id"], grade(expected, result["tool_calls"]), result["reply"]

    return await asyncio.gather(*(one(c) for c in cases))


async def run_routing(make_session, cases=ROUTING_CASES, concurrency=3):
    from agent.run import build_message

    semaphore = asyncio.Semaphore(concurrency)

    async def one(text, expected):
        async with semaphore:
            session = make_session()
            try:
                result = await session.send(await session.new_session(), build_message(text=text))
            except Exception as exc:  # noqa: BLE001
                result = {"tool_calls": [], "reply": str(exc)}
        transfers = [c["args"].get("agent_name") for c in result["tool_calls"] if c["name"] == "transfer_to_agent"]
        return text, expected, transfers[0] if transfers else None

    return await asyncio.gather(*(one(t, e) for t, e in cases))


def summarize(analyst, routing):
    graded = [g for _, g, _ in analyst if g.get("correct") is not None]
    summary = {
        "analyst_questions": len(graded),
        "analyst_correct_pct": round(100 * sum(g["correct"] for g in graded) / len(graded), 1) if graded else None,
        "analyst_avg_queries": round(sum(g["queries"] for g in graded) / len(graded), 2) if graded else None,
        "analyst_failures": {cid: {"found": g.get("found"), "reply": reply[:300]}
                             for cid, g, reply in analyst if g.get("correct") is False},
        "reference_errors": {cid: g["reference_error"] for cid, g, _ in analyst if "reference_error" in g},
    }
    if routing:
        right = [got == expected for _, expected, got in routing]
        summary["routing_correct_pct"] = round(100 * sum(right) / len(right), 1)
        summary["routing_failures"] = [{"message": t, "expected": e, "got": g} for t, e, g in routing if g != e]
    return summary


def print_summary(summary):
    print(f"Analyst: {summary['analyst_correct_pct']}% correct over {summary['analyst_questions']} questions "
          f"(average {summary['analyst_avg_queries']} queries each)")
    for cid, failure in summary["analyst_failures"].items():
        print(f"  wrong: {cid} {failure['found']}\n    reply: {failure['reply'][:200]}")
    for cid, error in summary["reference_errors"].items():
        print(f"  reference SQL failed for {cid}: {error}")
    if "routing_correct_pct" in summary:
        print(f"Routing: {summary['routing_correct_pct']}% correct")
        for failure in summary["routing_failures"]:
            print(f"  {failure}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--model")
    parser.add_argument("--only", help="Comma-separated analyst case ids")
    parser.add_argument("--skip-routing", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Questions run at once; keep low to stay under Vertex AI rate limits")
    args = parser.parse_args(argv)

    from google.cloud import bigquery

    from agent import config
    from agent.analyst import build_analyst_agent
    from agent.cli import configure_vertex
    from agent.router import build_root_agent
    from agent.run import AgentSession
    from agent.search import ProductSearch
    from agent.sql_tools import SqlRunner

    configure_vertex(args.project, config.MODEL_LOCATION)
    model = args.model or config.AGENT_MODEL
    runner = SqlRunner(bigquery.Client(project=args.project), args.project, config.DATASET, config.BQ_LOCATION)
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only.split(","))]
    answers = reference_answers(runner, cases, f"{args.project}.{config.DATASET}")

    analyst = asyncio.run(run_analyst(lambda: AgentSession(build_analyst_agent(runner, model=model)), cases, answers,
                                      concurrency=args.concurrency))
    routing = None
    if not args.skip_routing:
        search = ProductSearch.from_environment(args.project, config.DATASET, config.BQ_LOCATION, config.MODEL_LOCATION)
        routing = asyncio.run(run_routing(lambda: AgentSession(build_root_agent(search, runner, model=model)),
                                          concurrency=args.concurrency))

    summary = summarize(analyst, routing)
    summary["model"] = model
    summary["reference_answers"] = answers
    out = RESULTS_DIR / f"analytics_eval_{model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Model: {model}")
    print_summary(summary)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
