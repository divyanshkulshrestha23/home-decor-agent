"""Analyst SQL safety checks and the routed multi-agent flow, without network calls."""

import datetime as dt
import decimal
from typing import AsyncGenerator

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agent.router import build_root_agent
from agent.run import AgentSession, build_message
from agent.sql_tools import MAX_BYTES, MAX_ROWS, SqlRunner, describe_tables, single_statement


class TableRef:
    def __init__(self, project, dataset, table):
        self.project, self.dataset_id, self.table_id = project, dataset, table


class FakeRow(dict):
    pass


class FakeBigQuery:
    """Dry runs return the configured statement type, tables and bytes; real runs return rows."""

    def __init__(self, statement="SELECT", tables=(("p", "d", "order_items"),), scanned=1000, rows=(), error=None):
        self.statement, self.tables, self.scanned = statement, tables, scanned
        self.rows, self.error = list(rows), error
        self.calls = []

    def query(self, sql, job_config, location):
        self.calls.append((sql, job_config))
        fake = self

        class Job:
            statement_type = fake.statement
            referenced_tables = [TableRef(*t) for t in fake.tables]
            total_bytes_processed = fake.scanned

            def result(self, max_results=None):
                if fake.error:
                    raise fake.error
                return [FakeRow(r) for r in fake.rows[:max_results]]

        return Job()


def runner(**kwargs):
    return SqlRunner(FakeBigQuery(**kwargs), "p", "d", "asia-south1")


def test_select_runs_with_byte_cap_and_safe_values():
    r = runner(rows=[{"month": dt.date(2026, 1, 1), "revenue": decimal.Decimal("123.5"), "share": 0.123456}])
    result = r.run("SELECT 1;")
    assert result == {"columns": ["month", "revenue", "share"],
                      "rows": [{"month": "2026-01-01", "revenue": 123.5, "share": 0.1235}],
                      "row_count": 1, "truncated": False, "bytes_scanned": 1000}
    dry, real = r.bq.calls
    assert dry[1].dry_run is True and dry[0] == "SELECT 1"
    assert real[1].maximum_bytes_billed == MAX_BYTES


@pytest.mark.parametrize(
    "kwargs,sql,message",
    [
        ({"statement": "DELETE"}, "DELETE FROM x WHERE 1=1", "Only SELECT queries"),
        ({"tables": (("other", "secret", "users"),)}, "SELECT 1", "Only tables in p.d"),
        ({"scanned": MAX_BYTES + 1}, "SELECT 1", "would scan"),
        ({}, "SELECT 1; DROP TABLE x", "one statement"),
        ({}, "   ", "empty"),
        ({}, "SELECT " + "1," * 4000 + "1", "longer than"),
    ],
)
def test_unsafe_queries_are_refused_before_running(kwargs, sql, message):
    r = runner(**kwargs)
    result = r.run(sql)
    assert message in result["error"]
    assert all(call[1].dry_run for call in r.bq.calls)  # never ran for real


def test_results_are_truncated():
    rows = [{"n": i} for i in range(MAX_ROWS + 5)]
    result = runner(rows=rows).run("SELECT n FROM t")
    assert result["row_count"] == MAX_ROWS and result["truncated"] is True


def test_bigquery_errors_go_back_to_the_agent():
    error = Exception("Unrecognized name: revenu")
    result = runner(error=error).run("SELECT revenu FROM t")
    assert result["error"].startswith("BigQuery error: Unrecognized name")


def test_single_statement_strips_trailing_semicolon():
    assert single_statement(" SELECT 1 ; ") == "SELECT 1"


def test_describe_tables_lists_every_table_and_notes():
    text = describe_tables("p", "d")
    for table in ("products", "customers", "orders", "order_items"):
        assert f"`p.d.{table}`" in text
    assert "line_total_inr" in text and "Revenue:" in text


# --- routing ----------------------------------------------------------------

class RoutingLlm(BaseLlm):
    """Root transfers by keyword; the analyst queries once and reports; find_similar searches."""

    model: str = "scripted"

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        agent = llm_request.config.system_instruction or ""
        user = next(p.text for c in llm_request.contents if c.role == "user" for p in c.parts or [] if p.text)
        responses = [p.function_response for c in llm_request.contents for p in c.parts or [] if p.function_response]

        def call(name, args):
            return LlmResponse(content=types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(name=name, args=args))]))

        def say(text):
            return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))

        if "front desk" in str(agent):
            target = "store_analyst" if "revenue" in user else "find_similar"
            yield call("transfer_to_agent", {"agent_name": target})
        elif "data analyst" in str(agent):
            last = responses[-1] if responses else None
            if last is None or last.name == "transfer_to_agent":
                yield call("run_sql", {"sql": "SELECT 42 AS revenue"})
            else:
                yield say(f"Revenue was {last.response['rows'][0]['revenue']}.")
        else:
            last = responses[-1] if responses else None
            if last is None or last.name == "transfer_to_agent":
                yield call("find_similar_products", {"description": user})
            else:
                yield say("Found products.")


class FakeSearch:
    def __init__(self):
        self.calls = []

    def search(self, filters, image_bytes=None, image_mime=None, text=None, top_k=5):
        self.calls.append(text)
        return []


@pytest.mark.asyncio
async def test_router_sends_manager_questions_to_the_analyst():
    sql = runner(rows=[{"revenue": 42}])
    root = build_root_agent(FakeSearch(), sql, model=RoutingLlm())
    session = AgentSession(root)
    result = await session.send(await session.new_session(), build_message(text="What was revenue last month?"))
    names = [c["name"] for c in result["tool_calls"]]
    assert names == ["transfer_to_agent", "run_sql"]
    assert result["reply"].endswith("Revenue was 42.")


@pytest.mark.asyncio
async def test_router_sends_shoppers_to_find_similar():
    search = FakeSearch()
    root = build_root_agent(search, runner(), model=RoutingLlm())
    session = AgentSession(root)
    result = await session.send(await session.new_session(), build_message(text="a jute rug"))
    assert [c["name"] for c in result["tool_calls"]] == ["transfer_to_agent", "find_similar_products"]
    assert search.calls == ["a jute rug"]


class FakeToolContext:
    def __init__(self, invocation_id="inv-1"):
        self.invocation_id = invocation_id
        self.state = {}


def test_run_sql_tool_caps_queries_per_question():
    from agent.sql_tools import MAX_QUERIES_PER_QUESTION, make_sql_tools

    _, run_sql = make_sql_tools(runner(rows=[{"n": 1}]))
    context = FakeToolContext()
    results = [run_sql("SELECT 1", tool_context=context) for _ in range(MAX_QUERIES_PER_QUESTION + 1)]
    assert all("rows" in r for r in results[:-1])
    assert results[-1]["error"].startswith("Query limit reached")
    context.invocation_id = "inv-2"  # a new question gets a fresh allowance
    assert "rows" in run_sql("SELECT 1", tool_context=context)
