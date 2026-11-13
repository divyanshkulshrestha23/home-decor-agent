"""Read-only SQL access to the store's BigQuery tables, for the analyst agent.

The agent writes the SQL, so every query is checked before it runs:
1. one statement, of reasonable length;
2. a BigQuery dry run must report it as a SELECT (no INSERT, DELETE, DDL...);
3. it may only read tables in the store's own dataset;
4. it may not scan more than MAX_BYTES (the real run is also capped by
   maximum_bytes_billed, so BigQuery itself refuses anything larger).
Results are cut to MAX_ROWS rows so a large result can't flood the model.

When deployed, the service account should also have read-only BigQuery
roles; these checks are a second layer, not the only one.
"""

import datetime as dt
import decimal

from catalog.schema import PRODUCT_FIELDS
from sales.schema import TABLES as SALES_TABLES

MAX_BYTES = 1_000_000_000  # 1 GB; the whole dataset is far smaller
MAX_ROWS = 100
MAX_SQL_CHARS = 6000
MAX_QUERIES_PER_QUESTION = 6  # cost and latency cap; enforced per user message

TABLE_FIELDS = {"products": PRODUCT_FIELDS, **SALES_TABLES}

DATA_NOTES = """
- Sales run from 2025-01-01 to 2026-10-31. Treat 2026-10-31 as "today".
- Prices and revenue are in Indian rupees (INR).
- Revenue: SUM(order_items.line_total_inr) for lines that were not returned, in orders
  whose status is 'delivered'. Say so when you report revenue.
- Return rate: returned lines / lines in delivered orders.
- order_items.category and order_items.brand are copied from products, so most
  questions need only order_items (join orders for city, channel or status).
- Product sizes are in cm; longest_side_cm is the width a shopper means.
""".strip()


def describe_tables(project, dataset):
    """Text description of every table and column, for the agent."""
    lines = [f"Dataset `{project}.{dataset}` (use fully qualified table names).", ""]
    for table, fields in TABLE_FIELDS.items():
        lines.append(f"Table `{project}.{dataset}.{table}`:")
        lines.extend(f"  - {name} {kind}{' (list)' if mode == 'REPEATED' else ''}: {desc}"
                     for name, kind, mode, desc in fields)
        lines.append("")
    lines.append("Notes:")
    lines.append(DATA_NOTES)
    return "\n".join(lines)


def _json_safe(value):
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def single_statement(sql):
    """Reject empty, overlong or multi-statement SQL. Returns cleaned SQL or raises ValueError."""
    cleaned = (sql or "").strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("The query is empty.")
    if len(cleaned) > MAX_SQL_CHARS:
        raise ValueError(f"The query is longer than {MAX_SQL_CHARS} characters.")
    if ";" in cleaned:
        raise ValueError("Send one statement at a time (no ';').")
    return cleaned


class SqlRunner:
    """Validates and runs agent-written SQL. The BigQuery client is injected for tests."""

    def __init__(self, bq_client, project, dataset, location):
        self.bq = bq_client
        self.project = project
        self.dataset = dataset
        self.location = location

    def _config(self, **kwargs):
        from google.cloud import bigquery

        return bigquery.QueryJobConfig(use_query_cache=False, **kwargs)

    def check(self, sql):
        """Dry-run the query. Returns (cleaned_sql, bytes) or raises ValueError with a reason."""
        cleaned = single_statement(sql)
        job = self.bq.query(cleaned, job_config=self._config(dry_run=True), location=self.location)
        if job.statement_type != "SELECT":
            raise ValueError(f"Only SELECT queries are allowed (this is {job.statement_type}).")
        for table in job.referenced_tables or []:
            if (table.project, table.dataset_id) != (self.project, self.dataset):
                raise ValueError(f"Only tables in {self.project}.{self.dataset} may be queried "
                                 f"(found {table.project}.{table.dataset_id}.{table.table_id}).")
        scanned = job.total_bytes_processed or 0
        if scanned > MAX_BYTES:
            raise ValueError(f"The query would scan {scanned:,} bytes; the limit is {MAX_BYTES:,}.")
        return cleaned, scanned

    def run(self, sql):
        """Return a dict with columns and rows, or {"error": ...} the agent can act on."""
        try:
            cleaned, scanned = self.check(sql)
            job = self.bq.query(cleaned, job_config=self._config(maximum_bytes_billed=MAX_BYTES),
                                location=self.location)
            rows = []
            truncated = False
            columns = None
            for row in job.result(max_results=MAX_ROWS + 1):
                if columns is None:
                    columns = list(row.keys())
                if len(rows) == MAX_ROWS:
                    truncated = True
                    break
                rows.append({k: _json_safe(v) for k, v in row.items()})
            return {"columns": columns or [], "rows": rows, "row_count": len(rows),
                    "truncated": truncated, "bytes_scanned": scanned}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - BigQuery errors go back to the agent to fix
            message = getattr(exc, "message", None) or str(exc)
            return {"error": f"BigQuery error: {message[:500]}"}


def make_sql_tools(runner):
    """Build the two tools the analyst agent uses."""

    def describe_store_tables() -> str:
        """Describe the store's tables and columns, and how to compute revenue and return rate.

        Call this before writing your first query.
        """
        return describe_tables(runner.project, runner.dataset)

    def run_sql(sql: str, tool_context=None) -> dict:
        """Run one read-only BigQuery SELECT query on the store's tables.

        Use fully qualified table names. Aggregate in SQL (SUM, COUNT, GROUP BY) rather
        than fetching raw rows; at most 100 rows are returned. At most 6 queries per
        question.

        Args:
          sql: A single GoogleSQL SELECT statement.
        """
        if tool_context is not None:
            key = f"sql_queries:{tool_context.invocation_id}"
            used = tool_context.state.get(key, 0)
            if used >= MAX_QUERIES_PER_QUESTION:
                return {"error": f"Query limit reached ({MAX_QUERIES_PER_QUESTION} per question). "
                                 "Answer now with the results you already have."}
            tool_context.state[key] = used + 1
        return runner.run(sql)

    return [describe_store_tables, run_sql]
