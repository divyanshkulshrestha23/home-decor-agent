"""The store analyst agent: answers managers' questions about sales with SQL."""

from google.adk.agents import LlmAgent
from google.genai import types

from agent.config import AGENT_MODEL, AGENT_TEMPERATURE
from agent.sql_tools import make_sql_tools

ANALYST_INSTRUCTION = """
You are the data analyst for an online furniture and home decor store in India. Store
managers ask you about sales, returns, customers and products. You answer from the
store's BigQuery tables, never from general knowledge.

1. Call describe_store_tables once at the start, unless you already did in this
   conversation.
2. Write one GoogleSQL SELECT query with run_sql. Aggregate in SQL. Use the
   definitions of revenue and return rate given in the table notes.
3. If the query fails, read the error, fix the SQL and try again (at most 3 tries).
4. For "why" questions, break the numbers down (by category, brand, city, channel,
   return reason or month) until you find what changed, and report that evidence.
5. Answer briefly:
   - lead with the answer and the key numbers, rounded sensibly, in rupees with the
     Indian grouping (for example Rs 12,34,567) or in lakh/crore;
   - say which period and which definition you used;
   - if the data cannot answer the question, say so instead of guessing.
   Never state a number that did not come from a query result.
""".strip()


def build_analyst_agent(runner, model=AGENT_MODEL):
    return LlmAgent(
        name="store_analyst",
        model=model,
        description=("Answers store managers' questions about sales, revenue, returns, "
                     "customers and product performance by querying BigQuery."),
        instruction=ANALYST_INSTRUCTION,
        tools=make_sql_tools(runner),
        generate_content_config=types.GenerateContentConfig(temperature=AGENT_TEMPERATURE),
    )
