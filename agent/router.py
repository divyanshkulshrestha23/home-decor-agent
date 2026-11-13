"""The root agent: sends each message to the shopping or the analytics agent.

ADK gives an LlmAgent with sub_agents a transfer_to_agent tool; the root's
model reads the message and transfers to the sub-agent whose description
fits. The sub-agent then answers and keeps the conversation until the topic
changes.
"""

from google.adk.agents import LlmAgent

from agent.agent import build_find_similar_agent
from agent.analyst import build_analyst_agent
from agent.config import AGENT_MODEL, model_config

ROUTER_INSTRUCTION = """
You are the front desk of an online furniture and home decor store. You do not answer
questions yourself; you hand each message to the right specialist:

- find_similar: shoppers looking for products. Anything with a photo or voice note of
  furniture or decor, or a request to find, compare or recommend products.
- store_analyst: store managers asking about the business: sales, revenue, orders,
  returns, customers, cities, channels, trends, or why a number changed.

Transfer straight away when the intent is clear. If it is unclear (for example just
"hi"), say in one sentence that you can help shoppers find products or answer questions
about the store's sales, and ask which they need.
""".strip()


def build_root_agent(search, sql_runner, model=AGENT_MODEL):
    return LlmAgent(
        name="home_decor_assistant",
        model=model,
        description="Routes store messages to the shopping or analytics agent.",
        instruction=ROUTER_INSTRUCTION,
        sub_agents=[build_find_similar_agent(search, model=model),
                    build_analyst_agent(sql_runner, model=model)],
        generate_content_config=model_config(),
    )
