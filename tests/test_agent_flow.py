"""The agent loop end to end, with a scripted model and a fake search (no network)."""

from typing import AsyncGenerator

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agent.agent import build_find_similar_agent
from agent.run import AgentSession, build_message


class ScriptedLlm(BaseLlm):
    """Calls find_similar_products with fixed args, then summarises the first result."""

    model: str = "scripted"
    tool_args: dict = {}

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        responses = [
            part.function_response.response
            for content in llm_request.contents
            for part in content.parts or []
            if part.function_response
        ]
        if responses:
            last = responses[-1]
            if last.get("error"):
                text = f"Problem: {last['error']}"
            elif last.get("results"):
                top = last["results"][0]
                text = f"Try {top['name']} ({top['product_id']}) at Rs {top['price_inr']}."
            else:
                text = "Nothing matched."
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
            return
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(name="find_similar_products", args=self.tool_args))]))


class FakeSearch:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, filters, image_bytes=None, image_mime=None, text=None, top_k=5):
        self.calls.append({"filters": filters, "image": image_bytes, "mime": image_mime, "text": text})
        return self.results


PRODUCT = {
    "product_id": "B0SOFA", "name": "Navy Velvet Sofa", "brand": "Rivet", "category": "sofa",
    "price_inr": 18999, "colors": ["Navy"], "materials": ["Velvet"], "size_cm_w_d_h": "180 x 86 x 80",
    "similarity": 0.87, "image_path": "aa/x.jpg",
}


async def run(tool_args, message, results=(PRODUCT,)):
    search = FakeSearch(list(results))
    session = AgentSession(build_find_similar_agent(search, model=ScriptedLlm(tool_args=tool_args)))
    session_id = await session.new_session()
    return await session.send(session_id, message), search


@pytest.mark.asyncio
async def test_photo_and_voice_reach_the_search():
    message = build_message(image=(b"\xff\xd8photo", "image/jpeg"), audio=(b"voice", "audio/m4a"))
    args = {"description": "sofa like this in navy", "categories": ["sofa"], "max_price_inr": 20000}
    result, search = await run(args, message)

    call = search.calls[0]
    assert call["image"] == b"\xff\xd8photo" and call["mime"] == "image/jpeg"
    assert call["text"] == "sofa like this in navy"
    assert call["filters"].categories == ["sofa"] and call["filters"].max_price_inr == 20000
    assert "Navy Velvet Sofa (B0SOFA)" in result["reply"]
    tool_call = result["tool_calls"][0]
    assert tool_call["name"] == "find_similar_products"
    assert tool_call["response"]["used_photo"] is True
    assert tool_call["response"]["filters_applied"] == {"categories": ["sofa"], "max_price_inr": 20000}


@pytest.mark.asyncio
async def test_text_only_search_has_no_photo():
    result, search = await run({"description": "jute rug"}, build_message(text="a jute rug please"))
    assert search.calls[0]["image"] is None
    assert result["tool_calls"][0]["response"]["used_photo"] is False


@pytest.mark.asyncio
async def test_unknown_category_is_rejected_without_searching():
    result, search = await run({"description": "x", "categories": ["spaceship"]}, build_message(text="x"))
    assert search.calls == []
    assert "Unknown categories ['spaceship']" in result["reply"]


@pytest.mark.asyncio
async def test_no_results_adds_a_note():
    result, _ = await run({"description": "x", "max_width_cm": 10}, build_message(text="tiny sofa"), results=())
    response = result["tool_calls"][0]["response"]
    assert response["results"] == [] and "relaxing" in response["note"]
    assert result["reply"] == "Nothing matched."


class FailingSearch:
    def search(self, *args, **kwargs):
        raise RuntimeError("400 Unsupported query pattern")


@pytest.mark.asyncio
async def test_search_failure_is_reported_not_raised():
    session = AgentSession(build_find_similar_agent(FailingSearch(), model=ScriptedLlm(tool_args={"description": "x"})))
    session_id = await session.new_session()
    result = await session.send(session_id, build_message(text="a sofa"))
    assert result["tool_calls"][0]["response"]["error"].startswith("The product search failed (RuntimeError)")
    assert result["reply"].startswith("Problem:")


@pytest.mark.asyncio
async def test_plural_and_spaced_categories_are_normalized():
    _, search = await run({"description": "x", "categories": ["Rugs", "small storage", "rug"]}, build_message(text="x"))
    assert search.calls[0]["filters"].categories == ["rug", "small_storage"]


def test_agent_uses_low_temperature():
    agent = build_find_similar_agent(FakeSearch([]), model=ScriptedLlm())
    assert agent.generate_content_config.temperature == 0.2
    retry = agent.generate_content_config.http_options.retry_options
    assert retry.attempts == 5 and 429 in retry.http_status_codes


def test_build_message_needs_something():
    with pytest.raises(ValueError):
        build_message()
    parts = build_message(text="hi", image=(b"i", "image/png"), audio=(b"a", "audio/wav")).parts
    assert [p.inline_data.mime_type if p.inline_data else p.text for p in parts] == ["image/png", "audio/wav", "hi"]
