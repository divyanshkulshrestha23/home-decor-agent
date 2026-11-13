"""Run the agent on one shopper message (shared by the CLI, the server and tests)."""

import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "home_decor_agent"


def build_message(text=None, image=None, audio=None):
    """Build the shopper's message. image and audio are (bytes, mime_type) or None."""
    parts = []
    if image:
        parts.append(types.Part.from_bytes(data=image[0], mime_type=image[1]))
    if audio:
        parts.append(types.Part.from_bytes(data=audio[0], mime_type=audio[1]))
    if text:
        parts.append(types.Part.from_text(text=text))
    if not parts:
        raise ValueError("Send a photo, a voice note or some text.")
    return types.Content(role="user", parts=parts)


class AgentSession:
    """Holds a runner and session service so a conversation can continue."""

    def __init__(self, agent):
        self.sessions = InMemorySessionService()
        self.runner = Runner(agent=agent, app_name=APP_NAME, session_service=self.sessions)

    async def new_session(self, user_id="shopper"):
        session = await self.sessions.create_session(app_name=APP_NAME, user_id=user_id)
        return session.id

    async def send(self, session_id, message, user_id="shopper"):
        """Return {"reply": text, "tool_calls": [{"name", "args", "response"}]}."""
        reply, calls = [], {}
        async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
            for call in event.get_function_calls() or []:
                calls[call.id or str(uuid.uuid4())] = {"name": call.name, "args": dict(call.args or {}), "response": None}
            for response in event.get_function_responses() or []:
                entry = calls.get(response.id)
                if entry is None:
                    entry = calls.setdefault(response.id or str(uuid.uuid4()), {"name": response.name, "args": {}})
                entry["response"] = response.response
            if event.content and event.is_final_response():
                reply.extend(p.text for p in event.content.parts or [] if p.text)
        return {"reply": "\n".join(reply).strip(), "tool_calls": list(calls.values())}
