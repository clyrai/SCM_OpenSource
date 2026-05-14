from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import chat_router
from src.api.main import app


class _AgentMessage:
    def __init__(self, content: str, tool_calls: list[dict] | None = None):
        self.type = "ai"
        self.content = content
        self.tool_calls = tool_calls or []


def _extract_name(descriptions: list[str]) -> str | None:
    for desc in descriptions:
        match = re.search(r"\b(?:my name is|person:\s*)([A-Za-z][A-Za-z'-]+)\b", desc, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


class _StubAgent:
    def __init__(self, scm_client):
        self.scm = scm_client

    def _partial_store(self, user_text: str) -> str:
        text = "I have a milk allergy." if "milk" in user_text.lower() else "I have a peanut allergy."
        self.scm.add_memory(text=text, replaces_prior=False)
        return text

    def _name_reply(self) -> str:
        facts = self.scm.list_facts().get("facts") or []
        descriptions = [(f.get("description") or "").strip() for f in facts]
        name = _extract_name(descriptions)
        if not name:
            return "I don't have your name yet."
        return f"Your name is {name}."

    def invoke(self, payload, *_args, **_kwargs):
        user_text = payload["messages"][-1]["content"]
        if "what is my name" in user_text.lower():
            reply = self._name_reply()
            return {
                "messages": [
                    _AgentMessage(
                        content=reply,
                        tool_calls=[{"name": "get_user_profile", "args": {}}],
                    ),
                ],
            }
        partial = self._partial_store(user_text)
        return {
            "messages": [
                _AgentMessage(
                    content="Noted.",
                    tool_calls=[{"name": "add_memory", "args": {"text": partial}}],
                ),
            ],
        }

    async def astream_events(self, payload, *_args, **_kwargs):
        user_text = payload["messages"][-1]["content"]
        lower = user_text.lower()
        if "what is my name" in lower:
            yield {"event": "on_tool_start", "name": "get_user_profile", "data": {"input": {}}}
            yield {"event": "on_tool_end", "name": "get_user_profile", "data": {}}
            reply = self._name_reply()
        else:
            partial = self._partial_store(user_text)
            yield {
                "event": "on_tool_start",
                "name": "add_memory",
                "data": {"input": {"text": partial}},
            }
            yield {"event": "on_tool_end", "name": "add_memory", "data": {}}
            reply = "Noted."
        for token in reply.split():
            yield {
                "event": "on_chat_model_stream",
                "name": "chat_model",
                "tags": [],
                "data": {"chunk": SimpleNamespace(content=f"{token} ")},
            }


def _patch_agent_builders(monkeypatch):
    monkeypatch.setattr(chat_router, "_build_llm", lambda *_a, **_k: object())
    monkeypatch.setattr(chat_router, "_build_agent", lambda llm, scm_client: _StubAgent(scm_client))
    monkeypatch.setattr(chat_router, "_BYOKLLM", lambda *_a, **_k: chat_router._NoOpLLM())


def _base_payload(message: str) -> dict:
    return {
        "message": message,
        "llm_provider": "deepseek",
        "llm_api_key": "sk-test",
        "llm_model": "deepseek-chat",
    }


def test_chat_message_end_to_end_keeps_profile_after_partial_tool_calls(monkeypatch):
    _patch_agent_builders(monkeypatch)
    client = TestClient(app)
    slug = f"e2e_msg_{uuid.uuid4().hex[:8]}"

    intro = "Hi, I'm Alex. I'm a backend engineer in Lisbon, and I have a peanut allergy."
    correction = "Sorry it was not peanut but milk lol my bad"

    first = client.post(f"/chat/api/message/{slug}", json=_base_payload(intro))
    assert first.status_code == 200
    assert first.json()["tools_called"].count("add_memory") >= 2

    second = client.post(f"/chat/api/message/{slug}", json=_base_payload(correction))
    assert second.status_code == 200

    profile = client.get(f"/chat/api/profile/{slug}")
    assert profile.status_code == 200
    descriptions = [c["description"].lower() for c in profile.json()["concepts"]]
    assert any("alex" in d for d in descriptions), descriptions

    ask_name = client.post(f"/chat/api/message/{slug}", json=_base_payload("what is my name"))
    assert ask_name.status_code == 200
    assert "alex" in ask_name.json()["reply"].lower()


def test_chat_stream_end_to_end_keeps_profile_after_partial_tool_calls(monkeypatch):
    _patch_agent_builders(monkeypatch)
    client = TestClient(app)
    slug = f"e2e_stream_{uuid.uuid4().hex[:8]}"

    intro = "Hi, I'm Alex. I'm a backend engineer in Lisbon, and I have a peanut allergy."
    stream = client.post(f"/chat/api/stream/{slug}", json=_base_payload(intro))
    assert stream.status_code == 200

    events = []
    for line in stream.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    done = [e for e in events if e.get("type") == "done"]
    assert done, events
    assert done[-1]["tools_called"].count("add_memory") >= 2

    profile = client.get(f"/chat/api/profile/{slug}")
    assert profile.status_code == 200
    descriptions = [c["description"].lower() for c in profile.json()["concepts"]]
    assert any("alex" in d for d in descriptions), descriptions
