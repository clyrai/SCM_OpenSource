from __future__ import annotations

import json
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


def _context_blob(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            parts.append(str(m.get("content") or ""))
    return "\n".join(parts)


class _TaskContextStubAgent:
    def _reply(self, payload) -> str:
        messages = payload["messages"]
        user_text = str(messages[-1]["content"])
        lower = user_text.lower().strip()
        context = _context_blob(messages).lower()

        if "reach rome from swiss" in lower:
            return "Great question. Which city in Switzerland are you starting from?"
        if lower == "zurich":
            return "From Zurich, your best options are plane or train."
        if "which is better though" in lower:
            if "origin: zurich" in context:
                return "From Zurich, flying is usually faster while train is more scenic."
            return "It depends — which city are you starting from?"
        return "Got it."

    def invoke(self, payload, *_args, **_kwargs):
        reply = self._reply(payload)
        return {"messages": [_AgentMessage(content=reply, tool_calls=[])]}

    async def astream_events(self, payload, *_args, **_kwargs):
        reply = self._reply(payload)
        for token in reply.split():
            yield {
                "event": "on_chat_model_stream",
                "name": "chat_model",
                "tags": [],
                "data": {"chunk": SimpleNamespace(content=f"{token} ")},
            }


def _patch_agent_builders(monkeypatch):
    monkeypatch.setattr(chat_router, "_build_llm", lambda *_a, **_k: object())
    monkeypatch.setattr(
        chat_router,
        "_build_agent",
        lambda llm, scm_client: _TaskContextStubAgent(),
    )
    monkeypatch.setattr(chat_router, "_BYOKLLM", lambda *_a, **_k: chat_router._NoOpLLM())


def _base_payload(message: str) -> dict:
    return {
        "message": message,
        "llm_provider": "deepseek",
        "llm_api_key": "sk-test",
        "llm_model": "deepseek-chat",
    }


def test_task_context_captures_short_followup_without_fact_pollution(monkeypatch):
    _patch_agent_builders(monkeypatch)
    client = TestClient(app)
    slug = f"ctx_msg_{uuid.uuid4().hex[:8]}"

    q1 = client.post(
        f"/chat/api/message/{slug}",
        json=_base_payload("so how can I reach Rome from Swiss"),
    )
    assert q1.status_code == 200

    q2 = client.post(
        f"/chat/api/message/{slug}",
        json=_base_payload("Zurich"),
    )
    assert q2.status_code == 200

    ctx = client.get(f"/chat/api/context/{slug}")
    assert ctx.status_code == 200
    slots = {s["key"]: s["value"] for s in ctx.json()["slots"]}
    assert slots.get("origin", "").lower() == "zurich"

    profile = client.get(f"/chat/api/profile/{slug}")
    assert profile.status_code == 200
    descriptions = [c["description"].lower() for c in profile.json()["concepts"]]
    assert not any("zurich" in d for d in descriptions), descriptions

    q3 = client.post(
        f"/chat/api/message/{slug}",
        json=_base_payload("which is better though"),
    )
    assert q3.status_code == 200
    assert "zurich" in q3.json()["reply"].lower()


def test_task_context_updates_via_stream_route(monkeypatch):
    _patch_agent_builders(monkeypatch)
    client = TestClient(app)
    slug = f"ctx_stream_{uuid.uuid4().hex[:8]}"

    seed = client.post(
        f"/chat/api/message/{slug}",
        json=_base_payload("so how can I reach Rome from Swiss"),
    )
    assert seed.status_code == 200

    stream = client.post(
        f"/chat/api/stream/{slug}",
        json=_base_payload("Zurich"),
    )
    assert stream.status_code == 200

    events = []
    for line in stream.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    done = [e for e in events if e.get("type") == "done"]
    assert done, events

    ctx = client.get(f"/chat/api/context/{slug}")
    assert ctx.status_code == 200
    slots = {s["key"]: s["value"] for s in ctx.json()["slots"]}
    assert slots.get("origin", "").lower() == "zurich"
