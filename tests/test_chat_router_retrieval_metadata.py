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


class _SearchMetadataStubAgent:
    def __init__(self, scm_client):
        self.scm = scm_client

    def invoke(self, payload, *_args, **_kwargs):
        user_text = payload["messages"][-1]["content"]
        lower = str(user_text).lower()
        if "where do i work" in lower:
            result = self.scm.search_memory(query="where do i work", limit=5)
            memories = result.get("memories") or []
            top = (memories[0].get("description") if memories else "I need one more detail.")
            return {
                "messages": [
                    _AgentMessage(
                        content=f"From memory: {top}",
                        tool_calls=[{"name": "search_memory", "args": {"query": "where do i work"}}],
                    ),
                ],
            }
        self.scm.add_memory(text=user_text, replaces_prior=False)
        return {
            "messages": [
                _AgentMessage(
                    content="Saved.",
                    tool_calls=[{"name": "add_memory", "args": {"text": user_text}}],
                ),
            ],
        }

    async def astream_events(self, payload, *_args, **_kwargs):
        user_text = payload["messages"][-1]["content"]
        lower = str(user_text).lower()
        if "where do i work" in lower:
            yield {
                "event": "on_tool_start",
                "name": "search_memory",
                "data": {"input": {"query": "where do i work"}},
            }
            result = self.scm.search_memory(query="where do i work", limit=5)
            yield {"event": "on_tool_end", "name": "search_memory", "data": {"output": result}}
            memories = result.get("memories") or []
            reply = f"From memory: {(memories[0].get('description') if memories else 'unknown')}"
        else:
            yield {
                "event": "on_tool_start",
                "name": "add_memory",
                "data": {"input": {"text": str(user_text)}},
            }
            self.scm.add_memory(text=str(user_text), replaces_prior=False)
            yield {"event": "on_tool_end", "name": "add_memory", "data": {}}
            reply = "Saved."

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
        lambda llm, scm_client: _SearchMetadataStubAgent(scm_client),
    )
    monkeypatch.setattr(chat_router, "_BYOKLLM", lambda *_a, **_k: chat_router._NoOpLLM())


def _base_payload(message: str) -> dict:
    return {
        "message": message,
        "llm_provider": "deepseek",
        "llm_api_key": "sk-test",
        "llm_model": "deepseek-chat",
    }


def test_chat_message_returns_retrieval_snapshot_and_endpoint(monkeypatch):
    _patch_agent_builders(monkeypatch)
    client = TestClient(app)
    slug = f"retrieval_msg_{uuid.uuid4().hex[:8]}"

    seed = client.post(
        f"/chat/api/message/{slug}",
        json=_base_payload("I work at Atlas Labs in Zurich."),
    )
    assert seed.status_code == 200

    ask = client.post(
        f"/chat/api/message/{slug}",
        json=_base_payload("where do i work?"),
    )
    assert ask.status_code == 200
    body = ask.json()
    assert "retrieval" in body
    assert body["retrieval"]["fusion_mode"] == "weighted_rrf"
    assert "confidence" in body["retrieval"]

    retrieval = client.get(f"/chat/api/retrieval/{slug}")
    assert retrieval.status_code == 200
    payload = retrieval.json()
    assert payload["available"] is True
    assert payload["retrieval"]["fusion_mode"] == "weighted_rrf"
    assert "updated_at" in payload["retrieval"]


def test_chat_stream_done_event_includes_retrieval_snapshot(monkeypatch):
    _patch_agent_builders(monkeypatch)
    client = TestClient(app)
    slug = f"retrieval_stream_{uuid.uuid4().hex[:8]}"

    seed = client.post(
        f"/chat/api/message/{slug}",
        json=_base_payload("I work at Atlas Labs in Zurich."),
    )
    assert seed.status_code == 200

    stream = client.post(
        f"/chat/api/stream/{slug}",
        json=_base_payload("where do i work?"),
    )
    assert stream.status_code == 200

    events = []
    for line in stream.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    done = [e for e in events if e.get("type") == "done"]
    assert done, events
    assert "retrieval" in done[-1]
    assert done[-1]["retrieval"]["fusion_mode"] == "weighted_rrf"
