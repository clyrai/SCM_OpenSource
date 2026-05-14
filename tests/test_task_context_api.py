from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.integrations import memories_api


@pytest.fixture(autouse=True)
def _reset_memories_pool():
    pool = memories_api._pool
    if pool is not None:
        try:
            pool.stop()
        except Exception:
            pass
    memories_api._pool = None
    yield
    pool = memories_api._pool
    if pool is not None:
        try:
            pool.stop()
        except Exception:
            pass
    memories_api._pool = None


def test_task_context_update_get_clear_cycle():
    client = TestClient(app)
    user_id = f"ctx_{uuid.uuid4().hex[:8]}"

    up = client.post(
        f"/v1/users/{user_id}/context",
        json={
            "previous_assistant": "Which city in Switzerland are you starting from?",
            "message": "Zurich",
        },
    )
    assert up.status_code == 200, up.text
    payload = up.json()
    assert payload["ok"] is True
    assert any(s.get("key") == "origin" for s in payload["slots"])

    get1 = client.get(f"/v1/users/{user_id}/context")
    assert get1.status_code == 200
    slots = get1.json()["slots"]
    by_key = {s["key"]: s["value"] for s in slots}
    assert by_key.get("origin", "").lower() == "zurich"

    clear = client.delete(f"/v1/users/{user_id}/context")
    assert clear.status_code == 200
    assert clear.json()["ok"] is True

    get2 = client.get(f"/v1/users/{user_id}/context")
    assert get2.status_code == 200
    assert get2.json()["total"] == 0


def test_search_response_includes_task_context(monkeypatch):
    client = TestClient(app)
    user_id = f"ctx_{uuid.uuid4().hex[:8]}"

    seed = client.post(
        f"/v1/users/{user_id}/context",
        json={
            "previous_assistant": "Which city in Switzerland are you starting from?",
            "message": "Zurich",
        },
    )
    assert seed.status_code == 200

    monkeypatch.setattr(
        memories_api,
        "_invoke",
        lambda _tool, _args: {"ok": True, "memories": [], "memory_context": ""},
    )

    res = client.post(
        "/v1/memories/search",
        json={"user_id": user_id, "query": "best way to reach rome"},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert "task_context" in data
    slots = data["task_context"]["slots"]
    by_key = {s["key"]: s["value"] for s in slots}
    assert by_key.get("origin", "").lower() == "zurich"


def test_task_context_update_requires_message_or_assistant_hint():
    client = TestClient(app)
    user_id = f"ctx_{uuid.uuid4().hex[:8]}"
    res = client.post(f"/v1/users/{user_id}/context", json={})
    assert res.status_code == 400
