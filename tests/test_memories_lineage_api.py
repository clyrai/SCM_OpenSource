from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.integrations import memories_api


class _FakeLTM:
    def __init__(self) -> None:
        self.last_memory_id = None

    def get_lineage(self, memory_id: str):
        self.last_memory_id = memory_id
        if memory_id != "mem-123":
            return {}
        return {
            "memory_id": memory_id,
            "version_root": "root-1",
            "current_id": "mem-123",
            "version_count": 2,
            "conflict_count": 1,
            "versions": [
                {"id": "mem-old", "is_current_version": False},
                {"id": "mem-123", "is_current_version": True},
            ],
            "conflicts": [
                {"from": "mem-old", "to": "mem-123", "predicate": "contradicts"},
            ],
        }


class _FakeEngine:
    def __init__(self, ltm: _FakeLTM) -> None:
        self.long_term_memory = ltm


class _FakePool:
    def __init__(self, ltm: _FakeLTM) -> None:
        self._ltm = ltm
        self.calls = []

    def get_or_create(self, user_id: str, bump_activity: bool = False):
        self.calls.append((user_id, bump_activity))
        return _FakeEngine(self._ltm)


@pytest.fixture(autouse=True)
def _reset_memories_pool():
    old = memories_api._pool
    memories_api._pool = None
    yield
    memories_api._pool = old


def test_memory_lineage_endpoint_returns_payload(monkeypatch):
    ltm = _FakeLTM()
    pool = _FakePool(ltm)
    monkeypatch.setattr(memories_api, "get_pool", lambda: pool)

    client = TestClient(app)
    res = client.get("/v1/memories/mem-123/lineage", params={"user_id": "alice"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert body["user_id"] == "alice"
    assert body["memory_id"] == "mem-123"
    assert body["lineage"]["version_count"] == 2
    assert body["lineage"]["conflict_count"] == 1
    assert pool.calls == [("alice", False)]
    assert ltm.last_memory_id == "mem-123"


def test_memory_lineage_endpoint_returns_404_for_unknown_memory(monkeypatch):
    ltm = _FakeLTM()
    pool = _FakePool(ltm)
    monkeypatch.setattr(memories_api, "get_pool", lambda: pool)

    client = TestClient(app)
    res = client.get("/v1/memories/missing/lineage")
    assert res.status_code == 404
    assert "memory not found" in res.json()["detail"]
