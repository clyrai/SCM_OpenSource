"""
Tests for Phase 7 CrossSessionMemoryPool.

The pool is what gives SCM continuity across days. These tests are the
contract for what the pool must guarantee:

  - Disabled by default → returns [] (preserves legacy behavior).
  - When enabled, pulls episodes from prior sessions only — never from
    the current session.
  - Honors per-session and total-borrowed caps.
  - Honors the look-back time window.
  - Tags borrowed episodes with `_origin_session` and `_borrowed=True`.
  - Survives missing tables / empty DB / SQLite exceptions gracefully.
  - Hands borrowed Episodes to the existing sleep cycle without breaking it.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.cross_session_pool import (
    CrossSessionMemoryPool,
    CrossSessionPoolConfig,
)
from src.core.models import Episode, ImportanceVector, MemoryState
from src.core.time_utils import utc_now


# ─── Fake SQLite layer ──────────────────────────────────────────────────────


class FakeSQLite:
    """In-memory stand-in that matches the real sqlite_db API used by the pool."""

    def __init__(self):
        self.sessions: List[Dict[str, str]] = []  # [{session_id, last_active}]
        self.episodes: List[Dict[str, Any]] = []  # rows as the real DB would return

    def add_session(self, session_id: str, last_active: datetime):
        self.sessions.append({
            "session_id": session_id,
            "last_active": last_active.isoformat(),
        })

    def add_episode(
        self,
        session_id: str,
        ts: datetime,
        content: str,
        ep_id: Optional[str] = None,
        source: str = "user",
        concept_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        import json
        ep_id = ep_id or f"ep_{len(self.episodes)}"
        self.episodes.append({
            "id": ep_id,
            "timestamp": ts.isoformat(),
            "session_id": session_id,
            "raw_content": content,
            "concept_ids": json.dumps(concept_ids or []),
            "context": json.dumps(context or {}),
            "importance_json": json.dumps({}),
            "state": "active",
            "source": source,
        })

    # API shape expected by CrossSessionMemoryPool

    def list_recent_session_ids(
        self, limit: int = 5, since_iso: Optional[str] = None
    ) -> List[str]:
        rows = sorted(self.sessions, key=lambda r: r["last_active"], reverse=True)
        if since_iso:
            rows = [r for r in rows if r["last_active"] >= since_iso]
        return [r["session_id"] for r in rows[:limit]]

    def get_recent_episodes_for_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        since_iso: Optional[str] = None,
        max_per_session: int = 50,
        max_total: int = 200,
    ) -> List[Dict[str, Any]]:
        rows = list(self.episodes)
        if since_iso:
            rows = [r for r in rows if r["timestamp"] >= since_iso]
        if session_ids is not None:
            sid_set = set(session_ids)
            rows = [r for r in rows if r["session_id"] in sid_set]
            # apply per-session cap
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for r in sorted(rows, key=lambda x: x["timestamp"], reverse=True):
                grouped.setdefault(r["session_id"], []).append(r)
            capped: List[Dict[str, Any]] = []
            for sid, eps in grouped.items():
                capped.extend(eps[:max_per_session])
            capped.sort(key=lambda r: r["timestamp"], reverse=True)
            return capped[:max_total]
        rows.sort(key=lambda r: r["timestamp"], reverse=True)
        return rows[:max_total]

    def load_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        for s in self.sessions:
            if s["session_id"] == session_id:
                return dict(s)
        return None


def _make_pool(
    fake: FakeSQLite,
    current: str = "current_session",
    **cfg_kwargs,
) -> CrossSessionMemoryPool:
    cfg = CrossSessionPoolConfig(enabled=True, **cfg_kwargs)
    return CrossSessionMemoryPool(
        current_session_id=current,
        sqlite_factory=lambda: fake,
        config=cfg,
    )


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_disabled_returns_empty():
    fake = FakeSQLite()
    fake.add_session("other", utc_now())
    fake.add_episode("other", utc_now(), "should not be returned")
    pool = CrossSessionMemoryPool(
        current_session_id="me",
        sqlite_factory=lambda: fake,
        config=CrossSessionPoolConfig(enabled=False),
    )
    assert pool.gather() == []


def test_excludes_current_session():
    """Pool must never pull episodes from the current session — those come
    from working memory directly."""
    fake = FakeSQLite()
    now = utc_now()
    fake.add_session("me", now)
    fake.add_episode("me", now - timedelta(minutes=1), "current session content")
    fake.add_session("yesterday", now - timedelta(hours=24))
    fake.add_episode("yesterday", now - timedelta(hours=24), "prior content")

    pool = _make_pool(fake, current="me")
    borrowed = pool.gather()
    assert len(borrowed) == 1
    assert borrowed[0].raw_content == "prior content"
    assert borrowed[0].context.get("_origin_session") == "yesterday"
    assert borrowed[0].context.get("_borrowed") is True


def test_honors_lookback_hours():
    fake = FakeSQLite()
    now = utc_now()
    fake.add_session("recent", now - timedelta(hours=2))
    fake.add_episode("recent", now - timedelta(hours=2), "recent content")
    fake.add_session("ancient", now - timedelta(days=30))
    fake.add_episode("ancient", now - timedelta(days=30), "old content")

    # Only look back 24 hours
    pool = _make_pool(fake, current="me", lookback_hours=24.0)
    borrowed = pool.gather(now=now)
    assert len(borrowed) == 1
    assert borrowed[0].raw_content == "recent content"


def test_honors_max_sessions_cap():
    """If 10 sessions exist but config caps at 2, only 2 are pulled."""
    fake = FakeSQLite()
    now = utc_now()
    for i in range(10):
        sid = f"sess_{i}"
        ts = now - timedelta(hours=i + 1)
        fake.add_session(sid, ts)
        fake.add_episode(sid, ts, f"content_{i}")

    pool = _make_pool(fake, current="me", max_sessions=2, max_total_borrowed=100)
    borrowed = pool.gather(now=now)
    # Most recent 2 sessions only
    seen_sids = {ep.context["_origin_session"] for ep in borrowed}
    assert seen_sids == {"sess_0", "sess_1"}


def test_honors_max_episodes_per_session():
    fake = FakeSQLite()
    now = utc_now()
    fake.add_session("chatty", now - timedelta(hours=1))
    for i in range(50):
        fake.add_episode("chatty", now - timedelta(hours=1, seconds=i), f"msg_{i}")

    pool = _make_pool(fake, current="me", max_episodes_per_session=5, max_total_borrowed=100)
    borrowed = pool.gather(now=now)
    assert len(borrowed) == 5


def test_honors_max_total_borrowed():
    fake = FakeSQLite()
    now = utc_now()
    for s in range(5):
        sid = f"s_{s}"
        fake.add_session(sid, now - timedelta(hours=s + 1))
        for i in range(20):
            fake.add_episode(sid, now - timedelta(hours=s + 1, seconds=i), f"msg_{s}_{i}")

    pool = _make_pool(fake, current="me", max_total_borrowed=12)
    borrowed = pool.gather(now=now)
    assert len(borrowed) == 12


def test_borrowed_episodes_are_proper_episode_objects():
    fake = FakeSQLite()
    now = utc_now()
    fake.add_session("yesterday", now - timedelta(hours=2))
    fake.add_episode(
        "yesterday",
        now - timedelta(hours=2),
        "Caroline says: I work at GreenLeaf Cafe.",
        concept_ids=["c1", "c2"],
        context={"interlocutor": "user", "task_context": "conversation"},
    )

    pool = _make_pool(fake, current="me")
    borrowed = pool.gather(now=now)
    assert len(borrowed) == 1
    ep = borrowed[0]
    assert isinstance(ep, Episode)
    assert ep.raw_content == "Caroline says: I work at GreenLeaf Cafe."
    assert ep.concept_ids == ["c1", "c2"]
    # Original context preserved + borrowed markers added
    assert ep.context["interlocutor"] == "user"
    assert ep.context["_origin_session"] == "yesterday"
    assert ep.context["_borrowed"] is True


def test_handles_missing_methods_gracefully():
    """If sqlite layer doesn't have the new methods (older DB), pool returns []."""
    class OldSQLite:
        pass  # missing list_recent_session_ids and get_recent_episodes_for_sessions

    pool = CrossSessionMemoryPool(
        current_session_id="me",
        sqlite_factory=lambda: OldSQLite(),
        config=CrossSessionPoolConfig(enabled=True),
    )
    assert pool.gather() == []


def test_handles_factory_exceptions_gracefully():
    def broken_factory():
        raise RuntimeError("DB unreachable")
    pool = CrossSessionMemoryPool(
        current_session_id="me",
        sqlite_factory=broken_factory,
        config=CrossSessionPoolConfig(enabled=True),
    )
    assert pool.gather() == []


def test_stats_dict_populated_after_gather():
    fake = FakeSQLite()
    now = utc_now()
    fake.add_session("a", now - timedelta(hours=1))
    fake.add_episode("a", now - timedelta(hours=1), "x")
    fake.add_session("b", now - timedelta(hours=2))
    fake.add_episode("b", now - timedelta(hours=2), "y")
    pool = _make_pool(fake, current="me")
    pool.gather(now=now)
    stats = pool.stats_dict()
    assert stats["sessions_considered"] == 2
    assert stats["sessions_borrowed_from"] == 2
    assert stats["episodes_borrowed"] == 2
    assert set(stats["borrowed_session_ids"]) == {"a", "b"}


def test_no_qualifying_sessions_returns_empty():
    fake = FakeSQLite()
    pool = _make_pool(fake, current="me")
    assert pool.gather() == []


# ─── End-to-end: borrowed episodes feed sleep cycle ─────────────────────────


def test_e2e_sleep_consumes_cross_session_episodes():
    """
    A fresh ChatEngine with no current-session WM activity should still be
    able to run a sleep cycle that consolidates concepts from prior sessions
    via the pool. This is the killer test for the whole module.
    """
    from src.chat.engine import ChatEngine
    from src.core.encoder import MeaningEncoder

    class StubLLM:
        def extract_concepts(self, text): return []
        def _chat(self, *a, **kw): return ""

    # Build a fake SQLite that the pool will read from.
    fake = FakeSQLite()
    now = utc_now()
    for sid, msg in [
        ("y_session_1", "Caroline went to the support group on May 7th."),
        ("y_session_1", "Caroline lives in Seattle."),
        ("y_session_2", "Caroline started rock climbing in March."),
    ]:
        fake.add_session(sid, now - timedelta(hours=24))
        fake.add_episode(sid, now - timedelta(hours=24), msg)

    pool = CrossSessionMemoryPool(
        current_session_id="today",
        sqlite_factory=lambda: fake,
        config=CrossSessionPoolConfig(
            enabled=True,
            max_sessions=5,
            lookback_hours=72.0,
            max_total_borrowed=100,
        ),
    )

    engine = ChatEngine(
        llm=StubLLM(),
        encoder=MeaningEncoder(llm=None),
        session_id="today",
        profile="research",
        sandbox_mode=True,
        enable_persistence=False,
        enable_auto_sleep=False,
        cross_session_pool=pool,
    )
    # Add ONE current-session episode so sleep has something to anchor
    engine._extract_and_store("Caroline says: I'm thinking about climbing again.", source="user")

    result = engine.force_sleep(mode="deep")
    assert result is not None, "force_sleep returned None"
    assert result["mode"] == "deep"
    # The pool should have been called and recorded stats
    stats = pool.stats_dict()
    assert stats["episodes_borrowed"] >= 3, f"pool stats: {stats}"
    assert "y_session_1" in stats["borrowed_session_ids"]
    assert "y_session_2" in stats["borrowed_session_ids"]
