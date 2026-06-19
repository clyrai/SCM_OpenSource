"""
Tests for Phase 7 WakeSummaryBuilder.

The wake summary is the user-visible payoff of M1+M2+M3. These tests are the
contract for what users will see when they come back to their agent:

  - Empty case: no sleeps yet → polite "welcome back" with no fake data.
  - Idle duration is computed from IdleLearner activity timestamps.
  - Multiple cycles aggregate consolidated/forgotten/dreams.
  - Schema concepts (M3 output) become insights in the summary.
  - Insights are sorted: recurring_topic > cooccurrence > trajectory > cadence.
  - Narrative text is human-readable, never raw stats blob.
  - `since` parameter scopes the report.
  - to_dict() output is JSON-serializable and complete.
  - Cross-session pool stats surface in `sessions_consulted`.
  - Failure modes: missing engine fields, missing pool, missing learner all OK.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import (
    Concept,
    ConceptType,
    Episode,
    ImportanceVector,
    MemoryState,
)
from src.core.time_utils import utc_isoformat, utc_now
from src.lifecycle.wake_summary import (
    WakeInsight,
    WakeSummary,
    WakeSummaryBuilder,
)


# ─── Fakes ──────────────────────────────────────────────────────────────────


class FakeWM:
    def __init__(self, episodes=None):
        self._eps = episodes or []
    def get_all(self):
        return list(self._eps)


class FakeLTM:
    def __init__(self, concepts=None):
        self._concepts = concepts or []
    def get_all_concepts(self, include_suppressed=False):
        return list(self._concepts)


class FakeEngine:
    def __init__(
        self,
        session_id: str = "test_session",
        sleep_history: Optional[List[Dict[str, Any]]] = None,
        concepts: Optional[List[Concept]] = None,
        episodes: Optional[List[Episode]] = None,
        cross_session_pool: Any = None,
    ):
        self.session_id = session_id
        self._sleep_history = sleep_history or []
        self.long_term_memory = FakeLTM(concepts or [])
        self.working_memory = FakeWM(episodes or [])
        self.cross_session_pool = cross_session_pool


class FakeIdleLearner:
    def __init__(self, last_activity: Dict[str, datetime] = None,
                 history: List[Any] = None):
        self._last_activity = last_activity or {}
        self._history = history or []
    def get_stats(self):
        return {
            "last_activity": {
                sid: ts.isoformat() for sid, ts in self._last_activity.items()
            },
        }
    def get_history(self, session_id=None, since=None, limit=100):
        out = list(self._history)
        if session_id:
            out = [r for r in out if getattr(r, "session_id", None) == session_id]
        if since:
            out = [r for r in out if getattr(r, "triggered_at", None) is None
                   or r.triggered_at >= since]
        return out[:limit]


def _schema_concept(
    schema_type: str,
    description: str,
    entities: List[str],
    occurrence_count: int = 3,
    confidence: float = 0.6,
    source_sessions: Optional[List[str]] = None,
    created_offset_seconds: int = -100,
) -> Concept:
    """Build a Concept that looks like a schema produced by M3."""
    c = Concept(
        type=ConceptType.ABSTRACT,
        description=description,
        importance=ImportanceVector(),
        state=MemoryState.ACTIVE,
        confidence=confidence,
        salience_score=0.65,
    )
    c.created_at = utc_now() + timedelta(seconds=created_offset_seconds)
    c.context_tags.update({
        "_schema": True,
        "schema_type": schema_type,
        "entities": list(entities),
        "occurrence_count": occurrence_count,
        "source_sessions": source_sessions or [],
    })
    return c


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_empty_case_no_sleeps_yet():
    eng = FakeEngine()
    summary = WakeSummaryBuilder(eng).build()
    assert summary.session_id == "test_session"
    assert summary.sleep_cycles_run == 0
    assert summary.memories_consolidated == 0
    assert summary.insights == []
    assert "Welcome back" in summary.narrative


def test_to_dict_is_json_safe():
    import json
    eng = FakeEngine()
    summary = WakeSummaryBuilder(eng).build()
    d = summary.to_dict()
    # Should round-trip through JSON without exploding
    s = json.dumps(d)
    back = json.loads(s)
    assert back["session_id"] == "test_session"
    assert back["sleep_cycles_run"] == 0
    assert "narrative" in back


def test_idle_duration_from_idle_learner_activity():
    now = utc_now()
    last_active = now - timedelta(hours=8)
    learner = FakeIdleLearner(last_activity={"alice": last_active})
    eng = FakeEngine(session_id="alice")
    builder = WakeSummaryBuilder(eng, idle_learner=learner, now_fn=lambda: now)
    summary = builder.build()
    assert summary.idle_duration_seconds is not None
    assert abs(summary.idle_duration_seconds - 8 * 3600) < 60
    # Narrative should mention hours
    assert "hours" in summary.narrative


def test_aggregates_sleep_history():
    now = utc_now()
    eng = FakeEngine(
        sleep_history=[
            {
                "timestamp": utc_isoformat(now - timedelta(hours=4)),
                "mode": "deep",
                "consolidated": 12,
                "forgotten": 3,
                "dreams": 2,
            },
            {
                "timestamp": utc_isoformat(now - timedelta(hours=2)),
                "mode": "deep",
                "consolidated": 8,
                "forgotten": 5,
                "dreams": 1,
            },
        ],
    )
    learner = FakeIdleLearner(
        last_activity={"test_session": now - timedelta(hours=6)},
    )
    summary = WakeSummaryBuilder(
        eng, idle_learner=learner, now_fn=lambda: now,
    ).build()
    assert summary.sleep_cycles_run == 2
    assert summary.memories_consolidated == 20
    assert summary.memories_forgotten == 8
    assert summary.dreams_generated == 3
    # Narrative contains the totals
    assert "20" in summary.narrative
    assert "8" in summary.narrative


def test_wake_summary_surfaces_latest_dream_state():
    now = utc_now()
    dream_state = {
        "enabled": True,
        "dream_summary": "Sleep replay connected project deadline and team conflict.",
        "dream_emotional_tone": "anxious",
        "replayed_memories": ["project deadline", "team conflict"],
        "resolved_conflicts": [],
        "open_threads_for_today": ["project deadline"],
        "dream_count": 2,
    }
    eng = FakeEngine(
        sleep_history=[
            {
                "timestamp": utc_isoformat(now - timedelta(hours=2)),
                "mode": "deep",
                "consolidated": 3,
                "dreams": 2,
                "dream_state": dream_state,
            },
        ],
    )
    summary = WakeSummaryBuilder(eng, now_fn=lambda: now).build()

    assert summary.dream_state["dream_summary"] == dream_state["dream_summary"]
    assert summary.to_dict()["dream_state"]["dream_emotional_tone"] == "anxious"
    assert "Morning brief" in summary.narrative
    assert "project deadline" in summary.narrative
    assert "conscious" not in summary.narrative.lower()
    assert "real feelings" not in summary.narrative.lower()


def test_old_sleep_records_excluded_when_since_provided():
    now = utc_now()
    eng = FakeEngine(
        sleep_history=[
            {
                "timestamp": utc_isoformat(now - timedelta(days=2)),
                "mode": "deep",
                "consolidated": 100,
            },
            {
                "timestamp": utc_isoformat(now - timedelta(hours=1)),
                "mode": "deep",
                "consolidated": 5,
            },
        ],
    )
    summary = WakeSummaryBuilder(eng, now_fn=lambda: now).build(
        since=now - timedelta(hours=12),
    )
    # Only the recent record counts
    assert summary.memories_consolidated == 5
    assert summary.sleep_cycles_run == 1


def test_schema_concepts_become_insights():
    now = utc_now()
    schemas = [
        _schema_concept(
            "recurring_topic",
            "GreenLeaf is a recurring topic (5 mentions across 3 sessions).",
            entities=["GreenLeaf"],
            occurrence_count=5,
            confidence=0.85,
            source_sessions=["s1", "s2", "s3"],
        ),
        _schema_concept(
            "cooccurrence",
            "Caroline and Melanie are linked.",
            entities=["Caroline", "Melanie"],
            occurrence_count=3,
            confidence=0.7,
        ),
        _schema_concept(
            "temporal_cadence",
            "Caroline recurs on a weekly cadence.",
            entities=["Caroline"],
            occurrence_count=4,
            confidence=0.55,
        ),
    ]
    eng = FakeEngine(concepts=schemas)
    summary = WakeSummaryBuilder(eng, now_fn=lambda: now).build()
    assert len(summary.insights) == 3
    # Sorted: recurring_topic first
    assert summary.insights[0].insight_type == "recurring_topic"
    assert summary.insights[0].entities == ["GreenLeaf"]
    # Narrative includes them
    assert "noticed" in summary.narrative
    assert "GreenLeaf" in summary.narrative


def test_insights_capped_at_max():
    now = utc_now()
    schemas = [
        _schema_concept(
            "recurring_topic", f"Pattern {i}", entities=[f"E{i}"],
        )
        for i in range(20)
    ]
    eng = FakeEngine(concepts=schemas)
    summary = WakeSummaryBuilder(eng, now_fn=lambda: now).build(max_insights=3)
    assert len(summary.insights) == 3


def test_non_schema_concepts_excluded_from_insights():
    """Regular FACT concepts should never appear as insights."""
    now = utc_now()
    fact = Concept(
        type=ConceptType.FACT,
        description="Caroline lives in Seattle.",
        importance=ImportanceVector(),
    )
    fact.created_at = now
    eng = FakeEngine(concepts=[fact])
    summary = WakeSummaryBuilder(eng, now_fn=lambda: now).build()
    assert summary.insights == []


def test_narrative_handles_zero_state_gracefully():
    """No sleeps + no insights still produces a friendly narrative."""
    eng = FakeEngine()
    summary = WakeSummaryBuilder(eng).build()
    assert isinstance(summary.narrative, str)
    assert len(summary.narrative) > 0
    assert "Welcome back" in summary.narrative


def test_sessions_consulted_from_pool():
    """If the cross-session pool has last_stats, surface borrowed sessions."""
    class FakePool:
        from dataclasses import dataclass as _dc, field as _field
        from typing import List as _List

        class _Stats:
            def __init__(self, ids):
                self.borrowed_session_ids = ids
        def __init__(self, ids):
            self.last_stats = FakePool._Stats(ids)
        def stats_dict(self):
            return {"borrowed_session_ids": self.last_stats.borrowed_session_ids}

    eng = FakeEngine(cross_session_pool=FakePool(["yesterday", "monday"]))
    summary = WakeSummaryBuilder(eng).build()
    assert "yesterday" in summary.sessions_consulted
    assert "monday" in summary.sessions_consulted
    assert "2 prior session" in summary.narrative


def test_failure_modes_dont_crash():
    """Missing engine attributes, broken pool, missing learner — all should be safe."""
    class WeirdEngine:
        session_id = "weird"
        long_term_memory = FakeLTM([])
        working_memory = FakeWM([])
        # NO _sleep_history attribute
        # NO cross_session_pool attribute

    eng = WeirdEngine()
    summary = WakeSummaryBuilder(eng).build()
    # Should produce a valid summary, not crash
    assert summary.session_id == "weird"
    assert summary.sleep_cycles_run == 0


def test_diagnostics_payload_when_requested():
    now = utc_now()
    eng = FakeEngine(
        sleep_history=[{"timestamp": utc_isoformat(now), "consolidated": 1}],
    )
    summary = WakeSummaryBuilder(eng, now_fn=lambda: now).build(
        include_diagnostics=True,
    )
    assert "sleep_records" in summary.diagnostics
    assert summary.diagnostics["sleep_records"] == eng._sleep_history


def test_format_duration():
    """Direct test of the duration formatter."""
    fmt = WakeSummaryBuilder._format_duration
    assert "seconds" in fmt(45)
    assert "minutes" in fmt(900)
    assert "hours" in fmt(7200)
    assert "days" in fmt(2 * 86400)


# ─── End-to-end: real ChatEngine + IdleLearner + DeepSleep ─────────────────


def test_e2e_real_engine_real_sleep_real_summary():
    """
    Build a real ChatEngine, ingest some content, manually trigger sleep,
    and verify the wake summary surfaces the consolidation properly.
    """
    from src.chat.engine import ChatEngine
    from src.core.encoder import MeaningEncoder

    class StubLLM:
        def extract_concepts(self, text): return []
        def _chat(self, *a, **kw): return ""

    engine = ChatEngine(
        llm=StubLLM(),
        encoder=MeaningEncoder(llm=None),
        session_id="wake_e2e",
        profile="research",
        sandbox_mode=True,
        enable_persistence=False,
        enable_auto_sleep=False,
    )
    for msg in [
        "Caroline says: I work at GreenLeaf Cafe.",
        "Caroline says: I went to the LGBTQ support group on May 7th.",
        "Caroline says: My favorite hobby is rock climbing.",
        "Melanie says: I'm planning a trip to Boston.",
    ]:
        engine._extract_and_store(msg, source="user")
        engine._message_count += 1

    result = engine.force_sleep(mode="deep")
    assert result is not None, f"force_sleep failed: {result}"

    summary = WakeSummaryBuilder(engine).build()
    assert summary.sleep_cycles_run >= 1
    # Forced sleep counted in history (Phase 7 fix earlier)
    assert "Welcome back" in summary.narrative
