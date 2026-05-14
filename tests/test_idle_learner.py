"""
Unit + integration tests for the Phase 7 IdleLearner daemon.

The IdleLearner is the foundation of SCM's autonomous-learning vision, so
these tests are the contract for what the daemon must guarantee:

  - Records activity per-session, not globally.
  - Fires sleep only after `idle_threshold_seconds` of inactivity.
  - Does NOT fire on a session that just slept (cooldown).
  - Does NOT fire on a session that's still active.
  - Catches engine errors gracefully and keeps running.
  - Can be started/stopped cleanly with no thread leaks.
  - Maintains a bounded rolling history for the wake-summary endpoint.

We use a fake clock so tests simulate "10 minutes idle" in microseconds.
We use a fake engine so tests don't require Ollama/DeepSeek.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lifecycle.idle_learner import (
    IdleLearner,
    IdleLearnerConfig,
    IdleSleepRecord,
)


# ─── Fake clock ─────────────────────────────────────────────────────────────


class FakeClock:
    """Manually-advanced clock for deterministic time-based tests."""

    def __init__(self, start: Optional[datetime] = None):
        self._now = start or datetime(2026, 5, 1, 12, 0, 0)
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now = self._now + timedelta(seconds=seconds)


# ─── Fake engine ────────────────────────────────────────────────────────────


class FakeEngine:
    """Stand-in for ChatEngine. Records every force_sleep call."""

    def __init__(
        self,
        consolidated: int = 5,
        forgotten: int = 2,
        dreams: int = 1,
        raise_exc: Optional[Exception] = None,
        sleep_seconds: float = 0.0,
    ):
        self.calls: List[str] = []
        self.consolidated = consolidated
        self.forgotten = forgotten
        self.dreams = dreams
        self.raise_exc = raise_exc
        self.sleep_seconds = sleep_seconds

    def force_sleep(self, mode: str = "deep") -> Optional[Dict[str, Any]]:
        self.calls.append(mode)
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        if self.raise_exc is not None:
            raise self.raise_exc
        return {
            "consolidated": self.consolidated,
            "forgotten": self.forgotten,
            "dreams": self.dreams,
            "mode": mode,
        }


# ─── Test fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def _make_learner(
    engines: Dict[str, FakeEngine],
    clock: FakeClock,
    *,
    idle_threshold_seconds: float = 60.0,
    min_sleep_interval_seconds: float = 120.0,
    tick_interval_seconds: float = 0.05,
    sleep_mode: str = "deep",
    enabled: bool = True,
) -> IdleLearner:
    cfg = IdleLearnerConfig(
        idle_threshold_seconds=idle_threshold_seconds,
        min_sleep_interval_seconds=min_sleep_interval_seconds,
        tick_interval_seconds=tick_interval_seconds,
        sleep_mode=sleep_mode,
        enabled=enabled,
    )
    return IdleLearner(
        engine_provider=lambda: dict(engines),
        config=cfg,
        clock=clock.now,
    )


def _wait_until(predicate, timeout_s: float = 2.0, poll_s: float = 0.02) -> bool:
    """Spin-wait until predicate() is true or timeout. Used to wait on the daemon."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_records_activity_per_session(fake_clock):
    learner = _make_learner({}, fake_clock)
    learner.record_activity("alice")
    learner.record_activity("bob")
    fake_clock.advance(10)
    learner.record_activity("alice")  # alice updates, bob doesn't
    stats = learner.get_stats()
    assert "alice" in stats["last_activity"]
    assert "bob" in stats["last_activity"]


def test_does_not_fire_when_session_is_active(fake_clock):
    engine = FakeEngine()
    learner = _make_learner(
        {"alice": engine}, fake_clock,
        idle_threshold_seconds=60.0,
        tick_interval_seconds=0.05,
    )
    learner.record_activity("alice")
    learner.start()
    try:
        # Simulate < idle_threshold of time passing
        for _ in range(5):
            fake_clock.advance(5)
            time.sleep(0.05)
        # Should not have fired
        assert engine.calls == []
    finally:
        learner.stop()


def test_fires_sleep_after_idle_threshold(fake_clock):
    engine = FakeEngine()
    learner = _make_learner(
        {"alice": engine}, fake_clock,
        idle_threshold_seconds=60.0,
        tick_interval_seconds=0.02,
    )
    learner.record_activity("alice")
    learner.start()
    try:
        # Jump past the idle threshold
        fake_clock.advance(120)
        assert _wait_until(lambda: len(engine.calls) >= 1, timeout_s=2.0), \
            f"expected sleep to fire, calls={engine.calls}"
        assert engine.calls[0] == "deep"
    finally:
        learner.stop()

    # Verify history record has the right shape
    history = learner.get_history()
    assert len(history) == 1
    rec = history[0]
    assert rec.session_id == "alice"
    assert rec.success is True
    assert rec.consolidated == 5
    assert rec.forgotten == 2
    assert rec.dreams == 1
    assert rec.seconds_idle_when_triggered >= 60.0


def test_respects_cooldown_after_sleep(fake_clock):
    engine = FakeEngine()
    learner = _make_learner(
        {"alice": engine}, fake_clock,
        idle_threshold_seconds=60.0,
        min_sleep_interval_seconds=300.0,
        tick_interval_seconds=0.02,
    )
    learner.record_activity("alice")
    learner.start()
    try:
        fake_clock.advance(120)
        assert _wait_until(lambda: len(engine.calls) >= 1, timeout_s=2.0)

        # Advance another 60s — past idle threshold, but within cooldown
        for _ in range(3):
            fake_clock.advance(20)
            time.sleep(0.05)
        # Still only one call
        assert len(engine.calls) == 1, f"cooldown violated: calls={engine.calls}"

        # Now jump past the cooldown
        fake_clock.advance(400)
        assert _wait_until(lambda: len(engine.calls) >= 2, timeout_s=2.0), \
            f"expected second sleep after cooldown, calls={engine.calls}"
    finally:
        learner.stop()


def test_handles_engine_exceptions_gracefully(fake_clock):
    boom = FakeEngine(raise_exc=RuntimeError("simulated engine crash"))
    learner = _make_learner(
        {"unstable": boom}, fake_clock,
        idle_threshold_seconds=30.0,
        tick_interval_seconds=0.02,
    )
    learner.record_activity("unstable")
    learner.start()
    try:
        fake_clock.advance(60)
        assert _wait_until(lambda: len(boom.calls) >= 1, timeout_s=2.0)
        # Daemon must still be running after the exception
        assert learner.is_running()
    finally:
        learner.stop()

    history = learner.get_history()
    assert len(history) == 1
    assert history[0].success is False
    assert "RuntimeError" in (history[0].error or "")
    stats = learner.get_stats()
    assert stats["cycles_failed"] == 1


def test_engine_returning_none_is_recorded_as_failure(fake_clock):
    class NoneEngine:
        def __init__(self):
            self.calls = []
        def force_sleep(self, mode):
            self.calls.append(mode)
            return None  # e.g. nothing to consolidate

    engine = NoneEngine()
    learner = _make_learner(
        {"empty": engine}, fake_clock,
        idle_threshold_seconds=30.0,
        tick_interval_seconds=0.02,
    )
    learner.record_activity("empty")
    learner.start()
    try:
        fake_clock.advance(60)
        assert _wait_until(lambda: len(engine.calls) >= 1, timeout_s=2.0)
    finally:
        learner.stop()

    rec = learner.get_history()[0]
    assert rec.success is False
    assert "None" in (rec.error or "")


def test_multiple_sessions_independent_tracking(fake_clock):
    a = FakeEngine()
    b = FakeEngine()
    learner = _make_learner(
        {"alice": a, "bob": b}, fake_clock,
        idle_threshold_seconds=60.0,
        min_sleep_interval_seconds=600.0,
        tick_interval_seconds=0.02,
    )
    learner.record_activity("alice")
    learner.record_activity("bob")
    learner.start()
    try:
        # alice goes idle but bob keeps refreshing
        for _ in range(10):
            fake_clock.advance(8)
            learner.record_activity("bob")
            time.sleep(0.03)
        # alice has been idle ~80s; bob always active
        assert _wait_until(lambda: len(a.calls) >= 1, timeout_s=2.0)
        # bob should NOT have fired
        assert b.calls == [], f"bob should be active, but calls={b.calls}"
    finally:
        learner.stop()


def test_disabled_config_does_not_start(fake_clock):
    engine = FakeEngine()
    learner = _make_learner(
        {"alice": engine}, fake_clock, enabled=False,
    )
    learner.record_activity("alice")
    learner.start()
    fake_clock.advance(1000)
    time.sleep(0.1)
    assert not learner.is_running()
    assert engine.calls == []


def test_clean_shutdown_no_thread_leak(fake_clock):
    engine = FakeEngine()
    learner = _make_learner({"alice": engine}, fake_clock)
    learner.start()
    assert learner.is_running()
    learner.stop()
    assert not learner.is_running()


def test_get_history_filters_by_session_and_time(fake_clock):
    a = FakeEngine()
    b = FakeEngine()
    learner = _make_learner(
        {"alice": a, "bob": b}, fake_clock,
        idle_threshold_seconds=30.0,
        min_sleep_interval_seconds=10.0,
        tick_interval_seconds=0.02,
    )
    learner.record_activity("alice")
    learner.record_activity("bob")
    learner.start()
    try:
        fake_clock.advance(60)
        assert _wait_until(
            lambda: len(a.calls) >= 1 and len(b.calls) >= 1, timeout_s=2.0
        )
    finally:
        learner.stop()

    all_records = learner.get_history()
    assert len(all_records) >= 2
    only_alice = learner.get_history(session_id="alice")
    assert all(r.session_id == "alice" for r in only_alice)
    assert len(only_alice) >= 1


def test_history_buffer_is_bounded(fake_clock):
    engine = FakeEngine()
    cfg = IdleLearnerConfig(
        idle_threshold_seconds=10.0,
        min_sleep_interval_seconds=5.0,
        tick_interval_seconds=0.02,
        history_buffer_size=3,
    )
    learner = IdleLearner(
        engine_provider=lambda: {"alice": engine},
        config=cfg,
        clock=fake_clock.now,
    )
    learner.record_activity("alice")
    learner.start()
    try:
        # Force many rapid cycles
        for _ in range(10):
            fake_clock.advance(20)
            time.sleep(0.05)
    finally:
        learner.stop()
    history = learner.get_history()
    assert len(history) <= 3, f"history not bounded: len={len(history)}"


def test_stats_snapshot_includes_config(fake_clock):
    engine = FakeEngine()
    learner = _make_learner({"alice": engine}, fake_clock, sleep_mode="micro")
    stats = learner.get_stats()
    assert stats["config"]["sleep_mode"] == "micro"
    assert stats["config"]["enabled"] is True


def test_first_tick_does_not_fire_on_unseen_session(fake_clock):
    """
    When the daemon starts and a session has no recorded activity yet,
    it should NOT immediately consider it 'infinitely idle' and fire sleep.
    Instead, treat first sight as fresh activity.
    """
    engine = FakeEngine()
    # Note: we DO NOT call record_activity for "alice"
    learner = _make_learner({"alice": engine}, fake_clock,
                            idle_threshold_seconds=60.0,
                            tick_interval_seconds=0.02)
    learner.start()
    try:
        time.sleep(0.1)  # let a few ticks happen
        assert engine.calls == [], "should not fire on session with no recorded activity yet"
        # Now advance time and expect it to fire (because first-sight = now)
        fake_clock.advance(120)
        assert _wait_until(lambda: len(engine.calls) >= 1, timeout_s=2.0)
    finally:
        learner.stop()


# ─── End-to-end with a real ChatEngine ─────────────────────────────────────


def test_end_to_end_with_real_chat_engine():
    """
    Simulate a realistic flow: a real ChatEngine is created, a few messages
    are processed, the user goes 'idle' (clock advances), the daemon fires
    a real sleep cycle, and we verify the engine's state actually changed.
    """
    from src.chat.engine import ChatEngine
    from src.core.encoder import MeaningEncoder

    class StubLLM:
        def extract_concepts(self, text): return []
        def _chat(self, *a, **kw): return ""

    engine = ChatEngine(
        llm=StubLLM(),
        encoder=MeaningEncoder(llm=None),
        session_id="idle_e2e",
        profile="research",
        sandbox_mode=True,
        enable_persistence=False,
        enable_auto_sleep=False,  # we want IdleLearner to be the only sleep trigger
    )
    # Ingest a few messages so there's something to consolidate
    for msg in [
        "Caroline says: I work at GreenLeaf Cafe.",
        "Caroline says: I live in Seattle.",
        "Caroline says: My favorite hobby is rock climbing.",
        "Caroline says: I have a cat named Mochi.",
    ]:
        engine._extract_and_store(msg, source="user")
        engine._message_count += 1

    initial_concepts = len(engine.long_term_memory.get_all_concepts(include_suppressed=False))
    assert initial_concepts > 0, "test setup failed: no concepts ingested"

    clock = FakeClock()
    learner = IdleLearner(
        engine_provider=lambda: {"idle_e2e": engine},
        config=IdleLearnerConfig(
            idle_threshold_seconds=30.0,
            min_sleep_interval_seconds=10.0,
            tick_interval_seconds=0.02,
            sleep_mode="deep",
        ),
        clock=clock.now,
    )
    learner.record_activity("idle_e2e")
    learner.start()
    try:
        clock.advance(60)
        assert _wait_until(
            lambda: len(learner.get_history()) >= 1, timeout_s=10.0
        ), "IdleLearner did not fire on real engine"
    finally:
        learner.stop()

    rec = learner.get_history()[0]
    assert rec.session_id == "idle_e2e"
    assert rec.success is True, f"sleep failed: {rec.error}"
    assert rec.consolidated >= 0  # may be 0 for trivial input but shouldn't crash
    # Engine's sleep history should also have grown
    assert len(engine._sleep_history) >= 1
