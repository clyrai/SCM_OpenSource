"""
Tests for Phase 7 M6: LifecyclePolicy + state persistence.

Contracts under test:

  Policy:
    - AlwaysAllowPolicy is the no-op default.
    - BatteryPolicy blocks low battery on battery, allows on AC.
    - CPULoadPolicy blocks high system CPU, allows when low.
    - CompositePolicy: any blocker stops the chain.
    - Misbehaving sub-policy must NEVER kill the daemon.
    - psutil-missing path: configurable allow vs require.

  State store:
    - Round-trips activity map cleanly (datetime → iso → datetime).
    - Atomic write (no corrupted partial files on crash).
    - Bounded history serialization.
    - Corrupt JSON file resets to empty.

  IdleLearner integration:
    - Restores last_activity from disk on init.
    - Persists state every N ticks.
    - Policy block bumps `cycles_blocked_by_policy` and skips work.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lifecycle.lifecycle_policy import (
    AlwaysAllowPolicy,
    BatteryPolicy,
    CompositePolicy,
    CPULoadPolicy,
    LifecyclePolicy,
    PolicyDecision,
)
from src.lifecycle.state_store import IdleLearnerStateStore
from src.lifecycle.idle_learner import (
    IdleLearner,
    IdleLearnerConfig,
    IdleSleepRecord,
)


# ─── AlwaysAllow ────────────────────────────────────────────────────────────


def test_always_allow_returns_allowed():
    d = AlwaysAllowPolicy().evaluate()
    assert d.allowed is True
    assert d.decided_by == "always_allow"


# ─── Battery policy (mock psutil) ──────────────────────────────────────────


def _patch_battery(monkeypatch, percent, plugged):
    """Patch psutil.sensors_battery to return our fake reading."""
    from src.lifecycle import lifecycle_policy as lp_mod

    class FakeBatt:
        def __init__(self, p, pl):
            self.percent = p
            self.power_plugged = pl
            self.secsleft = 9999

    class FakePS:
        @staticmethod
        def sensors_battery():
            return FakeBatt(percent, plugged)

        @staticmethod
        def cpu_percent(interval=0.1):
            return 5.0

    monkeypatch.setattr(lp_mod, "_PSUTIL", FakePS)


def test_battery_policy_blocks_on_low_battery_unplugged(monkeypatch):
    _patch_battery(monkeypatch, percent=15, plugged=False)
    p = BatteryPolicy(min_battery_percent=30, allow_when_plugged_in=True)
    d = p.evaluate()
    assert d.allowed is False
    assert "15" in d.reason
    assert d.metadata["plugged"] is False


def test_battery_policy_allows_when_plugged(monkeypatch):
    _patch_battery(monkeypatch, percent=10, plugged=True)
    p = BatteryPolicy(min_battery_percent=30, allow_when_plugged_in=True)
    d = p.evaluate()
    assert d.allowed is True
    assert d.metadata["plugged"] is True


def test_battery_policy_allows_high_battery(monkeypatch):
    _patch_battery(monkeypatch, percent=80, plugged=False)
    p = BatteryPolicy(min_battery_percent=30)
    d = p.evaluate()
    assert d.allowed is True


def test_battery_policy_no_battery_means_desktop_allows(monkeypatch):
    from src.lifecycle import lifecycle_policy as lp_mod

    class FakePS:
        @staticmethod
        def sensors_battery():
            return None

        @staticmethod
        def cpu_percent(interval=0.1):
            return 5.0

    monkeypatch.setattr(lp_mod, "_PSUTIL", FakePS)
    p = BatteryPolicy()
    d = p.evaluate()
    assert d.allowed is True


def test_battery_policy_psutil_missing_default_allow(monkeypatch):
    from src.lifecycle import lifecycle_policy as lp_mod
    monkeypatch.setattr(lp_mod, "_PSUTIL", None)
    p = BatteryPolicy(require_psutil=False)
    d = p.evaluate()
    assert d.allowed is True


def test_battery_policy_psutil_missing_required_blocks(monkeypatch):
    from src.lifecycle import lifecycle_policy as lp_mod
    monkeypatch.setattr(lp_mod, "_PSUTIL", None)
    p = BatteryPolicy(require_psutil=True)
    d = p.evaluate()
    assert d.allowed is False


# ─── CPU policy ────────────────────────────────────────────────────────────


def _patch_cpu(monkeypatch, percent):
    from src.lifecycle import lifecycle_policy as lp_mod

    class FakePS:
        @staticmethod
        def sensors_battery():
            return None

        @staticmethod
        def cpu_percent(interval=0.1):
            return percent

    monkeypatch.setattr(lp_mod, "_PSUTIL", FakePS)


def test_cpu_policy_allows_when_idle(monkeypatch):
    _patch_cpu(monkeypatch, 5.0)
    p = CPULoadPolicy(max_cpu_percent=80, sample_interval_seconds=0.01)
    d = p.evaluate()
    assert d.allowed is True
    assert d.metadata["cpu_percent"] == 5.0


def test_cpu_policy_blocks_under_load(monkeypatch):
    _patch_cpu(monkeypatch, 95.0)
    p = CPULoadPolicy(max_cpu_percent=80, sample_interval_seconds=0.01)
    d = p.evaluate()
    assert d.allowed is False
    assert "95" in d.reason


# ─── Composite ─────────────────────────────────────────────────────────────


def test_composite_returns_first_blocker():
    class Allow(LifecyclePolicy):
        name = "allow"
        def evaluate(self):
            return PolicyDecision(allowed=True, reason="ok", decided_by="allow")
    class Block(LifecyclePolicy):
        name = "block"
        def evaluate(self):
            return PolicyDecision(allowed=False, reason="blocked here", decided_by="block")
    c = CompositePolicy([Allow(), Block(), Allow()])
    d = c.evaluate()
    assert d.allowed is False
    assert d.decided_by == "block"


def test_composite_all_allow():
    class Allow(LifecyclePolicy):
        name = "allow"
        def evaluate(self):
            return PolicyDecision(allowed=True, reason="ok", decided_by="allow")
    c = CompositePolicy([Allow(), Allow()])
    d = c.evaluate()
    assert d.allowed is True


def test_composite_misbehaving_subpolicy_does_not_crash():
    class Boom(LifecyclePolicy):
        name = "boom"
        def evaluate(self):
            raise RuntimeError("crash")
    class Allow(LifecyclePolicy):
        name = "allow"
        def evaluate(self):
            return PolicyDecision(allowed=True, reason="ok", decided_by="allow")
    c = CompositePolicy([Boom(), Allow()])
    d = c.evaluate()  # Boom is skipped silently
    assert d.allowed is True


def test_composite_empty_allows():
    c = CompositePolicy([])
    d = c.evaluate()
    assert d.allowed is True


# ─── State store ───────────────────────────────────────────────────────────


def test_state_store_save_and_load(tmp_path):
    p = tmp_path / "state.json"
    s = IdleLearnerStateStore(str(p))
    payload = {
        "last_activity": {"alice": "2026-05-01T12:00:00+00:00"},
        "tick_count": 42,
    }
    assert s.save(payload) is True
    out = s.load()
    assert out["tick_count"] == 42
    assert out["last_activity"]["alice"] == "2026-05-01T12:00:00+00:00"


def test_state_store_missing_file_returns_empty(tmp_path):
    s = IdleLearnerStateStore(str(tmp_path / "doesnt_exist.json"))
    assert s.load() == {}


def test_state_store_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json at all")
    s = IdleLearnerStateStore(str(p))
    assert s.load() == {}


def test_state_store_atomic_write_no_partial(tmp_path):
    """The store writes via .tmp + os.replace, so a crash mid-write
    can never leave a partial file at the target path. Test by checking
    no .tmp leftover after a successful save."""
    p = tmp_path / "atomic.json"
    s = IdleLearnerStateStore(str(p))
    s.save({"x": 1})
    assert p.exists()
    assert not (tmp_path / "atomic.json.tmp").exists()


def test_serialize_activity_map_round_trip():
    now = datetime(2026, 5, 1, 10, 30, 0)
    raw = IdleLearnerStateStore.serialize_activity_map({"alice": now})
    back = IdleLearnerStateStore.deserialize_activity_map(raw)
    assert back["alice"] == now


def test_serialize_history_handles_dataclass():
    now = datetime(2026, 5, 1)
    rec = IdleSleepRecord(
        session_id="alice", triggered_at=now, completed_at=now,
        seconds_idle_when_triggered=120.0, mode="deep", success=True,
        duration_seconds=2.0, consolidated=10, forgotten=2, dreams=1,
    )
    out = IdleLearnerStateStore.serialize_history([rec], limit=10)
    assert len(out) == 1
    assert out[0]["session_id"] == "alice"
    assert out[0]["consolidated"] == 10


# ─── IdleLearner integration with policy + state ───────────────────────────


class FakeClock:
    def __init__(self, start: Optional[datetime] = None):
        self._now = start or datetime(2026, 5, 1, 12, 0, 0)
        self._lock = threading.Lock()
    def now(self):
        with self._lock:
            return self._now
    def advance(self, sec):
        with self._lock:
            self._now = self._now + timedelta(seconds=sec)


class FakeEngine:
    def __init__(self):
        self.calls = []
    def force_sleep(self, mode="deep"):
        self.calls.append(mode)
        return {"consolidated": 1, "forgotten": 0, "dreams": 0}


class _BlockingPolicy(LifecyclePolicy):
    name = "test_block"
    def __init__(self):
        self.evaluations = 0
    def evaluate(self):
        self.evaluations += 1
        return PolicyDecision(allowed=False, reason="blocked-for-test",
                              decided_by="test_block")


def test_idle_learner_respects_blocking_policy():
    clock = FakeClock()
    engine = FakeEngine()
    pol = _BlockingPolicy()
    learner = IdleLearner(
        engine_provider=lambda: {"alice": engine},
        config=IdleLearnerConfig(
            idle_threshold_seconds=10.0,
            min_sleep_interval_seconds=10.0,
            tick_interval_seconds=0.02,
        ),
        clock=clock.now,
        policy=pol,
    )
    learner.record_activity("alice")
    learner.start()
    try:
        clock.advance(60)
        # Give the daemon time to tick
        time.sleep(0.4)
        # Engine should never have been called because policy blocks
        assert engine.calls == [], f"policy bypassed: {engine.calls}"
    finally:
        learner.stop()
    stats = learner.get_stats()
    assert stats["cycles_blocked_by_policy"] >= 1
    assert stats["last_policy_decision"]["allowed"] is False
    assert stats["last_policy_decision"]["decided_by"] == "test_block"


def test_idle_learner_persists_and_restores_activity(tmp_path):
    state_path = tmp_path / "state.json"
    store = IdleLearnerStateStore(str(state_path))
    clock = FakeClock()
    engine = FakeEngine()

    # First instance: record activity, persist
    l1 = IdleLearner(
        engine_provider=lambda: {"alice": engine},
        config=IdleLearnerConfig(tick_interval_seconds=0.02),
        clock=clock.now,
        state_store=store,
        persist_every_n_ticks=1,
    )
    l1.record_activity("alice")
    l1.start()
    time.sleep(0.2)
    l1.stop()

    # State file should exist
    assert state_path.exists()
    raw = json.loads(state_path.read_text())
    assert "alice" in raw["last_activity"]

    # Second instance: should restore alice's last_activity timestamp
    l2 = IdleLearner(
        engine_provider=lambda: {"alice": engine},
        config=IdleLearnerConfig(tick_interval_seconds=0.02),
        clock=clock.now,
        state_store=store,
    )
    assert "alice" in l2._last_activity


def test_idle_learner_policy_block_does_not_block_persistence(tmp_path):
    """Even when the policy blocks work, state should still be persisted
    so we don't lose track of activity timestamps."""
    state_path = tmp_path / "state.json"
    store = IdleLearnerStateStore(str(state_path))
    clock = FakeClock()
    engine = FakeEngine()
    pol = _BlockingPolicy()
    learner = IdleLearner(
        engine_provider=lambda: {"alice": engine},
        config=IdleLearnerConfig(tick_interval_seconds=0.02),
        clock=clock.now,
        policy=pol,
        state_store=store,
        persist_every_n_ticks=1,
    )
    learner.record_activity("alice")
    learner.start()
    time.sleep(0.3)
    learner.stop()
    # State file must have been written despite policy blocking
    assert state_path.exists()


def test_idle_learner_works_without_state_store():
    """No state_store → ephemeral mode, must still function."""
    clock = FakeClock()
    engine = FakeEngine()
    learner = IdleLearner(
        engine_provider=lambda: {"alice": engine},
        config=IdleLearnerConfig(
            idle_threshold_seconds=10.0,
            tick_interval_seconds=0.02,
        ),
        clock=clock.now,
        # No state_store, no policy → uses defaults
    )
    learner.record_activity("alice")
    learner.start()
    try:
        clock.advance(30)
        time.sleep(0.3)
        # Default AlwaysAllow + ephemeral works fine
    finally:
        learner.stop()
    stats = learner.get_stats()
    assert stats["config"]["persistence_enabled"] is False
    assert stats["config"]["policy_class"] == "AlwaysAllowPolicy"
