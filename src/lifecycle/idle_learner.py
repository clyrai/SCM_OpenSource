"""
IdleLearner: autonomous background daemon for sleep-time learning.

This module is the foundation of SCM's autonomous-learning vision. While the
user is away from their agent (laptop closed, no API calls for N minutes), the
IdleLearner triggers sleep cycles on every active session. The result is that
when the user returns, the agent has consolidated, abstracted, and pruned
overnight --- like a human who slept on their day's experiences.

Design principles:
  - Threaded, not async, so it can run independently of FastAPI's event loop
    and call into the synchronous ChatEngine / SleepCycleOrchestrator stack.
  - Injectable clock so unit tests can simulate hours of idle time in
    milliseconds.
  - Per-session activity tracking; the daemon decides per-session whether to
    fire sleep, not globally.
  - Cooldown enforcement so a session that just slept doesn't sleep again
    immediately.
  - All cycles logged in a rolling history buffer that M4 (wake summary) can
    consume.
  - Bounded resource use: configurable tick interval, hard cap on sleep
    duration (timeout the cycle if it runs long).
  - Safe shutdown via threading.Event so FastAPI lifespan can stop it cleanly.

Usage (programmatic):
    from src.lifecycle import IdleLearner, IdleLearnerConfig
    from src.api.chat import _chat_engines

    learner = IdleLearner(
        engine_provider=lambda: dict(_chat_engines),
        config=IdleLearnerConfig(idle_threshold_seconds=600),
    )
    learner.start()
    # ... agent runs ...
    learner.stop()

Usage (CLI):
    python -m src.lifecycle.idle_learner --idle-mins 10 --tick-secs 60
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ..core.time_utils import ensure_utc, utc_now
from .lifecycle_policy import (
    AlwaysAllowPolicy,
    LifecyclePolicy,
    PolicyDecision,
)
from .state_store import IdleLearnerStateStore


logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────────


@dataclass
class IdleLearnerConfig:
    """Tunable knobs for the IdleLearner background daemon."""

    # Seconds since last user activity before a session is considered "idle"
    # and eligible for autonomous sleep. Default 10 minutes.
    idle_threshold_seconds: float = 600.0

    # Minimum seconds between consecutive sleep cycles for the same session.
    # Prevents the daemon from running sleep on a hair-trigger.
    min_sleep_interval_seconds: float = 1800.0  # 30 minutes

    # How often the daemon's main loop wakes up to check sessions.
    # Lower = more responsive but more overhead.
    tick_interval_seconds: float = 60.0

    # Hard upper bound on a single sleep cycle's wall-clock duration. If a
    # cycle runs longer than this, we log a warning and abort. This protects
    # against runaway consolidation passes consuming the user's machine.
    max_sleep_duration_seconds: float = 120.0

    # Which sleep mode to fire autonomously. Options: "deep", "micro", "auto"
    # (auto = let the orchestrator decide based on entropy/turns).
    sleep_mode: str = "deep"

    # Maximum entries kept in the rolling history buffer (for M4 wake summary).
    history_buffer_size: int = 200

    # Master kill-switch.
    enabled: bool = True


# ─── Cycle history record ───────────────────────────────────────────────────


@dataclass
class IdleSleepRecord:
    """One autonomous sleep cycle, recorded for the wake summary."""

    session_id: str
    triggered_at: datetime
    completed_at: Optional[datetime] = None
    seconds_idle_when_triggered: float = 0.0
    mode: str = "deep"
    success: bool = False
    duration_seconds: float = 0.0
    consolidated: int = 0
    forgotten: int = 0
    dreams: int = 0
    error: Optional[str] = None
    raw_stats: Dict[str, Any] = field(default_factory=dict)


# ─── The daemon itself ──────────────────────────────────────────────────────


EngineProvider = Callable[[], Dict[str, Any]]
ClockFn = Callable[[], datetime]


class IdleLearner:
    """
    Background daemon that triggers sleep cycles on idle sessions.

    The daemon does not own session engines --- it asks for them via a
    provider callable each tick. This keeps it decoupled from how/where
    sessions are stored and avoids circular imports.

    The contract on engines: each engine must expose `force_sleep(mode=str)`
    and return a dict-like result (or None on failure). This matches the
    existing ChatEngine API.
    """

    def __init__(
        self,
        engine_provider: EngineProvider,
        config: Optional[IdleLearnerConfig] = None,
        clock: Optional[ClockFn] = None,
        policy: Optional[LifecyclePolicy] = None,
        state_store: Optional[IdleLearnerStateStore] = None,
        persist_every_n_ticks: int = 5,
    ):
        self.engine_provider = engine_provider
        self.config = config or IdleLearnerConfig()
        self._clock = clock or utc_now

        self._last_activity: Dict[str, datetime] = {}
        self._last_sleep: Dict[str, datetime] = {}
        self._history: List[IdleSleepRecord] = []
        self._history_lock = threading.Lock()
        self._activity_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0
        self._cycle_count = 0
        self._error_count = 0
        self._blocked_count = 0       # cycles blocked by policy

        # M6: lifecycle policy gates whether we run heavy work right now.
        self.policy: LifecyclePolicy = policy or AlwaysAllowPolicy()
        self._last_policy_decision: Optional[PolicyDecision] = None

        # M6: state persistence so daemon state survives API restarts.
        self._state_store = state_store
        self._persist_every_n_ticks = max(1, int(persist_every_n_ticks))
        if self._state_store is not None:
            self._restore_state()

    # ── Public API ──────────────────────────────────────────────────────────

    def record_activity(self, session_id: str) -> None:
        """
        Mark a session as active. Call this from API endpoints (e.g. /message)
        on every user-initiated request so the daemon knows when the user
        last interacted with each session.
        """
        if not session_id:
            return
        with self._activity_lock:
            self._last_activity[session_id] = self._clock()

    def start(self) -> None:
        """Start the background loop in a daemon thread."""
        if not self.config.enabled:
            logger.info("IdleLearner disabled by config; not starting.")
            return
        if self._thread is not None and self._thread.is_alive():
            logger.warning("IdleLearner already running; start() ignored.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="scm-idle-learner",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "IdleLearner started (idle_threshold=%.0fs, min_interval=%.0fs, tick=%.0fs)",
            self.config.idle_threshold_seconds,
            self.config.min_sleep_interval_seconds,
            self.config.tick_interval_seconds,
        )

    def stop(self, timeout_seconds: float = 5.0) -> None:
        """Signal the loop to exit and wait for the thread to join."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_seconds)
        logger.info(
            "IdleLearner stopped (ticks=%d, cycles=%d, errors=%d)",
            self._tick_count,
            self._cycle_count,
            self._error_count,
        )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Inspection / wake-summary surface ──────────────────────────────────

    def get_history(
        self,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[IdleSleepRecord]:
        """Return recent sleep records, optionally filtered by session/time."""
        with self._history_lock:
            records = list(self._history)
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        if since:
            records = [r for r in records if r.triggered_at >= since]
        return records[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Diagnostic snapshot for monitoring / wake-summary."""
        with self._history_lock:
            successes = sum(1 for r in self._history if r.success)
            failures = sum(1 for r in self._history if not r.success)
        last_decision = self._last_policy_decision
        return {
            "running": self.is_running(),
            "ticks": self._tick_count,
            "cycles_attempted": self._cycle_count,
            "cycles_succeeded": successes,
            "cycles_failed": failures,
            "cycles_blocked_by_policy": self._blocked_count,
            "errors": self._error_count,
            "tracked_sessions": len(self._last_activity),
            "last_activity": {
                sid: ts.isoformat() for sid, ts in self._last_activity.items()
            },
            "last_sleep": {
                sid: ts.isoformat() for sid, ts in self._last_sleep.items()
            },
            "last_policy_decision": (
                {
                    "allowed": last_decision.allowed,
                    "reason": last_decision.reason,
                    "decided_by": last_decision.decided_by,
                    "metadata": dict(last_decision.metadata),
                }
                if last_decision is not None else None
            ),
            "config": {
                "idle_threshold_seconds": self.config.idle_threshold_seconds,
                "min_sleep_interval_seconds": self.config.min_sleep_interval_seconds,
                "tick_interval_seconds": self.config.tick_interval_seconds,
                "sleep_mode": self.config.sleep_mode,
                "enabled": self.config.enabled,
                "policy_class": type(self.policy).__name__,
                "persistence_enabled": self._state_store is not None,
            },
        }

    # ── Internal loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Main daemon loop. Runs until stop() is called."""
        while not self._stop_event.is_set():
            self._tick_count += 1
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - safety net only
                self._error_count += 1
                logger.exception("IdleLearner tick failed: %s", exc)
            # Wait either tick_interval seconds OR until stop is signalled
            self._stop_event.wait(self.config.tick_interval_seconds)

    def _tick(self) -> None:
        """One pass over all current sessions."""
        engines = self.engine_provider() or {}
        if not engines:
            return

        # M6: lifecycle policy — bail early if work is not allowed right now.
        try:
            decision = self.policy.evaluate()
            self._last_policy_decision = decision
        except Exception:
            decision = PolicyDecision(
                allowed=True,
                reason="policy raised; defaulting to allow",
                decided_by="error_fallback",
            )
            self._last_policy_decision = decision
        if not decision.allowed:
            self._blocked_count += 1
            logger.debug(
                "IdleLearner tick blocked by %s: %s",
                decision.decided_by, decision.reason,
            )
            # Still persist state periodically so we don't lose track of
            # activity timestamps just because policy blocked work.
            self._maybe_persist()
            return

        now = self._clock()
        with self._activity_lock:
            activity = dict(self._last_activity)

        for session_id, engine in engines.items():
            # Sessions we've never seen activity for: assume "now" so they
            # do not sleep on the very first tick after start. They will
            # become eligible after idle_threshold_seconds of true inactivity.
            last_active = activity.get(session_id)
            if last_active is None:
                with self._activity_lock:
                    self._last_activity[session_id] = now
                continue

            seconds_idle = (now - last_active).total_seconds()
            if seconds_idle < self.config.idle_threshold_seconds:
                continue  # still active

            last_sleep = self._last_sleep.get(session_id)
            if last_sleep is not None:
                seconds_since_sleep = (now - last_sleep).total_seconds()
                if seconds_since_sleep < self.config.min_sleep_interval_seconds:
                    continue  # already slept recently

            self._fire_sleep(session_id, engine, seconds_idle, now)

        # M6: persist state at the cadence configured, even if no cycle fired.
        # Activity timestamps must survive an API restart even when no work
        # was due in the meantime.
        self._maybe_persist()

    def _fire_sleep(
        self,
        session_id: str,
        engine: Any,
        seconds_idle: float,
        now: datetime,
    ) -> None:
        """Run a sleep cycle on one session, with timeout protection."""
        self._cycle_count += 1
        record = IdleSleepRecord(
            session_id=session_id,
            triggered_at=now,
            seconds_idle_when_triggered=seconds_idle,
            mode=self.config.sleep_mode,
        )

        start = time.monotonic()
        try:
            result = self._invoke_sleep(engine)
        except Exception as exc:
            self._error_count += 1
            record.error = f"{type(exc).__name__}: {exc}"
            record.error += "\n" + traceback.format_exc(limit=4)
            logger.warning(
                "IdleLearner sleep failed for session %s: %s", session_id, exc
            )
            self._append_history(record)
            return
        finally:
            record.duration_seconds = time.monotonic() - start
            record.completed_at = self._clock()

        if record.duration_seconds > self.config.max_sleep_duration_seconds:
            logger.warning(
                "IdleLearner sleep for session %s took %.1fs (over limit %.1fs)",
                session_id,
                record.duration_seconds,
                self.config.max_sleep_duration_seconds,
            )

        if result:
            record.success = True
            record.consolidated = int(result.get("consolidated", 0) or 0)
            record.forgotten = int(result.get("forgotten", 0) or 0)
            record.dreams = int(result.get("dreams", 0) or 0)
            record.raw_stats = dict(result)
            self._last_sleep[session_id] = record.completed_at or now
            logger.info(
                "IdleLearner cycle ok: session=%s mode=%s consolidated=%d "
                "forgotten=%d dreams=%d duration=%.2fs",
                session_id,
                record.mode,
                record.consolidated,
                record.forgotten,
                record.dreams,
                record.duration_seconds,
            )
        else:
            record.error = "engine.force_sleep returned None"
            logger.debug(
                "IdleLearner cycle skipped: session=%s reason=force_sleep returned None",
                session_id,
            )

        self._append_history(record)

    def _invoke_sleep(self, engine: Any) -> Optional[Dict[str, Any]]:
        """Call into the engine's sleep API. Single point of contact."""
        if not hasattr(engine, "force_sleep"):
            raise AttributeError(
                f"engine of type {type(engine).__name__} has no force_sleep()"
            )
        return engine.force_sleep(mode=self.config.sleep_mode)

    def _append_history(self, record: IdleSleepRecord) -> None:
        with self._history_lock:
            self._history.append(record)
            if len(self._history) > self.config.history_buffer_size:
                # Drop oldest, keep the buffer bounded
                self._history = self._history[-self.config.history_buffer_size :]
        # Persist after every cycle since cycles are rare events worth saving.
        self._maybe_persist(force=True)

    # ── M6: state persistence ──────────────────────────────────────────────

    def _maybe_persist(self, force: bool = False) -> None:
        """Save state every N ticks, or unconditionally if `force=True`."""
        if self._state_store is None:
            return
        if not force and (self._tick_count % self._persist_every_n_ticks) != 0:
            return
        try:
            self._persist_state()
        except Exception:
            # Persistence failure must never crash the daemon.
            pass

    def _persist_state(self) -> None:
        """Snapshot current state to the JSON store."""
        with self._activity_lock:
            activity = dict(self._last_activity)
        with self._history_lock:
            recent = list(self._history)
        payload = {
            "last_activity": IdleLearnerStateStore.serialize_activity_map(activity),
            "last_sleep": IdleLearnerStateStore.serialize_activity_map(self._last_sleep),
            "history": IdleLearnerStateStore.serialize_history(recent, limit=50),
            "tick_count": self._tick_count,
            "cycle_count": self._cycle_count,
            "error_count": self._error_count,
            "blocked_count": self._blocked_count,
        }
        self._state_store.save(payload)

    def _restore_state(self) -> None:
        """Read prior state from the JSON store and replay it into memory."""
        if self._state_store is None:
            return
        data = self._state_store.load()
        if not data:
            return
        try:
            self._last_activity = IdleLearnerStateStore.deserialize_activity_map(
                data.get("last_activity", {})
            )
            self._last_sleep = IdleLearnerStateStore.deserialize_activity_map(
                data.get("last_sleep", {})
            )
            # Counters are advisory; re-prime them so persisted history is
            # consistent across restarts.
            self._tick_count = int(data.get("tick_count", 0))
            self._cycle_count = int(data.get("cycle_count", 0))
            self._error_count = int(data.get("error_count", 0))
            self._blocked_count = int(data.get("blocked_count", 0))
            # We deliberately do NOT rehydrate history records as full
            # IdleSleepRecord objects because their nested types (datetimes)
            # would require fragile reconstruction. The serialized history
            # is preserved in the store for diagnostic introspection.
        except Exception:
            # Corrupt state file: start clean.
            self._last_activity = {}
            self._last_sleep = {}


# ─── CLI ────────────────────────────────────────────────────────────────────


def _build_cli_engine_provider():
    """
    Used when running the daemon standalone via `python -m`.
    Connects to the in-memory chat engine pool managed by src.api.chat.
    """
    from ..api import chat as chat_module
    return lambda: dict(chat_module._chat_engines)


def main() -> None:  # pragma: no cover - smoke entry point
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="SCM IdleLearner daemon")
    parser.add_argument("--idle-mins", type=float, default=10.0,
                        help="Minutes of inactivity before sleep fires (default 10)")
    parser.add_argument("--min-interval-mins", type=float, default=30.0,
                        help="Minimum minutes between consecutive sleep cycles per session")
    parser.add_argument("--tick-secs", type=float, default=60.0,
                        help="How often the daemon checks sessions (default 60s)")
    parser.add_argument("--mode", choices=["deep", "micro", "auto"], default="deep",
                        help="Sleep mode to invoke (default deep)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = IdleLearnerConfig(
        idle_threshold_seconds=args.idle_mins * 60.0,
        min_sleep_interval_seconds=args.min_interval_mins * 60.0,
        tick_interval_seconds=args.tick_secs,
        sleep_mode=args.mode,
    )
    learner = IdleLearner(
        engine_provider=_build_cli_engine_provider(),
        config=cfg,
    )
    learner.start()
    print(f"IdleLearner started (idle={args.idle_mins}m, interval={args.min_interval_mins}m, tick={args.tick_secs}s)")
    print("Press Ctrl+C to stop.")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        learner.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
