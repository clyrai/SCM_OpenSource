"""
IdleLearner state persistence — Phase 7 production hardening.

When the API server restarts, we don't want the IdleLearner to forget what
it knew about session activity. Otherwise, a fresh restart would treat every
session as "infinitely idle" and immediately fire deep-sleep on all of them,
which is exactly the behavior we worked hard to avoid.

This module provides a tiny JSON-backed store for the daemon's per-session
state: last activity timestamp, last sleep timestamp, and a truncated
history buffer. The store is:

  - Atomic: writes go to .tmp + os.replace
  - Bounded: history capped before write
  - Safe: never raises; corrupt files reset to empty
  - Simple: no dependencies, just json + Path

Default path: `data/idle_learner_state.json`. Override via env var.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_PATH = "data/idle_learner_state.json"


class IdleLearnerStateStore:
    """JSON-backed persistence for IdleLearner state."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or _DEFAULT_PATH)
        self._lock = threading.Lock()

    # ── Read ───────────────────────────────────────────────────────────────

    def load(self) -> Dict[str, Any]:
        """Return the stored state dict, or {} on miss/corruption."""
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            return {}

    # ── Write ──────────────────────────────────────────────────────────────

    def save(self, state: Dict[str, Any]) -> bool:
        """
        Atomically persist `state` as JSON. Creates the parent dir if needed.
        Returns True on success, False if the write failed (never raises).
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with self._lock:
                with tmp.open("w") as f:
                    json.dump(state, f, default=_json_default, indent=2)
                # Atomic replace
                os.replace(tmp, self.path)
            return True
        except Exception:
            return False

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def serialize_activity_map(activity: Dict[str, datetime]) -> Dict[str, str]:
        """Convert {session_id: datetime} → JSON-safe."""
        return {sid: ts.isoformat() for sid, ts in activity.items() if ts}

    @staticmethod
    def deserialize_activity_map(raw: Dict[str, str]) -> Dict[str, datetime]:
        """Convert {session_id: iso_string} → {session_id: datetime}."""
        out: Dict[str, datetime] = {}
        if not isinstance(raw, dict):
            return out
        for sid, ts in raw.items():
            try:
                out[sid] = datetime.fromisoformat(ts)
            except Exception:
                continue
        return out

    @staticmethod
    def serialize_history(records: List[Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Convert IdleSleepRecord dataclasses to JSON-safe dicts."""
        out: List[Dict[str, Any]] = []
        for r in records[-limit:]:
            try:
                if is_dataclass(r):
                    out.append(asdict(r))
                elif isinstance(r, dict):
                    out.append(r)
                else:
                    # Best-effort attribute pull
                    out.append({k: getattr(r, k, None) for k in (
                        "session_id", "triggered_at", "completed_at",
                        "seconds_idle_when_triggered", "mode",
                        "success", "duration_seconds",
                        "consolidated", "forgotten", "dreams", "error",
                    )})
            except Exception:
                continue
        return out


def _json_default(o):
    """JSON serializer fallback for datetime/dataclasses."""
    if isinstance(o, datetime):
        return o.isoformat()
    if is_dataclass(o):
        return asdict(o)
    return str(o)
