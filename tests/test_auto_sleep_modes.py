from __future__ import annotations

from typing import Any

import src.core.sqlite_db as sqlite_db
import src.integrations.mcp_server as mcp_server
import src.lifecycle.circadian as circadian


class _StubSQLite:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.marked: list[dict[str, Any]] = []

    def get_user_sleep_config(self, user_id: str) -> dict[str, Any]:
        out = dict(self.cfg)
        out.setdefault("user_id", user_id)
        return out

    def mark_user_slept(
        self,
        user_id: str,
        when_iso: str | None = None,
        reason: str | None = None,
        create_if_missing: bool = True,
    ) -> None:
        self.marked.append(
            {
                "user_id": user_id,
                "when_iso": when_iso,
                "reason": reason,
                "create_if_missing": create_if_missing,
            }
        )


def _run_sweep(
    monkeypatch,
    cfg: dict[str, Any],
    *,
    last_activity: float,
    now: float = 1000.0,
    legacy_idle_mode: bool = True,
    global_idle_threshold: float = 300.0,
):
    pool = mcp_server.UserEnginePool(
        idle_threshold_sec=global_idle_threshold,
        sweep_interval_sec=1.0,
        auto_sleep=False,
        legacy_idle_mode=legacy_idle_mode,
    )
    pool._last_activity = {"u": last_activity}
    calls: list[dict[str, Any]] = []

    def _fake_fire(user_id: str, idle_for: float, scheduled: bool = False) -> None:
        calls.append(
            {"user_id": user_id, "idle_for": idle_for, "scheduled": scheduled}
        )

    monkeypatch.setattr(pool, "_fire_sleep_for", _fake_fire)
    sqlite = _StubSQLite(cfg)
    monkeypatch.setattr(sqlite_db, "get_memory", lambda: sqlite)
    monkeypatch.setattr(
        circadian,
        "should_fire",
        lambda conf: bool(conf.get("_should_fire", False)),
    )
    monkeypatch.setattr(mcp_server.time, "time", lambda: now)
    fired = pool._sweep_once()
    return fired, calls, sqlite


def test_mode_off_disables_autonomous_sleep(monkeypatch):
    fired, calls, sqlite = _run_sweep(
        monkeypatch,
        {"auto_sleep_mode": "off", "is_default": False, "_should_fire": True},
        last_activity=0.0,
    )
    assert fired == 0
    assert calls == []
    assert sqlite.marked == []


def test_mode_idle_only_uses_idle_trigger_and_custom_threshold(monkeypatch):
    fired, calls, sqlite = _run_sweep(
        monkeypatch,
        {"auto_sleep_mode": "idle_only", "is_default": False, "idle_timeout_sec": 120},
        last_activity=850.0,
    )
    assert fired == 1
    assert calls and calls[0]["scheduled"] is False
    assert calls[0]["idle_for"] >= 120
    assert sqlite.marked and sqlite.marked[0]["reason"] == "idle"


def test_mode_night_only_uses_circadian_trigger(monkeypatch):
    fired, calls, sqlite = _run_sweep(
        monkeypatch,
        {"auto_sleep_mode": "night_only", "is_default": False, "_should_fire": True},
        last_activity=990.0,
    )
    assert fired == 1
    assert calls and calls[0]["scheduled"] is True
    assert sqlite.marked and sqlite.marked[0]["reason"] == "scheduled"


def test_mode_auto_keeps_legacy_idle_fallback_for_default_users(monkeypatch):
    fired, calls, sqlite = _run_sweep(
        monkeypatch,
        {"auto_sleep_mode": "auto", "is_default": True, "_should_fire": False},
        last_activity=650.0,
        global_idle_threshold=300.0,
    )
    assert fired == 1
    assert calls and calls[0]["scheduled"] is False
    # Legacy fallback shouldn't create/persist sleep rows.
    assert sqlite.marked == []


def test_mode_auto_uses_circadian_for_configured_users(monkeypatch):
    fired, calls, sqlite = _run_sweep(
        monkeypatch,
        {"auto_sleep_mode": "auto", "is_default": False, "_should_fire": True},
        last_activity=990.0,
    )
    assert fired == 1
    assert calls and calls[0]["scheduled"] is True
    assert sqlite.marked and sqlite.marked[0]["reason"] == "scheduled"
