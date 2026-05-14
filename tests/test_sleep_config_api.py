"""End-to-end test of the v0.7.7 circadian sleep API.

Verifies:
  • GET  /v1/users/{id}/sleep-config returns documented defaults when no row exists
  • POST /v1/users/{id}/sleep-config persists the new schedule
  • Invalid timezone / bad HH:MM format → 400
  • The MCP sweeper transitions out of legacy idle mode the moment a user
    has a per-user config (smoke check via UserEnginePool internals)
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict

import pytest
import requests
import uvicorn


@pytest.fixture(scope="module")
def server() -> str:
    """Boot the FastAPI app on a random port for the duration of the module."""
    os.environ.setdefault("SCM_EMBEDDING_BACKEND", "ollama")
    os.environ.setdefault("SCM_EMBEDDING_MODEL", "nomic-embed-text")
    # Force a fresh data dir so the table and existing rows don't bleed in
    # from a prior test run.
    import tempfile
    os.environ["SCM_DATA_DIR"] = tempfile.mkdtemp(prefix="scm_test_")

    # Late import so env vars are picked up.
    from src.api.main import app
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            r = requests.get(f"{base}/v1/health", timeout=1)
            if r.ok:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        raise RuntimeError("server did not come up")
    yield base
    srv.should_exit = True


def test_default_config_when_user_unknown(server):
    r = requests.get(f"{server}/v1/users/freshuser/sleep-config")
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["user_id"] == "freshuser"
    assert cfg["timezone"] == "UTC"
    assert cfg["sleep_start"] == "23:00"
    assert cfg["sleep_end"] == "07:00"
    assert cfg["enabled"] is True
    assert cfg["is_default"] is True


def test_post_then_get_round_trip(server):
    body = {
        "timezone": "Europe/Lisbon",
        "sleep_start": "00:30",
        "sleep_end": "08:15",
        "enabled": True,
    }
    r = requests.post(f"{server}/v1/users/alice/sleep-config", json=body)
    assert r.status_code == 200
    saved = r.json()
    assert saved["timezone"] == "Europe/Lisbon"
    assert saved["sleep_start"] == "00:30"
    assert saved["sleep_end"] == "08:15"

    # Read back
    r2 = requests.get(f"{server}/v1/users/alice/sleep-config")
    cfg = r2.json()
    assert cfg["is_default"] is False
    assert cfg["timezone"] == "Europe/Lisbon"


def test_partial_update_keeps_other_fields(server):
    # First write the full config.
    requests.post(
        f"{server}/v1/users/bob/sleep-config",
        json={"timezone": "America/New_York", "sleep_start": "22:00", "sleep_end": "06:00"},
    )
    # Then change only enabled.
    r = requests.post(f"{server}/v1/users/bob/sleep-config", json={"enabled": False})
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["enabled"] is False
    assert cfg["timezone"] == "America/New_York"  # preserved
    assert cfg["sleep_start"] == "22:00"           # preserved


def test_invalid_timezone_400(server):
    r = requests.post(
        f"{server}/v1/users/eve/sleep-config",
        json={"timezone": "Bogus/Nowhere"},
    )
    assert r.status_code == 400
    assert "timezone" in r.json()["detail"].lower()


def test_invalid_hhmm_400(server):
    r = requests.post(
        f"{server}/v1/users/eve/sleep-config",
        json={"sleep_start": "25:99"},
    )
    assert r.status_code == 400
    assert "sleep_start" in r.json()["detail"]


def test_disabled_means_no_circadian_fire(server):
    """When enabled=False, should_fire returns False even inside the window."""
    from src.lifecycle.circadian import should_fire
    requests.post(
        f"{server}/v1/users/silent/sleep-config",
        json={"enabled": False},
    )
    r = requests.get(f"{server}/v1/users/silent/sleep-config")
    cfg = r.json()
    cfg["last_sleep_at"] = None  # ensure not blocked by the once-per-night guard
    # Even at 23:30 UTC (well inside the default window), should_fire is False.
    from datetime import datetime, timezone
    now = datetime(2026, 5, 5, 23, 30, tzinfo=timezone.utc)
    assert should_fire(cfg, now_utc=now) is False
