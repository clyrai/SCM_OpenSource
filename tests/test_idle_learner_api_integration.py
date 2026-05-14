"""
Integration test: boots the FastAPI app with IdleLearner enabled, sends
a couple of chat messages, simulates the user going away, and verifies
that the daemon fires a sleep cycle on the idle session and exposes it
via the /chat/idle-learner/* endpoints.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module", autouse=True)
def _enable_idle_learner_env():
    # Configure for fast test cycles
    prev = {}
    for k, v in {
        "IDLE_LEARNER_ENABLED": "true",
        "IDLE_LEARNER_IDLE_THRESHOLD_SECONDS": "0.5",
        "IDLE_LEARNER_MIN_SLEEP_INTERVAL_SECONDS": "0.1",
        "IDLE_LEARNER_TICK_INTERVAL_SECONDS": "0.1",
        "IDLE_LEARNER_SLEEP_MODE": "deep",
    }.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = v
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_app_boots_with_idle_learner_and_endpoint_responds():
    """
    Boot the FastAPI app, hit /chat/idle-learner/status — should report
    the daemon is running.
    """
    # Import AFTER env is set so config picks it up
    from fastapi.testclient import TestClient
    from src.api import main as main_module

    # Force reload of config so the env vars take effect
    import importlib
    from src.core import config as cfg_module
    importlib.reload(cfg_module)
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        r = client.get("/chat/idle-learner/status")
        assert r.status_code == 200
        body = r.json()
        # If daemon is enabled but no sessions exist, it should still be reported running
        assert body.get("running") is True or body.get("enabled") is False
        # Either way the endpoint must respond cleanly with the right schema
        assert "config" in body or "message" in body


def test_chat_message_records_activity_and_idle_fires_sleep(tmp_path):
    """
    Full flow: create a session, send a message, wait past the idle
    threshold, check the daemon fires sleep and the history endpoint
    surfaces it.
    """
    from fastapi.testclient import TestClient
    from src.api import main as main_module
    from src.api import chat as chat_module
    from src.core import config as cfg_module
    import importlib
    importlib.reload(cfg_module)
    importlib.reload(main_module)

    # Reset chat engine pool so we have a clean slate
    chat_module._chat_engines.clear()
    chat_module._session_runtime.clear()

    with TestClient(main_module.app) as client:
        # Create a session in sandbox mode so we don't touch the real DB
        sid = "idle_api_test"
        r = client.post(
            "/chat/session",
            json={
                "session_id": sid,
                "profile": "research",
                "sandbox": True,
                "reset": True,
            },
        )
        assert r.status_code == 200, r.text

        # Send a message — this should record activity for the daemon
        r = client.post(
            "/chat/message",
            json={"message": "My name is Alice. I work at GreenLeaf Cafe.", "session_id": sid, "sandbox": True},
        )
        assert r.status_code == 200, r.text

        # Wait past the idle threshold + a few ticks
        time.sleep(1.5)

        # Status endpoint should report activity tracked
        r = client.get("/chat/idle-learner/status")
        assert r.status_code == 200
        body = r.json()
        # The daemon should know about our session
        assert sid in body.get("last_activity", {})

        # History endpoint may or may not have a record yet depending on
        # timing — give it a few more iterations of the daemon loop.
        for _ in range(10):
            r = client.get(f"/chat/idle-learner/history?session_id={sid}")
            assert r.status_code == 200
            recs = r.json().get("records", [])
            if len(recs) >= 1:
                break
            time.sleep(0.3)
        else:
            pytest.fail(
                f"IdleLearner did not fire on session {sid} within timeout. "
                f"Final body: {r.json()}"
            )

        # The recorded cycle should be from our session and should have
        # completed_at populated
        rec = recs[0]
        assert rec["session_id"] == sid
        assert rec["completed_at"] is not None
