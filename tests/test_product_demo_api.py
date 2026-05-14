"""Product API regression tests for demo + readiness reporting."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from src.api.main import app


def _session(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def test_product_demo_endpoint_executes_full_story():
    client = TestClient(app)
    session_id = _session("product_demo")

    response = client.post(f"/chat/product-demo/{session_id}")
    assert response.status_code == 200

    payload = response.json()
    assert payload["success"] is True
    assert payload["session_id"] == session_id

    results = payload["results"]
    assert results["total_checks"] == 3
    assert results["passed_checks"] == results["total_checks"]
    assert results["checks"]["one_shot_name"] is True
    assert results["checks"]["one_shot_location"] is True
    assert results["checks"]["contradiction_update"] is True

    product_report = payload["product_report"]
    assert product_report["session_id"] == session_id
    assert product_report["readiness"]["score"] >= 35.0
    assert product_report["readiness"]["flags"]["runtime_signals_pass"] is True


def test_product_report_endpoint_shape_and_runtime_flags():
    client = TestClient(app)
    session_id = _session("product_report")

    # Seed conversation state before report fetch.
    client.post("/chat/message", json={"message": "My name is Alice.", "session_id": session_id})
    client.post("/chat/message", json={"message": "I live in Seattle.", "session_id": session_id})
    client.post("/chat/message", json={"message": "I prefer morning meetings.", "session_id": session_id})

    response = client.get(f"/chat/product-report/{session_id}")
    assert response.status_code == 200

    payload = response.json()
    report = payload["report"]
    diagnostics = report["diagnostics"]
    readiness = report["readiness"]
    packs = report["benchmarks"]["packs"]

    assert diagnostics["session_id"] == session_id
    assert "human_like_signals" in diagnostics
    assert diagnostics["human_like_signals"]["one_shot_ready"] is True

    assert "phase4" in packs
    assert "phase6_human" in packs
    assert "phase6_guardrails" in packs
    assert "phase6_demo" in packs

    assert readiness["max_score"] == 100.0
    assert 0.0 <= readiness["score"] <= 100.0
    assert "overall_pass" in readiness["flags"]


def test_backend_smoke_endpoint_returns_single_verdict():
    client = TestClient(app)
    session_id = _session("backend_smoke")

    response = client.post(f"/chat/backend-smoke/{session_id}")
    assert response.status_code == 200

    payload = response.json()
    report = payload["report"]
    checks = report["checks"]

    assert report["session_id"] == session_id
    assert report["overall_pass"] is True
    assert checks["demo_checks_pass"] is True
    assert checks["runtime_signals_pass"] is True
    assert checks["messages_recorded"] is True
    assert checks["sleep_cycles_recorded"] is True
    assert "how_to_replay" in report


def test_memory_report_endpoint_serializes_after_demo_session():
    client = TestClient(app)
    session_id = _session("memory_report")

    demo_response = client.post(f"/chat/product-demo/{session_id}")
    assert demo_response.status_code == 200

    response = client.get(f"/chat/memory/{session_id}")
    assert response.status_code == 200
    payload = response.json()
    assert "report" in payload
    assert payload["report"]["messages_exchanged"] >= 1
