"""Runtime productization API tests: profiles, sandbox, export/import, metrics."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from src.api.main import app


def _session(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def test_profiles_endpoint_lists_product_presets():
    client = TestClient(app)
    response = client.get("/chat/profiles")
    assert response.status_code == 200
    payload = response.json()
    profiles = payload["profiles"]
    assert "chatbot" in profiles
    assert "agent" in profiles
    assert "research" in profiles


def test_session_config_applies_sandbox_runtime():
    client = TestClient(app)
    session_id = _session("sandbox_cfg")

    resp = client.post(
        "/chat/session",
        json={
            "session_id": session_id,
            "profile": "agent",
            "sandbox": True,
            "reset": True,
        },
    )
    assert resp.status_code == 200
    runtime = resp.json()["session"]
    assert runtime["session_id"] == session_id
    assert runtime["profile"] == "agent"
    assert runtime["sandbox"] is True

    chat = client.post(
        "/chat/message",
        json={
            "session_id": session_id,
            "message": "My name is Alice.",
            "profile": "agent",
            "sandbox": True,
        },
    )
    assert chat.status_code == 200
    metadata = chat.json()["metadata"]
    assert metadata["runtime"]["profile"] == "agent"
    assert metadata["runtime"]["sandbox"] is True


def test_memory_export_import_round_trip_via_product_demo():
    client = TestClient(app)
    source_session = _session("export_src")
    target_session = _session("import_dst")

    demo = client.post(f"/chat/product-demo/{source_session}")
    assert demo.status_code == 200
    assert demo.json()["success"] is True

    exported = client.get(f"/chat/memory-export/{source_session}")
    assert exported.status_code == 200
    payload = exported.json()["payload"]
    assert payload["counts"]["concepts"] > 0

    imported = client.post(
        f"/chat/memory-import/{target_session}",
        json={
            "payload": payload,
            "replace_existing": True,
        },
    )
    assert imported.status_code == 200
    stats = imported.json()["import_stats"]
    assert stats["concepts_imported"] == payload["counts"]["concepts"]

    report = client.get(f"/chat/memory/{target_session}")
    assert report.status_code == 200
    total = report.json()["report"]["long_term_memory"]["total_concepts"]
    assert total >= payload["counts"]["concepts"]


def test_metrics_endpoint_exposes_prometheus_or_status_payload():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert body
    assert (
        "scm_http_requests_total" in body
        or '"enabled": false' in body
    )
