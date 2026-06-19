from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from src.api import chat_router
from src.api.main import app
import src.lifecycle.circadian as circadian


def _slug(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_chat_sleep_config_defaults_and_round_trip():
    client = TestClient(app)
    slug = _slug("sleepcfg")

    default_cfg = client.get(f"/chat/api/sleep-config/{slug}")
    assert default_cfg.status_code == 200
    body = default_cfg.json()
    assert body["auto_sleep_mode"] == "auto"
    assert body["effective_mode"] == "idle_only"
    assert body["is_default_schedule"] is True
    assert body["idle_timeout_sec"] is None

    saved = client.post(
        f"/chat/api/sleep-config/{slug}",
        json={
            "auto_sleep_mode": "night_only",
            "timezone": "Europe/Lisbon",
            "sleep_start": "22:30",
            "sleep_end": "06:45",
            "enabled": True,
        },
    )
    assert saved.status_code == 200
    payload = saved.json()
    assert payload["auto_sleep_mode"] == "night_only"
    assert payload["effective_mode"] == "night_only"
    assert payload["timezone"] == "Europe/Lisbon"
    assert payload["sleep_start"] == "22:30"
    assert payload["sleep_end"] == "06:45"
    assert payload["is_default_schedule"] is False


def test_chat_sleep_config_rejects_bad_inputs():
    client = TestClient(app)
    slug = _slug("sleepcfgbad")

    bad_mode = client.post(
        f"/chat/api/sleep-config/{slug}",
        json={"auto_sleep_mode": "sometimes"},
    )
    assert bad_mode.status_code == 400
    assert "auto_sleep_mode" in bad_mode.json()["detail"]

    bad_idle = client.post(
        f"/chat/api/sleep-config/{slug}",
        json={"auto_sleep_mode": "idle_only", "idle_timeout_sec": 0},
    )
    assert bad_idle.status_code == 400
    assert "idle_timeout_sec" in bad_idle.json()["detail"]


def test_chat_history_surfaces_pending_wake_summary_and_consumes_it():
    client = TestClient(app)
    slug = _slug("historywake")
    sess = chat_router._pool.get_or_create(slug)
    before = sess.last_activity
    with sess._lock:
        sess.pending_wake_summary = {
            "narrative": "While you were away, I noticed a pattern.",
            "generated_at": "2026-05-15T00:00:00+00:00",
            "reason": "scheduled",
        }

    response = client.get(f"/chat/api/history/{slug}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["wake_summary"]["narrative"].startswith("While you were away")
    assert sess.pending_wake_summary is None
    assert sess.last_activity >= before


def test_chat_pool_auto_sleep_supports_idle_and_night_modes(monkeypatch):
    pool = chat_router._ChatPool()
    pool._min_turns = 1
    sess = chat_router._ChatSession("auto_sleep_test")
    sess.transcript = [{"user": "hello"}]

    def fake_sleep(session, reason: str, byok=None):
        session.last_sleep_at = "2026-05-15T00:00:00+00:00"
        session.last_sleep_reason = reason
        return {
            "ok": True,
            "narrative": f"{reason} narrative",
            "fired_at": "2026-05-15T00:00:00+00:00",
        }

    monkeypatch.setattr(chat_router, "_run_chat_sleep_cycle", fake_sleep)

    sess.sleep_config.auto_sleep_mode = "idle_only"
    sess.sleep_config.idle_timeout_sec = 5
    sess.last_activity = 0.0
    monkeypatch.setattr(chat_router.time, "time", lambda: 100.0)
    assert pool._maybe_auto_sleep(sess) is True
    assert sess.pending_wake_summary["reason"] == "idle"
    assert sess.last_sleep_reason == "idle"

    with sess._lock:
        sess.pending_wake_summary = None
    sess.sleep_config.auto_sleep_mode = "night_only"
    monkeypatch.setattr(circadian, "should_fire", lambda *_a, **_k: True)
    assert pool._maybe_auto_sleep(sess) is True
    assert sess.pending_wake_summary["reason"] == "scheduled"
    assert sess.last_sleep_reason == "scheduled"


def test_chat_byok_capabilities_endpoint_reports_provider_support(monkeypatch):
    monkeypatch.setattr(
        chat_router._BYOKSemanticReranker,
        "_DEFAULT_MODELS",
        {
            "deepseek": "",
            "openai": "text-embedding-3-small",
            "groq": "",
            "together": "together-embed",
            "openrouter": "openai/text-embedding-3-small",
        },
    )
    client = TestClient(app)
    response = client.get("/chat/api/byok-capabilities")
    assert response.status_code == 200
    payload = response.json()["semantic_rerank"]["providers"]
    assert payload["openai"]["supported"] is True
    assert payload["deepseek"]["supported"] is False
    assert "DEEPSEEK" in payload["deepseek"]["note"]


def test_chat_byok_capabilities_probe_overrides_provider(monkeypatch):
    monkeypatch.setattr(
        chat_router._BYOKSemanticReranker,
        "_DEFAULT_MODELS",
        {
            "deepseek": "",
            "openai": "text-embedding-3-small",
            "groq": "",
            "together": "together-embed",
            "openrouter": "openai/text-embedding-3-small",
        },
    )
    monkeypatch.setattr(
        chat_router,
        "_probe_semantic_rerank_model",
        lambda provider, api_key: {
            "supported": provider == "deepseek" and api_key == "sk-probe",
            "model": "deepseek-embedding" if api_key == "sk-probe" else None,
            "note": "Available for on-demand query reranking." if api_key == "sk-probe" else "probe failed",
        },
    )
    client = TestClient(app)
    response = client.post(
        "/chat/api/byok-capabilities",
        json={"llm_provider": "deepseek", "llm_api_key": "sk-probe"},
    )
    assert response.status_code == 200
    payload = response.json()["semantic_rerank"]["providers"]
    assert payload["deepseek"]["supported"] is True
    assert payload["deepseek"]["model"] == "deepseek-embedding"
    assert payload["openai"]["supported"] is True


def test_build_semantic_reranker_uses_cached_probe_model(monkeypatch):
    monkeypatch.setattr(chat_router, "_requests", object())
    sess = chat_router._ChatSession("rerank_cache_test")
    chat_router._set_cached_semantic_model("deepseek", "sk-cache", "deepseek-embedding")
    reranker = chat_router._build_semantic_reranker("deepseek", "sk-cache", sess)
    assert reranker is not None
    assert reranker.model == "deepseek-embedding"
    chat_router._set_cached_semantic_model("deepseek", "sk-cache", "")
