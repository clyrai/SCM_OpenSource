"""End-to-end tests for SCM Cloud auth + per-tenant isolation.

Covers:
  • signup → mint initial key → use that key on /v1/memories
  • account-scoped /v1/cloud/me/* endpoints (whoami, mint key, list, revoke)
  • Bearer token enforcement: missing/invalid/revoked → 401
  • per-key rate limit triggers 429
  • cross-tenant isolation: two accounts can't read each other's memories
  • BYOK LLM config: set, decrypt under master key, never returned over API
  • cloud-auth-off mode: routes work without any token
"""
from __future__ import annotations

import os
import socket
import threading
import tempfile
import time
import uuid
from typing import Iterator

import pytest
import requests
import uvicorn


# ── Fixtures ────────────────────────────────────────────────────────────


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _spawn_server(port: int, env: dict) -> uvicorn.Server:
    for k, v in env.items():
        os.environ[k] = v

    # Force a clean data dir so this test doesn't bleed into other tests.
    os.environ["SCM_DATA_DIR"] = tempfile.mkdtemp(prefix="scm_cloud_test_")

    # Late import so env vars are applied before SCM modules read them.
    import importlib
    import src.api.main as _main
    importlib.reload(_main)
    config = uvicorn.Config(
        _main.app, host="127.0.0.1", port=port, log_level="error", lifespan="on",
    )
    srv = uvicorn.Server(config)
    threading.Thread(target=srv.run, daemon=True).start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            r = requests.get(f"{base}/v1/health", timeout=1)
            if r.ok:
                return srv
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not come up")


@pytest.fixture(scope="function")
def cloud_server() -> Iterator[str]:
    """Boot SCM with cloud auth ON. Each test gets a fresh server +
    fresh data dir so state doesn't leak."""
    port = _free_port()
    srv = _spawn_server(port, {
        "SCM_CLOUD_AUTH": "1",
        "SCM_CLOUD_SECRET_KEY": "test-secret-key-with-32-bytes-min-chars",
        "SCM_EMBEDDING_BACKEND": "ollama",
        "SCM_EMBEDDING_MODEL": "nomic-embed-text",
    })
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True


@pytest.fixture(scope="function")
def open_server() -> Iterator[str]:
    """SCM with cloud auth OFF — verifies the open-source self-hosted
    shape still works."""
    port = _free_port()
    srv = _spawn_server(port, {
        "SCM_CLOUD_AUTH": "0",
        "SCM_EMBEDDING_BACKEND": "ollama",
        "SCM_EMBEDDING_MODEL": "nomic-embed-text",
    })
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True


# ── Helpers ─────────────────────────────────────────────────────────────


def _signup_and_get_token(base: str, email: str) -> tuple[str, str]:
    r = requests.post(f"{base}/v1/cloud/accounts", json={"email": email})
    r.raise_for_status()
    aid = r.json()["id"]
    r = requests.post(f"{base}/v1/cloud/accounts/{aid}/keys/initial",
                      json={"label": "first"})
    r.raise_for_status()
    token = r.json()["token"]
    return aid, token


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Tests: cloud-auth ON ────────────────────────────────────────────────


def test_signup_then_use_key(cloud_server: str):
    base = cloud_server
    aid, token = _signup_and_get_token(base, f"alice-{uuid.uuid4().hex[:6]}@example.com")
    assert token.startswith("scm_live_")

    # whoami works with the new key
    r = requests.get(f"{base}/v1/cloud/me", headers=_h(token))
    assert r.status_code == 200
    assert r.json()["id"] == aid

    # /v1/memories works with the new key
    r = requests.post(
        f"{base}/v1/memories", headers=_h(token),
        json={"text": "Alice loves filter coffee.", "sync": True},
    )
    assert r.status_code == 200, r.text


def test_missing_bearer_returns_401(cloud_server: str):
    r = requests.post(f"{cloud_server}/v1/memories", json={"text": "x"})
    assert r.status_code == 401
    assert "Authorization" in r.json().get("error", "")


def test_invalid_token_returns_401(cloud_server: str):
    r = requests.post(
        f"{cloud_server}/v1/memories",
        headers={"Authorization": "Bearer scm_live_bogus"},
        json={"text": "x"},
    )
    assert r.status_code == 401


def test_revoked_key_returns_401(cloud_server: str):
    base = cloud_server
    aid, token = _signup_and_get_token(base, f"bob-{uuid.uuid4().hex[:6]}@example.com")
    keys = requests.get(f"{base}/v1/cloud/me/keys", headers=_h(token)).json()["keys"]
    key_id = keys[0]["id"]

    # Mint a SECOND key first so we can still call /me/keys/{id} after revoking
    r2 = requests.post(f"{base}/v1/cloud/me/keys", headers=_h(token), json={"label": "second"})
    second_token = r2.json()["token"]

    requests.delete(f"{base}/v1/cloud/me/keys/{key_id}", headers=_h(second_token))

    # Original token now 401s
    r = requests.get(f"{base}/v1/cloud/me", headers=_h(token))
    assert r.status_code == 401


def test_health_and_tools_are_public(cloud_server: str):
    """Public routes shouldn't require auth even when cloud auth is on."""
    assert requests.get(f"{cloud_server}/v1/health").status_code == 200
    assert requests.get(f"{cloud_server}/v1/tools").status_code == 200


def test_cross_tenant_isolation(cloud_server: str):
    """Alice's memory must not be visible to Bob even if Bob guesses the user_id."""
    base = cloud_server
    a_id, a_tok = _signup_and_get_token(base, f"alice-{uuid.uuid4().hex[:6]}@x.com")
    b_id, b_tok = _signup_and_get_token(base, f"bob-{uuid.uuid4().hex[:6]}@x.com")

    # Both Alice and Bob use user_id="default" — namespacing must keep them apart
    requests.post(
        f"{base}/v1/memories", headers=_h(a_tok),
        json={"text": "Alice's secret pet name is Whiskers.", "user_id": "default", "sync": True},
    ).raise_for_status()

    # Bob searches with the SAME user_id="default"
    r = requests.post(
        f"{base}/v1/memories/search", headers=_h(b_tok),
        json={"query": "Whiskers", "user_id": "default", "wait_for_pending": True},
    )
    r.raise_for_status()
    blob = (r.json().get("memory_context") or "").lower()
    descs = [m.get("description", "").lower() for m in (r.json().get("memories") or [])]
    assert "whiskers" not in blob
    assert not any("whiskers" in d for d in descs)


def test_byok_set_and_clear(cloud_server: str):
    """BYOK LLM key can be set, never echoed back, and cleared."""
    base = cloud_server
    _, token = _signup_and_get_token(base, f"carol-{uuid.uuid4().hex[:6]}@x.com")
    r = requests.post(
        f"{base}/v1/cloud/me/byok", headers=_h(token),
        json={"provider": "deepseek", "api_key": "sk-secret-llm-key-123",
              "base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # The api_key must NOT be returned anywhere — only provider+base_url
    assert "api_key" not in body
    assert body["provider"] == "deepseek"
    assert body["base_url"] == "https://api.deepseek.com"

    # /me must not leak it either
    me = requests.get(f"{base}/v1/cloud/me", headers=_h(token)).json()
    assert "byok_llm_api_key_enc" not in me
    assert "api_key" not in str(me)

    # Clear works
    r = requests.delete(f"{base}/v1/cloud/me/byok", headers=_h(token))
    assert r.status_code == 200


def test_byok_decryption_round_trip():
    """The cloud package's encrypt/decrypt round-trips under a known master key."""
    os.environ["SCM_CLOUD_SECRET_KEY"] = "test-secret-key-with-32-bytes-min-chars"
    # Force module-level reload so the master key gets re-read
    import importlib, src.cloud.accounts as a
    importlib.reload(a)
    plain = "sk-some-deepseek-secret"
    enc = a._encrypt(plain)
    assert enc != plain
    dec = a._decrypt(enc)
    assert dec == plain


def test_per_key_rate_limit(cloud_server: str):
    """Tier-default rate limit fires 429 once exceeded."""
    base = cloud_server
    aid, token = _signup_and_get_token(base, f"rate-{uuid.uuid4().hex[:6]}@x.com")
    # Mint a key with a TINY rate limit so the test can exhaust it deterministically
    r = requests.post(f"{base}/v1/cloud/me/keys", headers=_h(token),
                      json={"label": "tiny", "rate_limit_per_min": 3})
    tiny_tok = r.json()["token"]

    # Burn through the quota — a public endpoint /v1/users/{id}/sleep-config is
    # protected and cheap, perfect for stressing the limiter.
    statuses = []
    for _ in range(10):
        r = requests.get(f"{base}/v1/users/x/sleep-config", headers=_h(tiny_tok))
        statuses.append(r.status_code)

    assert 200 in statuses           # at least the first few succeed
    assert 429 in statuses           # at least one is throttled


def test_signup_gate_with_token_env():
    """When SCM_CLOUD_SIGNUP_TOKEN is set, signup requires the matching header."""
    port = _free_port()
    srv = _spawn_server(port, {
        "SCM_CLOUD_AUTH": "1",
        "SCM_CLOUD_SIGNUP_TOKEN": "open-sesame",
        "SCM_CLOUD_SECRET_KEY": "test-secret-key-with-32-bytes-min-chars",
        "SCM_EMBEDDING_BACKEND": "ollama",
        "SCM_EMBEDDING_MODEL": "nomic-embed-text",
    })
    try:
        base = f"http://127.0.0.1:{port}"
        # Without header → 403
        r = requests.post(f"{base}/v1/cloud/accounts", json={"email": "x@x.com"})
        assert r.status_code == 403
        # With header → 200
        r = requests.post(f"{base}/v1/cloud/accounts",
                          json={"email": "y@x.com"},
                          headers={"X-Signup-Token": "open-sesame"})
        assert r.status_code == 200
    finally:
        srv.should_exit = True
        # Clean the env so the next test isn't gated
        os.environ.pop("SCM_CLOUD_SIGNUP_TOKEN", None)


# ── Tests: cloud-auth OFF (open-source self-hosted shape) ──────────────


def test_open_mode_routes_work_without_token(open_server: str):
    """Self-hosted users with SCM_CLOUD_AUTH=0 can hit /v1/memories without
    any auth — preserves the OSS shape this entire codebase had pre-v0.7.8."""
    base = open_server
    r = requests.post(
        f"{base}/v1/memories",
        json={"text": "open-mode test", "user_id": "default", "sync": True},
    )
    assert r.status_code == 200
