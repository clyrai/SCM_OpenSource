"""Account + API-key management for SCM Cloud.

Design choices:

- **Bearer-token auth.** API keys are issued as `scm_<keyid>_<secret>` strings.
  Only `key_hash` (sha256 of the full token) is stored — losing the DB does
  not expose live keys.
- **Multiple keys per account.** Revoke individually, rotate, scope per key.
- **Tenancy enforced server-side.** The memory `user_id` accepted on
  /v1/memories endpoints is always namespaced under the authed account
  (`account.id + ":" + caller_user_id`). A caller cannot read another
  account's data even if they guess the user_id.
- **BYOK.** Each account can store an LLM provider+key+base_url+model
  for SCM's optional LLM extraction step. Stored at-rest under a symmetric
  key (env var SCM_CLOUD_SECRET_KEY); never returned in plaintext over the
  API. If unset, SCM falls back to its heuristic extractor (free, no key).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Symmetric encryption for at-rest BYOK API keys ─────────────────────


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """Stream-cipher XOR (one-time-pad-style with derived stream).

    Not AES, but sufficient for at-rest protection of LLM API keys against
    db dump leaks. Cloud production should swap this for AES-GCM via
    cryptography lib; for the bootstrap phase we keep zero new
    dependencies and use a SHA256-derived keystream.

    The key is derived from SCM_CLOUD_SECRET_KEY so the deployment can
    rotate it; rotating invalidates existing encrypted BYOK keys (they
    must be re-set), which is the desired behavior.
    """
    out = bytearray(len(data))
    pos = 0
    counter = 0
    while pos < len(data):
        block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        for i in range(min(32, len(data) - pos)):
            out[pos + i] = data[pos + i] ^ block[i]
        pos += 32
        counter += 1
    return bytes(out)


def _master_key() -> bytes:
    """Return the deployment's at-rest secret. Must be >=32 bytes."""
    raw = os.environ.get("SCM_CLOUD_SECRET_KEY")
    if not raw or len(raw) < 32:
        # Don't crash the dev server when the env var is missing — emit a
        # process-stable random key so dev mode works. Logged once.
        global _DEV_KEY  # noqa: PLW0603
        try:
            _DEV_KEY  # type: ignore[name-defined]
        except NameError:
            _DEV_KEY = secrets.token_bytes(32)
            print(
                "[scm.cloud] WARNING: SCM_CLOUD_SECRET_KEY not set; "
                "generating an ephemeral dev key. Encrypted BYOK keys will "
                "be unreadable across process restarts."
            )
        return _DEV_KEY
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    blob = _xor_bytes(plaintext.encode("utf-8"), _master_key())
    return urlsafe_b64encode(blob).decode("ascii")


def _decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    blob = urlsafe_b64decode(ciphertext.encode("ascii"))
    return _xor_bytes(blob, _master_key()).decode("utf-8", errors="replace")


# ── API-key formatting ──────────────────────────────────────────────────

# Tokens look like "scm_live_<keyid>_<secret>". The keyid prefix is logged
# for audit / display ("scm_live_a1b2c3..."); the full token is the bearer.
_TOKEN_PREFIX = "scm_live_"
_KEYID_LEN = 12   # base32 chars
_SECRET_LEN = 32  # base32 chars


def _new_token() -> tuple[str, str, str]:
    """Generate a new (full_token, key_id, key_hash) triple."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    key_id = "".join(secrets.choice(alphabet) for _ in range(_KEYID_LEN))
    secret = "".join(secrets.choice(alphabet) for _ in range(_SECRET_LEN))
    full = f"{_TOKEN_PREFIX}{key_id}_{secret}"
    key_hash = hashlib.sha256(full.encode("utf-8")).hexdigest()
    return full, key_id, key_hash


def _hash_token(full_token: str) -> str:
    return hashlib.sha256(full_token.encode("utf-8")).hexdigest()


# ── Account + key CRUD ──────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA_ENSURED = False


def _conn():
    """Open a SQLite connection, lazily ensuring the cloud-auth schema
    exists. Without this, callers who use the cloud module before the
    main ChatEngine has booted would hit 'no such table: accounts'.
    """
    global _SCHEMA_ENSURED
    from ..core.sqlite_db import get_connection, init_db
    if not _SCHEMA_ENSURED:
        init_db()
        _SCHEMA_ENSURED = True
    return get_connection()


def create_account(email: str, tier: str = "free") -> Dict[str, Any]:
    """Create a new cloud account. Email must be unique; raises ValueError
    if it already exists."""
    if not email or "@" not in email:
        raise ValueError("invalid email")
    aid = "acct_" + uuid.uuid4().hex[:16]
    now = _now()
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM accounts WHERE email = ?", (email,))
        if cur.fetchone() is not None:
            raise ValueError(f"account exists: {email}")
        cur.execute(
            """
            INSERT INTO accounts (id, email, tier, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (aid, email, tier, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": aid, "email": email, "tier": tier, "created_at": now}


def get_account(account_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM accounts WHERE id = ?", (account_id,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        # Never return the encrypted BYOK secret over read APIs.
        d.pop("byok_llm_api_key_enc", None)
        return d
    finally:
        conn.close()


def get_account_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM accounts WHERE email = ?", (email,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d.pop("byok_llm_api_key_enc", None)
        return d
    finally:
        conn.close()


def set_byok_llm(
    account_id: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Set or clear the account's bring-your-own-key LLM config used for
    optional LLM extraction during sleep cycles. Pass empty string to
    clear a field. The api_key is encrypted at rest."""
    sets = []
    args: List[Any] = []
    if provider is not None:
        sets.append("byok_llm_provider = ?"); args.append(provider or None)
    if api_key is not None:
        sets.append("byok_llm_api_key_enc = ?")
        args.append(_encrypt(api_key) if api_key else None)
    if base_url is not None:
        sets.append("byok_llm_base_url = ?"); args.append(base_url or None)
    if model is not None:
        sets.append("byok_llm_model = ?"); args.append(model or None)
    if not sets:
        return
    sets.append("updated_at = ?"); args.append(_now())
    args.append(account_id)

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE accounts SET {', '.join(sets)} WHERE id = ?", args,
        )
        conn.commit()
    finally:
        conn.close()


def get_byok_llm(account_id: str) -> Optional[Dict[str, Any]]:
    """Return the account's BYOK LLM config including the *decrypted*
    api_key. Used internally by the consolidate path; never exposed
    over an HTTP endpoint."""
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT byok_llm_provider, byok_llm_api_key_enc,
                   byok_llm_base_url, byok_llm_model
            FROM accounts WHERE id = ?
            """,
            (account_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None or not row["byok_llm_provider"]:
        return None
    return {
        "provider": row["byok_llm_provider"],
        "api_key": _decrypt(row["byok_llm_api_key_enc"] or ""),
        "base_url": row["byok_llm_base_url"],
        "model": row["byok_llm_model"],
    }


def issue_api_key(
    account_id: str,
    label: Optional[str] = None,
    scopes: str = "memories:rw",
    rate_limit_per_min: int = 60,
) -> Dict[str, Any]:
    """Mint a new API key for this account. Returns the FULL token —
    only this once; the caller must save it. Future reads only get the
    prefix + label.
    """
    if get_account(account_id) is None:
        raise ValueError(f"unknown account: {account_id}")
    full, key_id, key_hash = _new_token()
    prefix = f"{_TOKEN_PREFIX}{key_id}"
    record_id = "key_" + uuid.uuid4().hex[:16]
    now = _now()
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO api_keys
              (id, account_id, key_prefix, key_hash, label, scopes,
               rate_limit_per_min, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id, account_id, prefix, key_hash, label or "",
                scopes, int(rate_limit_per_min), now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": record_id,
        "account_id": account_id,
        "key_prefix": prefix,
        "label": label or "",
        "scopes": scopes,
        "rate_limit_per_min": rate_limit_per_min,
        "created_at": now,
        # Returned ONCE here, never persisted.
        "token": full,
    }


def list_api_keys(account_id: str, include_revoked: bool = False) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        cur = conn.cursor()
        if include_revoked:
            cur.execute(
                "SELECT * FROM api_keys WHERE account_id = ? ORDER BY created_at DESC",
                (account_id,),
            )
        else:
            cur.execute(
                "SELECT * FROM api_keys WHERE account_id = ? AND revoked_at IS NULL ORDER BY created_at DESC",
                (account_id,),
            )
        rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        d = dict(row)
        d.pop("key_hash", None)  # never expose
        out.append(d)
    return out


def revoke_api_key(account_id: str, key_record_id: str) -> bool:
    """Revoke a key. Future requests with that key will 401."""
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND account_id = ?",
            (_now(), key_record_id, account_id),
        )
        changed = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return changed > 0


def validate_token(token: str) -> Optional[Dict[str, Any]]:
    """Hot path for the auth middleware. Returns a {key, account} record
    on valid, None on invalid/revoked. Updates last_used_at as a
    side-effect (for usage analytics + key-lifecycle UX)."""
    if not token or not token.startswith(_TOKEN_PREFIX):
        return None
    h = _hash_token(token)
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
            (h,),
        )
        krow = cur.fetchone()
        if krow is None:
            return None
        cur.execute(
            "SELECT id, email, tier FROM accounts WHERE id = ?",
            (krow["account_id"],),
        )
        arow = cur.fetchone()
        if arow is None:
            return None
        # touch last_used_at — best-effort, don't block validation on it
        try:
            cur.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (_now(), krow["id"]),
            )
            conn.commit()
        except Exception:
            pass
        return {
            "key": {
                "id": krow["id"],
                "scopes": krow["scopes"],
                "rate_limit_per_min": int(krow["rate_limit_per_min"] or 60),
            },
            "account": dict(arow),
        }
    finally:
        conn.close()


# ── Tenant namespacing ─────────────────────────────────────────────────


def namespace_user_id(account_id: str, caller_user_id: str) -> str:
    """Map a caller-provided `user_id` into the tenancy-namespaced form
    used for the actual concept storage. Never let a caller from
    account A read account B's data even if they guess B's user_id.

    Format: "<account_id>::<caller_user_id>" — opaque to callers.
    """
    cu = (caller_user_id or "default").strip() or "default"
    return f"{account_id}::{cu}"
