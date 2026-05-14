"""HTTP endpoints for SCM Cloud account + key management.

Mounted at /v1/cloud/*. Two distinct call shapes:

1. **Signup-only routes** — public, no auth, used to create accounts and
   issue the first API key:
       POST /v1/cloud/accounts                   (create new account)
       POST /v1/cloud/accounts/{id}/keys/initial (mint first key for new acct)
   These are throttled separately and gated by an `X-Signup-Token` header
   when ``SCM_CLOUD_SIGNUP_TOKEN`` is set, so a public deployment can stop
   strangers from spamming new accounts. For dev mode (env var unset) the
   token check is skipped.

2. **Account-scoped routes** — require Bearer auth from the account whose
   resources are being read/written:
       GET    /v1/cloud/me                       (account info + tier)
       POST   /v1/cloud/me/byok                  (set LLM provider+key+model)
       DELETE /v1/cloud/me/byok                  (clear LLM config)
       GET    /v1/cloud/me/keys                  (list API keys)
       POST   /v1/cloud/me/keys                  (issue another key)
       DELETE /v1/cloud/me/keys/{id}             (revoke a key)

The token-on-issue is the only time the full secret is returned. Callers
must persist it on their end — there is no recovery flow.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import accounts as acct


router = APIRouter(prefix="/v1/cloud", tags=["cloud"])


# ── Public signup ──────────────────────────────────────────────────────


def _check_signup_gate(x_signup_token: Optional[str]) -> None:
    """If SCM_CLOUD_SIGNUP_TOKEN is set, require it as the X-Signup-Token
    header on signup endpoints. This stops random people creating accounts
    on a public deployment before we have a proper signup UI."""
    expected = os.environ.get("SCM_CLOUD_SIGNUP_TOKEN")
    if not expected:
        return
    if x_signup_token != expected:
        raise HTTPException(status_code=403, detail="signup is gated")


class _CreateAccountReq(BaseModel):
    email: str
    tier: str = "free"


@router.post("/accounts")
async def create_account(
    body: _CreateAccountReq,
    x_signup_token: Optional[str] = Header(None, alias="X-Signup-Token"),
) -> Dict[str, Any]:
    _check_signup_gate(x_signup_token)
    try:
        return acct.create_account(email=body.email, tier=body.tier)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class _InitialKeyReq(BaseModel):
    label: Optional[str] = "default"


@router.post("/accounts/{account_id}/keys/initial")
async def issue_initial_key(
    account_id: str,
    body: _InitialKeyReq,
    x_signup_token: Optional[str] = Header(None, alias="X-Signup-Token"),
) -> Dict[str, Any]:
    """Mint the FIRST API key for a freshly-created account. Only allowed
    when the account has zero existing keys. After this, callers use the
    Bearer-authed POST /v1/cloud/me/keys to mint additional keys."""
    _check_signup_gate(x_signup_token)
    if acct.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail="account not found")
    if acct.list_api_keys(account_id, include_revoked=False):
        raise HTTPException(
            status_code=409,
            detail="account already has at least one key; use the authed endpoint",
        )
    try:
        return acct.issue_api_key(account_id=account_id, label=body.label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Account-scoped (Bearer-authed) ────────────────────────────────────


def _require_account(request: Request) -> Dict[str, Any]:
    """Read the auth context the middleware stamped on the request."""
    account = getattr(request.state, "scm_account", None)
    if account is None:
        # Reach this only when SCM_CLOUD_AUTH is off and the caller hits
        # an /me/* route. Match the unauth-mode contract: 401.
        raise HTTPException(
            status_code=401,
            detail="cloud auth is off; /me/* requires SCM_CLOUD_AUTH=1",
        )
    return account


@router.get("/me")
async def whoami(request: Request) -> Dict[str, Any]:
    account = _require_account(request)
    return {
        "id": account["id"],
        "email": account["email"],
        "tier": account["tier"],
    }


class _BYOKReq(BaseModel):
    provider: str = Field(..., description="e.g. 'deepseek' | 'openai' | 'anthropic'")
    api_key: str = Field(..., description="the LLM provider API key")
    base_url: Optional[str] = Field(None, description="OpenAI-compatible base URL")
    model: Optional[str] = Field(None, description="default model name")


@router.post("/me/byok")
async def set_byok(request: Request, body: _BYOKReq) -> Dict[str, Any]:
    """Configure the account's bring-your-own-key LLM. The api_key is
    encrypted at rest; never returned in subsequent reads."""
    account = _require_account(request)
    acct.set_byok_llm(
        account_id=account["id"],
        provider=body.provider,
        api_key=body.api_key,
        base_url=body.base_url,
        model=body.model,
    )
    return {"ok": True, "provider": body.provider, "base_url": body.base_url}


@router.delete("/me/byok")
async def clear_byok(request: Request) -> Dict[str, Any]:
    account = _require_account(request)
    acct.set_byok_llm(
        account_id=account["id"],
        provider="",
        api_key="",
        base_url="",
        model="",
    )
    return {"ok": True}


@router.get("/me/keys")
async def list_keys(request: Request) -> Dict[str, Any]:
    account = _require_account(request)
    return {"keys": acct.list_api_keys(account["id"])}


class _MintKeyReq(BaseModel):
    label: Optional[str] = ""
    rate_limit_per_min: int = 60


@router.post("/me/keys")
async def mint_key(request: Request, body: _MintKeyReq) -> Dict[str, Any]:
    account = _require_account(request)
    return acct.issue_api_key(
        account_id=account["id"],
        label=body.label,
        rate_limit_per_min=body.rate_limit_per_min,
    )


@router.delete("/me/keys/{key_id}")
async def revoke_key(request: Request, key_id: str) -> Dict[str, Any]:
    account = _require_account(request)
    ok = acct.revoke_api_key(account_id=account["id"], key_record_id=key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="key not found")
    return {"ok": True, "revoked_id": key_id}
