"""FastAPI dependency + ASGI middleware that gates `/v1/*` routes
behind a Bearer-token API key.

Accepts ``Authorization: Bearer scm_live_<keyid>_<secret>``. Validates
against the api_keys table, applies per-key rate limit, and rewrites
the request payload so that any caller-supplied ``user_id`` becomes
namespaced under the authenticated account (cross-tenant safety).

Cloud-mode is opt-in via env var ``SCM_CLOUD_AUTH=1``. When unset (the
default), routes are unauthenticated — preserves the open-source
self-hosted shape where users run their own server with no auth.
"""
from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from .accounts import validate_token
from .rate_limit import acquire as rate_limit_acquire


def cloud_auth_enabled() -> bool:
    return os.environ.get("SCM_CLOUD_AUTH", "0") == "1"


# Routes that require an API key when cloud auth is on.
_PROTECTED_PREFIXES = (
    "/v1/memories",
    "/v1/wake-summary",
    "/v1/users",
    "/v1/cloud/me",          # account-scoped: whoami, byok, key mgmt
)
# Public exceptions:
#   /v1/health, /v1/tools, /v1/openapi.json   — discoverability
#   /v1/cloud/accounts                        — signup (gated by SCM_CLOUD_SIGNUP_TOKEN)
#   /v1/cloud/accounts/.../keys/initial       — first-key issuance for new accts
_PUBLIC = (
    "/v1/health",
    "/v1/tools",
    "/v1/openapi.json",
    "/v1/cloud/accounts",
)


def _is_protected(path: str) -> bool:
    if any(path == p or path.startswith(p + "/") for p in _PUBLIC):
        return False
    return any(path == p or path.startswith(p + "/") for p in _PROTECTED_PREFIXES)


class CloudAuthMiddleware(BaseHTTPMiddleware):
    """Validates Bearer tokens and stamps account context on the request.

    On success, ``request.state.scm_account`` and ``request.state.scm_key``
    are populated for downstream handlers. The ``/v1/memories/*`` handlers
    use ``scm_account.id`` to namespace ``user_id`` so a caller can't
    address another account's data.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not cloud_auth_enabled():
            return await call_next(request)

        if not _is_protected(request.url.path):
            return await call_next(request)

        # Extract Bearer token.
        auth_header = request.headers.get("authorization") or ""
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(
                {"error": "missing or malformed Authorization header",
                 "hint": "send: Authorization: Bearer scm_live_..."},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        token = auth_header[7:].strip()
        ctx = validate_token(token)
        if ctx is None:
            return JSONResponse(
                {"error": "invalid or revoked API key"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        # Rate limit per key.
        if not rate_limit_acquire(
            ctx["key"]["id"],
            ctx["key"]["rate_limit_per_min"],
        ):
            return JSONResponse(
                {"error": "rate limit exceeded",
                 "limit_per_min": ctx["key"]["rate_limit_per_min"]},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        # Stash the auth context on the request so handlers can read it.
        request.state.scm_account = ctx["account"]
        request.state.scm_key = ctx["key"]

        return await call_next(request)
