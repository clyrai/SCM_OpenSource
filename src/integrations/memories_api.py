"""
REST API for SCM, layered on the same five canonical tools.

Mounts under /v1 by default. Idle detection is automatic — once a user
hasn't called any endpoint for SCM_IDLE_THRESHOLD_SEC seconds (default
300), a sleep cycle fires in the background and the wake-summary is
cached. The next call surfaces the cached summary in the response.

Endpoints:
    POST   /v1/memories              add_memory tool
    POST   /v1/memories/search       search_memory tool
    POST   /v1/memories/consolidate  consolidate tool (manual override)
    GET    /v1/memories/{id}/lineage version lineage + conflict provenance
    GET    /v1/wake-summary          wake_summary tool
    DELETE /v1/memories/{id}         forget tool
    GET    /v1/tools                 export tool definitions in any format
    GET    /v1/health                liveness probe

This is the API a third-party agent service hits when integrating SCM.
The OpenAPI spec is served at /v1/openapi.json — paste that URL into a
ChatGPT Custom GPT to give it SCM memory in 30 seconds.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from .mcp_server import UserEnginePool
from .tools import (
    TOOLS,
    export_all_anthropic,
    export_all_gemini,
    export_all_openai,
    export_openapi_spec,
    get_tool,
)


router = APIRouter(prefix="/v1", tags=["memories"])


# Module-level singleton — one engine pool per server process.
_pool: Optional[UserEnginePool] = None


def get_pool() -> UserEnginePool:
    global _pool
    if _pool is None:
        idle_threshold = float(os.environ.get("SCM_IDLE_THRESHOLD_SEC", "300"))
        sweep_interval = float(os.environ.get("SCM_MCP_SWEEP_INTERVAL_SEC", "30"))
        auto_sleep = os.environ.get("SCM_AUTO_SLEEP_DISABLE", "0") != "1"
        _pool = UserEnginePool(
            idle_threshold_sec=idle_threshold,
            sweep_interval_sec=sweep_interval,
            auto_sleep=auto_sleep,
        )
        _pool.start()
    return _pool


def _user_id_from(payload: Dict[str, Any], header_user: Optional[str]) -> str:
    return (payload.get("user_id") or header_user or "default")


def _namespace_for_account(request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """If cloud auth is on AND this request was authed, namespace the
    `user_id` parameter under the calling account so cross-tenant reads
    are impossible. Mutates and returns the payload for caller convenience.

    No-op when running self-hosted (SCM_CLOUD_AUTH=0) — the caller's
    user_id passes through untouched, preserving the open-source shape.
    """
    account = getattr(getattr(request, "state", None), "scm_account", None)
    if account is None:
        return payload
    from ..cloud.accounts import namespace_user_id
    caller_uid = payload.get("user_id") or "default"
    payload["user_id"] = namespace_user_id(account["id"], caller_uid)
    return payload


# Tools that count as real user activity (reset the idle timer).
# System / admin tools (consolidate, wake_summary, forget) do NOT bump
# activity, so a manual consolidate doesn't make the user look "active"
# to the auto-sleep sweeper.
_USER_ACTIVITY_TOOLS = frozenset({"add_memory", "search_memory"})
_ALLOWED_AUTO_SLEEP_MODES = frozenset({"auto", "night_only", "idle_only", "off"})


def _invoke(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve user_id, get-or-create engine, call tool handler, append wake-summary if cached.

    v0.7.2 async ingest: add_memory enqueues to a per-user worker by
    default and returns a placeholder id in <100ms. Pass ``sync=true``
    in the body to wait for the LLM extractor to finish (slower; used
    by tests that need write-then-read consistency).

    search_memory accepts ``wait_for_pending=true`` to block until any
    in-flight ingests for this user have drained, giving callers an
    opt-in read-your-writes guarantee.
    """
    tool = get_tool(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"unknown tool: {tool_name}")

    user_id = args.get("user_id") or "default"
    pool = get_pool()
    bump_activity = tool_name in _USER_ACTIVITY_TOOLS

    # ── Special path: add_memory async by default ────────────────────────
    if tool_name == "add_memory":
        sync = bool(args.get("sync")) or os.environ.get("SCM_SYNC_INGEST") == "1"
        if not sync:
            text = (args.get("text") or "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="'text' is required")
            placeholder = pool.enqueue_ingest(
                user_id, text, args.get("metadata") or {}
            )
            return {
                "ok": True,
                "user_id": user_id,
                "memory_id": placeholder,
                "mode": "async",
                "status": "pending",
                "pending_count": pool.pending_count(user_id),
            }
        # Sync path: drain queue first so we have a stable engine state,
        # then run the handler (which calls engine.chat synchronously).
        pool.wait_for_pending(user_id, timeout=10.0)

    # ── Special path: consolidate goes through the pool's wake cache ─────
    if tool_name == "consolidate":
        # Drain pending ingests so consolidation sees the most recent state.
        pool.wait_for_pending(user_id, timeout=10.0)
        mode = args.get("mode") or "deep"
        result = pool.fire_sleep_now(user_id, mode=mode)
        result["user_id"] = user_id
        return result

    # ── Optional: search waits for pending ingests (read-your-writes) ────
    if tool_name == "search_memory" and args.get("wait_for_pending"):
        pool.wait_for_pending(user_id, timeout=10.0)

    engine = pool.get_or_create(user_id, bump_activity=bump_activity)
    result = tool.handler(args, engine)

    # Auto-surface a cached wake-summary on activity-resumption tools.
    if tool_name in _USER_ACTIVITY_TOOLS:
        cached = pool.cached_summary(user_id)
        if cached:
            result["wake_summary_pending"] = cached
            pool.clear_cached_summary(user_id)
    return result


def _memory_lineage_for_user(user_id: str, memory_id: str) -> Dict[str, Any]:
    pool = get_pool()
    engine = pool.get_or_create(user_id, bump_activity=False)
    ltm = getattr(engine, "long_term_memory", None)
    if ltm is None or not hasattr(ltm, "get_lineage"):
        raise HTTPException(status_code=503, detail="lineage unavailable for this engine")
    try:
        payload = ltm.get_lineage(memory_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load lineage: {type(exc).__name__}") from exc
    return payload if isinstance(payload, dict) else {}


# ─── Endpoints ────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> Dict[str, Any]:
    pool = get_pool()
    return {
        "ok": True,
        "active_users": len(pool._engines),
        "auto_sleep": pool._auto_sleep,
        "idle_threshold_sec": pool._idle_threshold,
    }


@router.post("/memories")
async def add_memory(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Add a memory. Body: {text, user_id?, metadata?}."""
    if "text" not in payload or not payload["text"]:
        raise HTTPException(status_code=400, detail="'text' is required")
    return _invoke("add_memory", _namespace_for_account(request, payload))


@router.post("/memories/search")
async def search_memory(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Search memories by associative retrieval. Body: {query, user_id?, limit?}."""
    if "query" not in payload or not payload["query"]:
        raise HTTPException(status_code=400, detail="'query' is required")
    args = _namespace_for_account(request, payload)
    result = _invoke("search_memory", args)
    user_id = args.get("user_id") or "default"
    slots = get_pool().task_context_snapshot(user_id)
    result["task_context"] = {"slots": slots, "total": len(slots)}
    return result


@router.post("/memories/consolidate")
async def consolidate(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Manually trigger a sleep cycle. Body: {user_id?, mode?}.

    Most callers do not need this — the auto-sleep sweeper fires it
    automatically when the user has been idle past the threshold.
    """
    return _invoke("consolidate", _namespace_for_account(request, payload or {}))


@router.get("/wake-summary")
async def wake_summary(
    request: Request,
    user_id: str = Query("default"),
    since_hours: float = Query(24.0, ge=0.5),
) -> Dict[str, Any]:
    """Return what the agent learned during recent idle time."""
    payload = _namespace_for_account(request, {"user_id": user_id, "since_hours": since_hours})
    return _invoke("wake_summary", payload)


@router.delete("/memories/{memory_id}")
async def forget(memory_id: str, request: Request, user_id: str = Query("default")) -> Dict[str, Any]:
    """Remove a specific memory by id."""
    payload = _namespace_for_account(request, {"memory_id": memory_id, "user_id": user_id})
    return _invoke("forget", payload)


@router.get("/memories/{memory_id}/lineage")
async def memory_lineage(
    memory_id: str,
    request: Request,
    user_id: str = Query("default"),
) -> Dict[str, Any]:
    """Return version lineage + contradiction links for one memory concept."""
    namespaced = _namespace_for_account(request, {"user_id": user_id}).get("user_id") or "default"
    lineage = _memory_lineage_for_user(namespaced, memory_id)
    if not lineage:
        raise HTTPException(status_code=404, detail=f"memory not found: {memory_id}")
    return {
        "ok": True,
        "user_id": user_id,
        "memory_id": memory_id,
        "lineage": lineage,
    }


# ─── Per-user circadian sleep schedule (v0.7.7+) ─────────────────────────


@router.get("/users/{user_id}/sleep-config")
async def get_sleep_config(user_id: str, request: Request) -> Dict[str, Any]:
    """Return this user's nightly consolidation schedule.

    Defaults to (UTC, 23:00→07:00, enabled) when the user hasn't set
    one. The MCP sweeper checks every 60s whether each user's local
    time has entered their window and fires once per night per user —
    the human-circadian model that replaces v0.7.6's fixed-idle timer.
    """
    from ..core.sqlite_db import get_memory
    args = _namespace_for_account(request, {"user_id": user_id})
    namespaced = args.get("user_id") or "default"
    cfg = get_memory().get_user_sleep_config(namespaced)
    return {
        "user_id": user_id,
        "timezone": cfg["timezone"],
        "sleep_start": cfg["sleep_start"],
        "sleep_end": cfg["sleep_end"],
        "enabled": bool(cfg["enabled"]),
        "auto_sleep_mode": cfg.get("auto_sleep_mode", "auto"),
        "idle_timeout_sec": cfg.get("idle_timeout_sec"),
        "is_default": bool(cfg.get("is_default", False)),
        "last_sleep_at": cfg.get("last_sleep_at"),
        "last_sleep_reason": cfg.get("last_sleep_reason"),
    }


@router.post("/users/{user_id}/sleep-config")
async def set_sleep_config(
    user_id: str, payload: Dict[str, Any], request: Request,
) -> Dict[str, Any]:
    """Update this user's nightly consolidation schedule.

    Body fields (all optional — partial updates supported):
        timezone:     IANA name like "Europe/Lisbon"
        sleep_start:  "HH:MM" in local tz, e.g. "23:00"
        sleep_end:    "HH:MM" in local tz, e.g. "07:00"
        enabled:      bool — disable to opt out of nightly sleep
        auto_sleep_mode: "auto" | "night_only" | "idle_only" | "off"
        idle_timeout_sec: positive number; null clears custom override
    """
    from zoneinfo import ZoneInfo
    from ..core.sqlite_db import get_memory
    from ..lifecycle.circadian import parse_hhmm

    args = _namespace_for_account(request, {"user_id": user_id})
    namespaced = args.get("user_id") or "default"

    tz = payload.get("timezone")
    if tz is not None:
        try:
            ZoneInfo(tz)
        except Exception:
            raise HTTPException(status_code=400, detail=f"unknown timezone: {tz!r}")
    for field in ("sleep_start", "sleep_end"):
        if payload.get(field) is not None and parse_hhmm(payload[field]) is None:
            raise HTTPException(status_code=400, detail=f"{field} must be 'HH:MM'")

    mode = payload.get("auto_sleep_mode")
    if mode is not None:
        mode = str(mode).strip().lower()
        if mode not in _ALLOWED_AUTO_SLEEP_MODES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "auto_sleep_mode must be one of: "
                    "auto, night_only, idle_only, off"
                ),
            )

    clear_idle_timeout = False
    idle_timeout_sec = None
    if "idle_timeout_sec" in payload:
        raw_idle = payload.get("idle_timeout_sec")
        if raw_idle in (None, ""):
            clear_idle_timeout = True
        else:
            try:
                idle_timeout_sec = float(raw_idle)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="idle_timeout_sec must be a number",
                )
            if idle_timeout_sec <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="idle_timeout_sec must be > 0",
                )

    cfg = get_memory().save_user_sleep_config(
        user_id=namespaced,
        timezone_name=tz,
        sleep_start=payload.get("sleep_start"),
        sleep_end=payload.get("sleep_end"),
        enabled=payload.get("enabled"),
        auto_sleep_mode=mode,
        idle_timeout_sec=idle_timeout_sec,
        clear_idle_timeout=clear_idle_timeout,
    )
    return {
        "user_id": user_id,
        "timezone": cfg["timezone"],
        "sleep_start": cfg["sleep_start"],
        "sleep_end": cfg["sleep_end"],
        "enabled": bool(cfg["enabled"]),
        "auto_sleep_mode": cfg.get("auto_sleep_mode", "auto"),
        "idle_timeout_sec": cfg.get("idle_timeout_sec"),
        "last_sleep_at": cfg.get("last_sleep_at"),
        "last_sleep_reason": cfg.get("last_sleep_reason"),
    }


@router.get("/users/{user_id}/context")
async def get_task_context(user_id: str, request: Request) -> Dict[str, Any]:
    """Return ephemeral task-context slots for this user."""
    args = _namespace_for_account(request, {"user_id": user_id})
    namespaced = args.get("user_id") or "default"
    slots = get_pool().task_context_snapshot(namespaced)
    return {
        "user_id": user_id,
        "slots": slots,
        "total": len(slots),
    }


@router.post("/users/{user_id}/context")
async def update_task_context(
    user_id: str, payload: Dict[str, Any], request: Request,
) -> Dict[str, Any]:
    """Update ephemeral task context from conversational text.

    Body:
      message?: str            # user message to parse into slots
      previous_assistant?: str # optional assistant question for one-word followups
      assistant_message?: str  # optional assistant text to prime pending slot
    """
    message = (payload.get("message") or "").strip()
    previous_assistant = (payload.get("previous_assistant") or "").strip()
    assistant_message = (payload.get("assistant_message") or "").strip()
    if not (message or assistant_message):
        raise HTTPException(
            status_code=400,
            detail="provide at least one of: message, assistant_message",
        )

    args = _namespace_for_account(request, {"user_id": user_id})
    namespaced = args.get("user_id") or "default"
    pool = get_pool()
    if assistant_message:
        pool.ingest_task_assistant(namespaced, assistant_message)
    updates: list[dict] = []
    if message:
        updates = pool.ingest_task_message(
            namespaced,
            text=message,
            previous_assistant=previous_assistant,
        )
    slots = pool.task_context_snapshot(namespaced)
    return {
        "ok": True,
        "user_id": user_id,
        "updates": updates,
        "slots": slots,
        "total": len(slots),
    }


@router.delete("/users/{user_id}/context")
async def clear_task_context(user_id: str, request: Request) -> Dict[str, Any]:
    """Clear ephemeral task-context slots for this user."""
    args = _namespace_for_account(request, {"user_id": user_id})
    namespaced = args.get("user_id") or "default"
    get_pool().clear_task_context(namespaced)
    return {"ok": True, "user_id": user_id}


# ─── Tool-definition exports ──────────────────────────────────────────────


@router.get("/tools")
async def list_tools(format: str = Query("openai")) -> Dict[str, Any]:
    """Export the five SCM tools in the requested function-calling format.

    format = openai | anthropic | gemini | openapi | all

    Use this to wire SCM into ChatGPT (openapi → Custom GPT Actions),
    Claude (anthropic → tool definitions), Gemini (gemini → function
    declarations), or any OpenAI-compatible LLM (openai).
    """
    fmt = format.lower()
    if fmt == "openai":
        return {"format": "openai", "tools": export_all_openai()}
    if fmt == "anthropic":
        return {"format": "anthropic", "tools": export_all_anthropic()}
    if fmt == "gemini":
        return {"format": "gemini", "tools": export_all_gemini()}
    if fmt == "openapi":
        # OpenAPI is structurally different — return the full spec.
        return export_openapi_spec()
    if fmt == "all":
        return {
            "openai": export_all_openai(),
            "anthropic": export_all_anthropic(),
            "gemini": export_all_gemini(),
            "openapi": export_openapi_spec(),
        }
    raise HTTPException(
        status_code=400,
        detail=f"unknown format: {format} (use openai, anthropic, gemini, openapi, or all)",
    )


@router.get("/openapi.json")
async def openapi_json() -> Dict[str, Any]:
    """OpenAPI 3.1 spec for the five tools.

    Paste the URL of this endpoint into a ChatGPT Custom GPT's Actions
    config to give the GPT SCM memory in one step.
    """
    server_url = os.environ.get(
        "SCM_PUBLIC_URL", "http://localhost:8000"
    )
    return export_openapi_spec(server_url=f"{server_url}/v1/tools")
