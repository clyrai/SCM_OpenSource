"""
The five canonical SCM tools, defined once and exported in every popular
function-calling format.

These tool definitions are the single source of truth. The MCP server,
REST API documentation, OpenAI function-calling clients, Anthropic tool
schemas, Gemini function declarations, and ChatGPT Custom GPT OpenAPI
spec all derive from this file.

Five tools, not seven: SCM collapses what other memory layers expose as
add / get / search / update / delete / history / users into "add" and
"search" because the lifecycle machinery handles consolidation
automatically; "consolidate" is exposed as a separate tool for callers
who want to force a sleep cycle, and "wake_summary" is the user-visible
report on what was learned during idle time — the SCM-specific hook with
no equivalent in stateless memory libraries.

Each tool definition includes:
    name        — stable identifier
    description — one-liner the LLM uses to choose when to call
    schema      — JSONSchema for the input
    handler     — Python callable; takes a dict, returns a dict

Handlers are pure functions that take an `engine_factory` callable for
test isolation; the integration entry points (MCP server, REST API,
LangChain adapter) inject the engine factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


# ─── Tool definitions ─────────────────────────────────────────────────────


@dataclass
class ToolDef:
    """One SCM tool. The single source of truth for cross-format export."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any], Any], Dict[str, Any]]
    examples: List[Dict[str, Any]] = field(default_factory=list)


# ─── The five tools ───────────────────────────────────────────────────────


def _add_memory_handler(args: Dict[str, Any], engine: Any) -> Dict[str, Any]:
    """Ingest a fact / observation into long-term memory.

    Pure ingest — does NOT return a generated chat reply. The agent that
    integrated SCM is responsible for generating its own response; SCM
    just remembers what was said.
    """
    text = args.get("text", "").strip()
    if not text:
        return {"ok": False, "error": "text is required"}
    user_id = args.get("user_id") or "default"

    # ChatEngine.chat ingests + retrieves + (optionally) generates. We
    # discard the response and return only memory metadata.
    #
    # `replaces_prior=True` (caller-supplied) flips on contradiction
    # versioning: any same-type, semantically-similar concept already in
    # memory will be SUPERSEDED. Caller is responsible for setting this
    # ONLY when the user explicitly corrects a prior fact ("I'm Saish,
    # not Alex") — not when a new speaker takes over the chat ("Hey,
    # I'm Shakira, a friend of the user") and not when a new fact is
    # additive ("I also like tea"). Defaulting to False is the safe
    # behavior; an aggressive default catastrophically wipes the user's
    # data when a friend briefly takes over the chat.
    replaces_prior = bool(args.get("replaces_prior"))
    _response, meta = engine.chat(text, force_versioning=replaces_prior)
    return {
        "ok": True,
        "user_id": user_id,
        "concepts_added": int(meta.get("concepts_added", 0) or 0),
        "concepts_total": int(meta.get("concepts_total", 0) or 0),
        "memory_id": meta.get("last_concept_id"),
    }


def _search_memory_handler(args: Dict[str, Any], engine: Any) -> Dict[str, Any]:
    """Retrieve memories relevant to a query, by association + recency.

    Three retrieval paths run in order, results merged and deduped:
      1. Spreading activation via SpreadingActivationRetriever
         (the right path for question-form queries that don't share
         tokens with the storage form — uses cues + graph propagation).
      2. Direct LTM text search (substring match — catches verbatim hits).
      3. Embedding fallback inside spreading activation when token cues
         find no seeds (already wired into _select_seeds).

    All paths filter is_current_version=False to avoid leaking superseded
    concepts (Phase 7 Bug 4 fix).
    """
    query = args.get("query", "").strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    user_id = args.get("user_id") or "default"
    limit = int(args.get("limit") or 5)

    seen_ids: set = set()
    concepts: List[Dict[str, Any]] = []

    def _add(c) -> None:
        cid = getattr(c, "id", "")
        if not cid or cid in seen_ids:
            return
        if not getattr(c, "is_current_version", True):
            return
        # Filter SelfModel/system internals — these are about the agent itself,
        # not the user's facts, and they shouldn't pollute user-facing retrieval.
        tags = getattr(c, "context_tags", None)
        if isinstance(tags, dict) and tags.get("_internal"):
            return
        seen_ids.add(cid)
        concepts.append({
            "id": cid,
            "description": getattr(c, "description", "") or "",
            "type": str(getattr(c, "type", "")),
            "confidence": float(getattr(c, "confidence", 1.0) or 1.0),
            "created_at": (
                c.created_at.isoformat()
                if getattr(c, "created_at", None) else None
            ),
        })

    # 1. Spreading activation — the strongest retrieval path. Returns the
    #    concepts the cue-driven graph propagation activated.
    memory_context: str = ""
    sa_stats: Dict[str, Any] = {}
    sa = getattr(engine, "_spreading_activation", None)
    if sa is not None:
        try:
            ctx_tags = {
                "session_id": getattr(engine, "session_id", None),
                "person": None,
            }
            activated, sa_stats = sa.retrieve(query, context_tags=ctx_tags)
            for c in activated[:limit * 2]:
                _add(c)
        except Exception:
            pass

    # 2. Get the formatted memory context too (for prompt injection).
    try:
        memory_context, _ = engine._retrieve_hme(query)
    except Exception:
        memory_context = ""

    # 3. LTM text search — catches verbatim substring hits the spreading
    #    activation might have missed.
    ltm = getattr(engine, "long_term_memory", None)
    if ltm is not None:
        try:
            results = ltm.search_by_text(query, limit=limit)
            for c in results:
                _add(c)
        except Exception:
            pass

    return {
        "ok": True,
        "user_id": user_id,
        "query": query,
        "memories": concepts[:limit],
        "memory_context": memory_context or "",
        "retrieved_count": len(concepts),
    }


def _consolidate_handler(args: Dict[str, Any], engine: Any) -> Dict[str, Any]:
    """Force a sleep cycle (consolidation + schema extraction + curiosity)."""
    user_id = args.get("user_id") or "default"
    mode = args.get("mode") or "deep"
    if mode not in ("deep", "micro"):
        return {"ok": False, "error": f"unknown mode: {mode}"}
    try:
        stats = engine.force_sleep(mode) or {}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    # Coerce to JSON-serialisable shape.
    return {
        "ok": True,
        "user_id": user_id,
        "mode": mode,
        "schemas_formed": int(stats.get("schemas_formed", 0)),
        "concepts_consolidated": int(stats.get("concepts_consolidated", 0)),
        "concepts_forgotten": int(stats.get("concepts_forgotten", 0)),
        "contradictions_resolved": int(stats.get("contradictions_resolved", 0)),
    }


def _wake_summary_handler(args: Dict[str, Any], engine: Any) -> Dict[str, Any]:
    """Return what the agent learned during recent idle time."""
    user_id = args.get("user_id") or "default"
    since_hours = float(args.get("since_hours") or 24.0)
    since = datetime.utcnow() - timedelta(hours=since_hours)

    builder = getattr(engine, "_wake_summary_builder", None)
    if builder is None:
        try:
            from src.lifecycle.wake_summary import WakeSummaryBuilder
            builder = WakeSummaryBuilder(engine=engine)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    try:
        summary = builder.build(since=since)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    insights = getattr(summary, "insights", []) or []
    return {
        "ok": True,
        "user_id": user_id,
        "since": since.isoformat(),
        "narrative": getattr(summary, "narrative", "") or "",
        "insights": [str(i) for i in insights[:20]],
        "schemas_formed": getattr(summary, "schemas_formed_count", len(insights)),
    }


def _forget_handler(args: Dict[str, Any], engine: Any) -> Dict[str, Any]:
    """Remove a specific memory by ID. Privacy / correction primitive."""
    memory_id = args.get("memory_id", "").strip()
    if not memory_id:
        return {"ok": False, "error": "memory_id is required"}
    user_id = args.get("user_id") or "default"

    ltm = getattr(engine, "long_term_memory", None)
    if ltm is None:
        return {"ok": False, "error": "long_term_memory unavailable"}

    try:
        removed = ltm.remove_concept(memory_id)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": bool(removed), "user_id": user_id, "memory_id": memory_id}


# ─── Schemas ──────────────────────────────────────────────────────────────


_USER_ID_FIELD = {
    "type": "string",
    "description": "Stable identifier for the end-user whose memory this is. Defaults to 'default' for single-user deployments. Use a per-user value (email, account ID) for multi-user systems.",
    "default": "default",
}


TOOLS: List[ToolDef] = [
    ToolDef(
        name="add_memory",
        description=(
            "Store a fact, observation, or event in long-term memory. "
            "Call this when the user says something worth remembering for "
            "future conversations: a preference, a fact about themselves, a "
            "recent event, a recurring routine. Returns a memory_id you can "
            "use later with the forget tool."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact or observation to remember, in natural language. Example: 'User prefers vegan food and lives in Seattle.'",
                },
                "user_id": _USER_ID_FIELD,
                "replaces_prior": {
                    "type": "boolean",
                    "description": (
                        "Set TRUE only when the user EXPLICITLY corrects a prior "
                        "fact they themselves stated ('I'm Saish, not Alex' / "
                        "'actually I moved to Seattle'). Flips on contradiction "
                        "versioning so the prior same-type, semantically-similar "
                        "fact gets superseded. Leave FALSE (default) for new "
                        "facts, additive details, or when a different person "
                        "joins the conversation. An aggressive default here "
                        "would wipe the user's stored data."
                    ),
                    "default": False,
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured tags (e.g., {'topic': 'food', 'source': 'profile_setup'}).",
                    "additionalProperties": True,
                },
            },
            "required": ["text"],
        },
        handler=_add_memory_handler,
        examples=[
            {"text": "User's name is Saish and they live in Bangalore."},
            {"text": "I prefer dark mode and decline meeting requests before 10am.", "user_id": "saish@example.com"},
            {"text": "My name is Saish.", "replaces_prior": True},
        ],
    ),
    ToolDef(
        name="search_memory",
        description=(
            "Retrieve memories relevant to a query. Uses associative "
            "spreading-activation retrieval (not just vector similarity), "
            "so question-form queries like 'what should I avoid for lunch?' "
            "can reach declarative facts like 'I have a peanut allergy' "
            "stored on a different day. Call this before responding to any "
            "message where prior context might matter."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for. Free text.",
                },
                "user_id": _USER_ID_FIELD,
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["query"],
        },
        handler=_search_memory_handler,
        examples=[
            {"query": "What does the user do for work?"},
            {"query": "What food preferences did they mention?", "limit": 3},
        ],
    ),
    ToolDef(
        name="consolidate",
        description=(
            "Trigger a sleep cycle to consolidate recent experiences into "
            "long-term patterns. The cycle runs schema extraction (REM-phase), "
            "contradiction resolution, adaptive forgetting, and curiosity-driven "
            "knowledge-gap filling. Most agents don't need to call this manually "
            "— SCM's idle-aware daemon fires it automatically when the session "
            "goes quiet. Call this only if you want to force consolidation "
            "before reading a wake-summary."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "user_id": _USER_ID_FIELD,
                "mode": {
                    "type": "string",
                    "enum": ["deep", "micro"],
                    "description": "deep = full NREM+REM consolidation (heavier); micro = lightweight replay (faster).",
                    "default": "deep",
                },
            },
            "required": [],
        },
        handler=_consolidate_handler,
    ),
    ToolDef(
        name="wake_summary",
        description=(
            "Return a structured report of what was learned during the most "
            "recent idle period — patterns the agent noticed, contradictions "
            "it resolved, knowledge gaps it filled. This is the 'while you "
            "were away' moment. Surface this to the user when they return "
            "from a long absence."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "user_id": _USER_ID_FIELD,
                "since_hours": {
                    "type": "number",
                    "description": "How far back to summarise. 24 = since yesterday morning. 168 = past week.",
                    "default": 24.0,
                    "minimum": 0.5,
                },
            },
            "required": [],
        },
        handler=_wake_summary_handler,
    ),
    ToolDef(
        name="forget",
        description=(
            "Permanently remove a specific memory. Use when the user requests "
            "deletion ('forget what I said about X') or when correcting an "
            "incorrectly-stored fact. Note: SCM's contradiction-versioning "
            "machinery already handles updates automatically (when the user "
            "says X then later says ~X), so you usually want add_memory with "
            "the new value, not forget. Use forget only for genuine deletion."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The id returned by a previous add_memory or search_memory call.",
                },
                "user_id": _USER_ID_FIELD,
            },
            "required": ["memory_id"],
        },
        handler=_forget_handler,
    ),
]


# ─── Format exporters ─────────────────────────────────────────────────────


def to_openai_tool(tool: ToolDef) -> Dict[str, Any]:
    """Render a ToolDef as an OpenAI function-calling tool spec.

    Compatible with: OpenAI chat-completions, OpenAI Assistants API,
    DeepSeek (OpenAI-compatible), Together AI, Anyscale, Voyage's
    function-calling endpoints, and anything that accepts the OpenAI
    'tools' array format.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def to_anthropic_tool(tool: ToolDef) -> Dict[str, Any]:
    """Render a ToolDef as an Anthropic Claude tool spec."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def to_gemini_function(tool: ToolDef) -> Dict[str, Any]:
    """Render a ToolDef as a Google Gemini FunctionDeclaration."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def to_openapi_path(tool: ToolDef, base_path: str = "/v1/tools") -> Dict[str, Any]:
    """Render a ToolDef as an OpenAPI 3.1 path object.

    Used for ChatGPT Custom GPT 'Actions' which require an OpenAPI spec.
    """
    return {
        f"{base_path}/{tool.name}": {
            "post": {
                "operationId": tool.name,
                "summary": tool.description,
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": tool.input_schema,
                        },
                    },
                },
                "responses": {
                    "200": {
                        "description": f"{tool.name} response",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"},
                            },
                        },
                    },
                },
            },
        },
    }


def export_all_openai() -> List[Dict[str, Any]]:
    return [to_openai_tool(t) for t in TOOLS]


def export_all_anthropic() -> List[Dict[str, Any]]:
    return [to_anthropic_tool(t) for t in TOOLS]


def export_all_gemini() -> List[Dict[str, Any]]:
    return [to_gemini_function(t) for t in TOOLS]


def export_openapi_spec(server_url: str = "https://api.scm.example.com") -> Dict[str, Any]:
    """Full OpenAPI 3.1 spec covering all five tools. Suitable for upload
    to a ChatGPT Custom GPT as the Actions schema."""
    paths: Dict[str, Any] = {}
    for tool in TOOLS:
        paths.update(to_openapi_path(tool))
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "SCM Memory API",
            "description": (
                "Memory layer that works like human memory — wake phase "
                "(attention, encoding, retrieval) plus sleep phase "
                "(consolidation, schema extraction, knowledge-gap filling)."
            ),
            "version": "0.1.0",
        },
        "servers": [{"url": server_url}],
        "paths": paths,
    }


def get_tool(name: str) -> Optional[ToolDef]:
    """Look up a tool by name."""
    for t in TOOLS:
        if t.name == name:
            return t
    return None
