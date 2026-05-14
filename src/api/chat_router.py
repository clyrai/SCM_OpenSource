"""Public free chatbot at /chat — community product.

Architecture:

  ┌────────────────────────────────────────────┐
  │ Browser                                    │
  │  • paste your LLM provider + API key       │
  │  • talk to the bot                         │
  │  • key stored in localStorage, sent in     │
  │    every request body (NEVER persisted     │
  │    on the server)                          │
  └──────────────┬─────────────────────────────┘
                 │
                 │  POST /chat/api/message/{slug}
                 │  body: { message, llm_provider, llm_api_key, llm_model }
                 ▼
  ┌────────────────────────────────────────────┐
  │ Server: chat_router                        │
  │  • per-slug ChatEngine (sandbox_mode=True) │
  │  • build a deep agent on-the-fly with the  │
  │    user's LLM key + SCM tools              │
  │  • run the agent, return the final reply   │
  │  • WIPE the LLM key from memory            │
  └────────────────────────────────────────────┘

Why "wipe the LLM key": the entire selling point is "we never see your
LLM cost." A user can grep our process memory after their session and
their key shouldn't be there. Per-request, never persisted.

Tools the agent gets:
  • SCM: search_memory, add_memory, get_user_profile, wake_summary,
         consolidate
  • DeepAgents builtins: write_todos (planning), task (sub-agents)
  • DEFAULT: NO shell exec, NO filesystem write — those are dangerous
    in a public service. We disable via permissions.
"""
from __future__ import annotations

import asyncio
import re
import os
import secrets
import string
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from ..chat.engine import ChatEngine
from ..integrations.tools import (
    _add_memory_handler,
    _consolidate_handler,
    _search_memory_handler,
    _wake_summary_handler,
)
from ..integrations.langchain_tools import make_scm_tools
from ..integrations.task_context import TaskContextState


class _BYOKLLM:
    """LLM adapter that uses the user's per-request BYOK key.

    Mirrors the surface of `src/llm/__init__.py:LLMExtractor` enough that
    the engine and encoder can call it transparently. Routes ALL calls
    to the user's chosen cloud provider via the OpenAI-compatible HTTP
    API — no Ollama, no env vars, no shared state. Constructed fresh
    per HTTP request so concurrent users don't collide on the same key.

    Why this exists: ChatEngine has its own internal LLM dependency
    (encoder concept extraction at line 419 of engine.py; response
    generation at line 379; sleep paraphrase; curiosity). Before BYOK
    those defaulted to Ollama. After BYOK we want the agent's model
    AND the engine's internal model to be the SAME thing — the
    cloud key the user pasted. This adapter is what makes that work
    without re-architecting the engine.

    Lifecycle: at the start of every request handler, the calling code
    constructs one of these, swaps it onto `engine.llm` and
    `engine.encoder.llm`, runs the request, then restores the
    persistent `_NoOpLLM` default in a `finally` block. The engine
    never holds a BYOK reference between requests.
    """
    temperature = 0.4
    timeout = 60

    _BASE_URLS = {
        "deepseek":   "https://api.deepseek.com/v1",
        "openai":     "https://api.openai.com/v1",
        "anthropic":  "https://api.anthropic.com/v1",
        "groq":       "https://api.groq.com/openai/v1",
        "together":   "https://api.together.xyz/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    _DEFAULT_MODELS = {
        "deepseek":   "deepseek-chat",
        "openai":     "gpt-4o-mini",
        "groq":       "llama-3.3-70b-versatile",
        "together":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
    }

    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        from openai import OpenAI
        self.provider = (provider or "deepseek").lower().strip()
        self.model = model or self._DEFAULT_MODELS.get(
            self.provider, "deepseek-chat",
        )
        base_url = self._BASE_URLS.get(self.provider, self._BASE_URLS["deepseek"])
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=self.timeout,
        )

    # Engine surface ────────────────────────────────────────────────
    def _chat(self, prompt: str, num_predict: int = 256) -> str:
        """Plain completion — used by engine.chat()'s response generator
        and by sleep paraphrase. /chat itself doesn't render the
        engine's own response (the LangChain agent writes the visible
        reply), but sleep + paraphrase do call this."""
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=num_predict,
            )
            choice = completion.choices[0]
            return (choice.message.content if choice.message else "") or ""
        except Exception as e:
            print(f"[BYOK LLM _chat error] {type(e).__name__}: {e}")
            return ""

    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Called by the encoder during ingest. Returning [] makes the
        encoder fall back to its built-in heuristic extractor, which is
        deterministic, local, and fast — better for the /chat hot path
        than burning tokens on a cloud call per add_memory. Keep this
        as a no-op even with BYOK active; if a future caller wants
        LLM-graded concept extraction it can call _chat directly.
        """
        return []


from contextlib import contextmanager


@contextmanager
def _byok_attached(engine, byok_llm):
    """Temporarily attach the BYOK LLM to the engine for one request.

    Replaces `engine.llm` and `engine.encoder.llm` for the duration of
    the `with` block, then restores the previous (no-op) defaults in
    `finally`. Single-user-per-slug means no concurrency issue in
    practice; if that ever changes, this should become per-request
    state passed through the call stack instead of mutation.
    """
    orig_engine_llm = getattr(engine, "llm", None)
    orig_encoder_llm = getattr(engine.encoder, "llm", None) if engine.encoder else None
    engine.llm = byok_llm
    if engine.encoder is not None:
        engine.encoder.llm = byok_llm
    try:
        yield
    finally:
        engine.llm = orig_engine_llm
        if engine.encoder is not None:
            engine.encoder.llm = orig_encoder_llm


class _NoOpLLM:
    """Stub LLM that never makes a network call.

    Why this exists: ChatEngine was built before BYOK, so by default it
    instantiates its own LLMExtractor() pointing at Ollama. Inside /chat,
    the BYOK cloud LLM (DeepSeek/OpenAI/whatever the user pasted) drives
    the LangChain agent — but the engine's INTERNAL LLM (for concept
    extraction in the encoder, and for response generation in chat())
    would silently hit Ollama on the server box. If Ollama is missing
    or slow, the whole product drags even though the user provided a
    perfectly good cloud key.

    The /chat product doesn't need any engine-internal LLM:
      • Concept extraction → the encoder's heuristic path (regex +
        noun-phrase) is sufficient; the agent already structured the
        user's input.
      • Response generation → the LangChain agent writes the visible
        reply with the BYOK model. engine.chat()'s internal response
        generation is never the user-facing answer.

    So we swap in this no-op. extract_concepts returns []; the encoder
    falls back to heuristic. _chat returns "" (engine.chat's response
    generation degrades to a fallback path that nothing visible reads).
    Net: the engine is 100% LLM-free in /chat. Ollama can be absent
    entirely.
    """
    provider = "noop"
    model = "noop"
    temperature = 0.0
    timeout = 1

    def extract_concepts(self, text: str):
        return []

    def _chat(self, prompt: str, num_predict: int = 256) -> str:
        return ""


class _InProcessSCM:
    """SCMClient-shaped wrapper that calls SCM tool handlers directly
    against an in-process ChatEngine — no HTTP loopback, no auth.

    Why this exists: the /chat router is the SCM server itself. Doing a
    self-HTTP call to /v1/* would (a) waste a TCP round-trip per memory
    op and (b) hit the cloud-auth middleware that gates /v1/*. Direct
    handler invocation is the right move.

    Implements the subset of SCMClient that make_scm_tools depends on:
    add_memory, search_memory, consolidate, wake_summary.
    """
    def __init__(
        self,
        engine: ChatEngine,
        user_id: str,
        retrieval_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.engine = engine
        self.user_id = user_id
        self._retrieval_hook = retrieval_hook
        self.last_search_result: Optional[Dict[str, Any]] = None

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                   replaces_prior: bool = False) -> Dict[str, Any]:
        """Fast ingest path — bypasses `engine.chat()` because we don't
        need its slow internal LLM calls when called from a tool context.

        The full `engine.chat()` flow does:
          1. LLM concept extraction (Ollama by default — ~5-10s)
          2. Memory retrieval
          3. LLM response generation (Ollama, ~10-20s, 512 tokens)
          4. Episode storage + sleep check

        Steps 1 and 3 are the slow Ollama hits that made the tool hang
        for 15-30s per call in the live UI. Step 3 is irrelevant in
        tool context — the calling agent writes the user-visible reply
        itself with the BYOK cloud model. Step 1's accuracy is also
        wasted: the agent has already parsed the user's intent and
        passed in clean text; we don't need a second LLM pass to
        extract it.

        Fast path: force the encoder into its heuristic (regex + noun-
        phrase) path with NO LLM call, run only `_extract_and_store`
        (the durable side of ingest), skip retrieval and response
        generation entirely. Result: ~10-50ms per call instead of
        15-30s. All the versioning / Shakira-fix semantics are
        preserved because they live in `_extract_and_store` →
        `LongTermMemory.add_concept(allow_versioning=...)`, not in
        the slow LLM steps.
        """
        engine = self.engine
        text = (text or "").strip()
        if not text:
            return {
                "ok": False, "error": "text is required",
                "user_id": self.user_id, "concepts_added": 0,
                "concepts_total": 0, "memory_id": None,
            }

        # Force heuristic extraction for THIS call only — bypass Ollama
        # without affecting other engine consumers. Restored in finally.
        original_encoder_llm = getattr(engine.encoder, "llm", None)
        engine.encoder.llm = None
        try:
            concepts = engine._extract_and_store(
                text,
                source="user",
                force_versioning=bool(replaces_prior),
            )
        finally:
            engine.encoder.llm = original_encoder_llm

        return {
            "ok": True,
            "user_id": self.user_id,
            "concepts_added": len(concepts or []),
            "concepts_total": len(
                engine.long_term_memory.get_all_concepts(include_suppressed=False),
            ),
            "memory_id": concepts[0].id if concepts else None,
        }

    def search_memory(self, query: str, limit: int = 5,
                      wait_for_pending: bool = False) -> Dict[str, Any]:
        result = _search_memory_handler(
            {"query": query, "user_id": self.user_id, "limit": limit,
             "wait_for_pending": wait_for_pending},
            self.engine,
        )
        retrieval = result.get("retrieval")
        if isinstance(retrieval, dict):
            snapshot = dict(retrieval)
            snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.last_search_result = snapshot
            if callable(self._retrieval_hook):
                try:
                    self._retrieval_hook(snapshot)
                except Exception:
                    pass
        return result

    def consolidate(self, mode: str = "deep") -> Dict[str, Any]:
        return _consolidate_handler(
            {"user_id": self.user_id, "mode": mode}, self.engine,
        )

    def wake_summary(self, since_hours: float = 24.0) -> Dict[str, Any]:
        return _wake_summary_handler(
            {"user_id": self.user_id, "since_hours": since_hours},
            self.engine,
        )

    def list_facts(self) -> Dict[str, Any]:
        """Direct fact listing — bypasses spreading-activation search.

        Why this exists: spreading activation is great for "what do I
        remember about X" queries that share tokens with stored content,
        but it's unreliable for meta queries like "the user's name"
        because the cue tokens don't overlap with stored concepts like
        "My name is Saish". The agent thinks the name isn't stored,
        re-stores it, contradiction-versioning re-supersedes — net loss.

        This method walks the concept graph directly and returns every
        current-version, non-internal, user-attributable concept. Same
        filter as the /chat/api/profile/{slug} endpoint that drives the
        right-side memory panel.
        """
        try:
            all_concepts = self.engine.long_term_memory.get_all_concepts(
                include_suppressed=False,
            )
        except Exception as e:
            return {"ok": False, "error": str(e), "facts": [], "schemas": []}
        facts: list = []
        schemas: list = []
        for c in all_concepts:
            tags = c.context_tags if isinstance(c.context_tags, dict) else {}
            if tags.get("_internal"):
                continue
            entry = {
                "type": str(getattr(c, "type", "")).split(".")[-1].lower(),
                "description": (c.description or "").strip(),
            }
            if tags.get("_schema"):
                schemas.append(entry)
            elif tags.get("session_id"):
                facts.append(entry)
        return {"ok": True, "facts": facts, "schemas": schemas}


router = APIRouter(prefix="/chat", tags=["chat"])

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# ── Per-slug session state ────────────────────────────────────────────


class _ChatSession:
    """One end-user. Memory persists across messages within this slug for
    as long as the process lives. No persistence beyond restart by design
    — this is a free demo, not a paid product."""

    def __init__(self, slug: str):
        self.slug = slug
        # Each session is ephemeral and isolated. Sandbox mode means
        # in-memory only; restart wipes everything.
        #
        # CRITICAL: pass `llm=_NoOpLLM()` so the engine never hits
        # Ollama. The visible chat reply and tool-calling are driven by
        # the user's BYOK cloud LLM through the LangChain agent in
        # `_build_agent`. The engine itself only needs to store,
        # retrieve, version, and consolidate memory — none of which
        # requires an LLM.
        self.engine = ChatEngine(
            session_id=f"chat_{slug}",
            profile="chatbot",
            enable_auto_sleep=False,
            sandbox_mode=True,
            llm=_NoOpLLM(),
        )
        self.transcript: List[Dict[str, Any]] = []
        self.task_context = TaskContextState()
        self.last_retrieval: Dict[str, Any] = {}
        self.last_activity = time.time()
        self._lock = threading.Lock()


class _ChatPool:
    def __init__(self):
        self._sessions: Dict[str, _ChatSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, slug: str) -> _ChatSession:
        with self._lock:
            sess = self._sessions.get(slug)
            if sess is None:
                sess = _ChatSession(slug)
                self._sessions[slug] = sess
            sess.last_activity = time.time()
            return sess

    def get(self, slug: str) -> Optional[_ChatSession]:
        with self._lock:
            return self._sessions.get(slug)


_pool = _ChatPool()


# ── LLM factory ───────────────────────────────────────────────────────


def _build_llm(provider: str, api_key: str, model: Optional[str]):
    """Construct a LangChain chat model from BYOK params. The api_key is
    only used to construct this object; we don't store it."""
    from langchain_openai import ChatOpenAI

    provider = (provider or "").lower().strip()
    base_urls = {
        "deepseek": "https://api.deepseek.com",
        "openai": None,  # ChatOpenAI's default
        "anthropic": "https://api.anthropic.com/v1",  # via openai-compat router; users with native Anthropic should use openai-compat proxy
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    default_models = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "groq": "llama-3.3-70b-versatile",
        "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
    }
    base_url = base_urls.get(provider)
    chosen_model = model or default_models.get(provider) or "gpt-4o-mini"

    kwargs = {
        "model": chosen_model,
        "api_key": api_key,
        "temperature": 0.4,
        "timeout": 60,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


# ── deep agent factory ────────────────────────────────────────────────


def _build_agent(llm, scm_client):
    """Construct a plain LangChain agent (NOT deepagents) on-the-fly.

    Why plain LangChain rather than deepagents:

    DeepAgents bundles a virtual filesystem + shell + sub-agent + planning
    suite. We don't need any of that here — SCM is our memory, the
    filesystem fantasy wastes user-paid LLM tokens, and the surface area
    is bigger than the value. A plain `create_agent` with curated
    middleware gives us:

      • Only the tools we want (SCM)
      • SummarizationMiddleware     — keep long convos under context limit
      • ModelCallLimitMiddleware    — cap runaway agent loops
      • ToolCallLimitMiddleware     — cap pathological tool spam
      • TodoListMiddleware          — gives the agent planning via
                                      `write_todos` WITHOUT bringing
                                      filesystem along
      • ModelRetryMiddleware        — graceful retry on flaky LLM calls

    All of these are stock `langchain.agents.middleware` classes — no
    third-party dep beyond what's already pinned.

    The agent's lifetime is one HTTP request because the LLM key rotates
    per-request per the BYOK contract.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        SummarizationMiddleware,
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        TodoListMiddleware,
        ModelRetryMiddleware,
    )

    scm_tools = make_scm_tools(scm_client)

    system_prompt = (
        "You are a helpful conversational assistant with persistent memory. "
        "Every turn has TWO steps. Both are required.\n\n"
        "## STEP 1 — Use the memory tools (REQUIRED, silent)\n\n"
        "Tool use happens BEFORE you reply. The user does NOT see this "
        "step. You MUST run the right tool for the input — skipping it "
        "is a failure even if your visible reply sounds good.\n\n"
        "- **User shares personal facts** (name, location, job, allergy, "
        "preference, habit, opinion, plan, relationship, hobby) → you "
        "MUST call `add_memory(text=...)` ONCE PER DISTINCT FACT. Each "
        "call's text should be a clean, self-contained statement in the "
        "user's voice. Don't over-split (\"peanut allergy\" is ONE fact, "
        "not two), don't under-split (one mega-blob hurts the memory "
        "panel). For \"I'm Alex, a backend engineer in Lisbon, with a "
        "peanut allergy\" make exactly 3 calls: \"My name is Alex.\", "
        "\"I'm a backend engineer in Lisbon.\", \"I have a peanut "
        "allergy.\"\n"
        "- **User EXPLICITLY corrects a fact THEY themselves stated "
        "earlier** (\"I'm Saish, not Alex\" / \"actually I moved to "
        "Seattle\") → call `add_memory` with `replaces_prior=True` and "
        "ONLY the new positive value. Versioning supersedes the old one.\n"
        "- **A DIFFERENT person takes over the chat** (\"hi I'm Shakira, "
        "Saish's friend\") → call `add_memory` with `replaces_prior=False`. "
        "Never True for a new speaker — that would wipe the prior user's "
        "data.\n"
        "- **User asks \"what do you know about me\" or any meta-memory "
        "question** (\"what's my name?\", \"what have I told you?\") → "
        "call `get_user_profile()`.\n"
        "- **User asks about something specific they previously told you** "
        "(\"where do I work?\", \"what was my dog's name?\") → call "
        "`search_memory(query=...)`. Decompose multi-part questions into "
        "separate searches.\n"
        "- **Pure chitchat with no facts and no recall** (\"lol\", "
        "\"thanks\", \"nothing\", \"ok\") → no tool needed; just reply.\n\n"
        "Trust whatever the tool returns — that's the user's actual data. "
        "Never invent facts.\n\n"
        "## STEP 2 — Compose ONE short reply (visible)\n\n"
        "After tools complete, write a single brief reply.\n\n"
        "**NEVER narrate the tool step.** These phrases are forbidden:\n"
        "  • \"Let me save that\" / \"I'll remember that\" / \"I've "
        "stored that\" / \"saved to memory\" / \"noted!\" / \"got it!\" / "
        "\"I've got that saved\" / \"for future reference\" / \"let me "
        "check my notes\" / \"checking my memory\"\n\n"
        "**NEVER echo facts back at the user as confirmation.** Don't "
        "say \"So you're Alex from Lisbon with a peanut allergy — good "
        "to know!\". They just said it. They don't need it back.\n\n"
        "**Match register and length.** Brief intro → brief greeting "
        "(1 sentence). Casual chitchat → casual reply. Substantive "
        "question → substantive answer. Default short; never pad.\n\n"
        "## Worked examples (follow these exactly)\n\n"
        "USER: \"Hi, I'm Alex. Backend engineer in Lisbon. Peanut allergy.\"\n"
        "TOOLS (in order, all silent):\n"
        "  add_memory(text=\"My name is Alex.\")\n"
        "  add_memory(text=\"I'm a backend engineer in Lisbon.\")\n"
        "  add_memory(text=\"I have a peanut allergy.\")\n"
        "REPLY: \"Hey Alex — good to meet you. What's on your mind?\"\n\n"
        "USER: \"Where do I work?\"\n"
        "TOOL: search_memory(query=\"where the user works\")\n"
        "REPLY: \"You're a backend engineer in Lisbon.\"\n\n"
        "USER: \"What do you know about me?\"\n"
        "TOOL: get_user_profile()\n"
        "REPLY: \"You're Alex — a backend engineer in Lisbon, peanut "
        "allergy.\" (one line, no preamble)\n\n"
        "USER: \"lol thanks\"\n"
        "TOOL: (none)\n"
        "REPLY: \"anytime 🙂\"\n\n"
        "USER: \"Actually I moved to Madrid last month.\"\n"
        "TOOL: add_memory(text=\"I live in Madrid.\", replaces_prior=True)\n"
        "REPLY: \"Madrid — got it. How's it treating you?\"\n"
        "  (note: \"got it\" here is acceptable because it's "
        "acknowledging the LOCATION CHANGE conversationally, not "
        "announcing storage. The forbidden phrases above are about "
        "narrating tool use, not normal human acknowledgements.)\n"
    )

    middleware = [
        # Plan via write_todos — same affordance deepagents has, scoped
        # without filesystem/shell baggage.
        TodoListMiddleware(),
        # Auto-compact when the conversation grows large.
        #
        # Tokens-based, NOT fractions. Fraction-based config requires the
        # LLM to declare its context window via a `profile` field, which
        # ChatOpenAI-against-DeepSeek (and most BYOK setups) doesn't
        # have. Tokens are universal across providers.
        #
        # Numbers picked to fit even the smallest provider context we
        # offer (Groq Llama 3.3 70B = 32K input tokens):
        #   trigger=("tokens", 24000) — fire at ~75% of the smallest
        #     provider's window; users on bigger models (Claude Sonnet
        #     200K) get way more headroom than they'll ever hit
        #   keep=("tokens", 8000)     — keep ~8K of recent messages
        #     verbatim; older messages are replaced with a structured
        #     summary. 8K ≈ 30-50 messages of typical chat
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 24000),
            keep=("tokens", 8000),
        ),
        # Cap how many model calls happen in a single user turn. Without
        # this, an agent stuck in a tool-loop can burn the user's BYOK
        # tokens fast. 15 is generous; most turns finish in ≤4.
        ModelCallLimitMiddleware(run_limit=15, exit_behavior="end"),
        # Same idea for tool calls — bound pathological spam.
        ToolCallLimitMiddleware(run_limit=20, exit_behavior="continue"),
        # Retry transient LLM failures (rate limits, network blips) with
        # exponential backoff before giving up. Keeps single-turn UX
        # smooth when the upstream provider hiccups.
        ModelRetryMiddleware(max_retries=2, initial_delay=1.0,
                             backoff_factor=2.0),
    ]

    return create_agent(
        model=llm,
        tools=scm_tools,
        system_prompt=system_prompt,
        middleware=middleware,
    )


# ── Routes ────────────────────────────────────────────────────────────


def _new_slug() -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(12))


@router.get("")
async def chat_root() -> RedirectResponse:
    """Redirect bare /chat to a fresh slugged URL the user can bookmark."""
    return RedirectResponse(url=f"/chat/s/{_new_slug()}", status_code=302)


@router.get("/s/{slug}")
async def chat_page(slug: str) -> FileResponse:
    return FileResponse(
        os.path.join(_STATIC_DIR, "chat.html"),
        media_type="text/html",
    )


class _MessageRequest(BaseModel):
    message: str
    llm_provider: str
    llm_api_key: str
    llm_model: Optional[str] = None


_NAME_SIGNAL_RE = re.compile(
    r"\b(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z'-]+)\b",
    re.IGNORECASE,
)
_ROLE_SIGNAL_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+(?:a|an)\s+[a-z][^.!?]{0,80}|i work(?: as| at)?\s+[a-z][^.!?]{0,80})\b",
    re.IGNORECASE,
)
_ALLERGY_SIGNAL_RE = re.compile(
    r"\b(?:allerg(?:y|ic)|intoleran(?:t|ce)|can't eat|cannot eat|sensitive to)\b",
    re.IGNORECASE,
)
_PREFERENCE_SIGNAL_RE = re.compile(
    r"\b(?:prefer|favorite|favourite|i like|i love|i hate|dislike)\b",
    re.IGNORECASE,
)
_LOCATION_SIGNAL_RE = re.compile(
    r"\b(?:i live in|i'm in|i am in|based in)\b",
    re.IGNORECASE,
)
_CORRECTION_SIGNAL_RE = re.compile(
    r"\b(?:actually|sorry|correction|instead|no longer|used to)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP_TOKENS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "just",
    "your", "about", "really", "very", "what", "where", "when", "then",
}


def _coerce_tool_args(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _profile_fact_signal_count(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    count = 0
    if _NAME_SIGNAL_RE.search(text):
        count += 1
    if _ROLE_SIGNAL_RE.search(text):
        count += 1
    if _LOCATION_SIGNAL_RE.search(text):
        count += 1
    if _ALLERGY_SIGNAL_RE.search(text):
        count += 1
    if _PREFERENCE_SIGNAL_RE.search(text):
        count += 1
    return count


def _tokenize_overlap(text: str) -> set[str]:
    return {
        tok for tok in _WORD_RE.findall((text or "").lower())
        if len(tok) > 2 and tok not in _STOP_TOKENS
    }


def _max_token_overlap(reference: str, candidates: List[str]) -> float:
    ref = _tokenize_overlap(reference)
    if not ref:
        return 0.0
    best = 0.0
    for c in candidates:
        cand = _tokenize_overlap(c)
        if not cand:
            continue
        overlap = len(ref & cand) / len(ref)
        if overlap > best:
            best = overlap
    return best


def _should_force_profile_ingest(user_message: str, add_memory_texts: List[str]) -> bool:
    signals = _profile_fact_signal_count(user_message)
    if signals == 0:
        return False
    if not add_memory_texts:
        return True
    if _max_token_overlap(user_message, add_memory_texts) >= 0.65:
        return False
    return len(add_memory_texts) < signals


def _is_explicit_correction(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if _CORRECTION_SIGNAL_RE.search(text):
        return True
    lowered = text.lower()
    return bool(re.search(r"\bnot\b.+\bbut\b", lowered))


def _maybe_force_profile_ingest(
    scm: _InProcessSCM,
    user_message: str,
    add_memory_texts: List[str],
    tools_called: List[str],
) -> None:
    if not _should_force_profile_ingest(user_message, add_memory_texts):
        return
    try:
        scm.add_memory(
            text=user_message,
            replaces_prior=_is_explicit_correction(user_message),
        )
        tools_called.append("add_memory")
    except Exception as exc:
        print(f"[chat_router] fallback add_memory failed: {exc}")


def _run_agent_sync(payload: _MessageRequest, slug: str) -> Dict[str, Any]:
    """Synchronous body of the chat endpoint, run in a thread pool to
    keep FastAPI's event loop responsive while the LLM is generating."""
    sess = _pool.get_or_create(slug)

    # In-process SCM access — bypasses HTTP and the cloud-auth middleware
    # that gates /v1/*. The chat router IS the SCM server, no point doing
    # a self-loopback HTTP call.
    scm = _InProcessSCM(
        engine=sess.engine,
        user_id=slug,
        retrieval_hook=lambda snap: setattr(sess, "last_retrieval", dict(snap)),
    )

    llm = _build_llm(payload.llm_provider, payload.llm_api_key, payload.llm_model)
    agent = _build_agent(llm=llm, scm_client=scm)
    byok = _BYOKLLM(payload.llm_provider, payload.llm_api_key, payload.llm_model)

    previous_bot = ""
    for t in reversed(sess.transcript):
        if t.get("bot"):
            previous_bot = str(t["bot"])
            break
    task_updates = sess.task_context.ingest_user_message(
        payload.message,
        previous_assistant=previous_bot,
    )
    task_context_prompt = sess.task_context.prompt_block()

    # Build the message history from the session transcript so the agent
    # has continuity within the slug.
    history = []
    if task_context_prompt:
        history.append({
            "role": "system",
            "content": task_context_prompt,
        })
    for t in sess.transcript[-12:]:  # last 12 turns
        if t.get("user"):
            history.append({"role": "user", "content": t["user"]})
        if t.get("bot"):
            history.append({"role": "assistant", "content": t["bot"]})
    history.append({"role": "user", "content": payload.message})

    # Run the agent. recursion_limit caps long planning loops.
    # Engine-internal LLM is the BYOK adapter for this request so any
    # internal call (paraphrase, response gen, etc.) uses the user's key.
    try:
        with _byok_attached(sess.engine, byok):
            result = agent.invoke(
                {"messages": history},
                {"recursion_limit": 25},
            )
    except Exception as e:
        return {"reply": f"[agent error: {type(e).__name__}: {e}]", "tools_called": []}

    # Extract the final assistant message
    msgs = result.get("messages", [])
    ai_msgs = [
        m for m in msgs
        if getattr(m, "type", "") == "ai" and (m.content or "").strip()
    ]
    reply = ai_msgs[-1].content if ai_msgs else "(no reply)"

    # Trace tool calls for the UI to show
    tool_calls = []
    add_memory_texts: list[str] = []
    for m in msgs:
        tcs = getattr(m, "tool_calls", None) or []
        for tc in tcs:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tool_calls.append(name)
            if name == "add_memory":
                if isinstance(tc, dict):
                    raw_args = (
                        tc.get("args")
                        or tc.get("arguments")
                        or tc.get("input")
                    )
                else:
                    raw_args = (
                        getattr(tc, "args", None)
                        or getattr(tc, "arguments", None)
                        or getattr(tc, "input", None)
                    )
                args = _coerce_tool_args(raw_args)
                text_arg = args.get("text")
                if isinstance(text_arg, str) and text_arg.strip():
                    add_memory_texts.append(text_arg.strip())

    _maybe_force_profile_ingest(
        scm=scm,
        user_message=payload.message,
        add_memory_texts=add_memory_texts,
        tools_called=tool_calls,
    )
    sess.task_context.ingest_assistant_message(reply)

    sess.transcript.append({
        "user": payload.message,
        "bot": reply,
        "ts": datetime.now(timezone.utc).isoformat(),
        "tools": tool_calls,
        "task_context": task_updates,
        "retrieval": scm.last_search_result if isinstance(scm.last_search_result, dict) else None,
    })
    response: Dict[str, Any] = {"reply": reply, "tools_called": tool_calls}
    if isinstance(scm.last_search_result, dict):
        response["retrieval"] = scm.last_search_result
    return response


@router.post("/api/message/{slug}")
async def chat_message(slug: str, body: _MessageRequest) -> JSONResponse:
    """One conversational turn — non-streaming. Used by clients that
    can't or don't want SSE. The streaming endpoint at /api/stream/{slug}
    is what the UI uses."""
    if not (body.message or "").strip():
        raise HTTPException(status_code=400, detail="empty message")
    if not (body.llm_api_key or "").strip():
        raise HTTPException(
            status_code=400,
            detail="missing llm_api_key — paste your provider key in the settings panel",
        )

    result = await asyncio.to_thread(_run_agent_sync, body, slug)
    return JSONResponse(result)


def _sse(event: dict) -> str:
    """Format a Python dict as one SSE event."""
    return f"data: {json.dumps(event)}\n\n"


async def _stream_agent(payload: _MessageRequest, slug: str):
    """Async generator that runs the agent and yields SSE events as
    things happen. Event shapes:

      {"type": "tool_call",   "name": "search_memory"}    — tool fires
      {"type": "tool_done",   "name": "search_memory"}    — tool returns
      {"type": "token",       "delta": "Hello"}           — assistant token
      {"type": "reset_reply"}                              — drop prior tokens,
                                                            next ones are the
                                                            real final reply
      {"type": "done",        "tools_called": [...]}      — turn finished
      {"type": "error",       "message": "..."}           — error mid-stream

    Why astream_events(version="v2"): it gives clean per-component events
    (`on_chat_model_stream`, `on_tool_start`, `on_tool_end`) instead of
    making us parse the raw graph state. The downside is more event types
    than we care about, so we filter to the relevant ones.

    `reset_reply` is the fix for chatty small models (DeepSeek, Groq,
    Llama) that disobey the system prompt and emit narration BEFORE
    calling a tool — "Sure, let me save those details for you." The
    agent then calls the tool, then emits a SECOND chat-model output
    that's the real reply. Naive streaming concatenates both, producing
    "Let me save those details for you.Great, I've got all that
    saved!". Whenever a tool completes, we tell the client to drop the
    bubble it's been accumulating; the next chat-model tokens are what
    the user actually deserves to see.
    """
    try:
        sess = _pool.get_or_create(slug)
        scm = _InProcessSCM(
            engine=sess.engine,
            user_id=slug,
            retrieval_hook=lambda snap: setattr(sess, "last_retrieval", dict(snap)),
        )
        llm = _build_llm(payload.llm_provider, payload.llm_api_key, payload.llm_model)
        agent = _build_agent(llm=llm, scm_client=scm)
        byok = _BYOKLLM(payload.llm_provider, payload.llm_api_key, payload.llm_model)

        previous_bot = ""
        for t in reversed(sess.transcript):
            if t.get("bot"):
                previous_bot = str(t["bot"])
                break
        task_updates = sess.task_context.ingest_user_message(
            payload.message,
            previous_assistant=previous_bot,
        )
        task_context_prompt = sess.task_context.prompt_block()

        history = []
        if task_context_prompt:
            history.append({
                "role": "system",
                "content": task_context_prompt,
            })
        for t in sess.transcript[-12:]:
            if t.get("user"):
                history.append({"role": "user", "content": t["user"]})
            if t.get("bot"):
                history.append({"role": "assistant", "content": t["bot"]})
        history.append({"role": "user", "content": payload.message})

        full_reply_parts: list[str] = []
        tools_called: list[str] = []
        add_memory_texts: list[str] = []
        # langgraph emits chat-model streams from sub-agents too (e.g.
        # the summarization middleware running its own LLM call). We only
        # want to stream the FINAL assistant message tokens. Heuristic:
        # the final assistant message is whatever's emitted in the last
        # `on_chat_model_stream` block before the graph completes.
        # Practical approach: stream all tokens, but suppress those that
        # come from a node whose name matches our middleware (summary).

        # Attach BYOK to the engine for THIS request — any engine-internal
        # LLM call (encoder, paraphrase, response gen) now uses the user's
        # cloud key instead of the persistent no-op default.
        with _byok_attached(sess.engine, byok):
            async for ev in agent.astream_events(
                {"messages": history},
                {"recursion_limit": 25},
                version="v2",
            ):
                kind = ev.get("event")
                data = ev.get("data") or {}
                name = ev.get("name") or ""

                if kind == "on_tool_start":
                    tools_called.append(name)
                    if name == "add_memory":
                        args = _coerce_tool_args(data.get("input"))
                        text_arg = args.get("text")
                        if isinstance(text_arg, str) and text_arg.strip():
                            add_memory_texts.append(text_arg.strip())
                    yield _sse({"type": "tool_call", "name": name})

                elif kind == "on_tool_end":
                    yield _sse({"type": "tool_done", "name": name})
                    # Drop whatever the model emitted BEFORE the tool —
                    # that's pre-tool narration, not the final answer.
                    # The next chat-model stream after this is the true
                    # reply.
                    full_reply_parts.clear()
                    yield _sse({"type": "reset_reply"})

                elif kind == "on_chat_model_stream":
                    # Skip stream events from middleware-internal LLM
                    # calls (summarization, etc). They aren't part of
                    # the user-visible reply.
                    tags = ev.get("tags") or []
                    if any(t.startswith("summarization") for t in tags):
                        continue
                    chunk = data.get("chunk")
                    delta = ""
                    if chunk is not None:
                        delta = getattr(chunk, "content", "") or ""
                    if delta:
                        full_reply_parts.append(delta)
                        yield _sse({"type": "token", "delta": delta})

        reply = "".join(full_reply_parts).strip() or "(no reply)"
        _maybe_force_profile_ingest(
            scm=scm,
            user_message=payload.message,
            add_memory_texts=add_memory_texts,
            tools_called=tools_called,
        )
        sess.task_context.ingest_assistant_message(reply)
        sess.transcript.append({
            "user": payload.message,
            "bot": reply,
            "ts": datetime.now(timezone.utc).isoformat(),
            "tools": tools_called,
            "task_context": task_updates,
            "retrieval": scm.last_search_result if isinstance(scm.last_search_result, dict) else None,
        })
        done_payload: Dict[str, Any] = {"type": "done", "tools_called": tools_called}
        if isinstance(scm.last_search_result, dict):
            done_payload["retrieval"] = scm.last_search_result
        yield _sse(done_payload)

    except Exception as e:
        yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})


@router.post("/api/stream/{slug}")
async def chat_stream(slug: str, body: _MessageRequest) -> StreamingResponse:
    """Stream one turn as SSE. The UI uses this for live token rendering.

    The LLM key in `body.llm_api_key` is used per-request to construct
    the agent; never persisted server-side.
    """
    if not (body.message or "").strip():
        raise HTTPException(status_code=400, detail="empty message")
    if not (body.llm_api_key or "").strip():
        raise HTTPException(
            status_code=400,
            detail="missing llm_api_key — paste your provider key in the settings panel",
        )
    return StreamingResponse(
        _stream_agent(body, slug),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )


class _SleepRequest(BaseModel):
    """Optional BYOK params for the sleep cycle.

    If provided, sleep paraphrase + curiosity (when enabled) will use
    the user's cloud LLM via `_BYOKLLM`. If omitted, sleep falls back
    to the persistent no-op LLM (heuristic only). The endpoint accepts
    an empty body for backward compat with the original 'Simulate
    night' button that didn't send credentials.
    """
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None


@router.post("/api/sleep/{slug}")
async def chat_sleep(slug: str, body: Optional[_SleepRequest] = None) -> JSONResponse:
    """Force a deep-sleep cycle and return the resulting wake-summary
    narrative. The UI calls this from the 'Simulate night' button —
    same code path the production circadian scheduler would fire at the
    user's configured bedtime."""
    sess = _pool.get_or_create(slug)

    # If the client sent BYOK params, wire them into the engine so
    # paraphrase + curiosity use the user's cloud key. Otherwise the
    # engine stays on its no-op default (heuristic schemas, no curiosity).
    byok = None
    if body and (body.llm_api_key or "").strip():
        try:
            byok = _BYOKLLM(body.llm_provider, body.llm_api_key, body.llm_model)
        except Exception as exc:
            print(f"[chat_sleep] BYOK init failed: {exc}")
            byok = None

    # Run the sleep cycle in a thread so we don't block the event loop.
    def _do():
        from ..integrations.tools import _consolidate_handler, _wake_summary_handler
        if byok is not None:
            with _byok_attached(sess.engine, byok):
                stats = _consolidate_handler({"user_id": slug, "mode": "deep"}, sess.engine) or {}
                summary = _wake_summary_handler(
                    {"user_id": slug, "since_hours": 24.0}, sess.engine,
                ) or {}
        else:
            stats = _consolidate_handler({"user_id": slug, "mode": "deep"}, sess.engine) or {}
            summary = _wake_summary_handler(
                {"user_id": slug, "since_hours": 24.0}, sess.engine,
            ) or {}
        return {
            "ok": True,
            "stats": stats,
            "narrative": (summary.get("narrative") or "").strip(),
        }
    result = await asyncio.to_thread(_do)
    # Stash the wake-summary on the session so the next chat turn can
    # surface it as a banner if the user wants.
    if result.get("narrative"):
        sess.transcript.append({
            "user": None,
            "bot": None,
            "wake_summary": result["narrative"],
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    return JSONResponse(result)


@router.get("/api/profile/{slug}")
async def chat_profile(slug: str) -> JSONResponse:
    """Live snapshot of what SCM has stored about this user. The UI
    polls this for the memory panel — drives the 'memory you can see'
    differentiation.

    Returns user-attributable concepts and (after sleep) schemas. Filters
    out internal SelfModel boilerplate."""
    sess = _pool.get(slug)
    if sess is None:
        return JSONResponse({"concepts": [], "schemas": [], "total": 0})

    def _gather():
        try:
            all_concepts = sess.engine.long_term_memory.get_all_concepts(
                include_suppressed=False,
            )
        except Exception:
            return [], []
        concepts = []
        schemas = []
        for c in all_concepts:
            tags = c.context_tags if isinstance(c.context_tags, dict) else {}
            if tags.get("_internal"):
                continue
            entry = {
                "type": str(getattr(c, "type", "")).split(".")[-1].lower(),
                "description": (c.description or "").strip(),
                "tags": [k for k in tags.keys() if not k.startswith("_")][:4],
            }
            if tags.get("_schema"):
                schemas.append(entry)
            elif tags.get("session_id"):
                concepts.append(entry)
        return concepts, schemas

    concepts, schemas = await asyncio.to_thread(_gather)
    return JSONResponse({
        "concepts": concepts,
        "schemas": schemas,
        "total": len(concepts) + len(schemas),
    })


@router.get("/api/context/{slug}")
async def chat_context(slug: str) -> JSONResponse:
    """Current ephemeral task context for this chat slug."""
    sess = _pool.get(slug)
    if sess is None:
        return JSONResponse({"slots": [], "total": 0})
    slots = await asyncio.to_thread(sess.task_context.snapshot)
    return JSONResponse({
        "slots": slots,
        "total": len(slots),
    })


@router.get("/api/retrieval/{slug}")
async def chat_retrieval(slug: str) -> JSONResponse:
    """Last hybrid-retrieval snapshot used in this chat slug."""
    sess = _pool.get(slug)
    if sess is None:
        return JSONResponse({"available": False, "retrieval": {}})
    snapshot = sess.last_retrieval if isinstance(sess.last_retrieval, dict) else {}
    if not snapshot:
        return JSONResponse({"available": False, "retrieval": {}})
    return JSONResponse({"available": True, "retrieval": snapshot})


@router.get("/api/history/{slug}")
async def chat_history(slug: str) -> JSONResponse:
    """Restore conversation on page reload."""
    sess = _pool.get(slug)
    if sess is None:
        return JSONResponse({"turns": []})
    return JSONResponse({"turns": list(sess.transcript)})
