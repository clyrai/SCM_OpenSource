"""
SCM MCP server — exposes the SCM memory layer to any Model Context Protocol
client (Claude Desktop, Cursor, ChatGPT-with-MCP, VS Code Continue, custom
agents).

Designed to behave like human memory:
  - The integrating agent only calls `add_memory` and `search_memory`.
  - SCM detects idle automatically (no message for IDLE_THRESHOLD seconds)
    and fires a sleep cycle in the background.
  - On the next user activity, the cached wake_summary is available via
    the `wake_summary` tool and via a notification on the next add_memory
    response.
  - `consolidate` is exposed as a manual override, but the agent doesn't
    need to call it under normal use.

Run as a stdio MCP server (default — works with Claude Desktop / Cursor):

    scm mcp           # via the CLI
    python -m src.integrations.mcp_server

Or as an HTTP MCP server (for clients that prefer HTTP transport):

    SCM_MCP_TRANSPORT=http python -m src.integrations.mcp_server

Environment:
    SCM_DATA_DIR             where to persist user memories (default ~/.scm)
    SCM_IDLE_THRESHOLD_SEC   idle threshold for auto-sleep (default 300)
    SCM_AUTO_SLEEP_DISABLE   set to '1' to disable autonomous sleep
    LLM_PROVIDER             ollama | deepseek | openai (default: heuristic)
    SCM_EMBEDDING_BACKEND    sentence_transformers | ollama | openai_compat
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .task_context import TaskContextState

logger = logging.getLogger("scm.mcp")


# ─── Per-user engine pool with auto-sleep ─────────────────────────────────


class UserEnginePool:
    """One ChatEngine per user_id, with automatic idle-driven consolidation.

    This is the central abstraction that makes SCM behave like human
    memory at the API surface: every tool call bumps the user's
    last-activity timestamp; a background sweeper checks every N seconds
    whether any user has been idle past threshold; idle users get a
    deep-sleep cycle automatically; the resulting wake-summary is
    cached for the next time the user returns.
    """

    def __init__(
        self,
        idle_threshold_sec: float = 300.0,
        sweep_interval_sec: float = 30.0,
        auto_sleep: bool = True,
        legacy_idle_mode: Optional[bool] = None,
    ):
        self._engines: Dict[str, Any] = {}
        self._last_activity: Dict[str, float] = {}
        self._cached_summaries: Dict[str, Any] = {}
        self._task_context: Dict[str, TaskContextState] = {}
        self._sleep_lock = threading.Lock()
        self._idle_threshold = idle_threshold_sec
        self._sweep_interval = sweep_interval_sec
        self._auto_sleep = auto_sleep
        # Backwards compat: when a deployment sets SCM_IDLE_THRESHOLD_SEC
        # without ever writing a per-user sleep config, keep the legacy
        # idle-timer behavior. Once the user POSTs a sleep-config, the
        # scheduler transitions automatically to the circadian model.
        if legacy_idle_mode is None:
            legacy_idle_mode = os.environ.get("SCM_LEGACY_IDLE_SLEEP", "1") == "1"
        self._legacy_idle_mode = bool(legacy_idle_mode)
        self._stop_flag = threading.Event()
        self._sweeper: Optional[threading.Thread] = None

        # Per-user async ingest queues (v0.7.2). Adding a memory enqueues
        # the text + metadata; a per-user worker thread drains the queue
        # and calls engine.chat() in the background. The user-facing
        # add_memory call returns in <100ms regardless of LLM speed.
        import queue as _queue
        self._ingest_queues: Dict[str, _queue.Queue] = {}
        self._ingest_workers: Dict[str, threading.Thread] = {}
        # pending counter per user — search_memory can wait for it to drain
        self._pending_lock = threading.Lock()
        self._pending_count: Dict[str, int] = {}
        # Notification: workers signal when they finish a task so
        # `wait_for_pending` can wake immediately instead of polling.
        self._pending_done = threading.Condition(self._pending_lock)

    def get_or_create(self, user_id: str, bump_activity: bool = True) -> Any:
        """Return the per-user ChatEngine, building it lazily.

        bump_activity:
            True  — this is real user activity (add_memory, search_memory),
                    so reset the user's idle timer.
            False — this is a system / admin operation (consolidate,
                    wake_summary, forget). Don't reset the timer; an
                    operator manually consolidating doesn't mean the user
                    came back.
        """
        if user_id in self._engines:
            if bump_activity:
                self._touch(user_id)
            return self._engines[user_id]
        engine = self._build_engine(user_id)
        self._engines[user_id] = engine
        # Always bump on first creation — the engine wouldn't exist if no
        # user activity had triggered it.
        self._touch(user_id)
        logger.info(f"[scm.mcp] created engine for user {user_id!r}")
        return engine

    def _touch(self, user_id: str) -> None:
        self._last_activity[user_id] = time.time()

    # ── Async ingest queue ──────────────────────────────────────────────────

    def enqueue_ingest(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Enqueue a memory for background ingest. Returns a placeholder id.

        The text won't be in LTM until the worker drains the queue (~1-2s
        depending on LLM extractor latency). A subsequent search_memory
        call won't see it unless the caller passes wait_for_pending=true.
        """
        import queue as _queue
        import uuid

        # Lazy-init the per-user queue + worker on first ingest.
        if user_id not in self._ingest_queues:
            q = _queue.Queue()
            self._ingest_queues[user_id] = q
            t = threading.Thread(
                target=self._ingest_worker,
                args=(user_id, q),
                name=f"scm-ingest-{user_id}",
                daemon=True,
            )
            t.start()
            self._ingest_workers[user_id] = t

        # Reserve a placeholder id; real concept id is assigned by SCM.
        placeholder = f"pending_{uuid.uuid4().hex[:12]}"
        with self._pending_lock:
            self._pending_count[user_id] = self._pending_count.get(user_id, 0) + 1
        self._ingest_queues[user_id].put({
            "text": text,
            "metadata": metadata or {},
            "placeholder": placeholder,
            "enqueued_at": time.time(),
        })
        # Bump activity — async ingest IS user activity.
        self._touch(user_id)
        return placeholder

    def _ingest_worker(self, user_id: str, q) -> None:
        """Per-user background worker that drains the ingest queue."""
        import queue as _queue
        while not self._stop_flag.is_set():
            try:
                task = q.get(timeout=1.0)
            except _queue.Empty:
                continue
            if task is None:  # shutdown sentinel
                break
            try:
                # Lazy-create engine on first ingest if not already.
                engine = self._engines.get(user_id)
                if engine is None:
                    engine = self._build_engine(user_id)
                    self._engines[user_id] = engine
                    logger.info(f"[scm.mcp] created engine for user {user_id!r} (via async ingest)")
                # The actual ingest — this is what was blocking the API path.
                engine.chat(task["text"])
            except Exception as e:
                logger.warning(f"[scm.mcp] async ingest failed for {user_id!r}: {e!r}")
            finally:
                q.task_done()
                with self._pending_lock:
                    self._pending_count[user_id] = max(
                        0, self._pending_count.get(user_id, 0) - 1
                    )
                    self._pending_done.notify_all()

    def pending_count(self, user_id: str) -> int:
        with self._pending_lock:
            return self._pending_count.get(user_id, 0)

    def wait_for_pending(self, user_id: str, timeout: float = 5.0) -> bool:
        """Block until the user's ingest queue is drained, or timeout.

        Returns True if drained, False if timed out. Tests / callers that
        need a write-then-read consistency guarantee can call this between
        add_memory and search_memory.
        """
        deadline = time.time() + timeout
        with self._pending_lock:
            while self._pending_count.get(user_id, 0) > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._pending_done.wait(timeout=remaining)
        return True

    def fire_sleep_now(self, user_id: str, mode: str = "deep") -> Dict[str, Any]:
        """Public method: fire a sleep cycle and cache the wake-summary.

        Used by the manual consolidate path so a programmatic consolidate
        produces a wake-summary the same way the autonomous sweeper would.
        """
        engine = self._engines.get(user_id)
        if engine is None:
            return {"ok": False, "error": f"no engine for user {user_id!r}"}
        with self._sleep_lock:
            try:
                stats = engine.force_sleep(mode) or {}
            except Exception as e:
                return {"ok": False, "error": str(e)}
            # Cache a wake-summary for this user. since=last_activity gives
            # the right window for a manual cycle.
            try:
                from datetime import timedelta
                last_act = self._last_activity.get(user_id, time.time())
                idle_for = max(60.0, time.time() - last_act + 60.0)
                since = datetime.now(timezone.utc) - timedelta(seconds=idle_for)
                summary = engine._wake_summary_builder.build(since=since)
                self._cached_summaries[user_id] = {
                    "narrative": getattr(summary, "narrative", "") or "",
                    "insights": [str(i) for i in (getattr(summary, "insights", []) or [])[:8]],
                    "fired_at": datetime.now(timezone.utc).isoformat(),
                    "trigger": "manual",
                    "mode": mode,
                }
            except Exception as e:
                logger.warning(f"wake-summary build failed for {user_id!r}: {e}")
            return {"ok": True, **stats}

    @staticmethod
    def _build_engine(user_id: str):
        """Build a ChatEngine using env-configured LLM + embedding backends."""
        from src.chat.engine import ChatEngine
        from src.chat import engine as engine_mod
        from src.core.encoder import MeaningEncoder
        from src.lifecycle.curiosity import (
            CuriosityConfig,
            CuriosityEngine,
            StaticDictionarySource,
        )
        from src.lifecycle.wake_summary import WakeSummaryBuilder
        from src.sleep.deep_sleep import DeepSleep
        from src.sleep.schema_extractor import SchemaExtractor, SchemaExtractorConfig
        from src.sleep.sleep_cycle import SleepCycleOrchestrator

        engine_mod.HME_ENABLED = True

        llm = None
        provider = os.environ.get("LLM_PROVIDER", "").lower()
        if provider in ("ollama", "deepseek", "openai"):
            try:
                from src.llm import LLMExtractor
                llm = LLMExtractor(provider=provider)
            except Exception as e:
                logger.warning(f"LLM extractor unavailable ({e}); falling back to heuristic.")

        encoder = MeaningEncoder(llm=llm)

        deep = DeepSleep(
            enable_synthesis=False,
            enable_schema_extraction=True,
            schema_extractor=SchemaExtractor(
                config=SchemaExtractorConfig(enabled=True),
            ),
            enable_paraphrase=True,
            enable_curiosity=True,
            curiosity_engine=CuriosityEngine(
                sources=[StaticDictionarySource({})],
                config=CuriosityConfig(enabled=True, min_occurrences=2,
                                       max_gaps_per_cycle=3),
            ),
        )
        orch = SleepCycleOrchestrator(deep_sleep=deep)

        # IMPORTANT: sandbox_mode=True for multi-user safety.
        # The underlying sqlite_db module is a process-global singleton,
        # so persisting from multiple users in one process leaks
        # concepts across users (Tier 6 of the brutal harness caught
        # this). Sandbox mode keeps each engine's graph in-memory only,
        # giving clean per-user isolation. Per-user SQLite persistence
        # is a v0.2 architectural change.
        engine = ChatEngine(
            llm=llm,
            encoder=encoder,
            sleep_orchestrator=orch,
            session_id=f"mcp_{user_id}",
            profile="research",
            sandbox_mode=True,
            enable_persistence=False,
            enable_auto_sleep=False,  # we drive sleep from the sweeper instead
        )
        engine._wake_summary_builder = WakeSummaryBuilder(engine=engine)
        return engine

    def cached_summary(self, user_id: str) -> Optional[Any]:
        return self._cached_summaries.get(user_id)

    def clear_cached_summary(self, user_id: str) -> None:
        self._cached_summaries.pop(user_id, None)

    # ─── Ephemeral task-context state ───────────────────────────────────

    def _task_context_for(self, user_id: str) -> TaskContextState:
        state = self._task_context.get(user_id)
        if state is None:
            state = TaskContextState()
            self._task_context[user_id] = state
        return state

    def ingest_task_message(
        self,
        user_id: str,
        text: str,
        previous_assistant: str = "",
    ) -> list[dict]:
        return self._task_context_for(user_id).ingest_user_message(
            text=text,
            previous_assistant=previous_assistant,
        )

    def ingest_task_assistant(self, user_id: str, text: str) -> None:
        self._task_context_for(user_id).ingest_assistant_message(text)

    def task_context_snapshot(self, user_id: str) -> list[dict]:
        return self._task_context_for(user_id).snapshot()

    def clear_task_context(self, user_id: str) -> None:
        self._task_context_for(user_id).clear()

    # ─── Background sweeper ───────────────────────────────────────────────

    def start(self) -> None:
        if self._sweeper is not None:
            return
        if not self._auto_sleep:
            logger.info("[scm.mcp] auto-sleep disabled; sweeper not started")
            return
        self._sweeper = threading.Thread(
            target=self._sweep_loop, name="scm-mcp-sweeper", daemon=True
        )
        self._sweeper.start()
        logger.info(f"[scm.mcp] idle sweeper started (threshold={self._idle_threshold}s)")

    def stop(self) -> None:
        self._stop_flag.set()
        # Signal each ingest worker to exit by enqueueing the sentinel.
        for q in self._ingest_queues.values():
            try:
                q.put(None)
            except Exception:
                pass
        if self._sweeper is not None:
            self._sweeper.join(timeout=5)
        for t in self._ingest_workers.values():
            t.join(timeout=2)

    def _sweep_loop(self) -> None:
        tick = 0
        while not self._stop_flag.wait(self._sweep_interval):
            tick += 1
            try:
                fired = self._sweep_once()
                # Every 10 ticks, log what state we're in so production
                # operators (and the brutal harness) can verify the
                # sweeper is alive.
                if tick % 10 == 0 or fired:
                    logger.info(
                        f"[scm.mcp] sweep tick={tick} users={len(self._last_activity)} "
                        f"cached={len(self._cached_summaries)} fired_this_tick={fired}"
                    )
            except Exception:
                logger.exception("sweep iteration failed")

    def _sweep_once(self) -> int:
        """v0.7.7+ circadian model: fire sleep when each user's *local time*
        enters their configured nightly window. Falls back to the legacy
        idle-timer model for backwards compat with deployments that set
        `SCM_IDLE_THRESHOLD_SEC` and never wrote a per-user config.

        Returns count of users whose sleep cycle was fired this tick.
        """
        from ..core.sqlite_db import get_memory
        from ..lifecycle.circadian import should_fire

        sqlite = get_memory()
        fired = 0
        now = time.time()

        for user_id in list(self._last_activity.keys()):
            # Already-cached + unconsumed wake summary? Don't re-fire.
            if user_id in self._cached_summaries:
                continue
            cfg = sqlite.get_user_sleep_config(user_id)

            if cfg.get("is_default") and self._legacy_idle_mode:
                # Legacy fallback: deployments using SCM_IDLE_THRESHOLD_SEC
                # without ever calling /v1/users/{id}/sleep-config keep
                # their old behavior. Once they POST a config, they
                # transition to the circadian model automatically.
                idle_for = now - self._last_activity[user_id]
                if idle_for < self._idle_threshold:
                    continue
                self._fire_sleep_for(user_id, idle_for)
                fired += 1
                continue

            if should_fire(cfg):
                self._fire_sleep_for(user_id, idle_for=0.0, scheduled=True)
                # Persist last_sleep_at so should_fire() returns False until
                # the next night (its once-per-night guard).
                sqlite.mark_user_slept(user_id)
                fired += 1
        return fired

    def _fire_sleep_for(
        self, user_id: str, idle_for: float, scheduled: bool = False,
    ) -> None:
        with self._sleep_lock:
            engine = self._engines.get(user_id)
            if engine is None:
                return
            reason = "scheduled (sleep window)" if scheduled else f"idle for {idle_for:.0f}s"
            logger.info(
                f"[scm.mcp] firing autonomous deep-sleep for user {user_id!r} ({reason})"
            )
            try:
                engine.force_sleep("deep")
            except Exception as e:
                logger.warning(f"deep-sleep failed for {user_id!r}: {e}")
                return

            # Build and cache the wake summary as of now.
            try:
                from datetime import timedelta
                # Scheduled cycles look back 24h (the night just ended);
                # legacy idle cycles look back idle_for + 60s.
                lookback_sec = 86400 if scheduled else (idle_for + 60)
                since = datetime.now(timezone.utc) - timedelta(seconds=lookback_sec)
                summary = engine._wake_summary_builder.build(since=since)
                self._cached_summaries[user_id] = {
                    "narrative": getattr(summary, "narrative", "") or "",
                    "insights": [str(i) for i in (getattr(summary, "insights", []) or [])[:8]],
                    "fired_at": datetime.now(timezone.utc).isoformat(),
                    "idle_seconds": idle_for,
                    "scheduled": scheduled,
                }
                logger.info(
                    f"[scm.mcp] wake-summary cached for {user_id!r}: "
                    f"{len(self._cached_summaries[user_id]['insights'])} insights"
                )
            except Exception as e:
                logger.warning(f"wake-summary build failed for {user_id!r}: {e}")


# ─── MCP server ───────────────────────────────────────────────────────────


def build_mcp_server(pool: UserEnginePool):
    """Construct the MCP Server instance with the five SCM tools registered."""
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    from .tools import TOOLS, get_tool

    server = Server("scm-memory")

    @server.list_tools()
    async def handle_list_tools() -> list:
        return [
            Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.input_schema,
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> list:
        tool = get_tool(name)
        if tool is None:
            return [TextContent(type="text", text=json.dumps({
                "ok": False,
                "error": f"unknown tool: {name}",
            }))]

        user_id = arguments.get("user_id") or "default"
        engine = pool.get_or_create(user_id)

        # Run the (potentially blocking) handler in a thread so we don't
        # stall the MCP event loop.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: tool.handler(arguments, engine)
        )

        # Auto-surface a cached wake-summary on the next activity after idle.
        if name in ("add_memory", "search_memory"):
            cached = pool.cached_summary(user_id)
            if cached:
                result["wake_summary_pending"] = cached
                pool.clear_cached_summary(user_id)

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


# ─── Entry points ─────────────────────────────────────────────────────────


def _read_idle_threshold() -> float:
    raw = os.environ.get("SCM_IDLE_THRESHOLD_SEC", "300")
    try:
        return float(raw)
    except ValueError:
        return 300.0


def _read_sweep_interval() -> float:
    raw = os.environ.get("SCM_MCP_SWEEP_INTERVAL_SEC", "30")
    try:
        return float(raw)
    except ValueError:
        return 30.0


async def run_stdio() -> None:
    """Run the MCP server over stdio (default; works with Claude Desktop)."""
    from mcp.server.stdio import stdio_server

    auto_sleep = os.environ.get("SCM_AUTO_SLEEP_DISABLE", "0") != "1"
    pool = UserEnginePool(
        idle_threshold_sec=_read_idle_threshold(),
        sweep_interval_sec=_read_sweep_interval(),
        auto_sleep=auto_sleep,
    )
    pool.start()
    server = build_mcp_server(pool)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        pool.stop()


async def run_http(host: str = "127.0.0.1", port: int = 7831) -> None:
    """Run the MCP server over HTTP (for clients that don't speak stdio)."""
    try:
        from mcp.server.streamable_http import streamable_http_server
    except Exception as e:
        raise RuntimeError(
            "HTTP MCP transport not available in this mcp version: "
            f"{e!s}. Use stdio transport instead."
        )

    auto_sleep = os.environ.get("SCM_AUTO_SLEEP_DISABLE", "0") != "1"
    pool = UserEnginePool(
        idle_threshold_sec=_read_idle_threshold(),
        sweep_interval_sec=_read_sweep_interval(),
        auto_sleep=auto_sleep,
    )
    pool.start()
    server = build_mcp_server(pool)
    try:
        async with streamable_http_server(server, host=host, port=port):
            print(f"SCM MCP server listening on http://{host}:{port}", file=sys.stderr)
            await asyncio.Event().wait()
    finally:
        pool.stop()


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("SCM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    transport = os.environ.get("SCM_MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        host = os.environ.get("SCM_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("SCM_MCP_PORT", "7831"))
        asyncio.run(run_http(host=host, port=port))
    else:
        asyncio.run(run_stdio())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
