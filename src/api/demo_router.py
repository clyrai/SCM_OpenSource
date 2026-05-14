"""Public demo: scm.run/demo

A fresh, standalone demo router. Does not reuse the legacy /chat/* routes
or the older index.html UI. The user gets a random session URL, chats with
the agent, can configure their own sleep window + timezone, and on a
return visit the wake-summary banner surfaces what SCM consolidated
during the configured sleep hours.

Wiring:
    GET  /demo                        → 302 to /demo/s/<random-slug>
    GET  /demo/s/{slug}               → serves static/demo.html
    POST /demo/api/chat/{slug}        → ingest + retrieve + LLM reply
    POST /demo/api/sleep/{slug}       → manual sleep override (rarely needed)
    POST /demo/api/skip-to-night/{slug} → demo aid: simulate clock advancing
                                          into the user's sleep window
    GET  /demo/api/wake/{slug}        → fetch the latest wake-summary
    GET  /demo/api/config/{slug}      → read sleep_window/timezone settings
    POST /demo/api/config/{slug}      → update sleep_window/timezone settings
    GET  /demo/api/history/{slug}     → restore conversation on page reload

Sleep model (the production-correct one — no polling timer):

    Each session has a `sleep_window` ("23:00-07:00") + `timezone`
    ("Europe/Lisbon"). A scheduler thread checks every 60 seconds whether
    the user's local time has just crossed into the window. If so AND
    there is unconsolidated content AND we haven't already slept this
    cycle, it fires one deep-sleep cycle. Like human circadian rhythm:
    once per night, at the time of day the user picked.

    For the live demo (which can't wait until midnight), POST
    /demo/api/skip-to-night fast-forwards: same code path, just sets the
    "current local time" reference into the window so the next scheduler
    tick fires immediately.

LLM: DeepSeek if `DEEPSEEK_API_KEY` is set, else a degraded fallback that
echoes retrieved memories without composing a sentence.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import string
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from ..chat.engine import ChatEngine
from ..integrations.tools import (
    _add_memory_handler,
    _consolidate_handler,
    _search_memory_handler,
    _wake_summary_handler,
)
from ..lifecycle.wake_summary import WakeSummaryBuilder

router = APIRouter(prefix="/demo", tags=["demo"])

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# ── Timezone helpers ──────────────────────────────────────────────────────


def _resolve_tz(name: str) -> tzinfo:
    """Return a tzinfo for a name like 'Europe/Lisbon'. Falls back to UTC.

    Uses zoneinfo from the stdlib (Python 3.9+); no extra dependencies.
    """
    if not name:
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _parse_hhmm(s: str) -> Optional[int]:
    """Parse 'HH:MM' into minute-of-day (0..1439). None on bad input."""
    if not s:
        return None
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h = int(parts[0]); m = int(parts[1])
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h * 60 + m


def _is_in_window(now_local: datetime, start_min: int, end_min: int) -> bool:
    """True if the local minute-of-day is within [start, end), wrapping midnight.

    Examples:
      start=23:00, end=07:00  → 22:30 NO, 23:30 YES, 02:00 YES, 07:30 NO
      start=00:00, end=06:00  → 05:59 YES, 06:00 NO
    """
    cur = now_local.hour * 60 + now_local.minute
    if start_min == end_min:
        return False  # zero-length window = disabled
    if start_min < end_min:
        return start_min <= cur < end_min
    # wraps midnight
    return cur >= start_min or cur < end_min


# ── Session state ─────────────────────────────────────────────────────────


@dataclass
class _SleepConfig:
    """Per-session sleep schedule. Defaults assume the user wants the
    standard human bedtime; they can override via the settings UI."""
    enabled: bool = True
    sleep_start: str = "23:00"      # local time
    sleep_end: str = "07:00"        # local time
    timezone_name: str = "UTC"      # IANA tz, e.g. "Europe/Lisbon"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sleep_start": self.sleep_start,
            "sleep_end": self.sleep_end,
            "timezone": self.timezone_name,
        }


class _DemoSession:
    """Holds one user's engine + chat history + sleep config + wake cache."""

    def __init__(self, slug: str):
        self.slug = slug
        # Each demo session is ephemeral and isolated:
        # • sandbox_mode=True → in-memory only, no SQLite writes
        # • enable_auto_sleep=False → sleep is triggered by the scheduler
        #   thread below using the user's configured window
        self.engine = ChatEngine(
            session_id=f"demo_{slug}",
            profile="chatbot",
            enable_auto_sleep=False,
            sandbox_mode=True,
        )
        self.transcript: List[Dict[str, str]] = []  # [{user, bot, ts}]
        self.config = _SleepConfig()
        self.wake_cache: Optional[Dict[str, Any]] = None
        self.wake_consumed_at: Optional[float] = None
        # Track when the most recent sleep cycle fired (to fire at most once
        # per local-day-window). UTC seconds; None = never slept yet.
        self.last_sleep_at: Optional[float] = None
        # Demo aid: when the user clicks "skip to night", we record an offset
        # that the scheduler adds to wall-clock when checking the window.
        # Lets visitors experience the moment without waiting until 23:00.
        self.demo_clock_offset_sec: int = 0
        self._lock = threading.Lock()

    def now_local(self) -> datetime:
        """Current local time in this session's timezone, with optional
        demo-skip offset applied. The scheduler uses this to decide whether
        we're in the configured sleep window."""
        tz = _resolve_tz(self.config.timezone_name)
        return (
            datetime.now(timezone.utc) + timedelta(seconds=self.demo_clock_offset_sec)
        ).astimezone(tz)


# ── Pool + scheduler ──────────────────────────────────────────────────────


class _DemoPool:
    """Process-wide registry of demo sessions, keyed by URL slug.

    Runs a scheduler thread (NOT a polling sweeper) that checks every
    SCM_DEMO_SCHED_INTERVAL_SEC seconds whether each session's local time
    is inside its configured sleep window. Fires one consolidation cycle
    per crossing — the human-circadian-rhythm model.
    """

    def __init__(self):
        self._sessions: Dict[str, _DemoSession] = {}
        self._lock = threading.Lock()
        self._sched: Optional[threading.Thread] = None
        self._sched_stop = threading.Event()
        # 60s by default — checking once a minute is plenty for a
        # window-aligned schedule.
        self._sched_interval = float(os.environ.get("SCM_DEMO_SCHED_INTERVAL_SEC", "60"))
        # Min turns before sleep can fire — 1-turn sessions produce empty
        # narratives that hurt the first-impression UX.
        self._min_turns = int(os.environ.get("SCM_DEMO_MIN_TURNS", "3"))

    def get_or_create(self, slug: str) -> _DemoSession:
        with self._lock:
            sess = self._sessions.get(slug)
            if sess is None:
                sess = _DemoSession(slug)
                self._sessions[slug] = sess
            return sess

    def get(self, slug: str) -> Optional[_DemoSession]:
        with self._lock:
            return self._sessions.get(slug)

    def start(self) -> None:
        if self._sched is not None:
            return
        self._sched = threading.Thread(
            target=self._sched_loop, name="demo-scheduler", daemon=True,
        )
        self._sched.start()

    def _sched_loop(self) -> None:
        """Run the schedule check on every interval tick."""
        while not self._sched_stop.wait(self._sched_interval):
            with self._lock:
                items = list(self._sessions.values())
            for sess in items:
                try:
                    self._maybe_fire(sess)
                except Exception:
                    pass

    def _maybe_fire(self, sess: _DemoSession) -> None:
        """Decide whether this session should sleep right now."""
        cfg = sess.config
        if not cfg.enabled:
            return
        # Need real material before sleep is meaningful.
        if len(sess.transcript) < self._min_turns:
            return
        # Time-of-day check, in the user's tz.
        start_min = _parse_hhmm(cfg.sleep_start)
        end_min = _parse_hhmm(cfg.sleep_end)
        if start_min is None or end_min is None:
            return
        local_now = sess.now_local()
        if not _is_in_window(local_now, start_min, end_min):
            return
        # Fire at most once per window crossing. Track local-date of the
        # window-start to detect "next night."
        # Use the date that the window STARTED on (so a 23:00→07:00 window
        # spanning midnight still counts as "one night").
        window_anchor_local = local_now.date()
        cur_local_min = local_now.hour * 60 + local_now.minute
        if start_min > end_min and cur_local_min < end_min:
            # We're past midnight inside a wrap-around window → the anchor
            # was yesterday's date.
            window_anchor_local = (local_now - timedelta(days=1)).date()
        if sess.last_sleep_at is not None:
            last_local = datetime.fromtimestamp(sess.last_sleep_at, tz=timezone.utc).astimezone(
                _resolve_tz(cfg.timezone_name)
            )
            last_anchor = last_local.date()
            if start_min > end_min and last_local.hour * 60 + last_local.minute < end_min:
                last_anchor = (last_local - timedelta(days=1)).date()
            if last_anchor == window_anchor_local:
                return  # already slept this night

        self._fire_sleep(sess)

    def _fire_sleep(self, sess: _DemoSession) -> None:
        with sess._lock:
            stats = sess.engine.force_sleep("deep") or {}
            try:
                builder = WakeSummaryBuilder(engine=sess.engine)
                since = datetime.now(timezone.utc) - timedelta(hours=24)
                summary = builder.build(since=since)
                narrative = (summary.narrative or "").strip() if summary else ""
                if narrative and _narrative_has_substance(narrative):
                    sess.wake_cache = {
                        "narrative": narrative,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "stats": stats,
                    }
                    sess.wake_consumed_at = None
                sess.last_sleep_at = time.time()
            except Exception:
                pass


def _narrative_has_substance(narrative: str) -> bool:
    """Decide whether a wake-summary narrative is worth surfacing.

    Heuristic: must contain at least 2 bullet-prefixed lines.
    """
    bullet_lines = [ln for ln in narrative.splitlines() if ln.strip().startswith("•")]
    return len(bullet_lines) >= 2


_pool = _DemoPool()
_pool.start()


# ── LLM (DeepSeek via OpenAI-compat) ──────────────────────────────────────


def _llm_reply(memory_context: str, history: List[Dict[str, str]], user_msg: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return _fallback_reply(memory_context, user_msg)
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        sys_prompt = (
            "You are a helpful assistant with persistent memory provided by SCM. "
            "The 'Relevant memories' block below was retrieved by the memory layer "
            "from prior turns; treat it as ground truth about the user. Use it "
            "naturally to personalize your reply. Don't invent facts not in memory "
            "or the current message. Keep replies to 1-2 short sentences.\n\n"
            f"Relevant memories:\n{memory_context or '(none yet)'}"
        )
        msgs = [{"role": "system", "content": sys_prompt}]
        for t in history[-6:]:
            if t.get("user"): msgs.append({"role": "user", "content": t["user"]})
            if t.get("bot"):  msgs.append({"role": "assistant", "content": t["bot"]})
        msgs.append({"role": "user", "content": user_msg})
        resp = client.chat.completions.create(
            model=os.environ.get("DEMO_LLM_MODEL", "deepseek-chat"),
            messages=msgs, max_tokens=150, temperature=0.4, timeout=20,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_reply(memory_context, user_msg, error=str(e))


def _fallback_reply(memory_context: str, user_msg: str, error: str = "") -> str:
    if memory_context:
        snippet = memory_context.split("\n", 4)
        head = "\n".join(snippet[:4])
        return f"Got it — I've noted that.\n\nWhat I remember about you so far:\n{head}"
    return "Got it — I've stored that. Tell me a few more things and I'll start to build a picture."


# ── Routes ────────────────────────────────────────────────────────────────


def _new_slug() -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(12))


@router.get("")
async def demo_root() -> RedirectResponse:
    """Bounce visitors to a fresh per-session URL."""
    return RedirectResponse(url=f"/demo/s/{_new_slug()}", status_code=302)


@router.get("/s/{slug}")
async def demo_page(slug: str) -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "demo.html"), media_type="text/html")


class _ChatRequest(BaseModel):
    message: str


@router.post("/api/chat/{slug}")
async def demo_chat(slug: str, body: _ChatRequest) -> JSONResponse:
    """One conversational turn — ingest, retrieve, generate."""
    user_msg = (body.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="empty message")

    sess = _pool.get_or_create(slug)

    # 1. Surface (and consume) any pending wake-summary FIRST.
    wake_payload = None
    with sess._lock:
        if sess.wake_cache and sess.wake_consumed_at is None:
            wake_payload = sess.wake_cache
            sess.wake_consumed_at = time.time()

    # 2. Retrieve memories relevant to what the user just said.
    try:
        retrieved = await asyncio.to_thread(
            _search_memory_handler,
            {"query": user_msg, "user_id": slug, "limit": 5},
            sess.engine,
        )
        memory_context = retrieved.get("memory_context", "") or ""
    except Exception:
        memory_context = ""

    # 3. Compose reply.
    reply = await asyncio.to_thread(
        _llm_reply, memory_context, sess.transcript, user_msg,
    )

    # 4. Store the user message.
    try:
        await asyncio.to_thread(
            _add_memory_handler,
            {"text": user_msg, "user_id": slug, "sync": True},
            sess.engine,
        )
    except Exception:
        pass

    # 5. Append to transcript.
    sess.transcript.append({
        "user": user_msg, "bot": reply,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    return JSONResponse({
        "reply": reply,
        "wake_summary": wake_payload,
    })


@router.post("/api/sleep/{slug}")
async def demo_sleep(slug: str) -> JSONResponse:
    """Manual sleep override — rarely needed in production. Kept for the
    'force a cycle now' use case (devs, testing, transparency)."""
    sess = _pool.get_or_create(slug)
    stats = await asyncio.to_thread(
        _consolidate_handler,
        {"user_id": slug, "mode": "deep"},
        sess.engine,
    )
    summary = await asyncio.to_thread(
        _wake_summary_handler,
        {"user_id": slug, "since_hours": 24.0},
        sess.engine,
    )
    narrative = summary.get("narrative") or ""
    payload = {"narrative": narrative, "stats": stats}
    if narrative and _narrative_has_substance(narrative):
        with sess._lock:
            sess.wake_cache = payload
            sess.wake_consumed_at = None
            sess.last_sleep_at = time.time()
    return JSONResponse({"ok": True, "stats": stats, "wake_summary": payload})


@router.post("/api/skip-to-night/{slug}")
async def demo_skip_to_night(slug: str) -> JSONResponse:
    """Demo-only affordance: shift this session's perceived clock to the
    start of its sleep window so the scheduler fires on the next tick.

    Same code path as the real scheduler. Visitors don't need to wait
    until midnight to experience the wake-summary moment.
    """
    sess = _pool.get_or_create(slug)
    cfg = sess.config
    start_min = _parse_hhmm(cfg.sleep_start)
    if start_min is None:
        return JSONResponse({"ok": False, "error": "invalid sleep_start"}, status_code=400)
    tz = _resolve_tz(cfg.timezone_name)

    # Compute the next moment that is sleep_start in the user's tz, then
    # set demo_clock_offset so `now_local()` returns that.
    now_local = datetime.now(timezone.utc).astimezone(tz)
    target_local = now_local.replace(
        hour=start_min // 60, minute=start_min % 60, second=30, microsecond=0,
    )
    if target_local <= now_local:
        target_local += timedelta(days=1)
    delta = (target_local - now_local).total_seconds()
    sess.demo_clock_offset_sec = int(delta) + sess.demo_clock_offset_sec
    # Reset last_sleep so the scheduler is allowed to fire again.
    sess.last_sleep_at = None
    # Force a scheduler check immediately rather than waiting up to 60s.
    _pool._maybe_fire(sess)
    # Reflect the new "now" the user sees.
    return JSONResponse({
        "ok": True,
        "skipped_seconds": int(delta),
        "now_local": sess.now_local().isoformat(),
        "wake_summary": sess.wake_cache,
    })


@router.get("/api/wake/{slug}")
async def demo_wake(slug: str) -> JSONResponse:
    sess = _pool.get(slug)
    if sess is None:
        return JSONResponse({"narrative": "", "note": "no session yet"})
    with sess._lock:
        cached = sess.wake_cache
    if cached and cached.get("narrative"):
        return JSONResponse(cached)
    summary = await asyncio.to_thread(
        _wake_summary_handler,
        {"user_id": slug, "since_hours": 24.0},
        sess.engine,
    )
    return JSONResponse({"narrative": summary.get("narrative") or ""})


class _ConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    sleep_start: Optional[str] = Field(None, description="HH:MM in local tz")
    sleep_end: Optional[str] = Field(None, description="HH:MM in local tz")
    timezone: Optional[str] = Field(None, description="IANA tz, e.g. Europe/Lisbon")


@router.get("/api/config/{slug}")
async def demo_get_config(slug: str) -> JSONResponse:
    sess = _pool.get_or_create(slug)
    return JSONResponse({
        **sess.config.to_dict(),
        "now_local": sess.now_local().isoformat(),
        "in_sleep_window": _is_in_window(
            sess.now_local(),
            _parse_hhmm(sess.config.sleep_start) or 0,
            _parse_hhmm(sess.config.sleep_end) or 0,
        ),
    })


@router.post("/api/config/{slug}")
async def demo_set_config(slug: str, body: _ConfigUpdate) -> JSONResponse:
    sess = _pool.get_or_create(slug)
    cfg = sess.config
    if body.enabled is not None:
        cfg.enabled = bool(body.enabled)
    if body.sleep_start is not None:
        if _parse_hhmm(body.sleep_start) is None:
            raise HTTPException(status_code=400, detail="sleep_start must be HH:MM")
        cfg.sleep_start = body.sleep_start
    if body.sleep_end is not None:
        if _parse_hhmm(body.sleep_end) is None:
            raise HTTPException(status_code=400, detail="sleep_end must be HH:MM")
        cfg.sleep_end = body.sleep_end
    if body.timezone is not None:
        # Try to resolve; if the lib can't, reject so the UI stays honest.
        try:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
            ZoneInfo(body.timezone)
        except Exception:
            raise HTTPException(status_code=400, detail=f"unknown timezone: {body.timezone!r}")
        cfg.timezone_name = body.timezone
    return JSONResponse({**cfg.to_dict(), "now_local": sess.now_local().isoformat()})


@router.get("/api/history/{slug}")
async def demo_history(slug: str) -> JSONResponse:
    sess = _pool.get(slug)
    if sess is None:
        return JSONResponse({"turns": [], "wake_summary": None, "config": _SleepConfig().to_dict()})
    with sess._lock:
        wake_payload = None
        if sess.wake_cache and sess.wake_consumed_at is None:
            wake_payload = sess.wake_cache
            sess.wake_consumed_at = time.time()
        return JSONResponse({
            "turns": list(sess.transcript),
            "wake_summary": wake_payload,
            "config": sess.config.to_dict(),
        })
