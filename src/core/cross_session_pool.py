"""
CrossSessionMemoryPool — Phase 7 cross-session memory window.

Default sleep cycles see only the *current* session's working memory. That
limits consolidation to a single conversation, which is wrong: humans
consolidate across days, not just within one session. This module gives the
sleep orchestrator a rolling window of prior-session episodes so that
NREM Hebbian co-activation, REM pattern detection, and adaptive forgetting
all operate on the agent's actual recent experience, not just today's chat.

Design:
  - The pool is an injectable component; ChatEngine constructs one per
    session if cross-session pooling is enabled.
  - It reads from the existing SQLite episodes table (now indexed by
    session_id thanks to the Phase 7 migration).
  - It applies fairness caps so one chatty session can't dominate the pool.
  - It tags every borrowed episode with `_origin_session` so post-sleep
    code can tell native vs. borrowed context apart.
  - Borrowed episodes feed into the same `episodes=` argument passed to
    `SleepCycleOrchestrator.begin_sleep_cycle`. No other code paths change.

Privacy contract:
  - Only the current agent's *own* prior sessions are pooled. No mechanism
    here reaches into another user's data.
  - The pool can be disabled per-engine. Default is OFF for backward compat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from typing import Any, Callable, Dict, List, Optional

from .models import Episode, ImportanceVector, MemoryState
from .time_utils import ensure_utc, utc_isoformat, utc_now


# ─── Configuration ──────────────────────────────────────────────────────────


@dataclass
class CrossSessionPoolConfig:
    """How aggressively the pool reaches across sessions."""

    # Master switch. False = legacy behavior (current session only).
    enabled: bool = False

    # Number of recent sessions (besides the current one) to pull from.
    max_sessions: int = 5

    # Look-back time window in hours. Episodes older than this are excluded.
    lookback_hours: float = 168.0  # 7 days

    # Per-session cap so a chatty session can't dominate the pool.
    max_episodes_per_session: int = 30

    # Total cap across all borrowed episodes.
    max_total_borrowed: int = 100

    # If True, restrict pooled sessions to those with the same `profile`
    # as the current engine (chatbot/agent/research). Off by default —
    # the agent should learn across modes, not just within one.
    restrict_to_same_profile: bool = False

    # Phase 7 brutal-fix: when True, the pool ALSO pulls older episodes
    # from the CURRENT session_id (not just from sibling sessions). This
    # is the right default for single-user multi-day deployments, where
    # there's only one session_id but the user has been chatting for days
    # and earlier episodes have been persisted but cleared from working
    # memory after deep sleep. Default False preserves legacy behavior.
    include_current_session: bool = False


# ─── Stats record (for the wake-summary endpoint) ──────────────────────────


@dataclass
class CrossSessionGatherStats:
    """One pool gather, for diagnostics."""
    requested_at: datetime = field(default_factory=utc_now)
    current_session_id: str = ""
    sessions_considered: int = 0
    sessions_borrowed_from: int = 0
    borrowed_session_ids: List[str] = field(default_factory=list)
    episodes_borrowed: int = 0
    skipped_due_to_age: int = 0
    skipped_due_to_caps: int = 0


# ─── The pool ──────────────────────────────────────────────────────────────


SqliteFactory = Callable[[], Any]


class CrossSessionMemoryPool:
    """
    Reaches into the episodes table to give sleep cycles a multi-session
    window. The pool itself is stateless across calls — every gather() does
    a fresh query. This keeps it simple and correct under concurrent edits
    from other engines.

    Usage:
        pool = CrossSessionMemoryPool(
            current_session_id="alice_2026_05",
            sqlite_factory=get_memory,           # from core.sqlite_db
            config=CrossSessionPoolConfig(enabled=True),
        )
        prior_episodes = pool.gather()
        # Hand prior_episodes to SleepCycleOrchestrator.begin_sleep_cycle
    """

    def __init__(
        self,
        current_session_id: str,
        sqlite_factory: SqliteFactory,
        config: Optional[CrossSessionPoolConfig] = None,
        current_profile: Optional[str] = None,
    ):
        self.current_session_id = current_session_id or ""
        self._sqlite_factory = sqlite_factory
        self.config = config or CrossSessionPoolConfig()
        self.current_profile = current_profile
        self.last_stats: Optional[CrossSessionGatherStats] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def gather(
        self,
        now: Optional[datetime] = None,
    ) -> List[Episode]:
        """
        Return Episode objects from prior sessions to feed into a sleep cycle.

        Returns [] if disabled, no SQLite, or no qualifying sessions found.
        """
        stats = CrossSessionGatherStats(
            requested_at=now or utc_now(),
            current_session_id=self.current_session_id,
        )
        self.last_stats = stats

        if not self.config.enabled:
            return []

        try:
            sqlite = self._sqlite_factory()
        except Exception:
            return []
        if sqlite is None:
            return []

        # 1. Identify candidate prior sessions.
        cutoff = (now or utc_now()) - timedelta(hours=self.config.lookback_hours)
        cutoff_iso = cutoff.isoformat()
        all_recent_ids = self._safe_list_sessions(sqlite, cutoff_iso)
        if self.config.include_current_session:
            # Single-user multi-day case: include this session too so older
            # persisted episodes (cleared from WM by prior deep-sleeps) can
            # be replayed for cross-day pattern detection.
            candidate_ids = list(all_recent_ids)
            if self.current_session_id and self.current_session_id not in candidate_ids:
                candidate_ids.insert(0, self.current_session_id)
        else:
            candidate_ids = [s for s in all_recent_ids if s != self.current_session_id]
        stats.sessions_considered = len(candidate_ids)

        # Optional profile restriction
        if self.config.restrict_to_same_profile and self.current_profile:
            candidate_ids = self._filter_to_profile(
                sqlite, candidate_ids, self.current_profile
            )

        # Trim to max_sessions (already ordered newest-first by the SQL)
        candidate_ids = candidate_ids[: self.config.max_sessions]

        if not candidate_ids:
            return []

        # 2. Pull recent episodes for those sessions, with fairness caps.
        rows = self._safe_query_episodes(
            sqlite,
            session_ids=candidate_ids,
            since_iso=cutoff_iso,
        )
        # 3. Convert rows to Episode objects.
        episodes: List[Episode] = []
        seen_session_ids = set()
        for row in rows:
            ep = self._row_to_episode(row)
            if ep is None:
                stats.skipped_due_to_caps += 1
                continue
            episodes.append(ep)
            sid = row.get("session_id")
            if sid:
                seen_session_ids.add(sid)
            if len(episodes) >= self.config.max_total_borrowed:
                break

        stats.episodes_borrowed = len(episodes)
        stats.borrowed_session_ids = sorted(seen_session_ids)
        stats.sessions_borrowed_from = len(seen_session_ids)
        return episodes

    # ── Internals ──────────────────────────────────────────────────────────

    def _safe_list_sessions(self, sqlite, since_iso: str) -> List[str]:
        try:
            return sqlite.list_recent_session_ids(
                limit=max(1, self.config.max_sessions * 3),  # over-fetch
                since_iso=since_iso,
            )
        except AttributeError:
            return []
        except Exception:
            return []

    def _safe_query_episodes(
        self,
        sqlite,
        session_ids: List[str],
        since_iso: str,
    ) -> List[Dict[str, Any]]:
        try:
            return sqlite.get_recent_episodes_for_sessions(
                session_ids=session_ids,
                since_iso=since_iso,
                max_per_session=self.config.max_episodes_per_session,
                max_total=self.config.max_total_borrowed,
            )
        except AttributeError:
            return []
        except Exception:
            return []

    def _filter_to_profile(self, sqlite, session_ids: List[str], profile: str) -> List[str]:
        # Best-effort profile match using session_meta. If the meta table
        # doesn't track profile (current schema doesn't), no-op.
        try:
            kept = []
            for sid in session_ids:
                meta = sqlite.load_session_meta(sid)
                # No profile column today — accept all.
                kept.append(sid)
            return kept
        except Exception:
            return session_ids

    @staticmethod
    def _row_to_episode(row: Dict[str, Any]) -> Optional[Episode]:
        try:
            timestamp = ensure_utc(row.get("timestamp")) or utc_now()
            context_raw = row.get("context") or "{}"
            context = json.loads(context_raw) if isinstance(context_raw, str) else (context_raw or {})
            importance_raw = row.get("importance_json") or "{}"
            importance_dict = (
                json.loads(importance_raw) if isinstance(importance_raw, str) else (importance_raw or {})
            )
            concept_ids_raw = row.get("concept_ids") or "[]"
            concept_ids = (
                json.loads(concept_ids_raw)
                if isinstance(concept_ids_raw, str)
                else (concept_ids_raw or [])
            )
            ep = Episode(
                id=row.get("id"),
                timestamp=timestamp,
                concept_ids=concept_ids,
                raw_content=row.get("raw_content") or "",
                context={**context, "_origin_session": row.get("session_id"), "_borrowed": True},
                importance=ImportanceVector(**importance_dict) if importance_dict else ImportanceVector(),
                state=MemoryState.ACTIVE,
                source=row.get("source") or "user",
            )
            return ep
        except Exception:
            return None

    # ── Diagnostics ────────────────────────────────────────────────────────

    def stats_dict(self) -> Dict[str, Any]:
        if self.last_stats is None:
            return {"never_called": True}
        s = self.last_stats
        return {
            "requested_at": utc_isoformat(s.requested_at),
            "current_session_id": s.current_session_id,
            "sessions_considered": s.sessions_considered,
            "sessions_borrowed_from": s.sessions_borrowed_from,
            "borrowed_session_ids": list(s.borrowed_session_ids),
            "episodes_borrowed": s.episodes_borrowed,
            "skipped_due_to_age": s.skipped_due_to_age,
            "skipped_due_to_caps": s.skipped_due_to_caps,
        }
