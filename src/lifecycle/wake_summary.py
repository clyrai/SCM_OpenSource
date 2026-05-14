"""
Wake-Up Summary — Phase 7's user-visible payoff.

When the user comes back to their agent after time away, they should see a
clear, human-readable report of what the agent did during that downtime.
This module assembles that report from the artifacts produced by M1
(IdleLearner sleep cycles), M2 (cross-session memory pool stats), and
M3 (schema extraction).

The summary is BOTH structured (for programmatic use / tests) AND
narrative (for direct display to the user). The narrative is generated
from heuristic templates — no LLM required, fully deterministic, free
to render any number of times.

Example narrative:

    While you were away (8.4 hours), I ran 2 deep-sleep cycles.
    I consolidated 47 memories and forgot 12 low-value items.
    I noticed 3 patterns:
      • Caroline appears in 5 sessions — likely a recurring topic.
      • Caroline and Melanie tend to come up together.
      • Your support-group routine recurs on a weekly cadence.
    I'm ready when you are.

Public API:

    builder = WakeSummaryBuilder(engine, idle_learner=...)
    summary = builder.build(since=optional_datetime)
    summary.narrative          # str — display this to the user
    summary.to_dict()          # dict — JSON-friendly for the REST endpoint
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core import linguistic_resources as lr
from ..core.models import Concept, MemoryState
from ..core.time_utils import ensure_utc, utc_isoformat, utc_now


# ─── Output schema ──────────────────────────────────────────────────────────


@dataclass
class WakeInsight:
    """One thing the agent noticed while idle."""
    insight_type: str       # "recurring_topic" | "cooccurrence" | "trajectory" | "temporal_cadence" | "general"
    text: str               # human-readable
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.5
    occurrence_count: int = 0
    source_sessions: List[str] = field(default_factory=list)


@dataclass
class WakeSummary:
    """Complete wake-up report — both structured and narrative."""
    session_id: str
    generated_at: datetime
    last_interaction_at: Optional[datetime]
    idle_duration_seconds: Optional[float]

    sleep_cycles_run: int = 0
    autonomous_cycles_run: int = 0      # subset: fired by IdleLearner, not user
    memories_consolidated: int = 0
    memories_forgotten: int = 0
    dreams_generated: int = 0
    sessions_consulted: List[str] = field(default_factory=list)
    insights: List[WakeInsight] = field(default_factory=list)

    narrative: str = ""

    # Optional diagnostic payload (full sleep records, schema metadata)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "generated_at": utc_isoformat(self.generated_at),
            "last_interaction_at": (
                utc_isoformat(self.last_interaction_at)
                if self.last_interaction_at else None
            ),
            "idle_duration_seconds": self.idle_duration_seconds,
            "idle_duration_hours": (
                round(self.idle_duration_seconds / 3600.0, 2)
                if self.idle_duration_seconds is not None else None
            ),
            "sleep_cycles_run": self.sleep_cycles_run,
            "autonomous_cycles_run": self.autonomous_cycles_run,
            "memories_consolidated": self.memories_consolidated,
            "memories_forgotten": self.memories_forgotten,
            "dreams_generated": self.dreams_generated,
            "sessions_consulted": list(self.sessions_consulted),
            "insights": [
                {
                    "type": i.insight_type,
                    "text": i.text,
                    "entities": list(i.entities),
                    "confidence": i.confidence,
                    "occurrence_count": i.occurrence_count,
                    "source_sessions": list(i.source_sessions),
                }
                for i in self.insights
            ],
            "narrative": self.narrative,
            "diagnostics": self.diagnostics,
        }


# ─── The builder ────────────────────────────────────────────────────────────


class WakeSummaryBuilder:
    """
    Assembles a `WakeSummary` for one chat engine.

    The builder reaches into:
      - engine._sleep_history           (forced + auto-triggered cycles)
      - engine.long_term_memory         (schema concepts + counts)
      - engine.cross_session_pool       (last gather stats, if available)
      - idle_learner (optional)         (autonomous-only cycles)

    All are read-only. The builder never mutates anything.
    """

    def __init__(
        self,
        engine: Any,
        idle_learner: Any = None,
        now_fn: Optional[callable] = None,
    ):
        self.engine = engine
        self.idle_learner = idle_learner
        self._now_fn = now_fn or utc_now

    # ── Public API ──────────────────────────────────────────────────────────

    def build(
        self,
        since: Optional[datetime] = None,
        max_insights: int = 6,
        include_diagnostics: bool = False,
    ) -> WakeSummary:
        now = self._now_fn()
        session_id = getattr(self.engine, "session_id", "default")

        last_interaction = self._last_interaction_time(session_id)
        idle_seconds = (
            (now - last_interaction).total_seconds()
            if last_interaction else None
        )

        # Default scope: cycles that ran *after* the user's last interaction.
        # If `since` provided, override.
        cutoff = since or last_interaction

        sleep_records = self._collect_sleep_records(cutoff)
        autonomous_records = self._collect_autonomous_records(session_id, cutoff)

        # Aggregate
        consolidated = sum(r.get("consolidated", 0) for r in sleep_records)
        forgotten = sum(r.get("forgotten", 0) for r in sleep_records)
        dreams = sum(r.get("dreams", 0) for r in sleep_records)
        sessions_consulted = self._collect_sessions_consulted()

        # Schema-based insights
        insights = self._collect_insights(cutoff, max_insights)

        summary = WakeSummary(
            session_id=session_id,
            generated_at=now,
            last_interaction_at=last_interaction,
            idle_duration_seconds=idle_seconds,
            sleep_cycles_run=len(sleep_records),
            autonomous_cycles_run=len(autonomous_records),
            memories_consolidated=consolidated,
            memories_forgotten=forgotten,
            dreams_generated=dreams,
            sessions_consulted=sessions_consulted,
            insights=insights,
        )
        summary.narrative = self._render_narrative(summary)

        if include_diagnostics:
            summary.diagnostics = {
                "sleep_records": sleep_records,
                "autonomous_records_count": len(autonomous_records),
                "cross_session_pool_last": self._cross_session_stats(),
            }

        return summary

    # ── Data gatherers ─────────────────────────────────────────────────────

    def _last_interaction_time(self, session_id: str) -> Optional[datetime]:
        """Most recent activity record. Tries IdleLearner first, then engine WM."""
        if self.idle_learner is not None:
            try:
                stats = self.idle_learner.get_stats()
                ts_iso = stats.get("last_activity", {}).get(session_id)
                if ts_iso:
                    return ensure_utc(ts_iso)
            except Exception:
                pass
        # Fall back to most recent episode timestamp in WM
        try:
            wm_episodes = self.engine.working_memory.get_all()
            if wm_episodes:
                return ensure_utc(wm_episodes[-1].timestamp)
        except Exception:
            pass
        return None

    def _collect_sleep_records(self, cutoff: Optional[datetime]) -> List[Dict[str, Any]]:
        """Return engine sleep_history entries newer than cutoff."""
        history = list(getattr(self.engine, "_sleep_history", []) or [])
        if not cutoff:
            return history
        cutoff_utc = ensure_utc(cutoff)
        kept = []
        for r in history:
            ts = ensure_utc(r.get("timestamp"))
            if ts is None or cutoff_utc is None or ts >= cutoff_utc:
                kept.append(r)
        return kept

    def _collect_autonomous_records(
        self,
        session_id: str,
        cutoff: Optional[datetime],
    ) -> List[Any]:
        """Subset of records from the IdleLearner daemon (autonomous only)."""
        if self.idle_learner is None:
            return []
        try:
            return self.idle_learner.get_history(
                session_id=session_id,
                since=cutoff,
                limit=200,
            )
        except Exception:
            return []

    def _collect_sessions_consulted(self) -> List[str]:
        """Sessions the cross-session pool borrowed from on its last call."""
        pool = getattr(self.engine, "cross_session_pool", None)
        if pool is None:
            return []
        try:
            return list(pool.last_stats.borrowed_session_ids) if pool.last_stats else []
        except Exception:
            return []

    def _cross_session_stats(self) -> Dict[str, Any]:
        pool = getattr(self.engine, "cross_session_pool", None)
        if pool is None:
            return {}
        try:
            return pool.stats_dict()
        except Exception:
            return {}

    def _collect_insights(
        self,
        cutoff: Optional[datetime],
        max_insights: int,
    ) -> List[WakeInsight]:
        """Pull schema + curiosity concepts from LTM, sort, and trim."""
        try:
            concepts = self.engine.long_term_memory.get_all_concepts(include_suppressed=False)
        except Exception:
            return []

        cutoff_utc = ensure_utc(cutoff)

        candidates: List[WakeInsight] = []
        for c in concepts:
            tags = c.context_tags if isinstance(c.context_tags, dict) else {}
            is_schema = bool(tags.get("_schema"))
            is_curiosity = bool(tags.get("_curiosity"))
            if not (is_schema or is_curiosity):
                continue
            # Only show concepts created/updated since the cutoff
            if cutoff_utc is not None:
                created = ensure_utc(getattr(c, "created_at", None))
                if created is None or created < cutoff_utc:
                    continue
            if is_schema:
                insight = WakeInsight(
                    insight_type=str(tags.get("schema_type", "general")),
                    text=c.description,
                    entities=list(tags.get("entities", []) or []),
                    confidence=float(getattr(c, "confidence", 0.5) or 0.5),
                    occurrence_count=int(tags.get("occurrence_count", 0) or 0),
                    source_sessions=list(tags.get("source_sessions", []) or []),
                )
            else:  # curiosity
                entity = tags.get("curiosity_entity") or "unknown"
                source = tags.get("curiosity_source") or "external"
                tmpl = lr.get_wake_summary_templates().get(
                    "curiosity_insight_template",
                    "I read up on {entity} (from {source}): {brief}",
                )
                insight_text = tmpl.format(
                    entity=entity, source=source, brief=c.description,
                )
                insight = WakeInsight(
                    insight_type="learned",
                    text=insight_text,
                    entities=[entity],
                    confidence=float(getattr(c, "confidence", 0.5) or 0.5),
                    occurrence_count=int(tags.get("occurrence_count", 0) or 0),
                    source_sessions=list(tags.get("source_sessions", []) or []),
                )
            candidates.append(insight)

        # Rank: learned (curiosity) → recurring_topic → cooccurrence → trajectory → cadence
        type_priority = {
            "learned": 0,
            "recurring_topic": 1,
            "cooccurrence": 2,
            "trajectory": 3,
            "temporal_cadence": 4,
            "general": 5,
        }
        candidates.sort(key=lambda i: (
            type_priority.get(i.insight_type, 9),
            -i.confidence,
            -i.occurrence_count,
        ))
        return candidates[:max_insights]

    def _collect_user_context_facts(self, limit: int = 4) -> List[str]:
        """Top user-attributable facts for a 'what I know about you' refresh.

        Returns descriptions of the most salient concepts that came from the
        user (have `session_id` in context_tags) and aren't internal/system
        concepts. Sorted by salience score, then importance, then recency.
        """
        try:
            concepts = self.engine.long_term_memory.get_all_concepts(
                include_suppressed=False,
            )
        except Exception:
            return []

        ranked: List[tuple] = []
        for c in concepts:
            tags = c.context_tags if isinstance(c.context_tags, dict) else {}
            if tags.get("_internal"):
                continue
            if not tags.get("session_id"):
                continue  # not user-attributable
            salience = float(getattr(c, "salience_score", 0.0) or 0.0)
            importance = (
                float(c.importance.overall) if getattr(c, "importance", None) else 0.0
            )
            created = ensure_utc(getattr(c, "created_at", None))
            ranked.append((salience, importance, created or 0, c.description))

        ranked.sort(key=lambda r: (-r[0], -r[1], -(r[2].timestamp() if hasattr(r[2], "timestamp") else 0)))
        seen_descriptions: set = set()
        out: List[str] = []
        for _, _, _, desc in ranked:
            d = (desc or "").strip()
            if not d or d in seen_descriptions:
                continue
            seen_descriptions.add(d)
            out.append(d)
            if len(out) >= limit:
                break
        return out

    # ── Narrative rendering ────────────────────────────────────────────────

    def _render_narrative(self, summary: WakeSummary) -> str:
        """Build a paragraph using templates loaded from linguistic_resources."""
        T = lr.get_wake_summary_templates()
        lines: List[str] = []

        # Greeting + idle duration
        if summary.idle_duration_seconds is None:
            lines.append(T.get("no_idle_known", "Welcome back."))
        else:
            duration_text = self._format_duration(summary.idle_duration_seconds, T)
            if summary.sleep_cycles_run == 0:
                lines.append(
                    T.get(
                        "idle_no_cycles",
                        "Welcome back. You were away for {duration_text}.",
                    ).format(duration_text=duration_text)
                )
            else:
                if summary.sleep_cycles_run == 1:
                    cycles_label = T.get("cycles_singular", "1 sleep cycle")
                else:
                    cycles_label = T.get(
                        "cycles_plural", "{n} sleep cycles"
                    ).format(n=summary.sleep_cycles_run)
                autonomous_note = ""
                if summary.autonomous_cycles_run > 0:
                    autonomous_note = T.get(
                        "autonomous_note",
                        " ({n} ran autonomously while you were away)",
                    ).format(n=summary.autonomous_cycles_run)
                lines.append(
                    T.get(
                        "idle_with_cycles",
                        "Welcome back. While you were away ({duration_text}), I ran {cycles_label}{autonomous_note}.",
                    ).format(
                        duration_text=duration_text,
                        cycles_label=cycles_label,
                        autonomous_note=autonomous_note,
                    )
                )

        # Memory totals
        if summary.memories_consolidated or summary.memories_forgotten or summary.dreams_generated:
            parts = []
            if summary.memories_consolidated:
                parts.append(T.get("totals_consolidated", "consolidated {n} memories").format(n=summary.memories_consolidated))
            if summary.memories_forgotten:
                parts.append(T.get("totals_forgotten", "forgot {n} items").format(n=summary.memories_forgotten))
            if summary.dreams_generated:
                parts.append(T.get("totals_dreams", "generated {n} dreams").format(n=summary.dreams_generated))
            lines.append(T.get("totals_wrap", "I {parts}.").format(parts=", ".join(parts)))

        # Cross-session reach
        if summary.sessions_consulted:
            n = len(summary.sessions_consulted)
            key = "sessions_consulted_singular" if n == 1 else "sessions_consulted_plural"
            tmpl = T.get(key, "Drew on {n} prior session(s).")
            lines.append(tmpl.format(n=n))

        # Insights
        if summary.insights:
            n = len(summary.insights)
            key = "patterns_intro_singular" if n == 1 else "patterns_intro_plural"
            tmpl = T.get(key, "I noticed {n} patterns:")
            lines.append(tmpl.format(n=n))
            for insight in summary.insights:
                lines.append(f"  • {insight.text}")
        elif summary.sleep_cycles_run > 0:
            # No formal schemas yet — fall back to a "what I know about you"
            # context refresh from the most salient user-attributable facts.
            # This makes the wake-summary moment land even on the first sleep
            # cycle, before there's enough data to form a recurring pattern.
            user_facts = self._collect_user_context_facts(limit=4)
            if user_facts:
                lines.append(T.get(
                    "context_refresh_intro",
                    "Here's what I have on you:",
                ))
                for fact in user_facts:
                    lines.append(f"  • {fact}")
            else:
                lines.append(T.get(
                    "no_patterns_yet",
                    "I didn't form any new patterns yet.",
                ))

        lines.append(T.get("closer", "I'm ready when you are."))
        return "\n".join(lines)

    @staticmethod
    def _format_duration(seconds: float, templates: Optional[Dict[str, str]] = None) -> str:
        T = templates or lr.get_wake_summary_templates()
        if seconds < 60:
            return T.get("duration_seconds", "{n} seconds").format(n=int(seconds))
        if seconds < 3600:
            return T.get("duration_minutes", "{n} minutes").format(n=round(seconds / 60.0, 1))
        if seconds < 86400:
            return T.get("duration_hours", "{n} hours").format(n=round(seconds / 3600.0, 1))
        return T.get("duration_days", "{n} days").format(n=round(seconds / 86400.0, 1))
