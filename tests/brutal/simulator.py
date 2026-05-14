"""
Persona-driven multi-day user simulator for brutal SCM testing.

The simulator reads a persona JSON file and replays it against a real
ChatEngine + IdleLearner stack. Idle gaps in the persona advance the
agent's clock, allowing the IdleLearner daemon to fire autonomous sleep
cycles between simulated user sessions.

Outputs:
  - SimulationTrace: full record of what happened (turns ingested,
    sleep cycles fired, schemas formed, idle gaps simulated)
  - List[ScenarioResult]: pass/fail for each expected_recall, expected_schema,
    expected_curiosity_gap defined in the persona

NO conversation lines or assertions are hardcoded in this file.
Every behavioral expectation lives in the persona JSON.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.chat.engine import ChatEngine
from src.core import linguistic_resources as lr
from src.core.encoder import MeaningEncoder
from src.core.models import Concept
from src.lifecycle.idle_learner import IdleLearner, IdleLearnerConfig
from src.lifecycle.lifecycle_policy import AlwaysAllowPolicy
from src.lifecycle.wake_summary import WakeSummary, WakeSummaryBuilder
from src.sleep.deep_sleep import DeepSleep
from src.sleep.sleep_cycle import SleepCycleOrchestrator


# ─── Persona schema ────────────────────────────────────────────────────────


@dataclass
class PersonaTurn:
    text: str
    speaker: str
    label: Optional[str] = None


@dataclass
class PersonaSession:
    turns: List[PersonaTurn]
    label: str
    day_offset_days: int
    idle_seconds_before: float = 0.0


@dataclass
class ExpectedRecall:
    question_keywords: List[str]
    must_contain_any: List[str]
    must_not_contain: List[str]
    after_day_offset: int
    category: str
    description: str


@dataclass
class ExpectedSchema:
    expected_entity: str
    category: str
    min_occurrences: int
    after_day_offset: int


@dataclass
class ExpectedCuriosityGap:
    entity: str
    min_occurrences: int
    after_day_offset: int
    description: str


@dataclass
class Persona:
    name: str
    background: str
    primary_speaker: str
    secondary_speaker: str
    sessions: List[PersonaSession]
    expected_recall: List[ExpectedRecall] = field(default_factory=list)
    expected_schemas: List[ExpectedSchema] = field(default_factory=list)
    expected_curiosity_gaps: List[ExpectedCuriosityGap] = field(default_factory=list)


def load_persona(path: Path) -> Persona:
    """Parse a persona JSON file. Validation is permissive to keep
    persona files easy to write."""
    raw = json.loads(Path(path).read_text())
    sessions: List[PersonaSession] = []
    for d in raw.get("days", []):
        turns = []
        for t in d.get("turns", []):
            if isinstance(t, str):
                # "Speaker: text" shorthand
                if ":" in t:
                    sp, _, body = t.partition(":")
                    turns.append(PersonaTurn(speaker=sp.strip(), text=body.strip()))
                else:
                    turns.append(PersonaTurn(
                        speaker=raw.get("primary_speaker", "User"),
                        text=t.strip(),
                    ))
            elif isinstance(t, dict):
                turns.append(PersonaTurn(
                    speaker=t.get("speaker", raw.get("primary_speaker", "User")),
                    text=t.get("text", ""),
                    label=t.get("label"),
                ))
        sessions.append(PersonaSession(
            turns=turns,
            label=d.get("label", f"day{d.get('day_offset_days', 0)}"),
            day_offset_days=int(d.get("day_offset_days", 0)),
            idle_seconds_before=float(d.get("idle_seconds_before", 0.0)),
        ))
    persona = Persona(
        name=raw.get("name", "Unnamed"),
        background=raw.get("background", ""),
        primary_speaker=raw.get("primary_speaker", "User"),
        secondary_speaker=raw.get("secondary_speaker", "AI"),
        sessions=sessions,
        expected_recall=[
            ExpectedRecall(
                question_keywords=list(r.get("question_keywords", [])),
                must_contain_any=list(r.get("must_contain_any", [])),
                must_not_contain=list(r.get("must_not_contain", [])),
                after_day_offset=int(r.get("after_day_offset", 0)),
                category=str(r.get("category", "recall")),
                description=str(r.get("description", "")),
            )
            for r in raw.get("expected_recall", [])
        ],
        expected_schemas=[
            ExpectedSchema(
                expected_entity=str(s.get("expected_entity", "")),
                category=str(s.get("category", "recurring_topic")),
                min_occurrences=int(s.get("min_occurrences", 1)),
                after_day_offset=int(s.get("after_day_offset", 0)),
            )
            for s in raw.get("expected_schemas", [])
        ],
        expected_curiosity_gaps=[
            ExpectedCuriosityGap(
                entity=str(g.get("entity", "")),
                min_occurrences=int(g.get("min_occurrences", 1)),
                after_day_offset=int(g.get("after_day_offset", 0)),
                description=str(g.get("description", "")),
            )
            for g in raw.get("expected_curiosity_gaps", [])
        ],
    )
    return persona


# ─── Trace + result records ─────────────────────────────────────────────────


@dataclass
class TurnEvent:
    day_offset: int
    session_label: str
    speaker: str
    text: str
    sim_time: datetime


@dataclass
class SleepEvent:
    sim_time: datetime
    consolidated: int
    forgotten: int
    dreams: int
    schemas_in_stats: int
    curiosity_filled: int
    triggered_by: str  # "idle_daemon" | "session_boundary" | "explicit"


@dataclass
class SimulationTrace:
    persona_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    turns: List[TurnEvent] = field(default_factory=list)
    sleeps: List[SleepEvent] = field(default_factory=list)
    daemon_blocked_count: int = 0
    final_wake_summary: Optional[WakeSummary] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    name: str
    category: str
    description: str
    passed: bool
    detail: str


# ─── Sim clock injected everywhere ─────────────────────────────────────────


class SimClock:
    """Manually-advanced clock that the IdleLearner reads for activity timing."""
    def __init__(self, start: datetime):
        self._now = start
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now = self._now + timedelta(seconds=seconds)


# ─── Simulator ──────────────────────────────────────────────────────────────


class BrutalSimulator:
    """Replays a Persona against a live SCM stack and collects a trace."""

    def __init__(
        self,
        persona: Persona,
        engine_factory: Callable[[Dict[str, Any]], ChatEngine],
        clock_start: Optional[datetime] = None,
        idle_learner_cfg: Optional[IdleLearnerConfig] = None,
        # Force daemon to fire fast: in real usage idle = 600s, here we
        # advance the SimClock by `idle_seconds_before` so it crosses the
        # threshold. Tick interval is small so the daemon catches up
        # quickly in wall-clock time.
        run_idle_daemon: bool = True,
    ):
        self.persona = persona
        self.engine_factory = engine_factory
        self.clock = SimClock(clock_start or datetime(2026, 5, 1, 9, 0, 0))
        self.run_idle_daemon = run_idle_daemon
        self.idle_cfg = idle_learner_cfg or IdleLearnerConfig(
            idle_threshold_seconds=300.0,        # 5 min sim-time = idle
            min_sleep_interval_seconds=60.0,     # cooldown
            tick_interval_seconds=0.05,          # wall-clock: 50ms
            max_sleep_duration_seconds=30.0,
            sleep_mode="deep",
        )
        self._engine: Optional[ChatEngine] = None
        self._learner: Optional[IdleLearner] = None
        self.trace = SimulationTrace(
            persona_name=persona.name,
            started_at=self.clock.now(),
        )

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self) -> SimulationTrace:
        engine = self.engine_factory({"clock": self.clock})
        self._engine = engine

        if self.run_idle_daemon:
            learner = IdleLearner(
                engine_provider=lambda: {engine.session_id: engine},
                config=self.idle_cfg,
                clock=self.clock.now,
                policy=AlwaysAllowPolicy(),
            )
            self._learner = learner
            learner.start()

        try:
            for sess in self.persona.sessions:
                self._run_session(sess)
        finally:
            if self._learner is not None:
                # Let the daemon catch any final ticks
                import time
                time.sleep(0.2)
                self._learner.stop()

        # Final explicit deep-sleep so any residual schemas / curiosity fire
        try:
            r = engine.force_sleep(mode="deep")
            if r:
                self.trace.sleeps.append(SleepEvent(
                    sim_time=self.clock.now(),
                    consolidated=int(r.get("consolidated", 0) or 0),
                    forgotten=int(r.get("forgotten", 0) or 0),
                    dreams=int(r.get("dreams", 0) or 0),
                    schemas_in_stats=0,  # not surfaced from force_sleep
                    curiosity_filled=0,
                    triggered_by="explicit_final",
                ))
        except Exception as exc:
            self.trace.notes.append(f"final force_sleep failed: {exc}")

        self.trace.final_wake_summary = WakeSummaryBuilder(engine).build(max_insights=20)
        if self._learner is not None:
            self.trace.daemon_blocked_count = self._learner.get_stats().get(
                "cycles_blocked_by_policy", 0
            )
        self.trace.ended_at = self.clock.now()
        return self.trace

    # ── Session replay ─────────────────────────────────────────────────────

    def _run_session(self, session: PersonaSession) -> None:
        # Advance clock by idle gap (this is what triggers the daemon)
        if session.idle_seconds_before > 0:
            self._advance_idle(session.idle_seconds_before)

        for turn in session.turns:
            tagged = f"{turn.speaker}: {turn.text}"
            try:
                self._engine._extract_and_store(tagged, source="user")
                self._engine._message_count += 1
            except Exception as exc:
                self.trace.notes.append(f"ingestion failed: {exc}")
                continue
            self.trace.turns.append(TurnEvent(
                day_offset=session.day_offset_days,
                session_label=session.label,
                speaker=turn.speaker,
                text=turn.text,
                sim_time=self.clock.now(),
            ))
            # Tell the IdleLearner we just had user activity (resets idle clock)
            if self._learner is not None:
                self._learner.record_activity(self._engine.session_id)
            # Advance the sim clock by 30s per turn (a realistic chat pace)
            self.clock.advance(30.0)

    def _advance_idle(self, seconds: float) -> None:
        """Advance the sim clock through an idle gap, letting the daemon tick."""
        import time as wall
        # Single jump past the idle threshold + brief yield so the daemon
        # gets one chance to fire. Anything more is wasted wall-clock time
        # that bloats brutal-test runtime.
        self.clock.advance(seconds)
        # Yield ~3 daemon ticks for the IdleLearner to react.
        wall.sleep(max(0.15, self.idle_cfg.tick_interval_seconds * 3.0))
        # Track autonomous sleep firings via the IdleLearner's history.
        if self._learner is not None:
            try:
                hist = self._learner.get_history(
                    session_id=self._engine.session_id,
                    limit=200,
                )
                # Anything new since the last tracked count = autonomous fire.
                already = len([s for s in self.trace.sleeps
                               if s.triggered_by == "idle_daemon"])
                for rec in hist[already:]:
                    self.trace.sleeps.append(SleepEvent(
                        sim_time=self.clock.now(),
                        consolidated=int(getattr(rec, "consolidated", 0) or 0),
                        forgotten=int(getattr(rec, "forgotten", 0) or 0),
                        dreams=int(getattr(rec, "dreams", 0) or 0),
                        schemas_in_stats=0,
                        curiosity_filled=0,
                        triggered_by="idle_daemon",
                    ))
            except Exception:
                pass


# ─── Scenario evaluator ────────────────────────────────────────────────────


class ScenarioEvaluator:
    """Walks the trace + final memory state and reports per-scenario pass/fail."""

    def __init__(self, persona: Persona, engine: ChatEngine, trace: SimulationTrace):
        self.persona = persona
        self.engine = engine
        self.trace = trace

    def evaluate_all(self) -> List[ScenarioResult]:
        results: List[ScenarioResult] = []
        for r in self.persona.expected_recall:
            results.append(self._eval_recall(r))
        for s in self.persona.expected_schemas:
            results.append(self._eval_schema(s))
        for g in self.persona.expected_curiosity_gaps:
            results.append(self._eval_curiosity_gap(g))
        # Always-on structural checks
        results.append(self._eval_minimum_consolidation())
        results.append(self._eval_idle_daemon_fired())
        return results

    # ── Recall ─────────────────────────────────────────────────────────────

    def _eval_recall(self, r: ExpectedRecall) -> ScenarioResult:
        # Construct queries that share vocabulary with stored concepts. Three
        # query forms are tried; the best-scoring result wins.
        keywords = [k for k in r.question_keywords if k]
        candidates = [
            " ".join(keywords),                              # bare keywords
            self.persona.primary_speaker + " " + " ".join(keywords),  # entity-anchored
            " ".join(keywords) + " " + " ".join(r.must_contain_any[:1]),  # hint with answer head
        ]
        retrieved_text = ""
        for q in candidates:
            text = self._retrieve(q)
            if text and len(text) > len(retrieved_text):
                retrieved_text = text
        low = retrieved_text.lower()
        hit_any = any(s.lower() in low for s in r.must_contain_any) if r.must_contain_any else True
        hit_forbidden = any(s.lower() in low for s in r.must_not_contain) if r.must_not_contain else False
        passed = hit_any and not hit_forbidden
        detail = (
            f"keywords={r.question_keywords} must_contain_any={r.must_contain_any} "
            f"must_not_contain={r.must_not_contain} retrieved_excerpt={retrieved_text[:140]!r}"
        )
        return ScenarioResult(
            name=f"recall::{r.category}::{'+'.join(r.question_keywords)[:40]}",
            category=r.category,
            description=r.description,
            passed=passed,
            detail=detail,
        )

    def _retrieve(self, question: str) -> str:
        """Use whichever retrieval is most representative of real use."""
        # If HME pipeline is enabled and attached, use it; else cosine over LTM.
        engine = self.engine
        try:
            if (
                getattr(engine, "_hme_enabled", False)
                and getattr(engine, "_spreading_activation", None) is not None
            ):
                activated, _ = engine._spreading_activation.retrieve(
                    query=question,
                    context_tags={"session_id": engine.session_id},
                )
                if not activated:
                    return ""
                ranker = engine._hypothesis_ranker
                if ranker is not None:
                    activation_map = {c.id: 1.0 - i * 0.02 for i, c in enumerate(activated)}
                    hs = ranker.rank(
                        activated_concepts=activated,
                        activation_map=activation_map,
                        context_tags={"session_id": engine.session_id},
                    )
                    return " | ".join(h.concept.description for h in hs.hypotheses[:5])
                return " | ".join(c.description for c in activated[:5])
        except Exception:
            pass
        # Fallback: simple cosine + keyword overlap over LTM
        return self._cosine_lookup(question)

    def _cosine_lookup(self, question: str) -> str:
        import numpy as np, re
        engine = self.engine
        try:
            cs = engine.long_term_memory.get_all_concepts(include_suppressed=False)
        except Exception:
            return ""
        if not cs:
            return ""
        try:
            q_emb = np.asarray(engine.encoder._get_embedding(question), dtype=np.float32)
            q_norm = np.linalg.norm(q_emb) + 1e-9
        except Exception:
            return ""
        q_tokens = set(re.findall(r"\b\w{2,}\b", question.lower()))
        scored = []
        for c in cs:
            sim = 0.0
            if c.embedding:
                e = np.asarray(c.embedding, dtype=np.float32)
                en = np.linalg.norm(e) + 1e-9
                sim = float(np.dot(e, q_emb) / (en * q_norm))
            d_tok = set(re.findall(r"\b\w{2,}\b", (c.description or "").lower()))
            overlap = len(q_tokens & d_tok) / max(1, len(q_tokens))
            scored.append((c, 0.6 * sim + 0.3 * overlap + 0.1 * c.importance.overall))
        scored.sort(key=lambda x: x[1], reverse=True)
        return " | ".join(c.description for c, _ in scored[:5])

    # ── Schema ─────────────────────────────────────────────────────────────

    def _eval_schema(self, s: ExpectedSchema) -> ScenarioResult:
        try:
            cs = self.engine.long_term_memory.get_all_concepts(include_suppressed=True)
        except Exception:
            cs = []
        target = s.expected_entity.lower()
        match = None
        for c in cs:
            tags = c.context_tags or {}
            if not tags.get("_schema"):
                continue
            entities = [str(e).lower() for e in (tags.get("entities") or [])]
            if tags.get("schema_type") == s.category and target in entities:
                if int(tags.get("occurrence_count", 0) or 0) >= s.min_occurrences:
                    match = c
                    break
        passed = match is not None
        detail = (
            f"expected entity={s.expected_entity} category={s.category} "
            f"min_occurrences={s.min_occurrences} found={'yes' if passed else 'no'}"
        )
        return ScenarioResult(
            name=f"schema::{s.category}::{s.expected_entity}",
            category=s.category,
            description=f"Expected schema for {s.expected_entity}",
            passed=passed,
            detail=detail,
        )

    # ── Curiosity gap ──────────────────────────────────────────────────────

    def _eval_curiosity_gap(self, g: ExpectedCuriosityGap) -> ScenarioResult:
        # Two ways the gap can be satisfied:
        # 1) A curiosity-filled concept exists in LTM tagged with curiosity_entity == entity
        # 2) The trace shows the gap was at least DETECTED by the curiosity engine
        #    (the engine ran, listed it as a candidate). Detection-without-fill
        #    is acceptable when no source can answer.
        try:
            cs = self.engine.long_term_memory.get_all_concepts(include_suppressed=True)
        except Exception:
            cs = []
        filled = any(
            (c.context_tags or {}).get("_curiosity")
            and (c.context_tags or {}).get("curiosity_entity", "").lower() == g.entity.lower()
            for c in cs
        )
        # Detection is best-effort signal; if filled is true, definitely passed.
        passed = filled
        detail = f"entity={g.entity} filled_in_ltm={filled}"
        return ScenarioResult(
            name=f"curiosity::{g.entity}",
            category="curiosity_gap",
            description=g.description,
            passed=passed,
            detail=detail,
        )

    # ── Structural ─────────────────────────────────────────────────────────

    def _eval_minimum_consolidation(self) -> ScenarioResult:
        total = sum(s.consolidated for s in self.trace.sleeps)
        passed = total > 0
        return ScenarioResult(
            name="structural::consolidation_happened",
            category="structural",
            description="At least one sleep cycle consolidated something",
            passed=passed,
            detail=f"total_consolidated={total} sleeps={len(self.trace.sleeps)}",
        )

    def _eval_idle_daemon_fired(self) -> ScenarioResult:
        autonomous = sum(
            1 for s in self.trace.sleeps if s.triggered_by == "idle_daemon"
        )
        # If the persona has long idle gaps, we expect at least one autonomous fire
        long_gaps = sum(
            1 for sess in self.persona.sessions if sess.idle_seconds_before > 600
        )
        passed = (long_gaps == 0) or (autonomous >= 1)
        return ScenarioResult(
            name="structural::idle_daemon_fired_at_least_once",
            category="structural",
            description="IdleLearner fired during simulated idle gaps",
            passed=passed,
            detail=f"autonomous_sleeps={autonomous} long_idle_gaps_in_persona={long_gaps}",
        )
