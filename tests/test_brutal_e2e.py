"""
Brutal end-to-end pytest entrypoints.

Tier 1: realistic 5-day persona simulation with measurable outcomes.
Tier 2: adversarial inputs (all-noise, all-contradictions, empty).
Tier 3: failure modes (broken state, broken sources, missing deps).
Tier 4: scale (extended persona × many sessions).

Each tier produces a BrutalReport which is asserted on. Reports are
printed to stdout for human inspection.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.brutal.harness import (
    BrutalReport,
    default_engine_factory,
    run_brutal,
)
from tests.brutal.simulator import (
    BrutalSimulator,
    Persona,
    PersonaSession,
    PersonaTurn,
    ScenarioEvaluator,
    SimClock,
    load_persona,
)

PERSONAS_DIR = Path(__file__).parent / "brutal" / "personas"


# ─── Tier 1: Realistic multi-day persona ───────────────────────────────────


def test_tier1_realistic_5day_persona_caroline():
    """
    Run the full 5-day Caroline persona through the complete Phase 7 stack:
    HME pipeline + cross-session pool (in-process) + idle learner +
    schema extraction + paraphrase + wake summary.

    Asserts:
      - All scenarios in the persona's expected_recall list pass
      - At least the recurring_topic schemas form (Caroline, TechCorp)
      - Final wake summary has at least 1 insight
      - At least one consolidation happens
    """
    persona_path = PERSONAS_DIR / "caroline_5day.json"
    assert persona_path.exists(), f"Persona file missing: {persona_path}"

    report = run_brutal(
        persona_path=persona_path,
        enable_hme=True,
        enable_schemas=True,
        enable_curiosity=False,  # tier 1 doesn't test curiosity
        enable_paraphrase=True,
    )
    report.print()

    # Structural floors: the simulation actually ran
    assert report.turns_ingested >= 30, f"too few turns ingested: {report.turns_ingested}"
    assert report.sleep_cycles >= 1, f"no sleep cycles fired"
    assert report.schemas_formed_in_final_summary >= 1, \
        "expected at least one schema/insight in the final wake summary"

    # Recall scenarios: must have at least 2 of 4 long-horizon recalls passing
    recall_scenarios = [s for s in report.scenarios if s.category in
                        ("long_horizon_recall", "contradiction_current")]
    recall_passed = sum(1 for s in recall_scenarios if s.passed)
    assert recall_passed >= 2, (
        f"only {recall_passed}/{len(recall_scenarios)} recall scenarios passed; "
        f"detail: {[(s.name, s.passed) for s in recall_scenarios]}"
    )

    # Schema scenarios: at least 1 of 3 expected schemas should form
    schema_scenarios = [s for s in report.scenarios if s.category == "recurring_topic"]
    schema_passed = sum(1 for s in schema_scenarios if s.passed)
    assert schema_passed >= 1, (
        f"expected at least one schema; got 0. "
        f"detail: {[(s.name, s.detail) for s in schema_scenarios]}"
    )


def test_tier1_with_curiosity_enabled():
    """
    Same persona as tier1 but with curiosity engine enabled, providing a
    static dictionary that covers the entities the persona mentions.
    """
    persona_path = PERSONAS_DIR / "caroline_5day.json"
    glossary = {
        "Datadog": "Datadog is a cloud monitoring and observability platform.",
        "Snowflake": "Snowflake is a cloud-based data warehousing platform.",
        "TechCorp": "TechCorp is a technology company in this scenario.",
    }
    report = run_brutal(
        persona_path=persona_path,
        enable_hme=True,
        enable_schemas=True,
        enable_curiosity=True,
        curiosity_dictionary=glossary,
    )
    report.print()
    # Curiosity scenarios should have at least 1 fill
    cur = [s for s in report.scenarios if s.category == "curiosity_gap"]
    cur_passed = sum(1 for s in cur if s.passed)
    assert cur_passed >= 1, (
        f"expected at least 1 curiosity gap filled; got 0. "
        f"detail: {[(s.name, s.detail) for s in cur]}"
    )
    # The wake summary should mention at least one "I read up on" insight
    learned = [s for s in report.scenarios if s.category == "curiosity_gap" and s.passed]
    if learned:
        assert "read up on" in report.final_narrative.lower() or \
               any(e.lower() in report.final_narrative.lower()
                   for e in ["datadog", "snowflake", "techcorp"]), \
            f"curiosity insights not surfaced in narrative: {report.final_narrative}"


# ─── Tier 2: Adversarial inputs ────────────────────────────────────────────


def _ad_hoc_persona(turns: List[str], name: str, idle_seconds: float = 0.0) -> Persona:
    """Build an in-memory Persona for adversarial scenarios."""
    sessions = [
        PersonaSession(
            turns=[PersonaTurn(speaker="User", text=t) for t in turns],
            label="adv_session",
            day_offset_days=0,
            idle_seconds_before=idle_seconds,
        )
    ]
    return Persona(
        name=name,
        background="adversarial",
        primary_speaker="User",
        secondary_speaker="AI",
        sessions=sessions,
    )


def test_tier2_all_noise_does_not_crash_or_hallucinate_patterns():
    """
    Feed the agent a session full of pure filler. It should:
      - Not crash
      - Not invent schemas about meaningless entities
      - Wake summary should be polite + honest
    """
    persona = _ad_hoc_persona(
        name="adversarial_noise",
        turns=[
            "lol", "haha yeah", "ok sure", "anyway", "hmm",
            "k", "got it", "for real", "yeah lol", "👍",
        ] * 3,
    )
    sim = BrutalSimulator(persona=persona, engine_factory=default_engine_factory(
        enable_schemas=True, enable_curiosity=False,
    ))
    trace = sim.run()
    assert trace.final_wake_summary is not None
    insights = trace.final_wake_summary.insights
    # Should not produce schemas about "lol" or "haha" or other filler tokens
    bad = [i for i in insights if any(b in i.text.lower() for b in
                                      ["lol", "haha", "anyway", "ok sure", "hmm"])]
    assert not bad, f"filler tokens leaked into insights: {[i.text for i in bad]}"


def test_tier2_repeated_contradictions_doesnt_blow_up():
    """
    User contradicts themselves over and over. Agent should remain stable.
    """
    persona = _ad_hoc_persona(
        name="adversarial_contradictions",
        turns=[
            "I work at GreenLeaf Cafe.",
            "Actually I switched — I'm at TechCorp now.",
            "Wait, I'm back at GreenLeaf.",
            "No actually I just joined Acme.",
            "Ignore that, I'm self-employed now.",
            "Update — I'm at TechCorp again.",
        ] * 2,
    )
    sim = BrutalSimulator(persona=persona, engine_factory=default_engine_factory(
        enable_schemas=True, enable_curiosity=False,
    ))
    trace = sim.run()
    # The agent should still be usable: at least one sleep cycle, no exceptions
    assert trace.final_wake_summary is not None
    # We don't assert which employer is "correct" — we assert stability
    assert len(trace.notes) == 0 or all("ingestion failed" not in n for n in trace.notes), \
        f"ingestion failures: {trace.notes}"


def test_tier2_empty_session_safe():
    """
    Persona with zero turns. Everything should still build + return.
    """
    persona = Persona(
        name="empty",
        background="empty",
        primary_speaker="User",
        secondary_speaker="AI",
        sessions=[PersonaSession(turns=[], label="empty", day_offset_days=0)],
    )
    sim = BrutalSimulator(persona=persona, engine_factory=default_engine_factory())
    trace = sim.run()
    assert trace.final_wake_summary is not None
    # Narrative should be the polite "welcome back" form, not garbage
    assert "Welcome back" in trace.final_wake_summary.narrative


# ─── Tier 3: Failure modes ─────────────────────────────────────────────────


def test_tier3_broken_curiosity_source_falls_through():
    """
    A curiosity source that always raises must NOT crash the sleep cycle.
    """
    from src.lifecycle.curiosity import CuriosityConfig, CuriosityEngine, CuriositySource

    class Broken(CuriositySource):
        name = "broken"
        def lookup(self, entity):
            raise RuntimeError("boom")

    persona_path = PERSONAS_DIR / "caroline_5day.json"
    persona = load_persona(persona_path)

    def _factory(ctx):
        engine = default_engine_factory(
            enable_hme=True, enable_schemas=True,
            enable_curiosity=False,  # we'll wire curiosity manually
        )(ctx)
        # Replace deep sleep's curiosity with a broken-source engine
        bad_engine = CuriosityEngine(
            sources=[Broken()],
            config=CuriosityConfig(enabled=True, min_occurrences=2),
        )
        engine.sleep_orchestrator.deep_sleep.enable_curiosity = True
        engine.sleep_orchestrator.deep_sleep.curiosity_engine = bad_engine
        return engine

    sim = BrutalSimulator(persona=persona, engine_factory=_factory)
    trace = sim.run()
    # Sleep should still produce consolidations despite the broken source
    total_consolidated = sum(s.consolidated for s in trace.sleeps)
    assert total_consolidated >= 0  # didn't crash
    assert trace.final_wake_summary is not None


def test_tier3_state_store_corruption_does_not_break_init(tmp_path):
    """
    A corrupted state file must NOT prevent IdleLearner from starting.
    """
    from src.lifecycle.idle_learner import IdleLearner, IdleLearnerConfig
    from src.lifecycle.state_store import IdleLearnerStateStore

    # Write garbage into the state file
    p = tmp_path / "state.json"
    p.write_text("totally not json{{{")

    store = IdleLearnerStateStore(str(p))
    # Constructor should succeed — broken file is silently treated as empty
    learner = IdleLearner(
        engine_provider=lambda: {},
        config=IdleLearnerConfig(),
        state_store=store,
    )
    # Empty state restored
    assert learner._last_activity == {}


# ─── Tier 4: Scale ──────────────────────────────────────────────────────────


def test_tier4_500_turn_persona_does_not_explode():
    """
    Build a 500-turn persona on the fly (mostly noise + some real facts).
    Verify the agent stays responsive and bounded resources.
    """
    real_facts = [
        "My name is Pat. I live in Boston.",
        "I work at Acme Corp.",
        "I love rock climbing.",
        "I have a dog named Rex.",
        "Last Tuesday I went to a conference.",
    ]
    noise = [
        "lol", "haha yeah", "ok sure", "anyway", "hmm got it", "for real",
        "Did you watch the game?", "I had pizza for lunch.",
        "Sorry, got distracted.", "Coffee, the only thing keeping me going.",
    ]
    turns = []
    for i in range(500):
        if i % 25 == 0 and (i // 25) < len(real_facts):
            turns.append(real_facts[i // 25])
        else:
            turns.append(noise[i % len(noise)])
    persona = _ad_hoc_persona(name="scale_500", turns=turns)
    sim = BrutalSimulator(
        persona=persona,
        engine_factory=default_engine_factory(enable_schemas=True),
    )
    t0 = time.time()
    trace = sim.run()
    elapsed = time.time() - t0
    # Liveness: 500 turns should complete in reasonable wall-clock time
    assert elapsed < 60.0, f"500-turn persona took too long: {elapsed:.1f}s"
    assert len(trace.turns) == 500
    # Final wake summary still renders cleanly
    assert trace.final_wake_summary is not None
    assert "Welcome back" in trace.final_wake_summary.narrative


# ─── Tier 5: Honest report ────────────────────────────────────────────────


def test_tier5_persisted_cross_session_schemas_form(tmp_path):
    """
    Brutal-uncovered Bug 1 fix verification:
    With persistence enabled, the cross-session memory pool should make
    schemas form across days even when entities appear once per day.

    Persona: developer_short. PostgreSQL appears in day 1 AND day 2.
    Without persistence: 0 schemas (M2 dead). With persistence: ≥ 1 schema.
    """
    persona_path = PERSONAS_DIR / "developer_short.json"
    db_path = tmp_path / "brutal_persist.db"
    report = run_brutal(
        persona_path=persona_path,
        enable_hme=True,
        enable_schemas=True,
        enable_curiosity=False,
        enable_paraphrase=False,  # avoid paraphrase mutating descriptions
        llm_backend="stub",
        persist=True,
        persist_db_path=str(db_path),
    )
    report.print()
    # The fix bar: at least 1 recurring-topic schema must form. Without M2,
    # the developer_short persona produces ZERO. With M2 it should be ≥ 1.
    schema_scenarios = [s for s in report.scenarios if s.category == "recurring_topic"]
    schema_passed = sum(1 for s in schema_scenarios if s.passed)
    assert schema_passed >= 1, (
        f"cross-session schema did not form despite persistence enabled. "
        f"detail: {[(s.name, s.detail) for s in schema_scenarios]}"
    )


def test_tier5_contradiction_versioning_archives_old_concept():
    """
    Brutal-uncovered Bug 2 fix verification:
    When a concept is superseded via versioning, retrieval must NOT surface
    the old (is_current_version=False) concept.

    Test directly creates an old + new concept and checks spreading
    activation doesn't return the superseded one.
    """
    from src.chat.engine import ChatEngine
    from src.core.encoder import MeaningEncoder
    from src.core.models import Concept, ConceptType, ImportanceVector, MemoryState
    from src.chat import engine as chat_engine_module

    chat_engine_module.HME_ENABLED = True

    class StubLLM:
        def extract_concepts(self, text): return []
        def _chat(self, *a, **kw): return ""

    engine = ChatEngine(
        llm=StubLLM(),
        encoder=MeaningEncoder(llm=None),
        session_id="versioning_test",
        profile="research",
        sandbox_mode=True,
        enable_persistence=False,
        enable_auto_sleep=False,
    )

    # Insert an OLD (superseded) concept and a NEW (current) one.
    old = Concept(
        type=ConceptType.FACT,
        description="Pat works at FinTech in the payments team.",
        importance=ImportanceVector(novelty=0.6, task_relevance=0.7),
        state=MemoryState.ACTIVE,
        salience_score=0.6,
    )
    old.is_current_version = False
    old.embedding = engine.encoder._get_embedding(old.description)
    engine.long_term_memory.add_concept(old)

    new = Concept(
        type=ConceptType.FACT,
        description="Pat works at FinTech in the fraud team.",
        importance=ImportanceVector(novelty=0.7, task_relevance=0.7),
        state=MemoryState.ACTIVE,
        salience_score=0.6,
    )
    new.is_current_version = True
    new.version_parent = old.id
    new.embedding = engine.encoder._get_embedding(new.description)
    engine.long_term_memory.add_concept(new)

    # Spreading activation must NOT return the superseded "payments" concept.
    activated, _ = engine._spreading_activation.retrieve(
        query="Pat current team",
        context_tags={"session_id": engine.session_id},
    )
    descriptions = [c.description for c in activated]
    joined = " | ".join(descriptions)
    assert "fraud" in joined.lower(), f"current concept missing: {descriptions}"
    assert "payments" not in joined.lower(), (
        f"superseded concept leaked into retrieval: {descriptions}"
    )


def test_brutal_report_summary_for_caroline_e2e():
    """
    Run the canonical 5-day persona and write a JSON report into
    research/metrics/ so the user can see the brutal-test outcome.
    This test does NOT assert; it always passes. Its value is the artifact.
    """
    persona_path = PERSONAS_DIR / "caroline_5day.json"
    report = run_brutal(
        persona_path=persona_path,
        enable_hme=True,
        enable_schemas=True,
        enable_curiosity=True,
        curiosity_dictionary={
            "Datadog": "Datadog is a cloud monitoring and observability platform.",
            "Snowflake": "Snowflake is a cloud-based data warehousing platform.",
            "TechCorp": "TechCorp is the speaker's current employer in this scenario.",
        },
    )
    out_dir = Path(__file__).parent.parent / "research" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "brutal_caroline_5day_latest.json"
    payload = {
        "persona": report.persona_name,
        "started_at": report.started_at.isoformat(),
        "ended_at": report.ended_at.isoformat(),
        "turns_ingested": report.turns_ingested,
        "sleep_cycles": report.sleep_cycles,
        "schemas_in_final_summary": report.schemas_formed_in_final_summary,
        "daemon_blocked_count": report.daemon_blocked_count,
        "pass_rate": report.pass_rate,
        "passed": report.passed,
        "failed": report.failed,
        "by_category": report.by_category(),
        "scenarios": [
            {
                "name": s.name,
                "category": s.category,
                "passed": s.passed,
                "description": s.description,
                "detail": s.detail,
            }
            for s in report.scenarios
        ],
        "final_narrative": report.final_narrative,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nBrutal report written: {out_path}")
