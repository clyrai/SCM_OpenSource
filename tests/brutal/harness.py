"""
Brutal test harness — assembles a full SCM stack with all Phase 7 features
enabled, runs a persona simulation against it, and reports pass/fail per
scenario.

Tier 1: realistic multi-day usage (the persona JSON drives this).
Tier 2: adversarial inputs (all-noise, all-contradictions).
Tier 3: failure modes (missing deps, broken sources).
Tier 4: scale (long persona × many sessions).

The harness is intentionally permissive about pass/fail — it produces a
report. Pytest entrypoints are in test_brutal_e2e.py and assert specific
gates from the report.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.chat import engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.encoder import MeaningEncoder
from src.lifecycle import (
    CuriosityConfig,
    CuriosityEngine,
    StaticDictionarySource,
)
from src.sleep.deep_sleep import DeepSleep
from src.sleep.schema_extractor import SchemaExtractor, SchemaExtractorConfig
from src.sleep.sleep_cycle import SleepCycleOrchestrator

from .simulator import (
    BrutalSimulator,
    Persona,
    ScenarioEvaluator,
    ScenarioResult,
    SimClock,
    SimulationTrace,
    load_persona,
)


# ─── Stub LLM (no API calls during brutal tests) ───────────────────────────


class _NoLLM:
    """Disables LLM calls so tests are fully offline + deterministic."""
    def extract_concepts(self, text):
        return []
    def _chat(self, *a, **kw):
        return ""


# ─── Engine factories ─────────────────────────────────────────────────────


def default_engine_factory(
    session_id: str = "brutal_user",
    profile: str = "research",
    enable_hme: bool = True,
    enable_schemas: bool = True,
    enable_curiosity: bool = False,
    curiosity_dictionary: Optional[Dict[str, str]] = None,
    enable_paraphrase: bool = True,
    llm_backend: str = "stub",   # "stub" | "ollama" | "deepseek"
    add_llm_curiosity_source: bool = False,
    persist: bool = False,        # M2: enable SQLite + cross-session pool
    persist_db_path: Optional[str] = None,   # if persist=True, where to store
) -> Callable[[Dict[str, Any]], ChatEngine]:
    """
    Build a ChatEngine with full Phase 7 stack (sandboxed, no persistence).
    Returns a factory function so the simulator can pass extra context.

    llm_backend:
      - "stub":     no LLM calls, fully offline, deterministic. Default.
      - "ollama":   local Ollama via the project's LLMExtractor. Slow but free.
      - "deepseek": OpenAI-compatible DeepSeek API. Fast but costs API tokens.
    add_llm_curiosity_source:
      When True AND enable_curiosity AND llm_backend != "stub", an LLMSource
      is appended to the curiosity engine. This is what gives the agent
      autonomous knowledge generation under real-LLM tests.
    """
    backend = (llm_backend or "stub").lower()
    if backend not in {"stub", "ollama", "deepseek"}:
        raise ValueError(f"unknown llm_backend: {backend!r}")

    # If a custom DB path is provided AND persistence is requested, redirect
    # the SQLite singleton to it. This must happen BEFORE the engine is
    # constructed (engine's _load_session reads from get_memory()).
    if persist and persist_db_path:
        from src.core import sqlite_db as _sqlite_mod
        _sqlite_mod.set_db_path(persist_db_path)

    def factory(ctx: Dict[str, Any]) -> ChatEngine:
        # HME flag is module-level
        chat_engine_module.HME_ENABLED = enable_hme

        # Build the LLM + encoder for this engine
        if backend == "stub":
            llm = _NoLLM()
            encoder = MeaningEncoder(llm=None)
        else:
            from src.llm import LLMExtractor
            llm = LLMExtractor(provider=backend)
            encoder = MeaningEncoder(llm=llm)

        curiosity_engine = None
        if enable_curiosity:
            sources = []
            if curiosity_dictionary:
                sources.append(StaticDictionarySource(curiosity_dictionary))
            if add_llm_curiosity_source and backend != "stub":
                from src.lifecycle.curiosity import LLMSource
                sources.append(LLMSource(llm=llm))
            if sources:
                curiosity_engine = CuriosityEngine(
                    sources=sources,
                    config=CuriosityConfig(
                        enabled=True,
                        min_occurrences=2,
                        max_gaps_per_cycle=2,  # bound cost
                    ),
                )

        deep = DeepSleep(
            enable_synthesis=False,  # keep tests deterministic
            enable_schema_extraction=enable_schemas,
            schema_extractor=SchemaExtractor(
                config=SchemaExtractorConfig(
                    enabled=enable_schemas,
                    min_repetitions=2,
                ),
            ),
            enable_paraphrase=enable_paraphrase,
            enable_curiosity=curiosity_engine is not None,
            curiosity_engine=curiosity_engine,
        )
        orch = SleepCycleOrchestrator(deep_sleep=deep)

        # Optionally enable real persistence + cross-session pool for brutal
        # tests that need M2 to actually function (cross-session schemas).
        if persist:
            from src.core.cross_session_pool import (
                CrossSessionMemoryPool,
                CrossSessionPoolConfig,
            )
            from src.core.sqlite_db import get_memory
            cs_pool = CrossSessionMemoryPool(
                current_session_id=session_id,
                sqlite_factory=get_memory,
                current_profile=profile,
                config=CrossSessionPoolConfig(
                    enabled=True,
                    max_sessions=5,
                    lookback_hours=720.0,  # 30 days for brutal tests
                    max_episodes_per_session=50,
                    max_total_borrowed=200,
                    # Single-user multi-day: include this session too so older
                    # persisted episodes are visible for cross-day patterns.
                    include_current_session=True,
                ),
            )
        else:
            cs_pool = None

        engine = ChatEngine(
            llm=llm,
            encoder=encoder,
            sleep_orchestrator=orch,
            session_id=session_id,
            profile=profile,
            sandbox_mode=not persist,
            enable_persistence=persist,
            enable_auto_sleep=False,  # the IdleLearner is the sole sleep trigger
            cross_session_pool=cs_pool,
        )
        return engine
    return factory


# ─── Report ────────────────────────────────────────────────────────────────


@dataclass
class BrutalReport:
    persona_name: str
    started_at: datetime
    ended_at: datetime
    turns_ingested: int
    sleep_cycles: int
    schemas_formed_in_final_summary: int
    daemon_blocked_count: int
    final_narrative: str
    scenarios: List[ScenarioResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scenarios)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scenarios if s.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def by_category(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for s in self.scenarios:
            d = out.setdefault(s.category, {"passed": 0, "failed": 0})
            if s.passed:
                d["passed"] += 1
            else:
                d["failed"] += 1
        return out

    def print(self) -> None:
        print(f"\n{'=' * 70}")
        print(f"BRUTAL REPORT — {self.persona_name}")
        print(f"{'=' * 70}")
        print(f"Turns ingested:        {self.turns_ingested}")
        print(f"Sleep cycles:          {self.sleep_cycles}")
        print(f"Schemas in final wake: {self.schemas_formed_in_final_summary}")
        print(f"Daemon blocked count:  {self.daemon_blocked_count}")
        print(f"Pass rate:             {self.passed}/{self.total} ({self.pass_rate:.0%})")
        print(f"\nBy category:")
        for cat, counts in self.by_category().items():
            total = counts["passed"] + counts["failed"]
            print(f"  {cat:24s}  {counts['passed']}/{total}")
        print(f"\nDetailed:")
        for s in self.scenarios:
            mark = "✓" if s.passed else "✗"
            print(f"  [{mark}] {s.name}")
            print(f"      {s.detail[:140]}")
        print(f"\nFinal wake narrative:")
        for line in self.final_narrative.splitlines():
            print(f"  {line}")
        print(f"{'=' * 70}\n")


# ─── Top-level runner ─────────────────────────────────────────────────────


def run_brutal(
    persona_path: Path,
    *,
    enable_hme: bool = True,
    enable_schemas: bool = True,
    enable_curiosity: bool = False,
    curiosity_dictionary: Optional[Dict[str, str]] = None,
    enable_paraphrase: bool = True,
    llm_backend: str = "stub",
    add_llm_curiosity_source: bool = False,
    persist: bool = False,
    persist_db_path: Optional[str] = None,
) -> BrutalReport:
    """
    Run a full brutal simulation against the persona at `persona_path`.
    Returns a BrutalReport ready to be inspected or asserted on.
    """
    persona = load_persona(persona_path)
    factory = default_engine_factory(
        enable_hme=enable_hme,
        enable_schemas=enable_schemas,
        enable_curiosity=enable_curiosity,
        curiosity_dictionary=curiosity_dictionary,
        enable_paraphrase=enable_paraphrase,
        llm_backend=llm_backend,
        add_llm_curiosity_source=add_llm_curiosity_source,
        persist=persist,
        persist_db_path=persist_db_path,
    )
    sim = BrutalSimulator(persona=persona, engine_factory=factory)
    trace = sim.run()
    evaluator = ScenarioEvaluator(persona=persona, engine=sim._engine, trace=trace)
    scenarios = evaluator.evaluate_all()
    final = trace.final_wake_summary
    return BrutalReport(
        persona_name=persona.name,
        started_at=trace.started_at,
        ended_at=trace.ended_at or trace.started_at,
        turns_ingested=len(trace.turns),
        sleep_cycles=len(trace.sleeps),
        schemas_formed_in_final_summary=len(final.insights) if final else 0,
        daemon_blocked_count=trace.daemon_blocked_count,
        final_narrative=final.narrative if final else "",
        scenarios=scenarios,
    )
