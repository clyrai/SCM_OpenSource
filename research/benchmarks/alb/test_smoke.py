"""
Smoke tests for the ALB benchmark plumbing.

Two synthetic adapters exercise the scorer end-to-end:

  1. PerfectAdapter — knows the persona's ground truth and reports it back
     verbatim. Should score ~1.0 on every metric. If it doesn't, the
     scoring or matching code is broken (NOT the adapter).

  2. NullAdapter — does nothing. Returns empty everywhere. Should score
     ~0.0 on idle-axis metrics and not crash.

These adapters are SCAFFOLDING for the test, not real systems. They
exist solely to verify the runner + scorer wiring.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from adapters import (
    BaseMemorySystem,
    Capability,
    Gap,
    IdleReport,
    Message,
    QueryResult,
    Schema,
    SystemStats,
    WakeSummary,
)
from runner import load_persona, run_persona, score_run


# ─── Test fixtures ─────────────────────────────────────────────────────────


class PerfectAdapter(BaseMemorySystem):
    """Knows the persona's ground truth. Should ace every metric."""

    def __init__(self, persona: Dict[str, Any]):
        self._persona = persona
        self._gt = persona["ground_truth"]
        self._reset_count = 0
        self._stats = SystemStats()
        self._idle_count = 0
        self._formed_schema_ids: List[str] = []

    @property
    def system_name(self) -> str:
        return "PerfectAdapter"

    @property
    def system_version(self) -> str:
        return "test-0.1"

    def supports(self) -> Set[Capability]:
        return {
            Capability.SCHEMA_EXTRACTION,
            Capability.GAP_TRACKING,
            Capability.AUTONOMOUS_FILL,
            Capability.WAKE_SUMMARY,
            Capability.VERSIONING,
            Capability.CROSS_SESSION_POOL,
            Capability.IDLE_PROCESSING,
        }

    def reset(self, persona_id: str, seed: int) -> None:
        self._reset_count += 1
        self._stats = SystemStats()
        self._idle_count = 0
        self._formed_schema_ids = []

    def ingest(self, message: Message, sim_time: datetime) -> None:
        self._stats.total_messages_ingested += 1

    def idle(self, duration_sim_seconds, sim_time, allow_compute=True) -> IdleReport:
        self._idle_count += 1
        self._stats.total_idle_periods += 1
        if allow_compute:
            # Form schemas progressively, one per idle period.
            for p in self._gt.get("patterns", []):
                if p["pattern_id"] not in self._formed_schema_ids:
                    self._formed_schema_ids.append(p["pattern_id"])
                    break
        return IdleReport(
            duration_sim_seconds=duration_sim_seconds,
            wall_clock_seconds=0.001,
            cpu_seconds=0.001,
            peak_rss_bytes=0,
            sleep_cycles_fired=1 if allow_compute else 0,
            schemas_formed=1 if allow_compute else 0,
        )

    def query(self, text: str, sim_time: datetime) -> QueryResult:
        self._stats.total_queries += 1
        # Look up the matching probe in persona to figure out what's being asked.
        for pq in self._persona.get("probe_queries", []):
            if pq["query"] != text:
                continue
            tm = pq["target_metric"]
            scoring = pq.get("scoring", {})
            if tm == "CGC_fill":
                gap_id = scoring.get("gap_id")
                for g in self._gt.get("gaps", []):
                    if g["gap_id"] == gap_id:
                        # Reply with all the expected definition keywords.
                        kws = g.get("expected_definition_keywords", [])
                        return QueryResult(
                            text=" ".join(kws),
                            metadata={"gap_id": gap_id},
                        )
            if tm == "CRAI_current":
                cid = scoring.get("contradiction_id")
                for c in self._gt.get("contradictions", []):
                    if c["contradiction_id"] == cid:
                        return QueryResult(text=c["new_value"])
            if tm == "CRAI_old":
                cid = scoring.get("contradiction_id")
                for c in self._gt.get("contradictions", []):
                    if c["contradiction_id"] == cid:
                        return QueryResult(text=c["old_value"])
            if tm == "CSS":
                qtxt = pq["query"]
                for q in self._gt.get("cross_session_questions", []):
                    if q["query"] == qtxt:
                        return QueryResult(text=" ".join(q["correct_answer_keywords"]))
        return QueryResult(text="")

    def list_schemas(self) -> List[Schema]:
        out: List[Schema] = []
        for pid in self._formed_schema_ids:
            for p in self._gt.get("patterns", []):
                if p["pattern_id"] == pid:
                    out.append(Schema(
                        schema_id=pid,
                        type=p["type"],
                        signature=p["signature"],
                        confidence=1.0,
                        raw_text=" ".join(
                            str(v) for v in p["signature"].values()
                            if isinstance(v, (str, int, float))
                        ) + " " + " ".join(
                            " ".join(map(str, v)) for v in p["signature"].values()
                            if isinstance(v, list)
                        ),
                    ))
                    break
        return out

    def list_open_questions(self) -> List[Gap]:
        return [
            Gap(gap_id=g["gap_id"], term=g["term"], has_been_filled=True,
                fill_source="static_dict")
            for g in self._gt.get("gaps", [])
        ]

    def wake_summary(self, since_sim_time: datetime) -> Optional[WakeSummary]:
        return WakeSummary(
            since_sim_time=since_sim_time,
            schemas_formed=self.list_schemas(),
            narrative="Reviewed recent conversations.",
        )

    def was_autonomous_fill(self, gap_id_or_term: str) -> bool:
        return True

    def stats(self) -> SystemStats:
        return self._stats


class NullAdapter(BaseMemorySystem):
    """Does nothing. Returns empty. Should not crash the runner."""

    def __init__(self):
        self._stats = SystemStats()

    @property
    def system_name(self) -> str:
        return "NullAdapter"

    @property
    def system_version(self) -> str:
        return "test-0.1"

    def supports(self) -> Set[Capability]:
        return set()

    def reset(self, persona_id, seed):
        self._stats = SystemStats()

    def ingest(self, message, sim_time):
        self._stats.total_messages_ingested += 1

    def idle(self, duration_sim_seconds, sim_time, allow_compute=True):
        return IdleReport(
            duration_sim_seconds=duration_sim_seconds,
            wall_clock_seconds=0.0,
            cpu_seconds=0.0,
            peak_rss_bytes=0,
        )

    def query(self, text, sim_time):
        return QueryResult(text="")

    def list_schemas(self):
        return []

    def list_open_questions(self):
        return []

    def wake_summary(self, since_sim_time):
        return None

    def was_autonomous_fill(self, gap_id_or_term):
        return False

    def stats(self):
        return self._stats


# ─── Tests ─────────────────────────────────────────────────────────────────


def main() -> int:
    persona_path = HERE / "personas" / "persona_001.json"
    persona = load_persona(persona_path)

    print("=" * 70)
    print("ALB smoke test — Perfect adapter (should ace metrics)")
    print("=" * 70)
    perfect = PerfectAdapter(persona)
    run = run_persona(persona, perfect, seed=0, idle_on=True)
    scored = score_run(run, persona, adapter=perfect)
    print(scored.summary())
    print()
    print(f"  capabilities reported: {run.capabilities}")
    print(f"  idle events: {len(run.idle_events)}")
    print(f"  probe outcomes: {len(run.probe_outcomes)}")
    print(f"  final schemas: {len(run.final_schemas)}")
    print(f"  final gaps:    {len(run.final_gaps)}")

    pdr_total = len(persona["ground_truth"]["patterns"])
    pdr_expected = min(len(run.idle_events), pdr_total) / pdr_total
    print(f"  expected PDR (one schema/idle, {len(run.idle_events)} idles, {pdr_total} patterns): {pdr_expected:.3f}")

    perfect_pdr_floor = 0.40 if len(run.idle_events) >= 2 else 0.20
    assert scored.pdr >= perfect_pdr_floor, f"PerfectAdapter PDR={scored.pdr} < {perfect_pdr_floor}"
    assert scored.cgc_id == 1.0, f"PerfectAdapter cgc_id={scored.cgc_id} != 1.0"
    assert scored.cgc_fill == 1.0, f"PerfectAdapter cgc_fill={scored.cgc_fill} != 1.0"
    assert scored.crai_current == 1.0, f"PerfectAdapter crai_current={scored.crai_current} != 1.0"
    assert scored.crai_old == 1.0, f"PerfectAdapter crai_old={scored.crai_old} != 1.0"
    assert scored.css == 1.0, f"PerfectAdapter css={scored.css} != 1.0"
    print("  PerfectAdapter assertions: PASS\n")

    print("=" * 70)
    print("ALB smoke test — Null adapter (should score ~0 on idle-axis)")
    print("=" * 70)
    null = NullAdapter()
    run2 = run_persona(persona, null, seed=0, idle_on=True)
    scored2 = score_run(run2, persona, adapter=null)
    print(scored2.summary())
    print()
    assert scored2.pdr == 0.0, f"NullAdapter pdr={scored2.pdr} != 0"
    assert scored2.cgc_id == 0.0
    assert scored2.cgc_fill == 0.0
    assert scored2.crai_current == 0.0
    assert scored2.crai_old == 0.0
    assert scored2.css == 0.0
    assert scored2.wsi_f1 == 0.0
    print("  NullAdapter assertions: PASS\n")

    print("=" * 70)
    print("ALB smoke test — Perfect adapter, idle_off (NIAL ablation)")
    print("=" * 70)
    perfect2 = PerfectAdapter(persona)
    run3 = run_persona(persona, perfect2, seed=0, idle_on=False)
    scored3 = score_run(run3, persona, adapter=perfect2)
    print(scored3.summary())
    print()
    assert scored3.pdr == 0.0, f"idle_off should yield no schemas, got pdr={scored3.pdr}"
    print(f"  NIAL on PDR (idle_on - idle_off): {scored.pdr - scored3.pdr:+.3f}")
    print("  NIAL ablation assertions: PASS\n")

    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
