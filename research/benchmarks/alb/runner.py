"""
ALB runner.

Drives a persona through an adapter and collects everything needed for scoring.

Usage:
    from runner import run_persona, score_run
    from adapters.scm_adapter import SCMAdapter   # or any baseline

    adapter = SCMAdapter()
    run = run_persona("personas/persona_001.json", adapter, seed=0, idle_on=True)
    scored = score_run(run, persona)              # all metrics in one call
    print(scored.summary())

The runner is system-agnostic: it talks to BaseMemorySystem only.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from adapters import (
    BaseMemorySystem,
    Capability,
    Gap,
    IdleReport,
    Message,
    QueryResult,
    Schema,
    WakeSummary,
)
from metrics import (
    aggregate_imc,
    score_cgc_fill,
    score_cgc_id,
    score_crai_current,
    score_crai_old,
    score_css,
    score_pdr,
    score_wsi,
)


WAKE_SUMMARY_QUERY_TOKEN = "__WAKE_SUMMARY__"


# ─── Result types ──────────────────────────────────────────────────────────


@dataclass
class IdleEvent:
    """One idle period during the run."""
    start_sim: datetime
    end_sim: datetime
    duration_sim_seconds: float
    schemas_before: List[Schema]
    schemas_after: List[Schema]
    gaps_before: List[Gap]
    gaps_after: List[Gap]
    report: IdleReport


@dataclass
class ProbeOutcome:
    """One probe query and its response."""
    query_id: str
    target_metric: str
    sim_time: datetime
    query_text: str
    response_text: str
    response: Optional[QueryResult] = None
    wake_summary_returned: Optional[WakeSummary] = None  # for WSI probes


@dataclass
class RunResult:
    """Everything the scorer needs from one persona × system × idle-condition run."""
    persona_id: str
    system_name: str
    system_version: str
    seed: int
    idle_on: bool
    started_at_wall: float
    ended_at_wall: float
    turns_ingested: int
    idle_events: List[IdleEvent] = field(default_factory=list)
    probe_outcomes: List[ProbeOutcome] = field(default_factory=list)
    final_schemas: List[Schema] = field(default_factory=list)
    final_gaps: List[Gap] = field(default_factory=list)
    final_wake_summary: Optional[WakeSummary] = None
    capabilities: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ScoredRun:
    """Per-metric scores for one run."""
    persona_id: str
    system_name: str
    system_version: str
    seed: int
    idle_on: bool
    pdr: float
    css: float
    cgc_id: float
    cgc_fill: float
    crai_current: float
    crai_old: float
    wsi_precision: float
    wsi_recall: float
    wsi_f1: float
    imc_wall_seconds: float
    imc_cpu_seconds: float
    imc_total_idle_periods: int
    raw_subscores: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== {self.system_name}@{self.system_version} | {self.persona_id} | seed={self.seed} | idle_on={self.idle_on} ===",
            f"  PDR              {self.pdr:.3f}",
            f"  CSS              {self.css:.3f}",
            f"  CGC_id           {self.cgc_id:.3f}",
            f"  CGC_fill         {self.cgc_fill:.3f}",
            f"  CRAI_current     {self.crai_current:.3f}",
            f"  CRAI_old         {self.crai_old:.3f}",
            f"  WSI (P/R/F1)     {self.wsi_precision:.3f} / {self.wsi_recall:.3f} / {self.wsi_f1:.3f}",
            f"  IMC wall         {self.imc_wall_seconds:.2f}s ({self.imc_total_idle_periods} idle periods)",
        ]
        return "\n".join(lines)


# ─── Persona loader ────────────────────────────────────────────────────────


def load_persona(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open() as f:
        return json.load(f)


def _parse_iso(s: str) -> datetime:
    """Parse ISO 8601, accepting 'Z' suffix as UTC."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ─── The runner ────────────────────────────────────────────────────────────


def run_persona(
    persona: Dict[str, Any] | str | Path,
    adapter: BaseMemorySystem,
    seed: int = 0,
    idle_on: bool = True,
) -> RunResult:
    """
    Drive `adapter` through `persona`, collecting all outputs for scoring.

    Args:
        persona: parsed persona dict OR path to a persona JSON file.
        adapter: the system under test.
        seed: passed to adapter.reset() and used by the persona seed shuffler.
        idle_on: whether to allow autonomous-learning work during idle.
                 False = NIAL ablation; idle() is called but allow_compute=False.
    """
    if not isinstance(persona, dict):
        persona = load_persona(persona)

    adapter.reset(persona_id=persona["persona_id"], seed=seed)
    started = time.monotonic()

    # Build a unified timeline: turns + probe queries, sorted by timestamp.
    events: List[Dict[str, Any]] = []
    for day in persona["days"]:
        for t in day["turns"]:
            events.append({
                "kind": "turn",
                "timestamp_utc": _parse_iso(t["timestamp_utc"]),
                "speaker": t["speaker"],
                "text": t["text"],
                "day_number": day["day_number"],
            })
    for q in persona.get("probe_queries", []):
        events.append({
            "kind": "probe",
            "timestamp_utc": _parse_iso(q["timestamp_utc"]),
            "query_id": q["query_id"],
            "query": q["query"],
            "target_metric": q["target_metric"],
            "scoring": q.get("scoring", {}),
        })
    events.sort(key=lambda e: e["timestamp_utc"])

    # Walk the timeline. Insert simulated idle periods between consecutive
    # events that span more than IDLE_GAP_THRESHOLD_SECONDS of sim time.
    # 6 hours keeps within-day breaks (lunch, meetings) from triggering idles
    # while still capturing afternoon-to-evening, overnight, and weekend gaps.
    IDLE_GAP_THRESHOLD = 60 * 60 * 6  # 6 hours

    idle_events: List[IdleEvent] = []
    probe_outcomes: List[ProbeOutcome] = []
    last_event_time: Optional[datetime] = None
    turns_ingested = 0

    for ev in events:
        sim_now = ev["timestamp_utc"]

        # Detect a substantive gap from the previous event → simulate idle.
        if last_event_time is not None:
            gap_sec = (sim_now - last_event_time).total_seconds()
            if gap_sec >= IDLE_GAP_THRESHOLD:
                idle_event = _do_idle(
                    adapter,
                    start_sim=last_event_time,
                    end_sim=sim_now,
                    allow_compute=idle_on,
                )
                idle_events.append(idle_event)

        # Process the event itself.
        if ev["kind"] == "turn":
            msg = Message(
                text=ev["text"],
                speaker=ev["speaker"],
                timestamp_utc=sim_now,
            )
            adapter.ingest(msg, sim_now)
            if ev["speaker"] == "user":
                turns_ingested += 1

        elif ev["kind"] == "probe":
            qtext = ev["query"]
            target = ev["target_metric"]
            outcome = ProbeOutcome(
                query_id=ev["query_id"],
                target_metric=target,
                sim_time=sim_now,
                query_text=qtext,
                response_text="",
            )
            if qtext == WAKE_SUMMARY_QUERY_TOKEN:
                # Special-case: WSI probe asks for the wake summary.
                since_iso = ev["scoring"].get("since")
                since = _parse_iso(since_iso) if since_iso else last_event_time or sim_now
                ws = adapter.wake_summary(since)
                outcome.wake_summary_returned = ws
                outcome.response_text = (ws.narrative if ws else "")
            else:
                resp = adapter.query(qtext, sim_now)
                outcome.response = resp
                outcome.response_text = resp.text if resp else ""
            probe_outcomes.append(outcome)

        last_event_time = sim_now

    # Final snapshots for scoring.
    final_schemas = list(adapter.list_schemas())
    final_gaps = list(adapter.list_open_questions())
    final_wake_summary = (
        adapter.wake_summary(last_event_time) if last_event_time else None
    )

    capabilities = sorted(c.value for c in adapter.supports())

    return RunResult(
        persona_id=persona["persona_id"],
        system_name=adapter.system_name,
        system_version=adapter.system_version,
        seed=seed,
        idle_on=idle_on,
        started_at_wall=started,
        ended_at_wall=time.monotonic(),
        turns_ingested=turns_ingested,
        idle_events=idle_events,
        probe_outcomes=probe_outcomes,
        final_schemas=final_schemas,
        final_gaps=final_gaps,
        final_wake_summary=final_wake_summary,
        capabilities=capabilities,
    )


def _do_idle(
    adapter: BaseMemorySystem,
    start_sim: datetime,
    end_sim: datetime,
    allow_compute: bool,
) -> IdleEvent:
    """Snapshot before/after, call adapter.idle(), return the event record."""
    schemas_before = list(adapter.list_schemas())
    gaps_before = list(adapter.list_open_questions())

    duration = (end_sim - start_sim).total_seconds()
    report = adapter.idle(
        duration_sim_seconds=duration,
        sim_time=start_sim,
        allow_compute=allow_compute,
    )

    schemas_after = list(adapter.list_schemas())
    gaps_after = list(adapter.list_open_questions())

    return IdleEvent(
        start_sim=start_sim,
        end_sim=end_sim,
        duration_sim_seconds=duration,
        schemas_before=schemas_before,
        schemas_after=schemas_after,
        gaps_before=gaps_before,
        gaps_after=gaps_after,
        report=report,
    )


# ─── Scoring ──────────────────────────────────────────────────────────────


def _index_responses(probes: Sequence[ProbeOutcome]) -> Dict[str, str]:
    """Map query_id → response text, for the keyword-match scorers."""
    return {p.query_id: p.response_text for p in probes}


def _was_autonomous_fn(adapter: BaseMemorySystem):
    """Wrapper so the CGC-fill scorer can ask the adapter post-hoc."""
    def fn(gap_id_or_term: str) -> bool:
        try:
            return bool(adapter.was_autonomous_fill(gap_id_or_term))
        except Exception:
            return False
    return fn


def score_run(
    run: RunResult,
    persona: Dict[str, Any],
    adapter: Optional[BaseMemorySystem] = None,
) -> ScoredRun:
    """
    Compute all metrics for one run.

    Args:
        run: the RunResult from run_persona().
        persona: the persona dict (for ground truth).
        adapter: optional, only needed if CGC-fill scoring needs to ask
            was_autonomous_fill(). If None, autonomous-fill is assumed False.

    Returns:
        ScoredRun with all metric values + raw subscores for the failure ledger.
    """
    gt = persona["ground_truth"]
    responses = _index_responses(run.probe_outcomes)

    # PDR
    pdr_res = score_pdr(gt.get("patterns", []), run.final_schemas)

    # CSS — match cross-session question IDs to their probe responses.
    # The persona's probe with target_metric="CSS" carries scoring.question_id.
    css_responses: Dict[str, str] = {}
    for probe in run.probe_outcomes:
        if probe.target_metric == "CSS":
            # The probe is keyed by query_id, but ground truth uses question_id.
            # Find which CSS question this probe scores by matching on text.
            for q in gt.get("cross_session_questions", []):
                if q["query"].strip().lower() in probe.query_text.strip().lower() \
                   or probe.query_text.strip().lower() in q["query"].strip().lower():
                    css_responses[q["question_id"]] = probe.response_text
                    break
    css_res = score_css(gt.get("cross_session_questions", []), css_responses)

    # CGC-id — uses adapter.list_open_questions() snapshot at end.
    cgc_id_res = score_cgc_id(gt.get("gaps", []), run.final_gaps)

    # CGC-fill — needs the response to a "define X" probe per gap, plus
    # the adapter's was_autonomous_fill verdict.
    fill_responses: Dict[str, str] = {}
    for probe in run.probe_outcomes:
        if probe.target_metric == "CGC_fill":
            gap_id = (probe.response.metadata.get("gap_id")  # type: ignore[union-attr]
                      if probe.response and probe.response.metadata else None)
            # Fallback: persona's scoring config carries gap_id directly.
            if not gap_id:
                # Find via the persona's probe config that produced this query.
                for pq in persona.get("probe_queries", []):
                    if pq["query_id"] == probe.query_id:
                        gap_id = pq.get("scoring", {}).get("gap_id")
                        break
            if gap_id:
                fill_responses[gap_id] = probe.response_text
    cgc_fill_res = score_cgc_fill(
        gt.get("gaps", []),
        fill_responses,
        was_autonomous_fn=_was_autonomous_fn(adapter) if adapter else (lambda _id: False),
    )

    # CRAI
    crai_cur_res = score_crai_current(gt.get("contradictions", []), responses)
    crai_old_res = score_crai_old(gt.get("contradictions", []), responses)

    # WSI — pick the LAST wake-summary probe and score against the schemas
    # actually formed during the corresponding idle window.
    wsi_p = wsi_r = wsi_f = 0.0
    for probe in reversed(run.probe_outcomes):
        if probe.target_metric == "WSI":
            window_event = _find_idle_event_ending_near(run.idle_events, probe.sim_time)
            actually_formed: List[Schema] = []
            if window_event:
                before_ids = {s.schema_id for s in window_event.schemas_before}
                actually_formed = [
                    s for s in window_event.schemas_after
                    if s.schema_id not in before_ids
                ]
            wsi_res = score_wsi(probe.wake_summary_returned, actually_formed)
            wsi_p, wsi_r, wsi_f = wsi_res.precision, wsi_res.recall, wsi_res.f1
            break

    # IMC
    imc_res = aggregate_imc([e.report for e in run.idle_events])

    return ScoredRun(
        persona_id=run.persona_id,
        system_name=run.system_name,
        system_version=run.system_version,
        seed=run.seed,
        idle_on=run.idle_on,
        pdr=pdr_res.score,
        css=css_res.score,
        cgc_id=cgc_id_res.score,
        cgc_fill=cgc_fill_res.score,
        crai_current=crai_cur_res.score,
        crai_old=crai_old_res.score,
        wsi_precision=wsi_p,
        wsi_recall=wsi_r,
        wsi_f1=wsi_f,
        imc_wall_seconds=imc_res.total_wall_seconds,
        imc_cpu_seconds=imc_res.total_cpu_seconds,
        imc_total_idle_periods=imc_res.total_idle_periods,
        raw_subscores={
            "pdr_per_pattern": pdr_res.per_pattern,
            "css_per_question": css_res.per_question,
            "cgc_id_per_gap": cgc_id_res.per_gap,
            "cgc_fill_per_gap": cgc_fill_res.per_gap,
            "crai_current_per_contradiction": crai_cur_res.per_contradiction,
            "crai_old_per_contradiction": crai_old_res.per_contradiction,
        },
    )


def _find_idle_event_ending_near(
    idle_events: Sequence[IdleEvent],
    target_time: datetime,
    tolerance_hours: float = 24.0,
) -> Optional[IdleEvent]:
    """The idle event whose end_sim is closest to target_time, within tolerance."""
    if not idle_events:
        return None
    best = None
    best_dist = None
    tol = tolerance_hours * 3600
    for e in idle_events:
        d = abs((target_time - e.end_sim).total_seconds())
        if d <= tol and (best_dist is None or d < best_dist):
            best = e
            best_dist = d
    return best
