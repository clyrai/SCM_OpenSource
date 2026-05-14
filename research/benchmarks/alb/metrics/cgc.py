"""
Curiosity Gap Coverage (CGC).

Two sub-metrics:
  CGC-id:    of planted gaps, how many does the system *identify* as open?
  CGC-fill:  of planted gaps, how many does the system fill *autonomously*?

Floor: a system without gap-tracking scores 0 on CGC-id (correct).
Floor: a system without an autonomous-fill mechanism scores 0 on CGC-fill
(correct — they answer reactively, not autonomously).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .match import score_keyword_response


@dataclass
class CGCIdResult:
    score: float
    total_gaps: int
    identified_gaps: int
    per_gap: List[Dict[str, Any]]


@dataclass
class CGCFillResult:
    score: float
    total_gaps: int
    autonomously_filled: int
    per_gap: List[Dict[str, Any]]


def _term_match(gt_term: str, candidate_term: str) -> bool:
    """Loose match: case-insensitive substring either direction."""
    a = (gt_term or "").lower().strip()
    b = (candidate_term or "").lower().strip()
    return bool(a) and bool(b) and (a in b or b in a)


def score_cgc_id(
    ground_truth_gaps: Sequence[Mapping[str, Any]],
    reported_gaps: Sequence[Any],
) -> CGCIdResult:
    """
    `reported_gaps` is what adapter.list_open_questions() returned.

    A gap is "identified" iff some reported gap matches the planted term.
    """
    if not ground_truth_gaps:
        return CGCIdResult(score=0.0, total_gaps=0, identified_gaps=0, per_gap=[])

    per = []
    found = 0
    for g in ground_truth_gaps:
        gt_term = g.get("term", "")
        match = None
        for r in reported_gaps:
            r_term = getattr(r, "term", "") or ""
            if _term_match(gt_term, r_term):
                match = r
                break
        ok = match is not None
        if ok:
            found += 1
        per.append({
            "gap_id": g.get("gap_id"),
            "term": gt_term,
            "identified": ok,
        })

    return CGCIdResult(
        score=found / len(ground_truth_gaps),
        total_gaps=len(ground_truth_gaps),
        identified_gaps=found,
        per_gap=per,
    )


def score_cgc_fill(
    ground_truth_gaps: Sequence[Mapping[str, Any]],
    fill_responses_by_gap_id: Mapping[str, str],
    was_autonomous_fn: Callable[[str], bool],
) -> CGCFillResult:
    """
    Score the autonomous-fill rate.

    Args:
        ground_truth_gaps: planted gaps from persona ground truth.
        fill_responses_by_gap_id: maps gap_id → adapter.query("define X") response.
        was_autonomous_fn: callable, takes a gap_id or term, returns True iff
            the system filled this gap autonomously during idle (NOT reactively
            in response to query()). For systems without curiosity, this is
            always False; CGC-fill then correctly scores 0.

    A gap is "filled" iff:
      (a) the response matches enough definition keywords, AND
      (b) was_autonomous_fn returns True.

    (b) is the strict gate that prevents reactive-only systems from claiming
    autonomous-fill credit.
    """
    if not ground_truth_gaps:
        return CGCFillResult(score=0.0, total_gaps=0, autonomously_filled=0, per_gap=[])

    per = []
    filled = 0
    for g in ground_truth_gaps:
        gid = g.get("gap_id", "")
        resp = fill_responses_by_gap_id.get(gid, "")
        keywords = g.get("expected_definition_keywords", [])
        min_match = g.get("min_keyword_match", 2)

        kw_score = score_keyword_response(
            resp,
            must_contain_any_of=keywords,
            must_not_contain=[],
            min_match_count=min_match,
        )
        autonomous = bool(was_autonomous_fn(gid)) if was_autonomous_fn else False

        passed = (kw_score >= 1.0) and autonomous
        if passed:
            filled += 1
        per.append({
            "gap_id": gid,
            "term": g.get("term"),
            "keyword_match_passed": kw_score >= 1.0,
            "autonomous_fill": autonomous,
            "passed": passed,
            "response_excerpt": (resp or "")[:200],
        })

    return CGCFillResult(
        score=filled / len(ground_truth_gaps),
        total_gaps=len(ground_truth_gaps),
        autonomously_filled=filled,
        per_gap=per,
    )
