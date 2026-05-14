"""
Contradiction Resolution Across Idle (CRAI).

Two sub-metrics:
  CRAI-current: when asked the current value, return new_value (not old_value).
  CRAI-old:     when asked the prior value, return old_value (versioning systems).

Systems without versioning will score 0 on CRAI-old. This is correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


@dataclass
class CRAIResult:
    score: float
    total_contradictions: int
    correct: int
    per_contradiction: List[Dict[str, Any]]


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle) and needle.lower() in (haystack or "").lower()


def score_crai_current(
    contradictions: Sequence[Mapping[str, Any]],
    responses_by_query_id: Mapping[str, str],
) -> CRAIResult:
    """
    Score the "current value" probe per contradiction.

    A contradiction is correctly resolved iff the response contains
    new_value AND does NOT contain old_value.
    """
    if not contradictions:
        return CRAIResult(score=0.0, total_contradictions=0, correct=0, per_contradiction=[])

    per = []
    correct = 0
    for c in contradictions:
        qid = c.get("current_probe_query_id", "")
        resp = responses_by_query_id.get(qid, "")
        new_v = c.get("new_value", "")
        old_v = c.get("old_value", "")
        ok = _contains(resp, new_v) and not _contains(resp, old_v)
        if ok:
            correct += 1
        per.append({
            "contradiction_id": c.get("contradiction_id"),
            "property": c.get("property"),
            "expected_new": new_v,
            "must_not_contain": old_v,
            "passed": ok,
            "response_excerpt": (resp or "")[:200],
        })

    return CRAIResult(
        score=correct / len(contradictions),
        total_contradictions=len(contradictions),
        correct=correct,
        per_contradiction=per,
    )


def score_crai_old(
    contradictions: Sequence[Mapping[str, Any]],
    responses_by_query_id: Mapping[str, str],
) -> CRAIResult:
    """
    Score the "previous value" probe per contradiction.

    Correct iff response contains old_value. Versioning systems score
    here; non-versioning systems will be near 0.
    """
    if not contradictions:
        return CRAIResult(score=0.0, total_contradictions=0, correct=0, per_contradiction=[])

    per = []
    correct = 0
    for c in contradictions:
        qid = c.get("old_probe_query_id", "")
        resp = responses_by_query_id.get(qid, "")
        old_v = c.get("old_value", "")
        ok = _contains(resp, old_v)
        if ok:
            correct += 1
        per.append({
            "contradiction_id": c.get("contradiction_id"),
            "property": c.get("property"),
            "expected_old": old_v,
            "passed": ok,
            "response_excerpt": (resp or "")[:200],
        })

    return CRAIResult(
        score=correct / len(contradictions),
        total_contradictions=len(contradictions),
        correct=correct,
        per_contradiction=per,
    )
