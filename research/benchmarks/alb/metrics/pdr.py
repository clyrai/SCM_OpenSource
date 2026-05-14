"""
Pattern Discovery Rate (PDR).

What: of the patterns planted in the persona, how many does the system
have in its abstraction set at the relevant probe time?

How: for each ground-truth pattern, run match_pattern() against the
schemas the system reports. Score = matched / total.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .match import match_pattern


@dataclass
class PDRResult:
    """One persona × system PDR scoring."""
    score: float                              # in [0, 1]
    total_patterns: int
    matched_patterns: int
    per_pattern: List[Dict[str, Any]]         # for the failure ledger


def score_pdr(
    ground_truth_patterns: Sequence[Dict[str, Any]],
    reported_schemas: Sequence[Any],
) -> PDRResult:
    """
    Score PDR for one (persona, system) pair.

    `reported_schemas` is the list returned by adapter.list_schemas() at
    scoring time. The matcher tolerates any system's signature shape.
    """
    if not ground_truth_patterns:
        return PDRResult(score=0.0, total_patterns=0, matched_patterns=0, per_pattern=[])

    per_pattern = []
    matched = 0
    for gt in ground_truth_patterns:
        sid, sub = match_pattern(gt, reported_schemas)
        hit = sid is not None and sub > 0
        if hit:
            matched += 1
        per_pattern.append({
            "pattern_id": gt.get("pattern_id"),
            "type": gt.get("type"),
            "matched": hit,
            "matched_schema_id": sid,
        })

    return PDRResult(
        score=matched / len(ground_truth_patterns),
        total_patterns=len(ground_truth_patterns),
        matched_patterns=matched,
        per_pattern=per_pattern,
    )
