"""
Wake-Summary Informativeness (WSI).

What: when the user "returns from idle," does the system report what it
learned, and is the report accurate against what the system actually did?

How: compare adapter.wake_summary(since) against the ground-truth diff
of (schemas, gaps, contradictions) between the start and end of the idle
window. Compute precision, recall, F1.

Note: WSI compares the summary against what the SYSTEM ACTUALLY DID
during idle (recorded by the runner via list_schemas() snapshots
before/after each idle), not against what the persona expected it to
do. This isolates "narrative fidelity" from "capability."

Systems without a wake-summary endpoint score 0 across the board.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class WSIResult:
    """Precision / recall / F1 over the schemas reported in the summary."""
    precision: float
    recall: float
    f1: float
    summary_present: bool
    reported_schema_count: int
    actual_schema_count: int
    matched_count: int
    notes: str = ""


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def score_wsi(
    summary: Optional[Any],
    actual_schemas_formed_in_window: Sequence[Any],
) -> WSIResult:
    """
    Args:
        summary: WakeSummary or None.
        actual_schemas_formed_in_window: schemas the system formed during
            this idle period. Computed by the runner as the set diff
            between list_schemas() before and after.
    """
    actual_ids = {getattr(s, "schema_id", None) for s in actual_schemas_formed_in_window}
    actual_ids.discard(None)
    n_actual = len(actual_ids)

    if summary is None:
        return WSIResult(
            precision=0.0, recall=0.0, f1=0.0,
            summary_present=False,
            reported_schema_count=0,
            actual_schema_count=n_actual,
            matched_count=0,
            notes="no wake_summary endpoint",
        )

    reported = list(getattr(summary, "schemas_formed", []) or [])
    reported_ids = {getattr(s, "schema_id", None) for s in reported}
    reported_ids.discard(None)

    matched = reported_ids & actual_ids
    n_reported = len(reported_ids)
    n_matched = len(matched)

    precision = n_matched / n_reported if n_reported else 0.0
    recall = n_matched / n_actual if n_actual else 0.0

    return WSIResult(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        summary_present=True,
        reported_schema_count=n_reported,
        actual_schema_count=n_actual,
        matched_count=n_matched,
    )
