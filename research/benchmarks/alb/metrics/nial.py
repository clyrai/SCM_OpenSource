"""
No-Idle Ablation Lift (NIAL).

For each metric, the difference between the system's score with idle
processing enabled vs disabled.

Used to answer: "Is the autonomous-learning machinery actually doing
useful work, or is the score attributable to retrieval alone?"

Floor systems (no idle work to disable) should have NIAL ≈ 0 across
metrics — that's the ablation control showing the lift is real for
systems that actually use idle time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class NIALResult:
    """Per-metric lift from disabling idle processing.

    Positive lift = idle processing helps that metric.
    Negative lift = idle processing hurts that metric (also informative).
    Zero lift = system is invariant to idle processing on this metric.
    """
    by_metric: Dict[str, float]            # metric_name → absolute lift
    by_metric_relative: Dict[str, float]   # metric_name → relative lift (idle-on / max(idle-off, eps))


EPS = 1e-6


def compute_nial(
    scores_idle_on: Dict[str, float],
    scores_idle_off: Dict[str, float],
) -> NIALResult:
    """
    Compute lift across metrics. Both inputs are dicts: metric_name → score.

    Metrics present in only one map are ignored (NIAL is paired by design).
    """
    metrics = set(scores_idle_on) & set(scores_idle_off)
    abs_lift: Dict[str, float] = {}
    rel_lift: Dict[str, float] = {}
    for m in sorted(metrics):
        on = float(scores_idle_on[m])
        off = float(scores_idle_off[m])
        abs_lift[m] = on - off
        rel_lift[m] = (on - off) / max(off, EPS)
    return NIALResult(by_metric=abs_lift, by_metric_relative=rel_lift)
