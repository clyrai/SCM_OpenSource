"""
Idle Maintenance Cost (IMC).

Aggregate cost stats over the idle periods of a single run.
Reported as a frontier alongside quality metrics — there's no "best,"
just a cost/benefit profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class IMCResult:
    """Per-run cost aggregate."""
    total_idle_periods: int
    total_wall_seconds: float
    total_cpu_seconds: float
    total_sleep_cycles: int
    peak_rss_bytes: int
    mean_wall_per_period: float
    mean_cpu_per_period: float


def aggregate_imc(idle_reports: Sequence) -> IMCResult:
    """
    Aggregate over the IdleReports collected during one run.

    Inputs are adapter-side IdleReport instances (see adapters.base).
    """
    n = len(idle_reports)
    if n == 0:
        return IMCResult(
            total_idle_periods=0,
            total_wall_seconds=0.0,
            total_cpu_seconds=0.0,
            total_sleep_cycles=0,
            peak_rss_bytes=0,
            mean_wall_per_period=0.0,
            mean_cpu_per_period=0.0,
        )

    total_wall = sum(getattr(r, "wall_clock_seconds", 0.0) for r in idle_reports)
    total_cpu = sum(getattr(r, "cpu_seconds", 0.0) for r in idle_reports)
    total_cycles = sum(getattr(r, "sleep_cycles_fired", 0) for r in idle_reports)
    peak_rss = max((getattr(r, "peak_rss_bytes", 0) for r in idle_reports), default=0)

    return IMCResult(
        total_idle_periods=n,
        total_wall_seconds=total_wall,
        total_cpu_seconds=total_cpu,
        total_sleep_cycles=total_cycles,
        peak_rss_bytes=peak_rss,
        mean_wall_per_period=total_wall / n,
        mean_cpu_per_period=total_cpu / n,
    )
