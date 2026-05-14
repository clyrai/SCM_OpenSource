"""
LifecyclePolicy — Phase 7 production hardening.

The IdleLearner can technically run sleep cycles whenever sessions go idle.
But on a real user laptop that's not always desirable: deep-sleep on a
4%-battery machine is rude, and aggressive consolidation while the user is
also recording video is worse. This module gives the daemon a pluggable
gate that decides whether work is allowed right now.

Policies are composable and stack-rankable. The default ships as a
"reasonable production policy":
  - Don't run heavy work below MIN_BATTERY_PERCENT
  - Don't run when system CPU is above MAX_CPU_PERCENT
  - Always allow when explicitly plugged in (override flag)

Every threshold is env-configurable. Nothing is hardcoded except the
hooks-of-last-resort defaults required for the dataclass to be valid.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
    _PSUTIL = psutil
except Exception:  # pragma: no cover
    _PSUTIL = None


# ─── Decision result ────────────────────────────────────────────────────────


@dataclass
class PolicyDecision:
    """One policy evaluation result."""
    allowed: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Which policy in a composite chain produced this decision.
    decided_by: str = "policy"


# ─── Abstract base ─────────────────────────────────────────────────────────


class LifecyclePolicy(ABC):
    """A predicate that decides whether the daemon should do work."""

    name: str = "abstract"

    @abstractmethod
    def evaluate(self) -> PolicyDecision:
        """Return whether work is allowed right now, with a reason."""
        ...


# ─── Built-in policies ─────────────────────────────────────────────────────


class AlwaysAllowPolicy(LifecyclePolicy):
    """Default: always permit work. The 'no policy' option."""
    name = "always_allow"

    def evaluate(self) -> PolicyDecision:
        return PolicyDecision(
            allowed=True,
            reason="always_allow_policy",
            decided_by=self.name,
        )


class BatteryPolicy(LifecyclePolicy):
    """
    Block work when on battery and battery percent is below threshold.

    `allow_when_plugged_in=True` (default) means the policy never blocks
    when on AC power, regardless of battery percent.
    """
    name = "battery"

    def __init__(
        self,
        min_battery_percent: float = 50.0,
        allow_when_plugged_in: bool = True,
        require_psutil: bool = False,
    ):
        self.min_battery_percent = float(min_battery_percent)
        self.allow_when_plugged_in = bool(allow_when_plugged_in)
        self.require_psutil = bool(require_psutil)

    def evaluate(self) -> PolicyDecision:
        if _PSUTIL is None:
            # No psutil → can't tell battery state. Default to allow unless
            # the caller explicitly required psutil.
            if self.require_psutil:
                return PolicyDecision(
                    allowed=False,
                    reason="psutil unavailable, blocking conservatively",
                    decided_by=self.name,
                )
            return PolicyDecision(
                allowed=True,
                reason="psutil unavailable, allowing by default",
                decided_by=self.name,
            )
        try:
            batt = _PSUTIL.sensors_battery()
        except Exception:
            return PolicyDecision(
                allowed=True,
                reason="battery sensor unreadable, allowing by default",
                decided_by=self.name,
            )
        if batt is None:
            # No battery (desktop) → always allow
            return PolicyDecision(
                allowed=True,
                reason="no battery (likely desktop)",
                decided_by=self.name,
                metadata={"battery": None},
            )
        if batt.power_plugged and self.allow_when_plugged_in:
            return PolicyDecision(
                allowed=True,
                reason=f"on AC power ({batt.percent}%)",
                decided_by=self.name,
                metadata={"battery_percent": batt.percent, "plugged": True},
            )
        if batt.percent >= self.min_battery_percent:
            return PolicyDecision(
                allowed=True,
                reason=f"battery {batt.percent}% >= {self.min_battery_percent}%",
                decided_by=self.name,
                metadata={"battery_percent": batt.percent, "plugged": batt.power_plugged},
            )
        return PolicyDecision(
            allowed=False,
            reason=f"battery {batt.percent}% < {self.min_battery_percent}%",
            decided_by=self.name,
            metadata={"battery_percent": batt.percent, "plugged": batt.power_plugged},
        )


class CPULoadPolicy(LifecyclePolicy):
    """Block work when system CPU is above threshold."""
    name = "cpu_load"

    def __init__(
        self,
        max_cpu_percent: float = 80.0,
        sample_interval_seconds: float = 0.2,
    ):
        self.max_cpu_percent = float(max_cpu_percent)
        self.sample_interval_seconds = float(sample_interval_seconds)

    def evaluate(self) -> PolicyDecision:
        if _PSUTIL is None:
            return PolicyDecision(
                allowed=True,
                reason="psutil unavailable, allowing by default",
                decided_by=self.name,
            )
        try:
            cpu = _PSUTIL.cpu_percent(interval=self.sample_interval_seconds)
        except Exception:
            return PolicyDecision(
                allowed=True,
                reason="CPU read failed, allowing by default",
                decided_by=self.name,
            )
        if cpu < self.max_cpu_percent:
            return PolicyDecision(
                allowed=True,
                reason=f"cpu {cpu}% < {self.max_cpu_percent}%",
                decided_by=self.name,
                metadata={"cpu_percent": cpu},
            )
        return PolicyDecision(
            allowed=False,
            reason=f"cpu {cpu}% >= {self.max_cpu_percent}%",
            decided_by=self.name,
            metadata={"cpu_percent": cpu},
        )


class CompositePolicy(LifecyclePolicy):
    """
    All sub-policies must allow. Returns the FIRST blocking decision so the
    caller can log the specific reason. If all allow, returns the LAST
    allow decision.
    """
    name = "composite"

    def __init__(self, policies: Optional[List[LifecyclePolicy]] = None):
        self.policies = list(policies or [])

    def evaluate(self) -> PolicyDecision:
        if not self.policies:
            return PolicyDecision(
                allowed=True,
                reason="no policies configured",
                decided_by=self.name,
            )
        last_allow: Optional[PolicyDecision] = None
        for p in self.policies:
            try:
                d = p.evaluate()
            except Exception as exc:
                # A misbehaving policy must NEVER block the agent.
                continue
            if not d.allowed:
                return PolicyDecision(
                    allowed=False,
                    reason=d.reason,
                    decided_by=d.decided_by,
                    metadata=d.metadata,
                )
            last_allow = d
        return last_allow or PolicyDecision(
            allowed=True,
            reason="all policies allowed",
            decided_by=self.name,
        )


# ─── Factory ───────────────────────────────────────────────────────────────


def build_default_policy_from_config() -> LifecyclePolicy:
    """
    Build the production-default lifecycle policy from env-driven config.
    Fully overridable: every threshold, including the master switch, is in
    src/core/config.py and exposed via .env.example.
    """
    from ..core.config import (
        LIFECYCLE_POLICY_ENABLED,
        LIFECYCLE_POLICY_MIN_BATTERY_PERCENT,
        LIFECYCLE_POLICY_ALLOW_WHEN_PLUGGED_IN,
        LIFECYCLE_POLICY_MAX_CPU_PERCENT,
        LIFECYCLE_POLICY_CPU_SAMPLE_SECONDS,
        LIFECYCLE_POLICY_REQUIRE_PSUTIL,
    )
    if not LIFECYCLE_POLICY_ENABLED:
        return AlwaysAllowPolicy()
    return CompositePolicy([
        BatteryPolicy(
            min_battery_percent=LIFECYCLE_POLICY_MIN_BATTERY_PERCENT,
            allow_when_plugged_in=LIFECYCLE_POLICY_ALLOW_WHEN_PLUGGED_IN,
            require_psutil=LIFECYCLE_POLICY_REQUIRE_PSUTIL,
        ),
        CPULoadPolicy(
            max_cpu_percent=LIFECYCLE_POLICY_MAX_CPU_PERCENT,
            sample_interval_seconds=LIFECYCLE_POLICY_CPU_SAMPLE_SECONDS,
        ),
    ])
