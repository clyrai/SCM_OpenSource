"""
Sleep Trigger: Detects when memory consolidation should begin
Based on entropy levels and conflict density in the memory system
"""
from typing import Dict, Tuple
from datetime import datetime, timedelta
import numpy as np

from ..core.config import (
    SLEEP_ENTROPY_THRESHOLD,
    SLEEP_CONFLICT_THRESHOLD,
    SLEEP_INTERVAL_MAX,
    MICRO_SLEEP_INTERVAL_TURNS,
    MICRO_SLEEP_ENTROPY_THRESHOLD,
    DEEP_SLEEP_MIN_IDLE_SECONDS,
    DEEP_SLEEP_SESSION_TURNS,
    DEEP_SLEEP_PRESSURE_THRESHOLD,
)
from ..core.models import MemoryState
from ..core.time_utils import ensure_utc, utc_now


class SleepTrigger:
    """
    Determines when to initiate sleep consolidation.

    Sleep is triggered when:
    1. Memory entropy exceeds threshold (too many competing patterns)
    2. Conflict density is high (contradicting memories)
    3. Time-based trigger (max interval exceeded)

    Inspired by: Sleep pressure in biological systems (SHY paper)
    """

    def __init__(
        self,
        entropy_threshold: float = SLEEP_ENTROPY_THRESHOLD,
        conflict_threshold: float = SLEEP_CONFLICT_THRESHOLD,
        max_interval: int = SLEEP_INTERVAL_MAX,
        micro_interval_turns: int = MICRO_SLEEP_INTERVAL_TURNS,
        micro_entropy_threshold: float = MICRO_SLEEP_ENTROPY_THRESHOLD,
        deep_min_idle_seconds: int = DEEP_SLEEP_MIN_IDLE_SECONDS,
        deep_session_turns: int = DEEP_SLEEP_SESSION_TURNS,
        deep_pressure_threshold: float = DEEP_SLEEP_PRESSURE_THRESHOLD,
    ):
        self.entropy_threshold = entropy_threshold
        self.conflict_threshold = conflict_threshold
        self.max_interval = max_interval
        self._last_sleep_time: datetime | None = None
        self._last_micro_sleep_time: datetime | None = None
        self._last_deep_sleep_time: datetime | None = None
        self._consecutive_checks = 0
        self.micro_interval_turns = micro_interval_turns
        self.micro_entropy_threshold = micro_entropy_threshold
        self.deep_min_idle_seconds = deep_min_idle_seconds
        self.deep_session_turns = deep_session_turns
        self.deep_pressure_threshold = deep_pressure_threshold

    def should_sleep(
        self,
        concepts: list,
        relations: list,
        recent_conflicts: int = 0,
        time_since_last_sleep: float | None = None
    ) -> Tuple[bool, str]:
        """
        Evaluate if sleep consolidation should begin.

        Returns:
            Tuple of (should_sleep: bool, reason: str)
        """
        if not concepts:
            return False, "No concepts to consolidate"

        entropy = self._compute_entropy(concepts)
        conflict_density = self._compute_conflict_density(relations, recent_conflicts)

        if entropy >= self.entropy_threshold:
            return True, f"High entropy: {entropy:.3f} >= {self.entropy_threshold}"

        if conflict_density >= self.conflict_threshold:
            return True, f"High conflict density: {conflict_density:.3f} >= {self.conflict_threshold}"

        if time_since_last_sleep and time_since_last_sleep >= self.max_interval:
            return True, f"Max interval exceeded: {time_since_last_sleep}s >= {self.max_interval}s"

        self._consecutive_checks = 0
        return False, f"Memory stable (entropy={entropy:.3f}, conflict={conflict_density:.3f})"

    def should_micro_sleep(
        self,
        concepts: list,
        relations: list,
        turns_since_last_micro: int = 0,
        recent_conflicts: int = 0,
    ) -> Tuple[bool, str]:
        """Evaluate whether a lightweight micro-sleep should run."""
        if not concepts:
            return False, "No concepts to consolidate"

        entropy = self._compute_entropy(concepts)
        conflict_density = self._compute_conflict_density(relations, recent_conflicts)

        if entropy >= self.micro_entropy_threshold:
            return True, f"Micro entropy spike: {entropy:.3f} >= {self.micro_entropy_threshold}"

        if turns_since_last_micro >= self.micro_interval_turns:
            return True, f"Micro turn interval reached: {turns_since_last_micro} >= {self.micro_interval_turns}"

        if conflict_density >= (self.conflict_threshold * 0.85):
            return True, f"Micro conflict pressure: {conflict_density:.3f}"

        return False, (
            f"Micro stable (entropy={entropy:.3f}, conflict={conflict_density:.3f}, "
            f"turns={turns_since_last_micro})"
        )

    def should_deep_sleep(
        self,
        concepts: list,
        relations: list,
        time_since_last_deep: float | None = None,
        session_turns: int = 0,
        recent_conflicts: int = 0,
    ) -> Tuple[bool, str]:
        """Evaluate whether full deep-sleep maintenance should run."""
        if not concepts:
            return False, "No concepts to consolidate"

        entropy = self._compute_entropy(concepts)
        conflict_density = self._compute_conflict_density(relations, recent_conflicts)
        pressure = max(entropy, conflict_density)

        if pressure >= self.deep_pressure_threshold:
            return True, f"Deep pressure high: {pressure:.3f} >= {self.deep_pressure_threshold}"

        if time_since_last_deep is not None and time_since_last_deep >= self.deep_min_idle_seconds:
            return True, (
                f"Deep idle window reached: {time_since_last_deep:.1f}s >= {self.deep_min_idle_seconds}s"
            )

        if session_turns >= self.deep_session_turns:
            return True, f"Deep long-session threshold: {session_turns} >= {self.deep_session_turns}"

        return False, (
            f"Deep stable (pressure={pressure:.3f}, idle={time_since_last_deep}, turns={session_turns})"
        )

    def _compute_entropy(self, concepts: list) -> float:
        """
        Compute normalized entropy of concept importance distribution.

        High entropy = concepts have similar importance (competition)
        Low entropy = clear hierarchy of importance
        """
        if len(concepts) < 2:
            return 0.0

        importances = [getattr(c, 'importance', None) for c in concepts]
        importance_values = []

        for imp in importances:
            if imp and hasattr(imp, 'overall'):
                importance_values.append(imp.overall)
            else:
                importance_values.append(0.5)

        importance_values = np.array(importance_values)

        if np.sum(importance_values) == 0:
            return 0.0

        probs = importance_values / np.sum(importance_values)
        probs = probs[probs > 0]

        if len(probs) == 0:
            return 0.0

        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(concepts))
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized

    def _compute_conflict_density(self, relations: list, recent_conflicts: int) -> float:
        """
        Compute conflict density from relations.

        Conflict occurs when:
        - Same subject has multiple contradictory relations
        - Recent interaction conflicts with existing memories
        """
        if not relations:
            return 0.0

        CONTRADICTION_PREDICATES = {'contradicts', 'opposite_of', 'conflicts_with'}
        total_relations = len(relations)

        contradictions = sum(
            1 for r in relations
            if getattr(r, 'predicate', None) in CONTRADICTION_PREDICATES
        )

        base_conflict = contradictions / total_relations if total_relations > 0 else 0.0
        recent_weight = recent_conflicts * 0.1
        density = min(1.0, base_conflict + recent_weight)

        return density

    def record_sleep(self, sleep_time: datetime | None = None, mode: str = "deep"):
        """Record that sleep occurred."""
        recorded = ensure_utc(sleep_time) or utc_now()
        self._last_sleep_time = recorded
        if mode == "micro":
            self._last_micro_sleep_time = recorded
        else:
            self._last_deep_sleep_time = recorded
        self._consecutive_checks = 0

    def time_since_last_sleep(self) -> float | None:
        """Get seconds since last sleep"""
        if self._last_sleep_time is None:
            return None
        return (utc_now() - ensure_utc(self._last_sleep_time)).total_seconds()

    def time_since_last_micro_sleep(self) -> float | None:
        """Get seconds since last micro-sleep."""
        if self._last_micro_sleep_time is None:
            return None
        return (utc_now() - ensure_utc(self._last_micro_sleep_time)).total_seconds()

    def time_since_last_deep_sleep(self) -> float | None:
        """Get seconds since last deep-sleep."""
        if self._last_deep_sleep_time is None:
            return None
        return (utc_now() - ensure_utc(self._last_deep_sleep_time)).total_seconds()

    def get_trigger_stats(self, concepts: list, relations: list) -> Dict:
        """Get detailed trigger statistics for debugging"""
        entropy = self._compute_entropy(concepts)
        conflict = self._compute_conflict_density(relations, 0)
        time_since = self.time_since_last_sleep()

        return {
            "current_entropy": entropy,
            "entropy_threshold": self.entropy_threshold,
            "conflict_density": conflict,
            "conflict_threshold": self.conflict_threshold,
            "time_since_last_sleep": time_since,
            "max_interval": self.max_interval,
            "should_sleep": entropy >= self.entropy_threshold or conflict >= self.conflict_threshold,
            "micro_sleep_interval_turns": self.micro_interval_turns,
            "micro_entropy_threshold": self.micro_entropy_threshold,
            "deep_min_idle_seconds": self.deep_min_idle_seconds,
            "deep_session_turns": self.deep_session_turns,
            "deep_pressure_threshold": self.deep_pressure_threshold,
        }
