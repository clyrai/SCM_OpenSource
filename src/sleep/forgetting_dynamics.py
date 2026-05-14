"""
ForgettingDynamics: adaptive memory retention, decay, and state transitions.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import math
from typing import Dict, List, Optional, Tuple

from ..core.config import (
    FORGETTING_ARCHIVE_THRESHOLD,
    FORGETTING_BASE_DECAY,
    FORGETTING_MIN_REHEARSAL_BEFORE_ARCHIVE,
    FORGETTING_PROTECT_SALIENCE,
    FORGETTING_SUPPRESS_THRESHOLD,
    FORGETTING_WEIGHT_ASSOCIATION,
    FORGETTING_WEIGHT_GRASP,
    FORGETTING_WEIGHT_INTERFERENCE,
    FORGETTING_WEIGHT_RECENCY,
    FORGETTING_WEIGHT_REHEARSAL,
    FORGETTING_WEIGHT_SALIENCE,
)
from ..core.models import Concept, MemoryState
from ..core.memory_scoring import refresh_consolidation_score
from ..core.time_utils import ensure_utc, utc_now


class ForgettingDynamics:
    """
    Adaptive forgetting that keeps strong traces alive and retires weak ones.

    Phase 5 behaviors:
    - retention scoring from grasp, salience, rehearsal, association density, recency, and interference
    - trace-specific decay rates
    - active -> suppressed -> archived state transitions
    - contradiction-safe handling of superseded versions
    """

    def __init__(
        self,
        grasp_weight: float = FORGETTING_WEIGHT_GRASP,
        salience_weight: float = FORGETTING_WEIGHT_SALIENCE,
        rehearsal_weight: float = FORGETTING_WEIGHT_REHEARSAL,
        association_weight: float = FORGETTING_WEIGHT_ASSOCIATION,
        recency_weight: float = FORGETTING_WEIGHT_RECENCY,
        interference_weight: float = FORGETTING_WEIGHT_INTERFERENCE,
        suppress_threshold: float = FORGETTING_SUPPRESS_THRESHOLD,
        archive_threshold: float = FORGETTING_ARCHIVE_THRESHOLD,
        base_decay: float = FORGETTING_BASE_DECAY,
        protect_salience: float = FORGETTING_PROTECT_SALIENCE,
        min_rehearsal_before_archive: int = FORGETTING_MIN_REHEARSAL_BEFORE_ARCHIVE,
        freshness_floor_hours: float = 1.0,
        freshness_importance_min: float = 0.20,
        importance_threshold: Optional[float] = None,
        decay_rate: Optional[float] = None,
        temporal_weight: Optional[float] = None,
        interference_factor: Optional[float] = None,
    ):
        if importance_threshold is not None:
            suppress_threshold = importance_threshold
        if decay_rate is not None:
            base_decay = decay_rate
        if temporal_weight is not None:
            recency_weight = temporal_weight
        if interference_factor is not None:
            interference_weight = interference_factor

        self.grasp_weight = grasp_weight
        self.salience_weight = salience_weight
        self.rehearsal_weight = rehearsal_weight
        self.association_weight = association_weight
        self.recency_weight = recency_weight
        self.interference_weight = interference_weight
        self.suppress_threshold = suppress_threshold
        self.archive_threshold = archive_threshold
        self.base_decay = base_decay
        # Phase 6 safety net: protect concepts above this salience floor from
        # state demotion. Default 0.0 keeps behavior identical to pre-fix code.
        self.protect_salience = max(0.0, min(1.0, protect_salience))
        self.min_rehearsal_before_archive = max(0, int(min_rehearsal_before_archive))
        # Freshness floor: substantive concepts younger than this stay ACTIVE.
        # Set hours <= 0 to disable (used by legacy tests that probe pure
        # forgetting dynamics without product-grade protection).
        self.freshness_floor_hours = max(0.0, float(freshness_floor_hours))
        self.freshness_importance_min = max(0.0, min(1.0, float(freshness_importance_min)))

    def apply(
        self,
        concepts: List[Concept],
        conflict_pairs: Optional[List[Tuple[str, str]]] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[List[Concept], Dict]:
        """
        Update retention, decay, and state for a batch of concepts.
        """
        now = ensure_utc(now) or utc_now()
        conflict_pairs = conflict_pairs or []
        conflict_counts = Counter()
        for left, right in conflict_pairs:
            conflict_counts[left] += 1
            conflict_counts[right] += 1

        stats = {
            "total_evaluated": len(concepts),
            "retained": 0,
            "suppressed": 0,
            "archived": 0,
            "forgotten": 0,
            "conflicts_resolved": len(conflict_pairs),
            "avg_retention": 0.0,
            "avg_decay_lambda": 0.0,
            "conflict_pairs": len(conflict_pairs),
        }

        updated: List[Concept] = []
        retention_scores: List[float] = []
        decay_lambdas: List[float] = []

        for concept in concepts:
            retention, interference = self.compute_retention_score(
                concept=concept,
                conflict_counts=conflict_counts,
                now=now,
            )
            concept.retention_score = retention
            concept.decay_rate = self.compute_decay_lambda(
                concept=concept,
                retention_score=retention,
                interference=interference,
            )

            age_hours = self._age_hours(concept, now)
            decay_factor = math.exp(-concept.decay_rate * max(0.25, age_hours / 24.0))
            concept.strength = max(0.05, concept.strength * decay_factor)

            self._transition_state(concept, retention, now)
            refresh_consolidation_score(concept, now=now)

            retention_scores.append(retention)
            decay_lambdas.append(concept.decay_rate)
            state_value = self._state_value(concept)
            if state_value == MemoryState.ACTIVE.value:
                stats["retained"] += 1
            elif state_value == MemoryState.SUPPRESSED.value:
                stats["suppressed"] += 1
            else:
                stats["archived"] += 1

            updated.append(concept)

        if retention_scores:
            stats["avg_retention"] = round(sum(retention_scores) / len(retention_scores), 4)
        if decay_lambdas:
            stats["avg_decay_lambda"] = round(sum(decay_lambdas) / len(decay_lambdas), 4)

        stats["forgotten"] = stats["suppressed"] + stats["archived"]
        return updated, stats

    def evaluate_forgetting(
        self,
        concepts: List[Concept],
        conflict_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Compatibility wrapper used by the sleep pipeline.
        """
        updated, stats = self.apply(concepts, conflict_pairs=conflict_pairs)
        forgotten_ids = [
            c.id
            for c in updated
            if self._state_value(c) != MemoryState.ACTIVE.value
        ]
        preserve_ids = [
            c.id
            for c in updated
            if self._state_value(c) == MemoryState.ACTIVE.value
        ]
        return forgotten_ids, preserve_ids, stats

    def compute_retention_score(
        self,
        concept: Concept,
        conflict_counts: Optional[Counter] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """
        Compute retention score using the Phase 5 formula.

        R = p1*Grasp + p2*Salience + p3*Rehearsal + p4*AssociationDensity
            + p5*Recency - p6*Interference
        """
        now = ensure_utc(now) or utc_now()
        conflict_counts = conflict_counts or Counter()

        grasp = self._clamp(self._concept_value(concept, "grasp_score", fallback_importance=True))
        salience = self._clamp(self._concept_value(concept, "salience_score", fallback_importance=True))
        rehearsal = self._clamp(min(1.0, getattr(concept, "rehearsal_count", 0) / 8.0))
        association = self._clamp(getattr(concept, "association_density", 0.0))
        recency = self._recency_score(concept, now)
        interference = self._interference_score(concept, conflict_counts)

        retention = (
            self.grasp_weight * grasp
            + self.salience_weight * salience
            + self.rehearsal_weight * rehearsal
            + self.association_weight * association
            + self.recency_weight * recency
            - self.interference_weight * interference
        )
        return self._clamp(retention), interference

    def compute_decay_lambda(
        self,
        concept: Concept,
        retention_score: float,
        interference: float = 0.0,
    ) -> float:
        """Compute a trace-specific decay rate."""
        retention_gap = 1.0 - self._clamp(retention_score)
        version_penalty = 0.15 if (not getattr(concept, "is_current_version", True)) else 0.0
        state_penalty = 0.10 if self._state_value(concept) == MemoryState.SUPPRESSED.value else 0.0
        decay_lambda = self.base_decay * (
            1.0
            + retention_gap * 1.5
            + interference * 0.8
            + version_penalty
            + state_penalty
        )
        return max(0.005, round(decay_lambda, 4))

    def compute_forgetting_threshold(
        self,
        concepts: List[Concept],
        target_forget_rate: float = 0.1,
    ) -> float:
        """
        Adaptive threshold that rises with memory load.
        """
        if not concepts:
            return self.suppress_threshold

        load_factor = min(1.0, len(concepts) / 100.0)
        threshold = self.suppress_threshold + load_factor * 0.18 + target_forget_rate * 0.05
        return min(0.85, round(threshold, 4))

    def get_forgetting_stats(self, concepts: List[Concept]) -> Dict:
        """Return aggregate forgetting diagnostics."""
        if not concepts:
            return {
                "total_concepts": 0,
                "forgettable": 0,
                "preserve": 0,
                "avg_retention": 0.0,
                "avg_decay": 0.0,
                "current_threshold": self.suppress_threshold,
            }

        retentions = []
        decays = []
        for concept in concepts:
            retention, _ = self.compute_retention_score(concept)
            decays.append(self.compute_decay_lambda(concept, retention))
            retentions.append(retention)

        threshold = self.compute_forgetting_threshold(concepts)
        forgettable = sum(1 for retention in retentions if retention < threshold)
        return {
            "total_concepts": len(concepts),
            "forgettable": forgettable,
            "preserve": len(concepts) - forgettable,
            "avg_retention": round(sum(retentions) / len(retentions), 4),
            "avg_decay": round(sum(decays) / len(decays), 4),
            "current_threshold": threshold,
        }

    def _transition_state(self, concept: Concept, retention: float, now: datetime) -> None:
        """
        Move a concept through the active/suppressed/archived lifecycle.

        Phase 6 protections (only active when protect_salience > 0):
          - Salience floor: concepts with salience >= protect_salience are kept ACTIVE
          - Rehearsal floor: fresh concepts (rehearsal < min) are not archived on
            their first sleep cycle, only suppressed
        """
        state_value = self._state_value(concept)
        association_density = getattr(concept, "association_density", 0.0)
        importance = concept.importance.overall if concept.importance else 0.0
        salience = getattr(concept, "salience_score", 0.0) or 0.0
        rehearsal = getattr(concept, "rehearsal_count", 0) or 0

        # Versioned (superseded) concepts always archive — protections do not apply.
        if not getattr(concept, "is_current_version", True):
            concept.state = MemoryState.ARCHIVED.value
            return

        if state_value == MemoryState.ARCHIVED.value:
            return

        # Freshness floor: a substantive concept the user just stated MUST
        # be retrievable. Anything younger than `freshness_floor_hours` AND
        # with non-trivial importance stays ACTIVE — a memory product that
        # forgets things you said 5 minutes ago is broken. Pure noise
        # (importance < freshness_importance_min) is NOT protected. Tests
        # can disable this floor entirely with freshness_floor_hours=0.
        # Note: importance.overall has a 0.10 baseline from `(emotional+1)/2`,
        # so the threshold default must be > 0.20 to actually exclude noise.
        if self.freshness_floor_hours > 0.0:
            age_hours = self._age_hours(concept, now)
            if age_hours < self.freshness_floor_hours and importance >= self.freshness_importance_min:
                concept.state = MemoryState.ACTIVE.value
                return

        # Salience floor: high-salience concepts cannot be demoted.
        if self.protect_salience > 0.0 and salience >= self.protect_salience:
            concept.state = MemoryState.ACTIVE.value
            return

        if (
            state_value == MemoryState.SUPPRESSED.value
            and association_density >= 0.4
            and importance < 0.7
        ):
            concept.state = MemoryState.ARCHIVED.value
            return

        if retention >= self.suppress_threshold:
            concept.state = MemoryState.ACTIVE.value
            return

        if retention >= self.archive_threshold:
            concept.state = MemoryState.SUPPRESSED.value
            return

        # Rehearsal floor: don't archive concepts that haven't survived a cycle.
        if rehearsal < self.min_rehearsal_before_archive:
            concept.state = MemoryState.SUPPRESSED.value
            return

        if state_value == MemoryState.SUPPRESSED.value:
            concept.state = MemoryState.ARCHIVED.value
        else:
            concept.state = MemoryState.SUPPRESSED.value

    def _interference_score(self, concept: Concept, conflict_counts: Counter) -> float:
        score = 0.0
        score += min(1.0, conflict_counts.get(concept.id, 0) * 0.15)
        if getattr(concept, "version_parent", None):
            score += 0.20 if not getattr(concept, "is_current_version", True) else 0.05
        if self._state_value(concept) == MemoryState.SUPPRESSED.value:
            score += 0.10
        if self._state_value(concept) == MemoryState.ARCHIVED.value:
            score += 0.20
        return self._clamp(score)

    def _recency_score(self, concept: Concept, now: datetime) -> float:
        last_accessed = getattr(concept, "last_accessed", None)
        if not last_accessed:
            return 0.5
        age_hours = max(0.0, self._age_hours(concept, now))
        if age_hours <= 1.0:
            return 1.0
        if age_hours >= 72.0:
            return 0.1
        return self._clamp(1.0 - (age_hours - 1.0) / 71.0)

    def _age_hours(self, concept: Concept, now: datetime) -> float:
        last_accessed = getattr(concept, "last_accessed", None)
        if not last_accessed:
            return 0.0
        last_accessed = ensure_utc(last_accessed)
        if not isinstance(last_accessed, datetime):
            return 0.0
        return max(0.0, (now - last_accessed).total_seconds() / 3600.0)

    @staticmethod
    def _concept_value(concept: Concept, field_name: str, fallback_importance: bool = False) -> float:
        value = getattr(concept, field_name, None)
        if value is not None:
            value = float(value)
            if fallback_importance and abs(value - 0.5) < 1e-6:
                importance = getattr(concept, "importance", None)
                if importance is not None:
                    is_generic_trace = (
                        getattr(concept, "rehearsal_count", 0) == 0
                        and getattr(concept, "activation_count", 0) == 0
                        and getattr(concept, "association_density", 0.0) <= 0.05
                    )
                    if is_generic_trace:
                        return float(importance.overall)
            return value
        if fallback_importance and getattr(concept, "importance", None):
            return float(concept.importance.overall)
        return 0.5

    @staticmethod
    def _state_value(concept: Concept) -> str:
        state = getattr(concept, "state", MemoryState.ACTIVE.value)
        return state.value if hasattr(state, "value") else str(state)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
