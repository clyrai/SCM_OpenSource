from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

from ..core.models import Concept, MemoryState
from ..core.memory_scoring import compute_consolidation_score
from ..core.time_utils import utc_now
from ..core import config as _config_module


class HypothesisConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ScoredHypothesis:
    concept: Concept
    hypothesis_score: float
    confidence: HypothesisConfidence
    evidence: Dict
    rank: int


@dataclass
class HypothesisSet:
    hypotheses: List[ScoredHypothesis]
    ensemble_score: float
    confidence: HypothesisConfidence
    coverage: float
    total_evidence_count: int


class HypothesisRanker:
    """
    Ranks activated concepts into scored hypotheses with confidence.

    Scoring dimensions:
    - Activation strength (from spreading activation)
    - Recency (time since last access)
    - Association density (how connected this concept is)
    - Importance (salience + grasp baked in)
    - Rehearsal count (repeated exposure = stronger)
    - Contradiction flag (reduces confidence if unresolved)

    Confidence thresholds:
    - HIGH:   score >= 0.70
    - MEDIUM: 0.40 <= score < 0.70
    - LOW:    0.15 <= score < 0.40
    - NONE:   score < 0.15 (concept excluded)

    No LLM calls in the critical path.
    """

    def __init__(
        self,
        activation_weight: float = 0.30,
        recency_weight: float = 0.15,
        density_weight: float = 0.15,
        importance_weight: float = 0.25,
        rehearsal_weight: float = 0.10,
        contradiction_penalty: float = 0.40,
        consolidation_weight: float = 0.08,
        high_confidence_threshold: float = 0.70,
        medium_confidence_threshold: float = 0.40,
        low_confidence_threshold: float = 0.15,
        max_hypotheses: int = 10,
    ):
        self.activation_weight = activation_weight
        self.recency_weight = recency_weight
        self.density_weight = density_weight
        self.importance_weight = importance_weight
        self.rehearsal_weight = rehearsal_weight
        self.contradiction_penalty = contradiction_penalty
        self.consolidation_weight = consolidation_weight
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.max_hypotheses = max_hypotheses

    def rank(
        self,
        activated_concepts: List[Concept],
        activation_map: Optional[Dict[str, float]] = None,
        context_tags: Optional[Dict] = None,
    ) -> HypothesisSet:
        """
        Main entry: convert activated concepts into ranked hypotheses.

        Args:
            activated_concepts: Concepts from spreading activation (ordered by activation).
            activation_map: Optional dict mapping concept_id -> activation score.
            context_tags: Optional context for relevance scoring.

        Returns:
            HypothesisSet with scored hypotheses, ensemble score, confidence, coverage.
        """
        context_tags = context_tags or {}
        activation_map = activation_map or {}
        now = utc_now()

        if not activated_concepts:
            return self._empty_set()

        scored = []
        for concept in activated_concepts:
            act_score = activation_map.get(concept.id, 0.0)
            rank_score, evidence = self._compute_hypothesis_score(
                concept, act_score, context_tags, now
            )
            confidence = self._classify_confidence(rank_score)

            if confidence == HypothesisConfidence.NONE:
                continue

            scored.append(ScoredHypothesis(
                concept=concept,
                hypothesis_score=rank_score,
                confidence=confidence,
                evidence=evidence,
                rank=0,
            ))

        scored.sort(key=lambda h: h.hypothesis_score, reverse=True)
        for i, h in enumerate(scored):
            h.rank = i + 1

        top = scored[:self.max_hypotheses]

        ensemble = self._ensemble_score(top)
        coverage = self._coverage_score(top, activated_concepts)
        total_evidence = sum(len(h.evidence) for h in top)
        set_confidence = self._ensemble_confidence(top, ensemble)

        return HypothesisSet(
            hypotheses=top,
            ensemble_score=ensemble,
            confidence=set_confidence,
            coverage=coverage,
            total_evidence_count=total_evidence,
        )

    def format_context(
        self,
        hypothesis_set: HypothesisSet,
        section_title: str = "Retrieved Memories",
        include_evidence: bool = False,
    ) -> str:
        """
        Format a hypothesis set into a readable context string.

        Args:
            hypothesis_set: Output from rank().
            section_title: Human-readable section heading.
            include_evidence: If True, include per-hypothesis evidence notes.

        Returns:
            Formatted multi-line string suitable for prompt injection.
        """
        if not hypothesis_set.hypotheses:
            return f"## {section_title}\n_No strong memory matches found._"

        lines = [f"## {section_title}"]
        conf_label = hypothesis_set.confidence.value.upper()
        lines[0] += f" (confidence: {conf_label}, coverage: {hypothesis_set.coverage:.0%})"

        for h in hypothesis_set.hypotheses:
            rank_marker = f"[Rank {h.rank}]" if h.rank else ""
            conf_marker = f"({h.confidence.value})"
            imp = h.concept.importance.overall if h.concept.importance else 0.0
            base = f"- {h.concept.description} {rank_marker} {conf_marker} [salience={imp:.2f}]"

            if include_evidence and h.evidence:
                ev_parts = [f"{k}={v}" for k, v in h.evidence.items() if v]
                if ev_parts:
                    base += f" | {', '.join(ev_parts)}"
            lines.append(base)

        return "\n".join(lines)

    def _compute_hypothesis_score(
        self,
        concept: Concept,
        activation_score: float,
        context_tags: Dict,
        now,
    ) -> Tuple[float, Dict]:
        """Compute composite hypothesis score and collect evidence."""
        evidence = {}

        act_component = activation_score
        evidence['activation'] = round(act_component, 3)

        recency = self._recency_score(concept.last_accessed)
        recency_component = recency * self.recency_weight
        evidence['recency'] = round(recency_component, 3)

        density = getattr(concept, 'association_density', 0.0)
        density_component = density * self.density_weight
        evidence['density'] = round(density_component, 3)

        importance = concept.importance.overall if concept.importance else 0.0
        importance_component = importance * self.importance_weight
        evidence['importance'] = round(importance_component, 3)

        consolidation = compute_consolidation_score(concept, now=now)
        consolidation_component = consolidation * self.consolidation_weight
        evidence['consolidation'] = round(consolidation_component, 3)
        evidence['consolidation_score'] = round(consolidation, 3)

        rehearsal = getattr(concept, 'rehearsal_count', 0)
        rehearsal_component = min(1.0, rehearsal / 10.0) * self.rehearsal_weight
        evidence['rehearsal'] = round(rehearsal_component, 3)

        raw_score = (
            self.activation_weight * act_component +
            recency_component +
            density_component +
            importance_component +
            consolidation_component +
            rehearsal_component
        )

        contradiction = self._contradiction_penalty(concept)
        evidence['contradiction_flag'] = bool(contradiction)
        if contradiction:
            raw_score *= (1.0 - self.contradiction_penalty)

        normalized = min(1.0, raw_score)
        evidence['raw_score'] = round(raw_score, 3)
        evidence['normalized'] = round(normalized, 3)

        return normalized, evidence

    def _recency_score(self, last_accessed) -> float:
        """Recency score in [0, 1]."""
        if last_accessed is None:
            return 0.5
        current_time = self._resolve_current_time()
        if isinstance(last_accessed, datetime):
            last_accessed = last_accessed.timestamp()
        age = max(0.0, current_time - float(last_accessed))
        hour = 3600.0
        if age <= hour:
            return 1.0
        if age >= 24 * hour:
            return 0.2
        return 1.0 - (age - hour) / (23 * hour)

    @staticmethod
    def _resolve_current_time() -> float:
        """
        Resolve current time for recency calculations.

        Supports benchmark overrides via `_config_module.current_time`, while
        using wall clock time during normal runtime.
        """
        configured = getattr(_config_module, "current_time", 0.0)
        try:
            configured_value = float(configured or 0.0)
        except Exception:
            configured_value = 0.0
        return configured_value if configured_value > 0.0 else time.time()

    def _contradiction_penalty(self, concept: Concept) -> bool:
        """Check if concept is a superseded or inactive version."""
        state = getattr(concept, "state", None)
        if hasattr(state, "value"):
            state = state.value

        if state in {MemoryState.ARCHIVED.value, MemoryState.SUPPRESSED.value}:
            return True

        if not getattr(concept, "version_parent", None):
            return False

        if not getattr(concept, "is_current_version", True):
            return True

        valid_to = getattr(concept, "valid_to", None)
        if valid_to is not None:
            return True

        if getattr(concept, "valid_from", None) is None:
            return True

        return False

    def _classify_confidence(self, score: float) -> HypothesisConfidence:
        if score >= self.high_confidence_threshold:
            return HypothesisConfidence.HIGH
        if score >= self.medium_confidence_threshold:
            return HypothesisConfidence.MEDIUM
        if score >= self.low_confidence_threshold:
            return HypothesisConfidence.LOW
        return HypothesisConfidence.NONE

    def _ensemble_score(self, hypotheses: List[ScoredHypothesis]) -> float:
        """Weighted ensemble: top hypothesis weighted 2x, others 1x."""
        if not hypotheses:
            return 0.0
        weights = [2.0] + [1.0] * (len(hypotheses) - 1)
        total_weight = sum(weights[:len(hypotheses)])
        return sum(h.hypothesis_score * w for h, w in zip(hypotheses, weights)) / total_weight

    def _coverage_score(
        self, top: List[ScoredHypothesis], all_activated: List[Concept]
    ) -> float:
        """Fraction of top-10 concepts that met the LOW confidence threshold."""
        if not all_activated:
            return 0.0
        return len(top) / min(len(all_activated), self.max_hypotheses)

    def _ensemble_confidence(
        self, hypotheses: List[ScoredHypothesis], ensemble: float
    ) -> HypothesisConfidence:
        if not hypotheses:
            return HypothesisConfidence.NONE
        avg_conf = sum(
            1.0 if h.confidence == HypothesisConfidence.HIGH else
            0.7 if h.confidence == HypothesisConfidence.MEDIUM else
            0.3 if h.confidence == HypothesisConfidence.LOW else 0.0
            for h in hypotheses
        ) / len(hypotheses)
        if avg_conf >= 0.8:
            return HypothesisConfidence.HIGH
        if avg_conf >= 0.5:
            return HypothesisConfidence.MEDIUM
        if avg_conf >= 0.2:
            return HypothesisConfidence.LOW
        return HypothesisConfidence.NONE

    def _empty_set(self) -> HypothesisSet:
        return HypothesisSet(
            hypotheses=[],
            ensemble_score=0.0,
            confidence=HypothesisConfidence.NONE,
            coverage=0.0,
            total_evidence_count=0,
        )
