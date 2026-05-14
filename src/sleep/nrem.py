"""
NREM Consolidation: Slow-wave sleep memory consolidation
Implements Hebbian strengthening and synaptic downscaling
"""
from typing import List, Dict, Tuple
import numpy as np

from ..core.models import Concept, Episode, Relation, MemoryState, PredicateType, ImportanceVector
from ..core.memory_scoring import refresh_consolidation_score
from ..core.config import NREM_DOWNSCALE_FACTOR, IMPORTANCE_THRESHOLD
from ..core.time_utils import utc_now


class NREMConsolidation:
    """
    NREM (Non-Rapid Eye Movement) sleep consolidation.

    Biological inspiration:
    - Slow-wave sleep (N3 stage) is critical for memory consolidation
    - Synaptic downscaling: overall connection strengths reduce during sleep
    - Hebbian consolidation: "neurons that fire together wire together"
    - Active system consolidation: repeated replay of memory traces

    This module implements:
    1. Hebbian strengthening: strengthen concept relations that co-occur
    2. Synaptic downscaling: normalize all strengths to prevent saturation
    3. Pattern separation: reduce interference between similar memories
    """

    def __init__(
        self,
        downscale_factor: float = NREM_DOWNSCALE_FACTOR,
        hebbian_learning_rate: float = 0.1,
        consolidation_threshold: float = 0.5
    ):
        self.downscale_factor = downscale_factor
        self.hebbian_lr = hebbian_learning_rate
        self.consolidation_threshold = consolidation_threshold

    def consolidate(
        self,
        concepts: List[Concept],
        episodes: List[Episode],
        existing_relations: List[Relation]
    ) -> Tuple[List[Concept], List[Relation], Dict]:
        """
        Perform NREM consolidation on concepts and episodes.

        Returns:
            Tuple of (updated_concepts, new_relations, stats)
        """
        if not concepts:
            return [], [], self._empty_stats()

        stats = {
            'concepts_processed': len(concepts),
            'episodes_processed': len(episodes),
            'relations_created': 0,
            'strength_increases': 0,
            'strength_decreases': 0,
            'consolidated_to_ltm': 0,
            'outlier_traces_softened': 0,
        }

        coactivation_map = self._build_coactivation_map(episodes, concepts)
        updated_concepts = self._apply_hebbian_strengthening(concepts, coactivation_map, stats)
        updated_concepts = self._apply_synaptic_downscaling(updated_concepts, stats)
        new_relations = self._create_consolidation_relations(updated_concepts, coactivation_map, stats)
        high_importance_concepts = self._filter_high_importance(updated_concepts)

        stats['consolidated_to_ltm'] = len(high_importance_concepts)

        return updated_concepts, new_relations, stats

    def _build_coactivation_map(
        self,
        episodes: List[Episode],
        concepts: List[Concept]
    ) -> Dict[str, List[str]]:
        """
        Build a map of which concepts co-activate in episodes.
        Used for Hebbian learning - concepts that appear together get stronger connections.
        """
        concept_ids = {c.id for c in concepts}
        coactivation: Dict[str, set] = {c.id: set() for c in concepts}

        for episode in episodes:
            episode_concepts = [
                cid for cid in episode.concept_ids
                if cid in concept_ids
            ]

            for i, cid1 in enumerate(episode_concepts):
                for cid2 in episode_concepts[i+1:]:
                    coactivation[cid1].add(cid2)
                    coactivation[cid2].add(cid1)

        return {k: list(v) for k, v in coactivation.items()}

    def _apply_hebbian_strengthening(
        self,
        concepts: List[Concept],
        coactivation_map: Dict[str, List[str]],
        stats: Dict
    ) -> List[Concept]:
        """
        Apply Hebbian learning: strengthen concepts that co-activate.

        For each concept pair that appears together in episodes,
        increase their strength (LTP - Long-Term Potentiation).
        """
        concept_map = {c.id: c for c in concepts}
        updated = []

        for concept in concepts:
            new_concept = concept.model_copy(
                update={
                    "state": MemoryState.CONSOLIDATING,
                    "last_accessed": utc_now(),
                    "access_count": concept.access_count + 1,
                    "rehearsal_count": getattr(concept, "rehearsal_count", 0) + (1 if coactivation_map.get(concept.id) else 0),
                    "activation_count": getattr(concept, "activation_count", 0) + (1 if coactivation_map.get(concept.id) else 0),
                }
            )

            coactivated = coactivation_map.get(concept.id, [])
            hubness = len(coactivated)
            new_concept.association_density = min(1.0, hubness / 20.0)
            if coactivated:
                strength_boost = self.hebbian_lr * len(coactivated) * 0.1
                old_strength = new_concept.strength
                new_concept.strength = min(2.0, new_concept.strength + strength_boost)
                hub_penalty = 1.0 / (1.0 + hubness / 10.0)
                new_concept.strength = max(0.05, new_concept.strength * hub_penalty)

                if hubness > 10 and new_concept.importance.overall < 0.7:
                    new_concept.state = MemoryState.SUPPRESSED
                    new_concept.strength = 0.05

                if new_concept.strength > old_strength:
                    stats['strength_increases'] += 1
                elif new_concept.strength < old_strength:
                    stats['strength_decreases'] += 1

            new_concept.importance = self._update_importance_after_consolidation(
                new_concept.importance, coactivated
            )
            refresh_consolidation_score(new_concept)

            updated.append(new_concept)

        return updated

    def _update_importance_after_consolidation(
        self,
        importance: ImportanceVector,
        coactivated: List[str]
    ) -> ImportanceVector:
        """
        Update importance vector after consolidation.
        Repetition increases, novelty decreases if co-activated concepts are similar.
        """
        new_repetition = min(1.0, importance.repetition + 0.1 * len(coactivated))

        return ImportanceVector(
            novelty=importance.novelty * 0.95,
            emotional=importance.emotional,
            task_relevance=importance.task_relevance,
            repetition=new_repetition
        )

    def _apply_synaptic_downscaling(
        self,
        concepts: List[Concept],
        stats: Dict
    ) -> List[Concept]:
        """
        Apply synaptic downscaling (homeostatic plasticity).

        During sleep, overall synaptic strengths are normalized to prevent
        saturation and maintain network stability. This is like the brain
        "resetting" itself.

        Biolgically inspired by: Tononi & Cirelli's synaptic homeostasis hypothesis.
        """
        if not concepts:
            return concepts

        strengths = [c.strength for c in concepts]
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths)

        updated = []
        for concept in concepts:
            scale = self.downscale_factor
            if std_strength > 0:
                z_score = (concept.strength - mean_strength) / std_strength
                if z_score > 1.5:
                    # Normalize strong outliers instead of dropping them.
                    # Versioned traces must survive sleep so the current line
                    # of memory remains intact after consolidation.
                    scale *= 0.75
                    stats['outlier_traces_softened'] += 1

            old_strength = concept.strength
            new_strength = concept.strength * scale

            new_concept = concept.model_copy(
                update={
                    "state": getattr(concept, "state", MemoryState.ACTIVE),
                    "strength": new_strength,
                }
            )

            if new_strength < old_strength:
                stats['strength_decreases'] += 1

            refresh_consolidation_score(new_concept)
            updated.append(new_concept)

        return updated

    def _create_consolidation_relations(
        self,
        concepts: List[Concept],
        coactivation_map: Dict[str, List[str]],
        stats: Dict
    ) -> List[Relation]:
        """
        Create new relations based on consolidation.
        Strong co-activation leads to SIMILAR_TO or RELATED_TO relations.
        """
        concept_map = {c.id: c for c in concepts}
        new_relations = []

        for concept_id, coactivated_ids in coactivation_map.items():
            if len(coactivated_ids) < 2:
                continue

            for other_id in coactivated_ids[:5]:
                relation = Relation(
                    subject_id=concept_id,
                    predicate=PredicateType.RELATED_TO,
                    object_id=other_id,
                    strength=0.7,
                    bidirectional=True
                )
                new_relations.append(relation)
                stats['relations_created'] += 1

        return new_relations

    def _filter_high_importance(self, concepts: List[Concept]) -> List[Concept]:
        """Filter concepts that meet threshold for long-term storage"""
        return [
            c for c in concepts
            if c.importance.overall >= self.consolidation_threshold
        ]

    def _empty_stats(self) -> Dict:
        return {
            'concepts_processed': 0,
            'episodes_processed': 0,
            'relations_created': 0,
            'strength_increases': 0,
            'strength_decreases': 0,
            'consolidated_to_ltm': 0,
            'outlier_traces_softened': 0,
        }

    def get_consolidation_strength(self, concept: Concept) -> float:
        """
        Calculate how strongly a concept should be consolidated.
        Based on importance, access count, and relation density.
        """
        base = concept.importance.overall
        access_factor = min(1.0, concept.access_count / 10)
        strength_factor = concept.strength

        return (base * 0.4 + access_factor * 0.3 + strength_factor * 0.3)
