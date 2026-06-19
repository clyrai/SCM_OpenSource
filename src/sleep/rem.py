"""
REM Dreaming: Generative replay during REM sleep
Synthesizes new connections and integrates memories
"""
import hashlib
from typing import List, Dict, Tuple
import numpy as np

from ..core.models import Concept, Episode, Relation, MemoryState, PredicateType
from ..core.memory_scoring import compute_consolidation_score
from ..core.config import REM_DREAM_COUNT
from ..core.time_utils import utc_now


class REMDreaming:
    """
    REM (Rapid Eye Movement) sleep dreaming module.

    Biological inspiration:
    - REM sleep is associated with emotional processing and memory integration
    - The brain generates novel combinations during dreaming (free association)
    - Hippocampal-cortical dialogue: memories are "replayed" and integrated

    This module implements:
    1. Memory replay: Reactivate concepts and episodes in novel combinations
    2. Semantic expansion: Create new relations between concepts
    3. Dream narrative: Generate coherent "dream" experiences for testing memory
    4. Integration: Connect new memories to existing semantic network
    """

    def __init__(
        self,
        dream_count: int = REM_DREAM_COUNT,
        novelty_factor: float = 0.3,
        integration_threshold: float = 0.4
    ):
        self.dream_count = dream_count
        self.novelty_factor = novelty_factor
        self.integration_threshold = integration_threshold

    def dream(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        recent_episodes: List[Episode]
    ) -> Tuple[List[Dict], List[Relation], Dict]:
        """
        Generate dreams from memory content.

        Returns:
            Tuple of (dreams, new_relations, stats)
        """
        if not concepts:
            return [], [], self._empty_stats()

        stats = {
            'dreams_generated': 0,
            'concepts_in_dreams': 0,
            'new_relations_created': 0,
            'integrated_concepts': 0
        }

        concept_map = {c.id: c for c in concepts}
        dreams = []

        active_concepts = self._select_dream_concepts(concepts, recent_episodes)

        for i in range(min(self.dream_count, len(active_concepts) // 2)):
            dream_concepts = self._generate_dream_sequence(active_concepts, i)
            dream_narrative = self._build_dream_narrative(dream_concepts, concept_map)

            dreams.append({
                'id': f"dream_{utc_now().timestamp()}_{i}",
                'sequence': [c.id for c in dream_concepts],
                'narrative': dream_narrative,
                'timestamp': utc_now().isoformat(),
                'emotional_tone': self._assess_dream_emotion(dream_concepts)
            })

            stats['dreams_generated'] += 1
            stats['concepts_in_dreams'] += len(dream_concepts)

        new_relations = self._create_novel_relations(active_concepts, relations, stats)
        integrated = self._integrate_concepts(active_concepts, new_relations)
        stats['integrated_concepts'] = len(integrated)

        return dreams, new_relations, stats

    def _select_dream_concepts(
        self,
        concepts: List[Concept],
        recent_episodes: List[Episode]
    ) -> List[Concept]:
        """
        Select concepts for dreaming based on activation patterns.
        Recent episodes should be replayed more often.
        """
        concept_scores = {}

        for concept in concepts:
            consolidation = compute_consolidation_score(concept)
            base_score = (
                concept.importance.overall * 0.35
                + consolidation * 0.35
                + min(1.0, concept.strength / 2.0) * 0.20
            )

            episode_count = sum(
                1 for ep in recent_episodes
                if concept.id in ep.concept_ids
            )

            concept_scores[concept.id] = base_score + episode_count * 0.2
            tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
            try:
                tension_boost = float(tags.get("unresolved_tension_score", 0.0) or 0.0)
            except Exception:
                tension_boost = 0.0
            concept_scores[concept.id] += min(0.4, max(0.0, tension_boost) * 0.4)

        sorted_concepts = sorted(
            concepts,
            key=lambda c: concept_scores.get(c.id, 0),
            reverse=True
        )

        max_dream_concepts = min(20, len(sorted_concepts))
        return sorted_concepts[:max_dream_concepts]

    def _generate_dream_sequence(
        self,
        concepts: List[Concept],
        dream_index: int
    ) -> List[Concept]:
        """
        Generate a sequence of concepts for a single dream.
        Uses activation spread and novelty to create interesting combinations.
        """
        if len(concepts) < 3:
            return concepts

        sequence = []
        remaining = list(concepts)
        step = 0

        first = remaining.pop(dream_index % len(remaining))
        sequence.append(first)

        while remaining and len(sequence) < 8:
            last_concept = sequence[-1]

            similarities = []
            for candidate in remaining:
                sim = self._compute_activation_spread(last_concept, candidate)
                similarities.append((candidate, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)

            if len(similarities) > 1 and self._deterministic_roll(
                dream_index,
                step,
                last_concept.id,
                similarities[0][0].id,
                similarities[1][0].id,
                len(remaining),
                len(concepts),
            ) < self.novelty_factor:
                candidate = similarities[1][0]
            else:
                candidate = similarities[0][0] if similarities else None

            if candidate:
                sequence.append(candidate)
                remaining.remove(candidate)
            step += 1

        return sequence

    def _compute_activation_spread(self, concept1: Concept, concept2: Concept) -> float:
        """
        Compute activation spread between two concepts.
        Based on embedding similarity, relation strength, and semantic distance.
        """
        base_similarity = 0.5

        if concept1.embedding and concept2.embedding:
            emb_sim = self._cosine_similarity(concept1.embedding, concept2.embedding)
            base_similarity = (base_similarity + emb_sim) / 2

        relation_boost = 0.0
        if concept1.type == concept2.type:
            relation_boost = 0.15

        novelty_boost = concept2.importance.novelty * 0.2

        return min(1.0, base_similarity + relation_boost + novelty_boost)

    def _build_dream_narrative(
        self,
        concepts: List[Concept],
        concept_map: Dict[str, Concept]
    ) -> str:
        """
        Build a text narrative from a dream sequence.
        For debugging/logging purposes.
        """
        descriptions = [
            concept_map.get(c.id, c).description if c else "unknown"
            for c in concepts
        ]

        narrative = " | ".join(descriptions[:6])
        return narrative if narrative else "No dream content"

    def _assess_dream_emotion(self, concepts: List[Concept]) -> str:
        """
        Assess the emotional tone of a dream based on its concepts.
        """
        if not concepts:
            return "neutral"

        avg_emotional = np.mean([c.importance.emotional for c in concepts])

        if avg_emotional > 0.3:
            return "positive"
        elif avg_emotional < -0.3:
            return "negative"
        else:
            return "neutral"

    def _create_novel_relations(
        self,
        concepts: List[Concept],
        existing_relations: List[Relation],
        stats: Dict
    ) -> List[Relation]:
        """
        Create novel relations between concepts during dreaming.
        This is where "creative" connections are made.
        """
        existing_pairs = {
            (r.subject_id, r.object_id) for r in existing_relations
        }

        new_relations = []

        for i, concept1 in enumerate(concepts[:10]):
            for concept2 in concepts[i+1:10]:
                pair_key = (concept1.id, concept2.id)
                reverse_key = (concept2.id, concept1.id)

                if pair_key in existing_pairs or reverse_key in existing_pairs:
                    continue

                activation = self._compute_activation_spread(concept1, concept2)

                if activation > self.integration_threshold:
                    relation = Relation(
                        subject_id=concept1.id,
                        predicate=PredicateType.RELATED_TO,
                        object_id=concept2.id,
                        strength=activation * 0.8,
                        bidirectional=True
                    )
                    new_relations.append(relation)
                    stats['new_relations_created'] += 1

        return new_relations

    def _integrate_concepts(
        self,
        concepts: List[Concept],
        new_relations: List[Relation]
    ) -> List[str]:
        """
        Mark concepts as integrated into the semantic network.
        Returns list of concept IDs that were integrated.
        """
        integrated_ids = set()

        for relation in new_relations:
            integrated_ids.add(relation.subject_id)
            integrated_ids.add(relation.object_id)

        return list(integrated_ids)

    def _empty_stats(self) -> Dict:
        return {
            'dreams_generated': 0,
            'concepts_in_dreams': 0,
            'new_relations_created': 0,
            'integrated_concepts': 0
        }

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Defensively returns 0.0 when shapes mismatch (e.g., a data dir
        contains concepts embedded by an older model — 384-dim MiniLM
        next to new 768-dim nomic-embed-text). Prevents the sleep cycle
        from crashing on mixed-vintage data; the affected pair just gets
        treated as dissimilar.
        """
        if not vec1 or not vec2:
            return 0.0
        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)

        if v1.shape != v2.shape:
            return 0.0

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def _deterministic_roll(*parts) -> float:
        """Return a stable pseudo-random value in [0, 1)."""
        payload = "|".join(str(part) for part in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big") / 2**64
