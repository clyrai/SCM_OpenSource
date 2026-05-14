"""
AssociationBinder: Link newly encoded memories into associative graph.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from .config import (
    ASSOCIATION_AGING_DECAY,
    ASSOCIATION_AGING_STALE_STEPS,
    ASSOCIATION_AGING_THRESHOLD,
    ASSOCIATION_LEARNING_RATE,
    ASSOCIATION_MAX_EDGES_PER_CONCEPT,
    ASSOCIATION_MIN_EDGE_STRENGTH,
    ASSOCIATION_SEMANTIC_THRESHOLD,
)
from .models import Concept, EventSchema, PredicateType, Relation


class AssociationBinder:
    """
    Binds event concepts into LTM graph using semantic/contextual rules.

    Core behavior:
    - Build co-occurrence links for concepts in the same event.
    - Link current event concepts to prior graph anchors.
    - Update relation strengths incrementally (Hebbian-like).
    - Age and prune stale weak links.
    """

    def __init__(
        self,
        learning_rate: float = ASSOCIATION_LEARNING_RATE,
        semantic_threshold: float = ASSOCIATION_SEMANTIC_THRESHOLD,
        max_edges_per_concept: int = ASSOCIATION_MAX_EDGES_PER_CONCEPT,
        min_edge_strength: float = ASSOCIATION_MIN_EDGE_STRENGTH,
        aging_decay: float = ASSOCIATION_AGING_DECAY,
        aging_stale_steps: int = ASSOCIATION_AGING_STALE_STEPS,
        aging_threshold: float = ASSOCIATION_AGING_THRESHOLD,
    ):
        self.learning_rate = learning_rate
        self.semantic_threshold = semantic_threshold
        self.max_edges_per_concept = max_edges_per_concept
        self.min_edge_strength = min_edge_strength
        self.aging_decay = aging_decay
        self.aging_stale_steps = aging_stale_steps
        self.aging_threshold = aging_threshold

    def bind_event(
        self,
        event: EventSchema,
        event_concepts: List[Concept],
        long_term_memory,
        candidate_pool: Optional[List[Concept]] = None,
    ) -> Dict[str, float]:
        """
        Create/update associative edges for one event.

        Returns stats including edge create/update counts and coverage.
        """
        stats = {
            "edges_created": 0,
            "edges_updated": 0,
            "edges_pruned": 0,
            "aged_edges": 0,
            "pairs_considered": 0,
            "pairs_bound": 0,
            "coverage": 0.0,
        }

        if not event_concepts:
            stats["edges_pruned"] += self.age_relations(long_term_memory, touched_edges=set())
            return stats

        pool = candidate_pool or long_term_memory.get_all_concepts(include_suppressed=False)
        pool_by_id = {c.id: c for c in pool}

        touched_edges: Set[Tuple[str, str]] = set()

        # 1) Intra-event co-occurrence links
        for left, right in combinations(event_concepts, 2):
            stats["pairs_considered"] += 1
            bound, created = self._upsert_association(
                event=event,
                left=left,
                right=right,
                long_term_memory=long_term_memory,
                semantic_score=self._semantic_similarity(left, right),
                token_overlap=self._token_overlap(left.description, right.description),
                force_bind=True,
            )
            if bound:
                touched_edges.add((bound.subject_id, bound.object_id))
                stats["pairs_bound"] += 1
                if created:
                    stats["edges_created"] += 1
                else:
                    stats["edges_updated"] += 1

        # 2) Event-to-prior anchors
        event_ids = {c.id for c in event_concepts}
        for concept in event_concepts:
            anchors = self._find_anchors(concept, pool, exclude_ids=event_ids)
            for anchor, semantic_score in anchors:
                stats["pairs_considered"] += 1
                bound, created = self._upsert_association(
                    event=event,
                    left=concept,
                    right=anchor,
                    long_term_memory=long_term_memory,
                    semantic_score=semantic_score,
                    token_overlap=self._token_overlap(concept.description, anchor.description),
                    force_bind=False,
                )
                if bound:
                    touched_edges.add((bound.subject_id, bound.object_id))
                    stats["pairs_bound"] += 1
                    if created:
                        stats["edges_created"] += 1
                    else:
                        stats["edges_updated"] += 1

        self._enforce_edge_caps(long_term_memory, [c.id for c in event_concepts], stats)
        stats["edges_pruned"] += self.age_relations(long_term_memory, touched_edges=touched_edges, stats=stats)
        self._update_association_density(long_term_memory, event_concepts)

        if stats["pairs_considered"] > 0:
            stats["coverage"] = round(stats["pairs_bound"] / stats["pairs_considered"], 4)

        return stats

    def age_relations(
        self,
        long_term_memory,
        touched_edges: Set[Tuple[str, str]],
        stats: Optional[Dict[str, float]] = None,
    ) -> int:
        """Age stale weak links and prune if they decay too far."""
        pruned = 0
        to_remove: List[Tuple[str, str]] = []

        for subject_id, object_id, data in long_term_memory.graph.edges(data=True):
            edge_key = (subject_id, object_id)
            if edge_key in touched_edges:
                data["age_steps"] = 0
                continue

            age_steps = int(data.get("age_steps", 0)) + 1
            data["age_steps"] = age_steps

            strength = float(data.get("strength", 0.0))
            if age_steps >= self.aging_stale_steps and strength < self.aging_threshold:
                decayed = round(strength * self.aging_decay, 4)
                if decayed < self.min_edge_strength:
                    to_remove.append((subject_id, object_id))
                else:
                    data["strength"] = decayed
                    if stats is not None:
                        stats["aged_edges"] += 1

        for subject_id, object_id in to_remove:
            long_term_memory.graph.remove_edge(subject_id, object_id)
            pruned += 1

        return pruned

    def _upsert_association(
        self,
        event: EventSchema,
        left: Concept,
        right: Concept,
        long_term_memory,
        semantic_score: float,
        token_overlap: float,
        force_bind: bool,
    ) -> Tuple[Optional[Relation], bool]:
        predicate = self._infer_predicate(event, left, right, semantic_score, token_overlap)
        context_score = self._context_score(event, left, right)

        should_bind = force_bind or semantic_score >= self.semantic_threshold or token_overlap >= 0.35 or context_score >= 0.7
        if not should_bind:
            return None, False

        strength_delta = self._strength_delta(event, left, right, semantic_score, token_overlap, context_score)
        edge = long_term_memory.graph.get_edge_data(left.id, right.id, default=None)

        if edge:
            current = float(edge.get("strength", 0.0))
            updated = round(min(1.0, current + strength_delta), 4)
            long_term_memory.graph[left.id][right.id]["strength"] = updated
            long_term_memory.graph[left.id][right.id]["predicate"] = predicate.value
            long_term_memory.graph[left.id][right.id]["age_steps"] = 0

            relation = Relation(
                subject_id=left.id,
                predicate=predicate,
                object_id=right.id,
                strength=updated,
                bidirectional=edge.get("bidirectional", True),
            )
            return relation, False

        initial_strength = round(max(self.min_edge_strength, strength_delta), 4)
        if initial_strength < self.min_edge_strength:
            return None, False

        relation = Relation(
            subject_id=left.id,
            predicate=predicate,
            object_id=right.id,
            strength=min(1.0, initial_strength),
            bidirectional=True,
        )
        long_term_memory.add_relation(relation)
        # Track relation staleness directly on graph edge.
        if long_term_memory.graph.has_edge(left.id, right.id):
            long_term_memory.graph[left.id][right.id]["age_steps"] = 0
        return relation, True

    def _strength_delta(
        self,
        event: EventSchema,
        left: Concept,
        right: Concept,
        semantic_score: float,
        token_overlap: float,
        context_score: float,
    ) -> float:
        g_left = max(0.1, float(getattr(left, "grasp_score", 0.5) or 0.5))
        g_right = max(0.1, float(getattr(right, "grasp_score", 0.5) or 0.5))

        sal_left = max(0.1, float(getattr(left, "salience_score", 0.5) or 0.5))
        sal_right = max(0.1, float(getattr(right, "salience_score", 0.5) or 0.5))
        sal_event = max(0.1, float(event.salience or 0.5))

        salience_term = (sal_left + sal_right + sal_event) / 3.0
        signal = 0.45 * semantic_score + 0.25 * token_overlap + 0.30 * context_score
        signal = max(0.05, min(1.0, signal))

        delta = self.learning_rate * ((g_left + g_right) / 2.0) * salience_term * signal
        return max(0.01, min(1.0, delta))

    def _infer_predicate(
        self,
        event: EventSchema,
        left: Concept,
        right: Concept,
        semantic_score: float,
        token_overlap: float,
    ) -> PredicateType:
        left_type = left.type.value if hasattr(left.type, "value") else str(left.type)
        right_type = right.type.value if hasattr(right.type, "value") else str(right.type)
        text = event.what.lower()

        if event.is_contradiction and token_overlap >= 0.25:
            return PredicateType.CONTRADICTS
        if "because" in text or "due to" in text or "therefore" in text:
            return PredicateType.CAUSED_BY
        if left_type == "person" and right_type == "preference":
            return PredicateType.PREFERS
        if left_type == "person" and right_type == "location":
            return PredicateType.LOCATED_AT
        if event.when:
            return PredicateType.TEMPORAL
        if event.where:
            return PredicateType.SPATIAL
        if left_type == right_type and semantic_score >= (self.semantic_threshold + 0.08):
            return PredicateType.SIMILAR_TO
        return PredicateType.RELATED_TO

    def _find_anchors(
        self,
        concept: Concept,
        pool: List[Concept],
        exclude_ids: Set[str],
        limit: int = 3,
    ) -> List[Tuple[Concept, float]]:
        candidates: List[Tuple[Concept, float]] = []
        for candidate in pool:
            if candidate.id in exclude_ids or candidate.id == concept.id:
                continue

            semantic = self._semantic_similarity(concept, candidate)
            overlap = self._token_overlap(concept.description, candidate.description)
            score = 0.7 * semantic + 0.3 * overlap

            if semantic >= self.semantic_threshold or overlap >= 0.4:
                candidates.append((candidate, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:limit]

    def _context_score(self, event: EventSchema, left: Concept, right: Concept) -> float:
        score = 0.2
        if event.when:
            score += 0.2
        if event.where:
            score += 0.2
        if event.why:
            score += 0.2

        who = (event.who or "").lower()
        if who and (who in left.description.lower() or who in right.description.lower()):
            score += 0.1

        if event.certainty >= 0.7:
            score += 0.1

        return min(1.0, score)

    def _enforce_edge_caps(self, long_term_memory, source_ids: List[str], stats: Dict[str, float]) -> None:
        for source_id in source_ids:
            if source_id not in long_term_memory.graph:
                continue

            outgoing = list(long_term_memory.graph.out_edges(source_id, data=True))
            if len(outgoing) <= self.max_edges_per_concept:
                continue

            outgoing.sort(key=lambda row: float(row[2].get("strength", 0.0)))
            remove_count = len(outgoing) - self.max_edges_per_concept

            for subject_id, object_id, _ in outgoing[:remove_count]:
                long_term_memory.graph.remove_edge(subject_id, object_id)
                stats["edges_pruned"] += 1

    def _update_association_density(self, long_term_memory, concepts: List[Concept]) -> None:
        for concept in concepts:
            if concept.id not in long_term_memory.graph:
                continue

            out_degree = long_term_memory.graph.out_degree(concept.id)
            in_degree = long_term_memory.graph.in_degree(concept.id)
            degree = out_degree + in_degree

            density = round(min(1.0, degree / max(1.0, self.max_edges_per_concept * 2.0)), 4)
            concept.association_density = density
            concept.activation_count = int(getattr(concept, "activation_count", 0)) + 1

            if concept.id in long_term_memory._concept_cache:
                long_term_memory._concept_cache[concept.id] = concept
            if concept.id in long_term_memory.graph.nodes:
                long_term_memory.graph.nodes[concept.id]["association_density"] = density
                long_term_memory.graph.nodes[concept.id]["activation_count"] = concept.activation_count

    def _semantic_similarity(self, left: Concept, right: Concept) -> float:
        if left.embedding and right.embedding and len(left.embedding) == len(right.embedding):
            dot = 0.0
            norm_left = 0.0
            norm_right = 0.0
            for a, b in zip(left.embedding, right.embedding):
                dot += a * b
                norm_left += a * a
                norm_right += b * b

            if norm_left > 0 and norm_right > 0:
                return max(0.0, min(1.0, dot / ((norm_left ** 0.5) * (norm_right ** 0.5))))

        return self._token_overlap(left.description, right.description)

    def _token_overlap(self, text_a: str, text_b: str) -> float:
        tokens_a = {t for t in text_a.lower().split() if t}
        tokens_b = {t for t in text_b.lower().split() if t}
        if not tokens_a or not tokens_b:
            return 0.0
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return inter / union if union > 0 else 0.0
