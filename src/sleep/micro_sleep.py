"""
MicroSleep: lightweight online memory maintenance for HME Phase 4.

Runs frequent, low-latency consolidation passes:
- replay top unstable traces
- reinforce repeated co-occurrence links
- merge near-duplicate traces
- apply light decay to weak, low-value traces
"""
from collections import defaultdict
from itertools import combinations
import re
from typing import Dict, List, Set, Tuple

from ..core.config import (
    IMPORTANCE_THRESHOLD,
    MICRO_SLEEP_LIGHT_DECAY_FACTOR,
    MICRO_SLEEP_REPLAY_TOP_K,
)
from ..core.models import Concept, Episode, MemoryState, PredicateType, Relation
from ..core.memory_scoring import refresh_consolidation_score
from ..core.time_utils import utc_now


class MicroSleep:
    """Frequent low-cost replay and cleanup pass."""

    def __init__(
        self,
        replay_top_k: int = MICRO_SLEEP_REPLAY_TOP_K,
        light_decay_factor: float = MICRO_SLEEP_LIGHT_DECAY_FACTOR,
        duplicate_similarity: float = 0.86,
        min_pair_repeats: int = 2,
        episode_window: int = 20,
    ):
        self.replay_top_k = max(1, replay_top_k)
        self.light_decay_factor = min(1.0, max(0.80, light_decay_factor))
        self.duplicate_similarity = min(1.0, max(0.5, duplicate_similarity))
        self.min_pair_repeats = max(2, min_pair_repeats)
        self.episode_window = max(5, episode_window)

    def run(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        episodes: List[Episode],
    ) -> Tuple[List[Concept], List[Relation], Dict]:
        """
        Execute one micro-sleep pass.

        Returns:
            (updated_concepts, updated_relations, stats)
        """
        if not concepts:
            return [], list(relations), self._empty_stats()

        stats = self._empty_stats()
        concept_map = {c.id: c for c in concepts}
        updated_relations = list(relations)

        replay_ids = self._select_unstable_trace_ids(concepts)
        self._replay_unstable_traces(concept_map, replay_ids, stats)

        pair_counts = self._build_pair_counts(episodes, concept_map.keys())
        updated_relations = self._reinforce_repeated_links(
            updated_relations=updated_relations,
            pair_counts=pair_counts,
            concept_map=concept_map,
            stats=stats,
        )

        merge_map = self._merge_near_duplicates(concepts)
        if merge_map:
            updated_relations = self._rewrite_relations_for_merges(updated_relations, merge_map)
            concepts = [c for c in concepts if c.id not in merge_map]
            stats["duplicates_merged"] = len(merge_map)
            stats["merged_ids"] = sorted(merge_map.keys())

        self._apply_light_decay(concepts, replay_ids, stats)
        for concept in concepts:
            refresh_consolidation_score(concept)

        updated_relations = self._dedupe_relations(updated_relations)

        stats["concepts_processed"] = len(concepts)
        stats["relations_processed"] = len(updated_relations)
        return concepts, updated_relations, stats

    def _select_unstable_trace_ids(self, concepts: List[Concept]) -> List[str]:
        scored: List[Tuple[str, float]] = []
        for concept in concepts:
            retention = getattr(concept, "retention_score", 0.5)
            prediction_error = getattr(concept, "prediction_error", 0.0)
            novelty = concept.importance.novelty if concept.importance else 0.5
            strength_norm = min(1.0, max(0.0, concept.strength / 2.0))

            instability = (
                (1.0 - retention) * 0.40
                + prediction_error * 0.30
                + novelty * 0.20
                + (1.0 - strength_norm) * 0.10
            )
            scored.append((concept.id, instability))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[: self.replay_top_k]]

    def _replay_unstable_traces(self, concept_map: Dict[str, Concept], replay_ids: List[str], stats: Dict):
        now = utc_now()
        replayed = 0
        for concept_id in replay_ids:
            concept = concept_map.get(concept_id)
            if concept is None:
                continue
            concept.rehearsal_count = getattr(concept, "rehearsal_count", 0) + 1
            concept.activation_count = getattr(concept, "activation_count", 0) + 1
            concept.retention_score = min(1.0, getattr(concept, "retention_score", 0.5) + 0.05)
            concept.strength = min(2.5, concept.strength + (0.02 + concept.importance.overall * 0.03))
            concept.last_accessed = now
            concept.state = MemoryState.ACTIVE
            replayed += 1
        stats["replayed"] = replayed

    def _build_pair_counts(self, episodes: List[Episode], known_ids: Set[str]) -> Dict[Tuple[str, str], int]:
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        window = episodes[-self.episode_window :]
        for episode in window:
            concept_ids = [cid for cid in episode.concept_ids if cid in known_ids]
            if len(concept_ids) < 2:
                continue
            unique_ids = sorted(set(concept_ids))
            for left, right in combinations(unique_ids, 2):
                pair_counts[(left, right)] += 1
        return pair_counts

    def _reinforce_repeated_links(
        self,
        updated_relations: List[Relation],
        pair_counts: Dict[Tuple[str, str], int],
        concept_map: Dict[str, Concept],
        stats: Dict,
    ) -> List[Relation]:
        if not pair_counts:
            return updated_relations

        index: Dict[Tuple[str, str], int] = {}
        for i, rel in enumerate(updated_relations):
            index[(rel.subject_id, rel.object_id)] = i

        reinforced = 0
        created = 0

        for (left, right), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            if count < self.min_pair_repeats:
                continue
            if left not in concept_map or right not in concept_map:
                continue

            direct_key = (left, right)
            reverse_key = (right, left)

            if direct_key in index:
                rel = updated_relations[index[direct_key]]
                rel.strength = min(2.0, rel.strength + min(0.30, 0.06 * count))
                reinforced += 1
            elif reverse_key in index:
                rel = updated_relations[index[reverse_key]]
                rel.strength = min(2.0, rel.strength + min(0.30, 0.06 * count))
                reinforced += 1
            else:
                rel = Relation(
                    subject_id=left,
                    predicate=PredicateType.RELATED_TO,
                    object_id=right,
                    strength=min(1.0, 0.22 + count * 0.08),
                    bidirectional=True,
                )
                index[(left, right)] = len(updated_relations)
                updated_relations.append(rel)
                created += 1

        stats["relations_reinforced"] = reinforced
        stats["relations_created"] = created
        return updated_relations

    def _merge_near_duplicates(self, concepts: List[Concept]) -> Dict[str, str]:
        """
        Returns mapping duplicate_id -> primary_id.
        """
        if len(concepts) < 2:
            return {}

        ranked = sorted(
            concepts,
            key=lambda c: (c.importance.overall * c.strength, c.access_count, c.rehearsal_count),
            reverse=True,
        )

        merge_map: Dict[str, str] = {}
        taken: Set[str] = set()

        for i, anchor in enumerate(ranked):
            if anchor.id in taken:
                continue
            for candidate in ranked[i + 1 :]:
                if candidate.id in taken:
                    continue
                if candidate.type != anchor.type:
                    continue
                similarity = self._description_similarity(anchor.description, candidate.description)
                if similarity >= self.duplicate_similarity:
                    merge_map[candidate.id] = anchor.id
                    taken.add(candidate.id)

        return merge_map

    def _rewrite_relations_for_merges(self, relations: List[Relation], merge_map: Dict[str, str]) -> List[Relation]:
        rewritten: List[Relation] = []
        for rel in relations:
            subject = merge_map.get(rel.subject_id, rel.subject_id)
            object_ = merge_map.get(rel.object_id, rel.object_id)
            if subject == object_:
                continue
            rewritten.append(
                Relation(
                    id=rel.id,
                    subject_id=subject,
                    predicate=rel.predicate,
                    object_id=object_,
                    strength=rel.strength,
                    created_at=rel.created_at,
                    bidirectional=rel.bidirectional,
                )
            )
        return rewritten

    def _apply_light_decay(self, concepts: List[Concept], replay_ids: List[str], stats: Dict):
        replay_id_set = set(replay_ids)
        decayed = 0
        for concept in concepts:
            if concept.id in replay_id_set:
                continue
            if concept.importance.overall < IMPORTANCE_THRESHOLD and concept.strength < 1.0:
                concept.strength = max(0.05, concept.strength * self.light_decay_factor)
                concept.retention_score = max(
                    0.0,
                    getattr(concept, "retention_score", 0.5) * self.light_decay_factor,
                )
                decayed += 1
        stats["lightly_decayed"] = decayed

    def _dedupe_relations(self, relations: List[Relation]) -> List[Relation]:
        deduped: Dict[Tuple[str, str, str], Relation] = {}
        for rel in relations:
            predicate = rel.predicate.value if hasattr(rel.predicate, "value") else str(rel.predicate)
            key = (rel.subject_id, predicate, rel.object_id)
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = rel
                continue
            if rel.strength > existing.strength:
                existing.strength = rel.strength
            existing.bidirectional = existing.bidirectional or rel.bidirectional
        return list(deduped.values())

    @staticmethod
    def _description_similarity(left: str, right: str) -> float:
        left_tokens = MicroSleep._normalize(left)
        right_tokens = MicroSleep._normalize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        inter = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return inter / union if union else 0.0

    @staticmethod
    def _normalize(text: str) -> Set[str]:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        tokens = [tok for tok in cleaned.split() if tok and len(tok) > 2]
        return set(tokens)

    @staticmethod
    def _empty_stats() -> Dict:
        return {
            "mode": "micro",
            "concepts_processed": 0,
            "relations_processed": 0,
            "replayed": 0,
            "relations_reinforced": 0,
            "relations_created": 0,
            "duplicates_merged": 0,
            "merged_ids": [],
            "lightly_decayed": 0,
        }
