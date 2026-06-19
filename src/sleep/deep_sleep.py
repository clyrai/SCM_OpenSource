"""
DeepSleep: full replay/downscale/synthesis/pruning pass for HME Phase 4.
"""
from typing import Any, Dict, List, Tuple

from ..core.config import (
    DEEP_SLEEP_ENABLE_SYNTHESIS,
    DEEP_SLEEP_GLOBAL_DOWNSCALE_FACTOR,
    DREAM_STATE_ENABLED,
)
from ..core.models import Concept, Episode, MemoryState, Relation
from ..core.memory_scoring import refresh_consolidation_score
from .dream_state import DreamStateBuilder
from .forgetting import ForgettingModule
from .nrem import NREMConsolidation
from .paraphrase import SleepParaphraser
from .rem import REMDreaming
from .schema_extractor import SchemaExtractor, SchemaExtractorConfig
from .tension import TensionDetector


def _state_value(concept: Concept) -> str:
    state = getattr(concept, "state", MemoryState.ACTIVE.value)
    return state.value if hasattr(state, "value") else str(state)


class DeepSleep:
    """Full maintenance cycle with broad replay and pruning."""

    def __init__(
        self,
        nrem: NREMConsolidation | None = None,
        rem: REMDreaming | None = None,
        forgetting: ForgettingModule | None = None,
        paraphraser: SleepParaphraser | None = None,
        schema_extractor: SchemaExtractor | None = None,
        curiosity_engine: Any = None,  # CuriosityEngine - circular-import-free
        global_downscale_factor: float = DEEP_SLEEP_GLOBAL_DOWNSCALE_FACTOR,
        enable_synthesis: bool = DEEP_SLEEP_ENABLE_SYNTHESIS,
        enable_dream_state: bool = DREAM_STATE_ENABLED,
        enable_paraphrase: bool = False,
        enable_schema_extraction: bool = False,
        enable_curiosity: bool = False,
    ):
        self.nrem = nrem or NREMConsolidation()
        self.rem = rem or REMDreaming()
        self.forgetting = forgetting or ForgettingModule()
        self.global_downscale_factor = min(1.0, max(0.70, global_downscale_factor))
        self.enable_synthesis = enable_synthesis
        self.enable_dream_state = bool(enable_dream_state)
        self.dream_state_builder = DreamStateBuilder()
        self.tension_detector = TensionDetector()
        # Phase 6: optional sleep-time paraphrase for retrieval-friendly storage.
        # min_rehearsals=0 → paraphrase every concept on its first sleep so
        # benchmarks see the effect immediately. Production deployments may
        # raise this to 1 or 2 to delay paraphrase until a concept proves
        # durable (saves LLM cost when using LLMParaphraser backend).
        self.enable_paraphrase = bool(enable_paraphrase)
        self.paraphraser = paraphraser or SleepParaphraser(min_rehearsals=0)
        # Phase 7: optional schema extraction during REM. Detects recurring
        # patterns across episodes and emits ABSTRACT-type concepts that the
        # wake-summary endpoint can surface to the user.
        self.enable_schema_extraction = bool(enable_schema_extraction)
        self.schema_extractor = schema_extractor or SchemaExtractor()
        # Phase 7: optional curiosity engine. Detects entity gaps and
        # ingests external knowledge from configured sources. Default off.
        self.enable_curiosity = bool(enable_curiosity)
        self.curiosity_engine = curiosity_engine

    def run(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        episodes: List[Episode],
    ) -> Tuple[List[Concept], List[Relation], Dict]:
        """
        Execute deep-sleep pass.

        Returns:
            (updated_concepts, updated_relations, stats)
        """
        if not concepts:
            return [], list(relations), self._empty_stats()

        active_concepts = [
            c for c in concepts
            if _state_value(c) not in {
                MemoryState.SUPPRESSED.value,
                MemoryState.ARCHIVED.value,
            }
        ]

        nrem_concepts, nrem_new_relations, nrem_stats = self.nrem.consolidate(
            concepts=active_concepts,
            episodes=episodes,
            existing_relations=relations,
        )

        relation_pool = list(relations) + list(nrem_new_relations)
        self._apply_global_downscale(nrem_concepts, relation_pool)

        # Phase 6: paraphrase pass — rewrite consolidated concept descriptions
        # into clean, retrieval-friendly fact statements. Cost is paid once
        # per concept that survived consolidation, not per ingestion turn.
        if self.enable_paraphrase:
            self.paraphraser.apply(nrem_concepts)

        self._refresh_consolidation_scores(nrem_concepts)
        closed_threads = self.tension_detector.close_resolved(
            concepts=nrem_concepts,
            episodes=episodes,
        )
        unresolved_tensions = self.tension_detector.detect(
            concepts=nrem_concepts,
            relations=relation_pool,
            episodes=episodes,
        )
        self.tension_detector.annotate(nrem_concepts, unresolved_tensions)

        dreams: List[Dict] = []
        rem_new_relations: List[Relation] = []
        if self.enable_synthesis:
            recent_episodes = self._select_replay_episodes(episodes)
            dreams, rem_new_relations, rem_stats = self.rem.dream(
                concepts=nrem_concepts,
                relations=relation_pool,
                recent_episodes=recent_episodes,
            )
            relation_pool.extend(rem_new_relations)
        else:
            rem_stats = self.rem._empty_stats()

        dream_state = (
            self.dream_state_builder.build(
                dreams,
                tensions=unresolved_tensions,
                closed_threads=closed_threads,
            )
            if self.enable_dream_state
            else self.dream_state_builder.empty()
        )

        # Phase 7: schema extraction. Detect recurring patterns across the
        # episode set (which now spans sessions thanks to M2) and emit
        # ABSTRACT-type schema concepts. These are added to the consolidated
        # set so they participate in normal retrieval and forgetting passes.
        schema_concepts: List[Concept] = []
        schema_stats: Dict[str, Any] = {}
        if self.enable_schema_extraction:
            try:
                schemas = self.schema_extractor.extract(episodes, nrem_concepts)
                schema_stats = dict(self.schema_extractor.last_stats)
                for s in schemas:
                    c = s.to_concept()
                    schema_concepts.append(c)
                # Add to the consolidated set so they're returned from this run
                # and synced back to LTM by the engine's _apply_sleep_updates.
                nrem_concepts.extend(schema_concepts)
            except Exception as exc:
                schema_stats = {"error": str(exc)}

        # Phase 7: curiosity pass. Detect entity gaps and (with permission)
        # fetch external briefs to fill them. Filled gaps become FACT-type
        # concepts tagged `_curiosity=True` for audit. Default off.
        curiosity_concepts: List[Concept] = []
        curiosity_stats: Dict[str, Any] = {}
        if self.enable_curiosity and self.curiosity_engine is not None:
            try:
                filled = self.curiosity_engine.run(episodes, nrem_concepts)
                curiosity_stats = dict(self.curiosity_engine.last_stats)
                for fg in filled:
                    if fg.concept is not None:
                        curiosity_concepts.append(fg.concept)
                nrem_concepts.extend(curiosity_concepts)
            except Exception as exc:
                curiosity_stats = {"error": str(exc)}

        conflict_pairs = self._extract_conflict_pairs(relation_pool)
        forgotten_ids, preserved_ids, forgetting_stats = self.forgetting.evaluate_forgetting(
            concepts=nrem_concepts,
            conflict_pairs=conflict_pairs,
        )
        self._refresh_consolidation_scores(nrem_concepts)

        retired_concepts = [c for c in nrem_concepts if _state_value(c) != MemoryState.ACTIVE.value]
        updated_concepts = [c for c in nrem_concepts if _state_value(c) == MemoryState.ACTIVE.value]
        updated_relations = self._dedupe_relations(relation_pool)

        stats = {
            "mode": "deep",
            "nrem": nrem_stats,
            "rem": rem_stats,
            "forgetting": forgetting_stats,
            "dreams": dreams,
            "dream_state": dream_state,
            "unresolved_tensions": unresolved_tensions,
            "closed_threads": closed_threads,
            "global_downscale_factor": self.global_downscale_factor,
            "relations_created_nrem": len(nrem_new_relations),
            "relations_created_rem": len(rem_new_relations),
            "preserved": len(preserved_ids),
            "forgotten_ids": forgotten_ids,
            "retired_ids": [c.id for c in retired_concepts],
            "retired_concepts": retired_concepts,
            # Phase 7: surfaced for the wake summary
            "schemas_extracted": [
                {
                    "id": c.id,
                    "description": c.description,
                    "schema_type": c.context_tags.get("schema_type"),
                    "occurrence_count": c.context_tags.get("occurrence_count"),
                    "entities": c.context_tags.get("entities"),
                    "source_sessions": c.context_tags.get("source_sessions"),
                }
                for c in schema_concepts
            ],
            "schema_stats": schema_stats,
            # Phase 7: curiosity engine outputs
            "curiosity_filled": [
                {
                    "id": c.id,
                    "entity": c.context_tags.get("curiosity_entity"),
                    "source": c.context_tags.get("curiosity_source"),
                    "description": c.description,
                    "occurrence_count": c.context_tags.get("occurrence_count"),
                }
                for c in curiosity_concepts
            ],
            "curiosity_stats": curiosity_stats,
        }
        return updated_concepts, updated_relations, stats

    def _apply_global_downscale(self, concepts: List[Concept], relations: List[Relation]):
        # Phase 6 fix: protect high-salience concepts from aggressive downscaling
        # so contradictory anchor facts and explicit user-stated facts survive
        # multiple sleep cycles even if entropy fires constantly.
        from ..core.config import FORGETTING_PROTECT_SALIENCE
        protect_floor = FORGETTING_PROTECT_SALIENCE
        for concept in concepts:
            sal = getattr(concept, "salience_score", 0.0) or 0.0
            factor = self.global_downscale_factor
            if protect_floor > 0.0 and sal >= protect_floor:
                # Half the downscale; protected concepts decay slower.
                factor = (factor + 1.0) / 2.0
            concept.strength = max(0.05, concept.strength * factor)
        for relation in relations:
            relation.strength = max(0.05, relation.strength * self.global_downscale_factor)

    @staticmethod
    def _refresh_consolidation_scores(concepts: List[Concept]):
        for concept in concepts:
            refresh_consolidation_score(concept)

    @staticmethod
    def _extract_conflict_pairs(relations: List[Relation]) -> List[Tuple[str, str]]:
        conflicts: List[Tuple[str, str]] = []
        for relation in relations:
            predicate = relation.predicate.value if hasattr(relation.predicate, "value") else str(relation.predicate)
            lower = predicate.lower()
            if "contradict" in lower or "conflict" in lower or "opposite" in lower:
                conflicts.append((relation.subject_id, relation.object_id))
        return conflicts

    @staticmethod
    def _dedupe_relations(relations: List[Relation]) -> List[Relation]:
        merged: Dict[Tuple[str, str, str], Relation] = {}
        for relation in relations:
            predicate = relation.predicate.value if hasattr(relation.predicate, "value") else str(relation.predicate)
            key = (relation.subject_id, predicate, relation.object_id)
            existing = merged.get(key)
            if existing is None:
                merged[key] = relation
                continue
            if relation.strength > existing.strength:
                existing.strength = relation.strength
            existing.bidirectional = existing.bidirectional or relation.bidirectional
        return list(merged.values())

    @staticmethod
    def _empty_stats() -> Dict:
        return {
            "mode": "deep",
            "nrem": {},
            "rem": {},
            "forgetting": {},
            "dreams": [],
            "dream_state": DreamStateBuilder.empty(),
            "unresolved_tensions": [],
            "closed_threads": [],
            "global_downscale_factor": DEEP_SLEEP_GLOBAL_DOWNSCALE_FACTOR,
            "relations_created_nrem": 0,
            "relations_created_rem": 0,
            "preserved": 0,
            "forgotten_ids": [],
            "retired_ids": [],
            "retired_concepts": [],
        }

    @staticmethod
    def _select_replay_episodes(episodes: List[Episode]) -> List[Episode]:
        """
        Select a replay window that preserves coverage across the session.

        The benchmark datasets shuffle episode order, so simply taking the tail
        can under-sample parts of memory. We sort by timestamp and take an even
        sample with a bounded window size so deep sleep sees a representative
        trace without replaying everything.
        """
        if not episodes:
            return []

        ordered = sorted(
            episodes,
            key=lambda episode: (
                episode.timestamp.timestamp()
                if hasattr(episode.timestamp, "timestamp")
                else float(episode.timestamp)
            ),
        )

        window_size = min(len(ordered), max(20, min(48, len(ordered) // 2)))
        if window_size >= len(ordered):
            return ordered
        if window_size <= 1:
            return [ordered[-1]]

        step = (len(ordered) - 1) / (window_size - 1)
        indices: List[int] = []
        for i in range(window_size):
            idx = int(round(i * step))
            if idx not in indices:
                indices.append(idx)

        if len(indices) < window_size:
            for idx in range(len(ordered)):
                if idx not in indices:
                    indices.append(idx)
                if len(indices) == window_size:
                    break

        return [ordered[idx] for idx in indices]
