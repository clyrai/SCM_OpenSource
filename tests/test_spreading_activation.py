"""Tests for SpreadingActivationRetriever."""

from __future__ import annotations

import random
import time
import uuid
from datetime import timedelta
from typing import List

from src.chat.engine import ChatEngine
from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, Episode, ImportanceVector, MemoryState, Relation, PredicateType
from src.core.time_utils import utc_now
from src.retrieval.spreading_activation import SpreadingActivationRetriever


def _embedding(seed: int) -> List[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _concept(
    description: str,
    ctype: ConceptType,
    seed: int,
    salience: float = 0.8,
    grasp: float = 0.8,
    ltm: LongTermMemory = None,
) -> Concept:
    concept = Concept(
        type=ctype,
        description=description,
        embedding=_embedding(seed),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.8, repetition=0.2),
        salience_score=salience,
        grasp_score=grasp,
    )
    if ltm is not None:
        concept.id = f"concept_{seed}_{uuid.uuid4().hex[:6]}"
        ltm.add_concept(concept)
    return concept


def _add_relation(ltm: LongTermMemory, from_id: str, to_id: str, strength: float, predicate: str):
    """Add a relation to LTM using the correct Relation object."""
    rel = Relation(
        subject_id=from_id,
        object_id=to_id,
        predicate=PredicateType.RELATED_TO,
        strength=strength,
    )
    ltm.add_relation(rel)
    if ltm.graph.has_edge(from_id, to_id):
        ltm.graph[from_id][to_id]['strength'] = strength
        ltm.graph[from_id][to_id]['predicate'] = predicate


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    return ltm


def _wm_with_episodes(count: int = 3) -> List[Episode]:
    return [
        Episode(
            concept_ids=[],
            raw_content=f"User mentioned about {i} topics",
            importance=ImportanceVector(novelty=0.5, task_relevance=0.5),
            source="user",
        )
        for i in range(count)
    ]


class _DummyLLM:
    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        return "ok"


class _StubEncoder:
    def extract(self, text: str) -> List[Concept]:
        return []


class _DummyEmbedding:
    def encode(self, text: str) -> List[float]:
        return _embedding(sum(ord(c) for c in text))


class TestSpreadingActivationRetriever:
    def test_cue_extraction(self):
        ltm = _fast_ltm()
        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)
        cues = retriever._extract_cues("I really love traveling to Paris")
        assert "really" not in cues
        assert "love" not in cues
        assert "traveling" in cues
        assert "paris" in cues

    def test_seed_selection_by_cue_match(self):
        ltm = _fast_ltm()
        c1 = _concept("Paris is a beautiful city in France", ConceptType.LOCATION, seed=1, ltm=ltm)
        c2 = _concept("I work on Machine Learning projects", ConceptType.FACT, seed=2, ltm=ltm)
        c3 = _concept("Coffee improves morning productivity", ConceptType.FACT, seed=3, ltm=ltm)

        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)
        seeds = retriever._select_seeds(["paris", "france"], limit=5)

        assert len(seeds) >= 1
        seed_ids = {c.id for c, _ in seeds}
        assert c1.id in seed_ids
        assert c2.id not in seed_ids

    def test_seed_selection_prefers_consolidated_memory_on_tie(self):
        ltm = _fast_ltm()
        from src.core.working_memory import WorkingMemory

        noise = Concept(
            id="noise",
            type=ConceptType.FACT,
            description="project token42 record",
            embedding=_embedding(101),
            importance=ImportanceVector(novelty=0.08, emotional=0.0, task_relevance=0.08, repetition=0.1),
            salience_score=0.1,
            grasp_score=0.1,
            strength=1.7,
            retention_score=0.2,
            rehearsal_count=0,
            activation_count=0,
            association_density=0.02,
        )
        key = Concept(
            id="key",
            type=ConceptType.FACT,
            description="project token42 record",
            embedding=_embedding(102),
            importance=ImportanceVector(novelty=0.88, emotional=0.1, task_relevance=0.92, repetition=0.75),
            salience_score=0.9,
            grasp_score=0.9,
            strength=1.0,
            retention_score=0.8,
            rehearsal_count=4,
            activation_count=3,
            association_density=0.7,
        )
        ltm.add_concept(noise)
        ltm.add_concept(key)

        wm = WorkingMemory()
        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)
        seeds = retriever._select_seeds(["project", "token42"], limit=3)

        assert len(seeds) >= 2
        assert seeds[0][0].id == "key"

    def test_fallback_seeds_when_no_match(self):
        ltm = _fast_ltm()
        c1 = _concept("Paris is in France", ConceptType.LOCATION, seed=1, ltm=ltm)
        c2 = _concept("London is in England", ConceptType.LOCATION, seed=2, ltm=ltm)

        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)
        seeds = retriever._select_seeds(["zzzz_unknown_word"], limit=3)

        assert len(seeds) == 2
        assert all(score > 0 for _, score in seeds)

    def test_activation_propagation_one_step(self):
        ltm = _fast_ltm()
        c1 = _concept("Paris is in France", ConceptType.LOCATION, seed=1, ltm=ltm)
        c2 = _concept("France is in Europe", ConceptType.LOCATION, seed=2, ltm=ltm)
        c3 = _concept("Coffee is a drink", ConceptType.FACT, seed=3, ltm=ltm)

        _add_relation(ltm, c1.id, c2.id, strength=0.9, predicate="located_in")
        _add_relation(ltm, c2.id, c3.id, strength=0.3, predicate="related_to")

        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(
            working_memory=wm,
            long_term_memory=ltm,
            spreading_steps=1,
            spreading_decay=0.45,
            activation_threshold=0.01,
        )
        seeds = [(c1, 1.0)]
        activated, stats = retriever._propagate_activation(seeds, {}, steps=1)

        assert stats['propagation_steps'] == 1
        activated_ids = {c.id for c in activated}
        assert c1.id in activated_ids
        assert c2.id in activated_ids

    def test_activation_propagation_suppressed_concepts_excluded(self):
        ltm = _fast_ltm()
        c1 = _concept("Paris is in France", ConceptType.LOCATION, seed=1, ltm=ltm)
        c2 = _concept("Forgotten idea", ConceptType.FACT, seed=2, ltm=ltm)

        ltm.remove_concept(c2.id, soft=True)

        _add_relation(ltm, c1.id, c2.id, strength=0.8, predicate="related_to")

        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(
            working_memory=wm, long_term_memory=ltm, activation_threshold=0.01
        )
        seeds = [(c1, 1.0)]
        activated, _ = retriever._propagate_activation(seeds, {}, steps=2)

        activated_ids = {c.id for c in activated}
        assert c2.id not in activated_ids

    def test_retrieve_returns_stats(self):
        ltm = _fast_ltm()
        _concept("Paris is in France", ConceptType.LOCATION, seed=1, ltm=ltm)

        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)
        activated, stats = retriever.retrieve("Paris France travel")

        assert 'cues' in stats
        assert 'seeds' in stats
        assert 'steps' in stats
        assert 'propagation_steps' in stats
        assert 'total_concepts_activated' in stats
        assert isinstance(activated, list)

    def test_context_gate_recency_boost(self):
        from src.core.working_memory import WorkingMemory
        from src.core import config as _config_module
        ltm = _fast_ltm()
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)

        old_time = getattr(_config_module, 'current_time', 0.0)

        last_accessed_ts = 1000000.0
        _config_module.current_time = 1003600.0
        one_hour_old = retriever._recency_factor(last_accessed_ts)

        _config_module.current_time = 1020000.0
        five_hour_old = retriever._recency_factor(last_accessed_ts)

        _config_module.current_time = old_time

        assert one_hour_old > five_hour_old
        assert one_hour_old == 1.0

    def test_retrieve_empty_query_uses_fallback(self):
        ltm = _fast_ltm()
        _concept("Paris", ConceptType.LOCATION, seed=1, ltm=ltm)
        _concept("London", ConceptType.LOCATION, seed=2, ltm=ltm)

        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)
        activated, stats = retriever.retrieve("")

        assert isinstance(activated, list)
        assert 'seeds' in stats

    def test_retrieve_with_recency_boost_applies_signal(self):
        from src.core.working_memory import WorkingMemory
        from src.core import config as _config_module

        ltm = _fast_ltm()
        older = _concept("Preference topic meeting window", ConceptType.PREFERENCE, seed=51, ltm=ltm)
        newer = _concept("Preference topic meeting window", ConceptType.PREFERENCE, seed=52, ltm=ltm)

        older.last_accessed = utc_now() - timedelta(days=3)
        newer.last_accessed = utc_now()
        ltm.add_concept(older)
        ltm.add_concept(newer)

        wm = WorkingMemory()
        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)

        old_time = getattr(_config_module, "current_time", 0.0)
        _config_module.current_time = utc_now().timestamp()
        ranked, stats = retriever.retrieve_with_recency_boost(
            "preference meeting window", recency_boost=1.0
        )
        _config_module.current_time = old_time

        assert stats.get("recency_boost_applied") is True
        assert "max_boosted_activation" in stats
        assert ranked
        ranked_ids = {concept.id for concept in ranked}
        assert older.id in ranked_ids
        assert newer.id in ranked_ids


class TestSpreadingActivationBrutal:
    def test_throughput_large_graph(self):
        ltm = _fast_ltm()
        concepts = []
        for i in range(200):
            c = _concept(f"Concept {i} about topic {i % 20}", ConceptType.FACT, seed=i, ltm=ltm)
            concepts.append(c)

        for i in range(0, 180, 2):
            _add_relation(ltm, concepts[i].id, concepts[i + 1].id, strength=0.6, predicate="related")

        from src.core.working_memory import WorkingMemory
        wm = WorkingMemory()

        retriever = SpreadingActivationRetriever(
            working_memory=wm,
            long_term_memory=ltm,
            spreading_steps=3,
            activation_threshold=0.02,
        )

        start = time.time()
        for _ in range(10):
            activated, stats = retriever.retrieve("topic 5 topic 12 topic 3")
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Too slow: {elapsed:.2f}s for 10 retrieves"
        assert stats['total_concepts_activated'] >= 0


class TestContextGateNoneSafety:
    """Regression tests for the v0.7.1 fix to spreading_activation._context_gate.

    Background: when working_memory has no recent user episode (e.g., right
    after a deep-sleep cycle clears WM), `context_tags['person']` is None.
    The previous code did `context_tags['person'].lower()` which crashed
    the entire retrieval path, surfacing as "search returns nothing" in
    the brutal LangChain harness.
    """

    def _build(self):
        from src.core.working_memory import WorkingMemory
        ltm = _fast_ltm()
        c1 = _concept("user is allergic to seafood", ConceptType.FACT, seed=1, ltm=ltm)
        c2 = _concept("user lives in Bangalore", ConceptType.FACT, seed=2, ltm=ltm)
        c1.context_tags = {"session_id": "alice", "person": "alice"}
        c2.context_tags = {"session_id": "alice", "person": "alice"}
        wm = WorkingMemory(capacity=7)
        retriever = SpreadingActivationRetriever(working_memory=wm, long_term_memory=ltm)
        return retriever

    def test_retrieve_with_none_person_tag_does_not_crash(self):
        """Bug fix: context_tags={'person': None} used to raise AttributeError."""
        retriever = self._build()
        # Explicit None — simulates "no recent user episode in WM."
        activated, stats = retriever.retrieve(
            "What food am I allergic to?",
            context_tags={"session_id": "alice", "person": None},
        )
        # Must complete without exception. May or may not return concepts
        # (depends on cue/concept overlap), but the call must not crash.
        assert isinstance(activated, list)
        assert isinstance(stats, dict)

    def test_retrieve_with_no_context_tags_does_not_crash(self):
        """Same fix path: pass empty context_tags."""
        retriever = self._build()
        activated, stats = retriever.retrieve("seafood allergy", context_tags={})
        assert isinstance(activated, list)
        assert isinstance(stats, dict)

    def test_retrieve_with_missing_person_key_does_not_crash(self):
        """And the case where 'person' key isn't present at all."""
        retriever = self._build()
        activated, stats = retriever.retrieve(
            "seafood allergy",
            context_tags={"session_id": "alice"},
        )
        assert isinstance(activated, list)
        assert isinstance(stats, dict)
