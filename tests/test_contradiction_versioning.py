"""Tests for Phase 5 contradiction versioning."""

from __future__ import annotations

from datetime import timedelta
from typing import List

from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, Episode, ImportanceVector, PredicateType, MemoryState
from src.core.time_utils import utc_now
from src.sleep.deep_sleep import DeepSleep


def _embedding(seed: int) -> List[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _state_value(concept: Concept) -> str:
    state = concept.state
    return state.value if hasattr(state, "value") else str(state)


def test_contradiction_creates_version_chain_and_hides_old_version():
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False

    common_tags = {"session_id": "phase5", "person": "user", "task": "conversation"}

    old = Concept(
        id="old_pref",
        type=ConceptType.PREFERENCE,
        description="I prefer morning meetings",
        embedding=_embedding(10),
        importance=ImportanceVector(novelty=0.8, emotional=0.1, task_relevance=0.9, repetition=0.7),
        context_tags=dict(common_tags),
        valid_from=utc_now() - timedelta(days=1),
    )
    ltm.add_concept(old, context_tags=common_tags, allow_versioning=False)

    new = Concept(
        id="new_pref",
        type=ConceptType.PREFERENCE,
        description="I prefer evening meetings",
        embedding=_embedding(10),
        importance=ImportanceVector(novelty=0.9, emotional=0.1, task_relevance=0.95, repetition=0.6),
        context_tags=dict(common_tags),
    )
    stored = ltm.add_concept(new, context_tags=common_tags, allow_versioning=True)

    assert stored.version_parent == old.id
    assert stored.version_root == old.id
    assert stored.is_current_version is True
    assert _state_value(stored) == MemoryState.ACTIVE.value

    assert old.is_current_version is False
    assert _state_value(old) == MemoryState.ARCHIVED.value
    assert old.valid_to is not None

    assert ltm.graph.has_edge(old.id, stored.id)
    assert ltm.graph[old.id][stored.id]["predicate"] == PredicateType.CONTRADICTS.value

    current_ids = {concept.id for concept in ltm.get_all_concepts()}
    history_ids = {concept.id for concept in ltm.get_all_concepts(include_superseded=True)}
    assert stored.id in current_ids
    assert old.id not in current_ids
    assert old.id in history_ids

    retrieved = ltm.search_by_embedding(stored.embedding, limit=3)
    assert retrieved
    assert retrieved[0].id == stored.id


def test_versioned_concepts_keep_audit_chain():
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False

    root_tags = {"session_id": "phase5", "person": "user"}

    base = Concept(
        id="root",
        type=ConceptType.FACT,
        description="The office is in Mumbai",
        embedding=_embedding(20),
        importance=ImportanceVector(novelty=0.7, task_relevance=0.8),
        context_tags=dict(root_tags),
    )
    ltm.add_concept(base, context_tags=root_tags, allow_versioning=False)

    mid = Concept(
        id="mid",
        type=ConceptType.FACT,
        description="The office is in Pune",
        embedding=_embedding(20),
        importance=ImportanceVector(novelty=0.9, task_relevance=0.9),
        context_tags=dict(root_tags),
    )
    ltm.add_concept(mid, context_tags=root_tags, allow_versioning=True)

    assert _state_value(ltm.get_concept("root")) == MemoryState.ARCHIVED.value
    assert ltm.get_concept("mid").version_parent == "root"
    assert ltm.get_concept("mid").version_root == "root"
    assert any(
        (rel.predicate.value if hasattr(rel.predicate, "value") else rel.predicate) == PredicateType.CONTRADICTS.value
        for rel in ltm.get_all_relations(include_history=True)
    )


def test_deep_sleep_preserves_versioned_preference_traces():
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False

    tags = {"session_id": "phase5", "person": "user", "task": "conversation"}

    old = Concept(
        id="sleep_old_pref",
        type=ConceptType.PREFERENCE,
        description="I prefer morning meetings",
        embedding=_embedding(30),
        importance=ImportanceVector(novelty=0.8, emotional=0.1, task_relevance=0.9, repetition=0.7),
        context_tags=dict(tags),
    )
    ltm.add_concept(old, context_tags=tags, allow_versioning=False)

    for i in range(18):
        filler = Concept(
            id=f"sleep_noise_{i}",
            type=ConceptType.FACT,
            description=f"Noise trace {i}",
            embedding=_embedding(1000 + i),
            importance=ImportanceVector(novelty=0.1, emotional=0.0, task_relevance=0.1, repetition=0.0),
            strength=0.55,
            context_tags=dict(tags),
        )
        ltm.add_concept(filler, context_tags=tags, allow_versioning=False)

    new = Concept(
        id="sleep_new_pref",
        type=ConceptType.PREFERENCE,
        description="I prefer evening meetings",
        embedding=_embedding(30),
        importance=ImportanceVector(novelty=0.9, emotional=0.1, task_relevance=0.95, repetition=0.6),
        context_tags=dict(tags),
        strength=3.0,
    )
    stored = ltm.add_concept(new, context_tags=tags, allow_versioning=True)

    deep = DeepSleep()
    concepts = ltm.get_all_concepts(include_suppressed=False)
    relations = ltm.get_all_relations(include_history=False)
    episodes = [
        Episode(
            raw_content="I prefer evening meetings",
            concept_ids=[stored.id],
            source="user",
        )
    ]

    updated_concepts, updated_relations, stats = deep.run(concepts, relations, episodes)
    combined = {concept.id: concept for concept in updated_concepts + stats["retired_concepts"]}

    assert stored.id in combined
    assert combined[stored.id].version_parent == old.id
    assert combined[stored.id].version_root == old.id
    assert combined[stored.id].is_current_version is True
