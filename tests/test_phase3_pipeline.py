"""Phase 3 integration tests for ChatEngine pipeline (SpreadingActivation + HypothesisRanker)."""

from __future__ import annotations

import uuid
from typing import List

import src.chat.engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.models import Concept, ConceptType, ImportanceVector, Relation, PredicateType


class _DummyLLM:
    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        return "That's interesting."


def _stub_embedding(seed: int) -> List[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _add_relation(ltm, from_id: str, to_id: str, strength: float, predicate: str = "related"):
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


class _StubEncoderWithConcepts:
    def __init__(self, concepts: List[Concept]):
        self._concepts = concepts
        self._counter = 0

    def extract(self, text: str) -> List[Concept]:
        self._counter += 1
        return [
            Concept(
                type=ConceptType.PERSON,
                description=f"User about topic {self._counter}",
                embedding=_stub_embedding(self._counter),
                importance=ImportanceVector(novelty=0.8, task_relevance=0.8),
                salience_score=0.8,
                grasp_score=0.8,
            )
        ]

    def _get_embedding(self, text: str) -> List[float]:
        seed = sum(ord(ch) for ch in text) % 97
        return _stub_embedding(seed)


class TestPhase3Pipeline:
    def test_spreading_activation_retriever_initialized_when_hme_enabled(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoderWithConcepts([]),
            enable_auto_sleep=False,
            session_id=f"phase3_{uuid.uuid4().hex}",
        )

        assert engine._spreading_activation is not None
        assert engine._hypothesis_ranker is not None

    def test_retrieve_hme_returns_formatted_context(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)
        monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
        from src.core import database as db_module
        db_module.DATABASE_URL = "sqlite:///:memory:"

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoderWithConcepts([]),
            enable_auto_sleep=False,
            session_id=f"phase3_{uuid.uuid4().hex}",
        )

        engine.long_term_memory._persist_concept = lambda c: None
        engine.long_term_memory._persist_relation = lambda r: None

        c1 = Concept(
            type=ConceptType.LOCATION,
            description="Paris is the capital of France",
            embedding=_stub_embedding(1),
            importance=ImportanceVector(novelty=0.8, task_relevance=0.8),
            salience_score=0.8,
            grasp_score=0.8,
        )
        c1.id = "c1"
        engine.long_term_memory.add_concept(c1)
        _add_relation(engine.long_term_memory, "c1", "c2", strength=0.7, predicate="related_to")

        memory_context, stats = engine._retrieve_hme("Tell me about Paris", None)

        assert isinstance(memory_context, str)
        assert "Retrieved Memories" in memory_context
        assert 'total_concepts_activated' in stats
        assert 'hypothesis_count' in stats
        assert 'hypothesis_confidence' in stats

    def test_hme_retrieval_uses_seeds_from_ltm(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoderWithConcepts([]),
            enable_auto_sleep=False,
            session_id=f"phase3_{uuid.uuid4().hex}",
        )

        engine.long_term_memory._persist_concept = lambda c: None
        engine.long_term_memory._persist_relation = lambda r: None

        c1 = Concept(
            type=ConceptType.LOCATION,
            description="I live in Seattle",
            embedding=_stub_embedding(5),
            importance=ImportanceVector(novelty=0.8, task_relevance=0.9),
            salience_score=0.85,
            grasp_score=0.8,
        )
        c1.id = "seattle_1"
        engine.long_term_memory.add_concept(c1)

        memory_context, stats = engine._retrieve_hme("Seattle", None)

        assert stats['seeds'] >= 1
        assert stats['total_concepts_activated'] >= 1

    def test_phase3_retrieval_with_no_ltm_concepts(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoderWithConcepts([]),
            enable_auto_sleep=False,
            session_id=f"phase3_{uuid.uuid4().hex}",
        )

        memory_context, stats = engine._retrieve_hme("unknown word xyz123", None)

        assert isinstance(memory_context, str)
        assert 'total_concepts_activated' in stats

    def test_phase3_retrieval_context_tags_included(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoderWithConcepts([]),
            enable_auto_sleep=False,
            session_id="test_session_123",
        )

        engine.long_term_memory._persist_concept = lambda c: None
        engine.long_term_memory._persist_relation = lambda r: None

        memory_context, stats = engine._retrieve_hme("Paris France", None)

        assert 'coverage' in stats
        assert 'hypothesis_ensemble' in stats


class TestPhase3PipelineStress:
    def test_phase3_retrieval_throughput(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoderWithConcepts([]),
            enable_auto_sleep=False,
            session_id=f"phase3_stress_{uuid.uuid4().hex}",
        )

        engine.long_term_memory._persist_concept = lambda c: None
        engine.long_term_memory._persist_relation = lambda r: None

        for i in range(50):
            c = Concept(
                type=ConceptType.FACT,
                description=f"Fact about topic {i}",
                embedding=_stub_embedding(i),
                importance=ImportanceVector(novelty=0.7, task_relevance=0.7),
                salience_score=0.7,
                grasp_score=0.7,
            )
            c.id = f"c_{i}"
            engine.long_term_memory.add_concept(c)
            if i > 0:
                _add_relation(
                    engine.long_term_memory,
                    f"c_{i-1}", f"c_{i}", strength=0.6, predicate="related"
                )

        import time
        start = time.time()
        for _ in range(20):
            ctx, stats = engine._retrieve_hme("topic 10 topic 25", None)
        elapsed = time.time() - start

        assert elapsed < 3.0, f"Phase 3 retrieval too slow: {elapsed:.2f}s for 20 calls"
        assert stats['total_concepts_activated'] >= 0
