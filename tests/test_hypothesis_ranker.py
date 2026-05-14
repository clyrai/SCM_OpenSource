"""Tests for HypothesisRanker."""

from __future__ import annotations

import random
import uuid
from typing import List

from src.core.models import Concept, ConceptType, ImportanceVector, MemoryState
from src.retrieval.hypothesis_ranker import (
    HypothesisConfidence,
    HypothesisRanker,
    HypothesisSet,
    ScoredHypothesis,
)


def _embedding(seed: int) -> List[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _concept(
    description: str,
    ctype: ConceptType,
    seed: int,
    salience: float = 0.8,
    grasp: float = 0.8,
    rehearsal: int = 0,
    last_accessed_timestamp: float = None,
    association_density: float = 0.5,
) -> Concept:
    from datetime import datetime
    concept = Concept(
        type=ctype,
        description=description,
        embedding=_embedding(seed),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.8, repetition=0.2),
        salience_score=salience,
        grasp_score=grasp,
        rehearsal_count=rehearsal,
        association_density=association_density,
    )
    if last_accessed_timestamp is not None:
        concept.last_accessed = datetime.fromtimestamp(last_accessed_timestamp)
    return concept


class TestHypothesisRanker:
    def test_confidence_classification_high(self):
        ranker = HypothesisRanker()
        assert ranker._classify_confidence(0.80) == HypothesisConfidence.HIGH
        assert ranker._classify_confidence(1.0) == HypothesisConfidence.HIGH

    def test_confidence_classification_medium(self):
        ranker = HypothesisRanker()
        assert ranker._classify_confidence(0.50) == HypothesisConfidence.MEDIUM
        assert ranker._classify_confidence(0.69) == HypothesisConfidence.MEDIUM

    def test_confidence_classification_low(self):
        ranker = HypothesisRanker()
        assert ranker._classify_confidence(0.20) == HypothesisConfidence.LOW
        assert ranker._classify_confidence(0.39) == HypothesisConfidence.LOW

    def test_confidence_classification_none(self):
        ranker = HypothesisRanker()
        assert ranker._classify_confidence(0.05) == HypothesisConfidence.NONE
        assert ranker._classify_confidence(0.0) == HypothesisConfidence.NONE

    def test_ranking_orders_by_score(self):
        c1 = _concept("Paris is in France", ConceptType.LOCATION, seed=1, salience=0.3, grasp=0.3)
        c2 = _concept("Machine learning is AI", ConceptType.FACT, seed=2, salience=0.9, grasp=0.9)
        c3 = _concept("Coffee is a drink", ConceptType.FACT, seed=3, salience=0.6, grasp=0.6)

        ranker = HypothesisRanker()
        hypothesis_set = ranker.rank([c1, c2, c3])

        assert len(hypothesis_set.hypotheses) == 3
        scores = [h.hypothesis_score for h in hypothesis_set.hypotheses]
        assert scores == sorted(scores, reverse=True)

    def test_contradiction_penalty_reduces_score(self):
        c1 = _concept("Paris is in France", ConceptType.LOCATION, seed=1, salience=0.8, grasp=0.8)
        c2 = _concept("Old wrong fact", ConceptType.FACT, seed=2, salience=0.8, grasp=0.8)
        c2.version_parent = "previous_version"

        ranker = HypothesisRanker()
        set_normal = ranker.rank([c1])
        set_contradicted = ranker.rank([c2])

        assert set_normal.hypotheses[0].hypothesis_score >= set_contradicted.hypotheses[0].hypothesis_score

    def test_empty_concepts_returns_empty_set(self):
        ranker = HypothesisRanker()
        result = ranker.rank([])
        assert isinstance(result, HypothesisSet)
        assert len(result.hypotheses) == 0
        assert result.confidence == HypothesisConfidence.NONE

    def test_activation_map_bias_ranking(self):
        c1 = _concept("Paris is in France", ConceptType.LOCATION, seed=1, salience=0.5, grasp=0.5)
        c2 = _concept("London is in England", ConceptType.LOCATION, seed=2, salience=0.5, grasp=0.5)

        ranker = HypothesisRanker()
        activation_map = {c1.id: 0.9, c2.id: 0.2}
        result = ranker.rank([c1, c2], activation_map=activation_map)

        assert result.hypotheses[0].concept.id == c1.id
        assert result.hypotheses[0].hypothesis_score > result.hypotheses[1].hypothesis_score

    def test_consolidation_score_biases_ranking(self):
        high = _concept("Project token42 record", ConceptType.FACT, seed=1, salience=0.9, grasp=0.9, rehearsal=5, association_density=0.8)
        high.retention_score = 0.85
        high.strength = 1.2

        low = _concept("Project token42 record", ConceptType.FACT, seed=2, salience=0.1, grasp=0.1, rehearsal=0, association_density=0.05)
        low.retention_score = 0.2
        low.strength = 1.7

        ranker = HypothesisRanker()
        result = ranker.rank([low, high], activation_map={low.id: 0.8, high.id: 0.8})

        assert result.hypotheses[0].concept.id == high.id
        assert result.hypotheses[0].evidence["consolidation_score"] >= result.hypotheses[1].evidence["consolidation_score"]

    def test_format_context_with_evidence(self):
        c1 = _concept("Paris is in France", ConceptType.LOCATION, seed=1, salience=0.8, grasp=0.8)
        c2 = _concept("Machine learning", ConceptType.FACT, seed=2, salience=0.5, grasp=0.5)

        ranker = HypothesisRanker()
        result = ranker.rank([c1, c2])
        formatted = ranker.format_context(result, include_evidence=True)

        assert "Retrieved Memories" in formatted
        assert "Paris" in formatted
        assert "confidence:" in formatted

    def test_format_context_empty(self):
        ranker = HypothesisRanker()
        result = ranker.rank([])
        formatted = ranker.format_context(result)
        assert "No strong memory matches found" in formatted

    def test_ensemble_score_weighted(self):
        c1 = _concept("Paris", ConceptType.LOCATION, seed=1, salience=0.9, grasp=0.9)
        c2 = _concept("London", ConceptType.LOCATION, seed=2, salience=0.3, grasp=0.3)
        c3 = _concept("Berlin", ConceptType.LOCATION, seed=3, salience=0.3, grasp=0.3)

        ranker = HypothesisRanker()
        result = ranker.rank([c1, c2, c3])

        assert result.ensemble_score > 0
        assert result.hypotheses[0].rank == 1

    def test_recency_score_fresh(self):
        from src.core import config as _config_module
        ranker = HypothesisRanker()

        old_time = getattr(_config_module, 'current_time', 0.0)
        _config_module.current_time = 10000.0

        score = ranker._recency_score(9999.0)

        _config_module.current_time = old_time

        assert score > 0.9

    def test_recency_score_old(self):
        from src.core import config as _config_module
        ranker = HypothesisRanker()

        old_time = getattr(_config_module, 'current_time', 0.0)
        _config_module.current_time = 100000.0

        score = ranker._recency_score(0.0)

        _config_module.current_time = old_time

        assert score < 0.5

    def test_hypothesis_ranker_respects_max_hypotheses(self):
        concepts = [
            _concept(f"Concept {i}", ConceptType.FACT, seed=i, salience=max(0.1, 0.9 - i * 0.05))
            for i in range(20)
        ]
        ranker = HypothesisRanker(max_hypotheses=5)
        result = ranker.rank(concepts)
        assert len(result.hypotheses) <= 5


class TestHypothesisRankerBrutal:
    def test_ranker_throughput_many_concepts(self):
        concepts = [
            _concept(
                f"Concept {i} about topic {i % 15}",
                ConceptType.FACT,
                seed=i,
                salience=0.3 + (i % 7) * 0.1,
                grasp=0.3 + (i % 7) * 0.1,
                rehearsal=i % 10,
                association_density=0.3 + (i % 5) * 0.1,
            )
            for i in range(100)
        ]

        import time
        ranker = HypothesisRanker()

        start = time.time()
        for _ in range(50):
            result = ranker.rank(concepts)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Ranker too slow: {elapsed:.2f}s for 50 ranks"
        assert len(result.hypotheses) > 0
