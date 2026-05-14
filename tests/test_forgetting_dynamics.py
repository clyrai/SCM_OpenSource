"""Tests for Phase 5 forgetting dynamics."""

from __future__ import annotations

from datetime import timedelta

from src.core.models import Concept, ConceptType, ImportanceVector, MemoryState
from src.core.time_utils import utc_now
from src.sleep.forgetting_dynamics import ForgettingDynamics


def _state_value(concept: Concept) -> str:
    state = concept.state
    return state.value if hasattr(state, "value") else str(state)


def _concept(
    concept_id: str,
    description: str,
    *,
    grasp: float,
    salience: float,
    rehearsal: int,
    association: float,
    last_accessed_offset_hours: float,
) -> Concept:
    concept = Concept(
        id=concept_id,
        type=ConceptType.FACT,
        description=description,
        importance=ImportanceVector(
            novelty=0.7,
            emotional=0.1,
            task_relevance=0.8,
            repetition=0.7,
        ),
        grasp_score=grasp,
        salience_score=salience,
        rehearsal_count=rehearsal,
        association_density=association,
        strength=1.0,
    )
    concept.last_accessed = utc_now() - timedelta(hours=last_accessed_offset_hours)
    return concept


def test_retention_scores_drive_state_transitions():
    dynamics = ForgettingDynamics(suppress_threshold=0.32, archive_threshold=0.16)

    strong = _concept(
        "strong",
        "durable memory trace",
        grasp=0.92,
        salience=0.88,
        rehearsal=5,
        association=0.75,
        last_accessed_offset_hours=0.5,
    )
    weak = _concept(
        "weak",
        "noisy memory trace",
        grasp=0.12,
        salience=0.10,
        rehearsal=0,
        association=0.05,
        last_accessed_offset_hours=72.0,
    )

    updated, stats = dynamics.apply([strong, weak])
    updated_map = {concept.id: concept for concept in updated}

    assert stats["total_evaluated"] == 2
    assert stats["retained"] == 1
    assert stats["forgotten"] == 1
    assert updated_map["strong"].retention_score > updated_map["weak"].retention_score
    assert _state_value(updated_map["strong"]) == MemoryState.ACTIVE.value
    assert _state_value(updated_map["weak"]) in {
        MemoryState.SUPPRESSED.value,
        MemoryState.ARCHIVED.value,
    }


def test_evaluate_forgetting_returns_ids():
    dynamics = ForgettingDynamics()

    high = _concept(
        "high",
        "high value",
        grasp=0.9,
        salience=0.9,
        rehearsal=3,
        association=0.7,
        last_accessed_offset_hours=1.0,
    )
    low = _concept(
        "low",
        "low value",
        grasp=0.05,
        salience=0.05,
        rehearsal=0,
        association=0.02,
        last_accessed_offset_hours=48.0,
    )

    forgotten, preserved, stats = dynamics.evaluate_forgetting([high, low])

    assert "low" in forgotten
    assert "high" in preserved
    assert stats["forgotten"] >= 1


def test_threshold_rises_with_memory_load():
    dynamics = ForgettingDynamics()
    small = [_concept(f"s{i}", f"small {i}", grasp=0.5, salience=0.5, rehearsal=1, association=0.3, last_accessed_offset_hours=1.0) for i in range(5)]
    large = [_concept(f"l{i}", f"large {i}", grasp=0.5, salience=0.5, rehearsal=1, association=0.3, last_accessed_offset_hours=1.0) for i in range(80)]

    assert dynamics.compute_forgetting_threshold(large) >= dynamics.compute_forgetting_threshold(small)
