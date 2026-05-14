"""
Stable scoring helpers for memory consolidation and retrieval.

These utilities provide one shared definition of how "well consolidated"
a memory trace is, so sleep, retrieval, and ranking code can stay aligned.
"""
from __future__ import annotations

from typing import Dict

from .models import Concept, MemoryState
from .time_utils import ensure_utc, utc_now


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _state_value(concept: Concept) -> str:
    state = getattr(concept, "state", MemoryState.ACTIVE.value)
    return state.value if hasattr(state, "value") else str(state)


def _recency_score(concept: Concept, now=None) -> float:
    last_accessed = getattr(concept, "last_accessed", None)
    if last_accessed is None:
        return 0.5

    now = ensure_utc(now) or utc_now()
    last_accessed = ensure_utc(last_accessed)
    if last_accessed is None:
        return 0.5

    age_hours = max(0.0, (now - last_accessed).total_seconds() / 3600.0)
    if age_hours <= 1.0:
        return 1.0
    if age_hours >= 72.0:
        return 0.1
    return _clamp(1.0 - (age_hours - 1.0) / 71.0)


def consolidation_profile(concept: Concept, now=None) -> Dict[str, float]:
    """
    Return a transparent score breakdown for a concept.

    The composite score is intentionally sleep-centric: traces with real replay
    evidence should outrank raw, unslept traces. This keeps the score useful for
    human-like consolidation instead of just rewarding static importance.
    """
    now = ensure_utc(now) or utc_now()

    importance = concept.importance.overall if concept.importance else 0.0
    salience = _clamp(getattr(concept, "salience_score", importance))
    grasp = _clamp(getattr(concept, "grasp_score", importance))
    retention = _clamp(getattr(concept, "retention_score", importance))
    rehearsal = _clamp(min(1.0, max(0.0, getattr(concept, "rehearsal_count", 0) / 8.0)))
    activation = _clamp(min(1.0, max(0.0, getattr(concept, "activation_count", 0) / 8.0)))
    association = _clamp(getattr(concept, "association_density", 0.0))
    strength = _clamp(min(1.0, max(0.0, getattr(concept, "strength", 1.0) / 2.0)))
    association_density = _clamp(getattr(concept, "association_density", 0.0))
    association_specificity = _clamp(1.0 - association_density)

    version_bonus = 0.0
    if getattr(concept, "version_parent", None):
        version_bonus += 0.12 if getattr(concept, "is_current_version", True) else -0.30

    state_value = _state_value(concept)
    if state_value == MemoryState.SUPPRESSED.value:
        version_bonus -= 0.20
    elif state_value == MemoryState.ARCHIVED.value:
        version_bonus -= 0.35

    has_sleep_evidence = (
        rehearsal > 0.0
        or activation > 0.0
        or getattr(concept, "version_parent", None) is not None
    )

    if not has_sleep_evidence and state_value == MemoryState.ACTIVE.value:
        # Unslept traces stay intentionally neutral so retrieval is decided by
        # direct cue match, not by static importance alone.
        raw_score = (
            0.12 * retention
            + 0.22 * strength
            + 0.05 * association_specificity
        )
    else:
        raw_score = (
            0.05 * importance
            + 0.05 * salience
            + 0.05 * grasp
            + 0.10 * retention
            + 0.25 * rehearsal
            + 0.25 * activation
            + 0.15 * association_specificity
            + 0.10 * strength
            + version_bonus
        )

    score = _clamp(raw_score)
    return {
        "importance": round(importance, 4),
        "salience": round(salience, 4),
        "grasp": round(grasp, 4),
        "retention": round(retention, 4),
        "rehearsal": round(rehearsal, 4),
        "activation": round(activation, 4),
        "association": round(association, 4),
        "association_specificity": round(association_specificity, 4),
        "strength": round(strength, 4),
        "version_bonus": round(version_bonus, 4),
        "score": round(score, 4),
    }


def compute_consolidation_score(concept: Concept, now=None) -> float:
    """Return the composite consolidation score for a concept."""
    return consolidation_profile(concept, now=now)["score"]


def refresh_consolidation_score(concept: Concept, now=None) -> float:
    """
    Update a concept in place with its latest consolidation score.

    The score is mirrored into context tags so it can survive persistence in
    the current schema without a migration.
    """
    score = compute_consolidation_score(concept, now=now)
    concept.consolidation_score = score

    context_tags = getattr(concept, "context_tags", None) or {}
    context_tags["consolidation_score"] = score
    concept.context_tags = context_tags

    return score
