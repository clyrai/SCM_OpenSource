import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import Concept, ConceptType, Episode, ImportanceVector, PredicateType, Relation
from src.sleep.tension import TensionDetector


def test_tension_detector_ranks_unfinished_emotional_threads():
    detector = TensionDetector()
    tense = Concept(
        id="c1",
        type=ConceptType.FACT,
        description="project deadline is still unclear and blocking launch",
        importance=ImportanceVector(
            novelty=0.5,
            emotional=-0.8,
            task_relevance=0.95,
            repetition=0.8,
        ),
    )
    neutral = Concept(
        id="c2",
        type=ConceptType.FACT,
        description="user likes green tea",
        importance=ImportanceVector(
            novelty=0.4,
            emotional=0.0,
            task_relevance=0.2,
            repetition=0.2,
        ),
    )

    tensions = detector.detect([neutral, tense], relations=[], episodes=[])

    assert tensions[0]["concept_id"] == "c1"
    assert "goal_open" in tensions[0]["signals"]
    assert "uncertainty" in tensions[0]["signals"]
    assert not any(t["concept_id"] == "c2" for t in tensions)


def test_tension_detector_marks_contradiction_relations():
    detector = TensionDetector()
    first = Concept(
        id="old",
        type=ConceptType.FACT,
        description="morning meetings are preferred",
        importance=ImportanceVector(task_relevance=0.8),
    )
    second = Concept(
        id="new",
        type=ConceptType.FACT,
        description="evening meetings are preferred",
        importance=ImportanceVector(task_relevance=0.8),
    )
    relation = Relation(
        subject_id="old",
        predicate=PredicateType.CONTRADICTS,
        object_id="new",
    )

    tensions = detector.detect([first, second], relations=[relation], episodes=[])

    assert {t["concept_id"] for t in tensions} == {"old", "new"}
    assert all("contradiction" in t["signals"] for t in tensions)


def test_tension_detector_uses_episode_text_for_linked_concepts():
    detector = TensionDetector()
    concept = Concept(
        id="c1",
        type=ConceptType.FACT,
        description="launch plan",
        importance=ImportanceVector(task_relevance=0.9, emotional=-0.4),
    )
    episode = Episode(
        concept_ids=["c1"],
        raw_content="I am stressed because the launch plan is still not done.",
    )

    tensions = detector.detect([concept], relations=[], episodes=[episode])

    assert tensions
    assert "stress" in tensions[0]["signals"]
    assert "unfinished_task" in tensions[0]["signals"]


def test_tension_detector_marks_matching_thread_closed():
    detector = TensionDetector()
    concept = Concept(
        id="c1",
        type=ConceptType.FACT,
        description="project deadline is still unclear and blocking launch",
        importance=ImportanceVector(task_relevance=0.9, emotional=-0.7),
    )
    episode = Episode(
        concept_ids=[],
        raw_content="The project deadline issue is resolved now, and launch is no longer an issue.",
    )

    closed = detector.close_resolved([concept], [episode])
    tensions = detector.detect([concept], relations=[], episodes=[episode])

    assert len(closed) == 1
    assert closed[0]["concept_id"] == "c1"
    assert concept.context_tags["tension_resolved"] is True
    assert "tension_resolved_at" in concept.context_tags
    assert tensions == []


def test_tension_detector_does_not_close_negated_resolution():
    detector = TensionDetector()
    concept = Concept(
        id="c1",
        type=ConceptType.FACT,
        description="project deadline is still unclear and blocking launch",
        importance=ImportanceVector(task_relevance=0.9, emotional=-0.7),
    )
    episode = Episode(
        concept_ids=["c1"],
        raw_content="The project deadline is not resolved yet.",
    )

    closed = detector.close_resolved([concept], [episode])
    tensions = detector.detect([concept], relations=[], episodes=[episode])

    assert closed == []
    assert tensions
