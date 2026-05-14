"""Tests for EventCompiler."""

import pytest

from src.core.event_compiler import EventCompiler
from src.core.models import Episode, ImportanceVector


class TestEventCompiler:
    def setup_method(self):
        self.compiler = EventCompiler()

    def test_compile_episode_extracts_core_fields(self):
        episode = Episode(
            raw_content="I met Alice in Seattle today because we had a planning meeting.",
            source="user",
            importance=ImportanceVector(task_relevance=0.8),
            salience_score=0.82,
            grasp_score=0.79,
        )

        event = self.compiler.compile_episode(episode, interlocutor="saish")

        assert event.who == "saish"
        assert "planning meeting" in event.what.lower()
        assert event.when is not None and "today" in event.when.lower()
        assert event.where == "Seattle"
        assert event.why is not None and "planning meeting" in event.why.lower()
        assert 0.0 <= event.certainty <= 1.0
        assert event.event_key is not None

    def test_compile_episode_assistant_sets_who(self):
        episode = Episode(
            raw_content="I can help with that now.",
            source="assistant",
            importance=ImportanceVector(),
            salience_score=0.5,
            grasp_score=0.5,
        )

        event = self.compiler.compile_episode(episode)
        assert event.who == "assistant"
        assert event.source == "assistant"

    def test_uncertainty_terms_lower_certainty(self):
        clear_episode = Episode(
            raw_content="I live in Boston.",
            source="user",
            salience_score=0.8,
            grasp_score=0.8,
        )
        unsure_episode = Episode(
            raw_content="I think maybe I might live in Boston.",
            source="user",
            salience_score=0.8,
            grasp_score=0.8,
        )

        clear = self.compiler.compile_episode(clear_episode, interlocutor="user")
        unsure = self.compiler.compile_episode(unsure_episode, interlocutor="user")
        assert unsure.certainty < clear.certainty

    def test_contradiction_detection(self):
        episode = Episode(
            raw_content="Actually no, I was wrong. I changed my mind.",
            source="user",
            salience_score=0.7,
            grasp_score=0.6,
        )
        event = self.compiler.compile_episode(episode)
        assert event.is_contradiction is True

    def test_duplicate_detection_true_for_similar_event(self):
        e1 = self.compiler.compile_episode(
            Episode(
                raw_content="I work at Google in Seattle today.",
                source="user",
                salience_score=0.7,
                grasp_score=0.7,
            ),
            interlocutor="user",
        )
        e2 = self.compiler.compile_episode(
            Episode(
                raw_content="Today I work at Google in Seattle.",
                source="user",
                salience_score=0.75,
                grasp_score=0.72,
            ),
            interlocutor="user",
        )

        assert self.compiler.is_duplicate(e2, [e1]) is True

    def test_duplicate_detection_false_for_distinct_event(self):
        e1 = self.compiler.compile_episode(
            Episode(
                raw_content="I work at Google in Seattle.",
                source="user",
                salience_score=0.7,
                grasp_score=0.7,
            ),
            interlocutor="user",
        )
        e2 = self.compiler.compile_episode(
            Episode(
                raw_content="I enjoy pizza on weekends.",
                source="user",
                salience_score=0.6,
                grasp_score=0.6,
            ),
            interlocutor="user",
        )

        assert self.compiler.is_duplicate(e2, [e1]) is False
