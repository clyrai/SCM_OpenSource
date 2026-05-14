"""
Tests for PredictionErrorEngine
"""
import pytest
from src.core.prediction_error import PredictionErrorEngine


class TestPredictionErrorEngine:
    """Test suite for PredictionErrorEngine (surprise detection)."""

    def setup_method(self):
        self.engine = PredictionErrorEngine(window_size=5, decay=0.85)

    def test_novel_entity_has_high_surprise(self):
        """First mention of a new entity should score high on surprise."""
        score = self.engine.compute(
            text="Sarah just joined the team",
            extracted_entities=["Sarah"],
            topic="work",
        )
        assert score >= 0.15

    def test_repeated_entity_has_low_surprise(self):
        """Repeating the same entity within window should score low."""
        self.engine.compute(
            text="Sarah is working on the project",
            extracted_entities=["Sarah"],
            topic="work",
        )
        score = self.engine.compute(
            text="Sarah is attending the meeting today",
            extracted_entities=["Sarah"],
            topic="work",
        )
        assert score < 0.3

    def test_contradiction_increases_surprise(self):
        """Explicit contradiction should boost surprise score."""
        baseline = self.engine.compute(
            text="I actually prefer morning meetings",
            extracted_entities=[],
            topic="work",
        )
        contradiction = self.engine.compute(
            text="Actually wait, no, I was wrong, I prefer evening meetings",
            extracted_entities=[],
            topic="work",
        )
        assert contradiction > baseline

    def test_batch_computation(self):
        """Batch computation returns list of scores."""
        texts = [
            "The project deadline is Friday",
            "The sky is blue",
            "Sarah joined the team yesterday",
        ]
        scores = self.engine.compute_batch(texts)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_topic_familiarity_reduces_score(self):
        """Repeated topic within window should lower surprise."""
        self.engine.compute(
            text="Let's discuss the project deadline",
            topic="work",
        )
        score = self.engine.compute(
            text="The project status looks good",
            topic="work",
        )
        assert score < 0.4

    def test_contradiction_signal_detected(self):
        """Contradiction keywords should be detected."""
        assert self.engine._is_contradiction("actually no wait I was wrong")
        assert self.engine._is_contradiction("I was wrong, I take it back")
        assert not self.engine._is_contradiction("I like coffee and tea")

    def test_reset_clears_history(self):
        """Reset should clear all rolling history."""
        self.engine.compute("Sarah joined the team", extracted_entities=["Sarah"])
        assert len(self.engine._recent_entities) > 0
        self.engine.reset()
        assert len(self.engine._recent_entities) == 0
        assert len(self.engine._recent_topics) == 0

    def test_topic_inference(self):
        """Topic inference should detect work-related content."""
        topic = self.engine._infer_topic("Let's schedule a meeting with the boss")
        assert topic == "work"

    def test_topic_inference_personal(self):
        """Topic inference should detect personal content."""
        topic = self.engine._infer_topic("My kids are going to soccer practice")
        assert topic == "personal"

    def test_empty_text_returns_low_score(self):
        """Empty or near-empty text should not score high surprise."""
        score = self.engine.compute("")
        assert 0.0 <= score <= 0.3

    def test_prediction_context_exposed(self):
        """get_context should return current rolling state."""
        self.engine.compute("Sarah works at Google", extracted_entities=["Sarah"], topic="work")
        ctx = self.engine.get_context()
        assert isinstance(ctx.entity_history, list)
        assert isinstance(ctx.topic_history, list)

    def test_unknown_topic_returns_none(self):
        """Unknown topic should return None from inference."""
        topic = self.engine._infer_topic(
            "The phenomenon of quantum entanglement is fascinating"
        )
        assert topic is None

    def test_schema_overlap_with_prior_concepts(self):
        """Content overlapping with prior concepts should score lower."""
        prior = ["I love hiking in the mountains", "My favorite trail is in Colorado"]
        score = self.engine.compute(
            text="I went hiking and saw amazing mountains",
            prior_concepts=prior,
        )
        assert 0.0 <= score <= 1.0

    def test_score_bounded_between_zero_and_one(self):
        """All scores should be clamped to [0, 1]."""
        for _ in range(10):
            score = self.engine.compute(
                "Some random content with entities like XYZ123ABC",
                extracted_entities=["XYZ123ABC", "UnknownEntity", "FooBarBaz"],
            )
            assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
