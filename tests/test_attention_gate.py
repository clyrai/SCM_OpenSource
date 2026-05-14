"""
Tests for AttentionGate
"""
import pytest
from src.core.attention_gate import AttentionGate
from src.core.models import (
    Concept,
    ImportanceVector,
    EncodeIntensity,
    ConceptType,
)
from src.core.value_tagger import ValueTagger


class TestAttentionGate:
    """Test suite for AttentionGate (selective encoding)."""

    def setup_method(self):
        self.gate = AttentionGate()
        self.tagger = ValueTagger()

    def test_high_salience_triggers_strong_encode(self):
        """High novelty + task relevance should trigger STRONG encoding."""
        importance = ImportanceVector(
            novelty=0.95,
            emotional=0.8,
            task_relevance=0.9,
            repetition=0.1,
        )
        result = self.gate.evaluate(
            text="My name is Saish and I work at Google as an engineer",
            importance=importance,
        )
        assert result.intensity == EncodeIntensity.STRONG
        assert result.salience >= 0.6
        assert result.should_encode is True

    def test_low_salience_triggers_skip_encode(self):
        """Low salience + high noise penalty should trigger SKIP."""
        importance = ImportanceVector(
            novelty=0.05,
            emotional=0.0,
            task_relevance=0.05,
            repetition=0.0,
        )
        result = self.gate.evaluate(
            text="um uh like you know basically um so yeah",
            importance=importance,
        )
        assert result.intensity in (
            EncodeIntensity.SKIP,
            EncodeIntensity.WEAK,
        )

    def test_normal_salience_triggers_normal_encode(self):
        """Medium salience should trigger NORMAL encoding."""
        importance = ImportanceVector(
            novelty=0.5,
            emotional=0.3,
            task_relevance=0.5,
            repetition=0.3,
        )
        result = self.gate.evaluate(
            text="The weather is nice today",
            importance=importance,
        )
        assert result.intensity in (
            EncodeIntensity.NORMAL,
            EncodeIntensity.WEAK,
        )
        assert result.should_encode is True

    def test_grasp_score_high_for_clear_familiar_content(self):
        """Clear, well-formed statements should have high grasp."""
        importance = ImportanceVector(
            novelty=0.7,
            emotional=0.0,
            task_relevance=0.8,
            repetition=0.2,
        )
        result = self.gate.evaluate(
            text="My name is Alice and I work at Google",
            importance=importance,
            prior_concepts=["My name is Alice", "I work at Google"],
        )
        assert result.grasp >= 0.3

    def test_grasp_score_low_for_complex_content(self):
        """Long complex sentences should score lower on grasp."""
        importance = ImportanceVector(
            novelty=0.7,
            emotional=0.0,
            task_relevance=0.8,
            repetition=0.2,
        )
        result = self.gate.evaluate(
            text="Although the aforementioned methodology would theoretically "
            "yield improved outcomes, nevertheless, the implementation "
            "requires substantial resources which furthermore may not be "
            "available given however the existing constraints.",
            importance=importance,
        )
        assert result.grasp <= 0.6

    def test_clarity_scoring(self):
        """Short declarative sentences score high on clarity; complex ones score lower."""
        high_clarity = self.gate._compute_clarity("My name is Alice.")
        assert high_clarity >= 0.6

        low_clarity = self.gate._compute_clarity(
            "However therefore furthermore nevertheless whereas consequently"
        )
        assert low_clarity < 0.65

    def test_cognitive_load_scoring(self):
        """Complex text scores high on cognitive load."""
        low_load = self.gate._compute_cognitive_load("I like coffee.")
        high_load = self.gate._compute_cognitive_load(
            "However although nevertheless furthermore therefore "
            "consequently whereas consequently therefore."
        )
        assert high_load > low_load

    def test_noise_penalty_applied_to_filler_text(self):
        """Filler-heavy text should receive noise penalty."""
        importance = ImportanceVector(
            novelty=0.6,
            emotional=0.0,
            task_relevance=0.6,
            repetition=0.2,
        )
        result = self.gate.evaluate(
            text="um um like you know basically um so yeah",
            importance=importance,
        )
        assert result.noise_penalty > 0.1

    def test_batch_evaluation(self):
        """Batch evaluation should return one result per input."""
        texts = [
            "My name is Alice",
            "I work at Google",
            "The sky is blue",
        ]
        importances = [
            ImportanceVector(novelty=0.9, emotional=0.5, task_relevance=0.8, repetition=0.1),
            ImportanceVector(novelty=0.9, emotional=0.3, task_relevance=0.8, repetition=0.1),
            ImportanceVector(novelty=0.5, emotional=0.0, task_relevance=0.3, repetition=0.3),
        ]
        results = self.gate.evaluate_batch(texts, importances)
        assert len(results) == 3
        assert all(r.salience >= 0.0 for r in results)

    def test_episode_evaluation_helper(self):
        """evaluate_episode convenience method should work."""
        from src.core.models import Episode

        ep = Episode(
            raw_content="My name is Saish",
            importance=ImportanceVector(
                novelty=0.9, emotional=0.5, task_relevance=0.8, repetition=0.1
            ),
        )
        result = self.gate.evaluate_episode(ep)
        assert result.should_encode is True
        assert result.intensity in [EncodeIntensity.STRONG, EncodeIntensity.NORMAL]

    def test_grasp_result_detail(self):
        """compute_grasp_result should return full breakdown."""
        result = self.gate.compute_grasp_result(
            text="My name is Alice",
            salience=0.7,
            schema_overlap=0.3,
        )
        assert hasattr(result, "grasp_score")
        assert hasattr(result, "one_shot_capable")
        assert 0.0 <= result.grasp_score <= 1.0

    def test_update_prior_concepts(self):
        """prior_concepts should update rolling history."""
        concepts = [
            "I live in Mumbai",
            "I work at a startup",
            "I studied at MIT",
        ]
        self.gate.update_prior_concepts(concepts)
        assert len(self.gate._prior_concepts) == 3

    def test_schema_overlap_computed_when_prior_given(self):
        """Schema overlap should be computed when prior concepts exist."""
        importance = ImportanceVector(
            novelty=0.5,
            emotional=0.0,
            task_relevance=0.5,
            repetition=0.5,
        )
        concept = Concept(
            type=ConceptType.FACT,
            description="Alice works at Google",
        )
        result = self.gate.evaluate(
            text="Alice works at Google",
            importance=importance,
            concept=concept,
            prior_concepts=["Bob works at Google", "Alice lives in Mumbai"],
        )
        assert 0.0 <= result.schema_overlap <= 1.0

    def test_encode_result_always_has_fields(self):
        """Every EncodeResult should have all required fields."""
        importance = ImportanceVector(
            novelty=0.5, emotional=0.0, task_relevance=0.5, repetition=0.5
        )
        result = self.gate.evaluate(
            text="Hello",
            importance=importance,
        )
        assert hasattr(result, "should_encode")
        assert hasattr(result, "intensity")
        assert hasattr(result, "salience")
        assert hasattr(result, "grasp")
        assert hasattr(result, "prediction_error")
        assert hasattr(result, "reason")
        assert hasattr(result, "noise_penalty")
        assert hasattr(result, "schema_overlap")


class TestAttentionGateIntegration:
    """Integration tests for AttentionGate with ValueTagger."""

    def setup_method(self):
        self.gate = AttentionGate()
        self.tagger = ValueTagger()

    def test_full_pipeline_high_value_content(self):
        """Tag + gate pipeline should classify high-value content as STRONG or NORMAL."""
        concept = Concept(
            type=ConceptType.PERSON,
            description="My name is Alice",
        )
        importance = self.tagger.tag(concept)
        result = self.gate.evaluate(
            text="My name is Alice",
            importance=importance,
            concept=concept,
        )
        assert result.intensity in [
            EncodeIntensity.STRONG,
            EncodeIntensity.NORMAL,
        ], f"Expected STRONG or NORMAL, got {result.intensity}"
        assert result.salience >= 0.5

    def test_full_pipeline_low_value_content(self):
        """Tag + gate pipeline should classify low-value content as WEAK or SKIP."""
        concept = Concept(
            type=ConceptType.FACT,
            description="um yeah like whatever",
        )
        importance = self.tagger.tag(concept)
        result = self.gate.evaluate(
            text="um yeah like whatever so um basically",
            importance=importance,
            concept=concept,
        )
        assert result.intensity in [
            EncodeIntensity.WEAK,
            EncodeIntensity.SKIP,
        ]

    def test_prediction_error_boosts_salience(self):
        """High prediction error should boost salience."""
        importance = ImportanceVector(
            novelty=0.4,
            emotional=0.0,
            task_relevance=0.4,
            repetition=0.4,
        )
        result_without = self.gate.evaluate(
            text="Alice works at Google",
            importance=importance,
        )

        self.gate.enable_prediction_error = True
        result_with = self.gate.evaluate(
            text="Alice works at a completely different company now",
            importance=importance,
            prior_concepts=["Alice works at Google"],
        )

        assert result_with.salience >= result_without.salience


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
