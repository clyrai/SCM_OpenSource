"""
Tests for ValueTagger
"""
import pytest
from src.core.value_tagger import ValueTagger
from src.core.models import Concept, ConceptType, ImportanceVector


class TestValueTagger:
    """Test ValueTagger functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tagger = ValueTagger()

    def test_tag_positive_emotion(self):
        """Test positive emotional tagging"""
        concept = Concept(
            type=ConceptType.PREFERENCE,
            description="I love morning meetings"
        )
        importance = self.tagger.tag(concept)

        assert importance.emotional > 0

    def test_tag_negative_emotion(self):
        """Test negative emotional tagging"""
        concept = Concept(
            type=ConceptType.PREFERENCE,
            description="I hate early morning meetings"
        )
        importance = self.tagger.tag(concept)

        assert importance.emotional < 0

    def test_tag_novelty_increases_for_new_concept(self):
        """Test that novelty is high for unseen concepts"""
        concept = Concept(
            type=ConceptType.PERSON,
            description="Person: JohnDoeUnique12345"
        )
        importance = self.tagger.tag(concept)

        # First time seeing this, should be novel
        assert importance.novelty > 0.7

    def test_tag_repetition_for_seen_concept(self):
        """Test that repetition increases for seen concepts"""
        concept = Concept(
            type=ConceptType.FACT,
            description="The sky is blue"
        )

        # First tag
        imp1 = self.tagger.tag(concept)
        self.tagger.update_history(concept)

        # Second tag
        imp2 = self.tagger.tag(concept)

        # Repetition should be higher second time
        assert imp2.repetition >= imp1.repetition

    def test_importance_overall_calculation(self):
        """Test overall importance score"""
        vector = ImportanceVector(
            novelty=1.0,
            emotional=1.0,
            task_relevance=1.0,
            repetition=1.0
        )

        overall = vector.overall
        assert overall == 1.0

    def test_batch_tagging(self):
        """Test batch tagging multiple concepts"""
        concepts = [
            Concept(type=ConceptType.PERSON, description="Person: Alice"),
            Concept(type=ConceptType.PREFERENCE, description="Likes coffee"),
            Concept(type=ConceptType.FACT, description="Works at Google"),
        ]

        vectors = self.tagger.tag_batch(concepts)

        assert len(vectors) == 3
        for v in vectors:
            assert 0 <= v.novelty <= 1
            assert -1 <= v.emotional <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])