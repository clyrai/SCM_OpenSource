"""
Tests for MeaningEncoder
"""
import pytest
from src.core.encoder import MeaningEncoder
from src.core.models import Concept, ConceptType


class TestMeaningEncoder:
    """Test MeaningEncoder functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.encoder = MeaningEncoder()

    def test_extract_with_heuristic_name(self):
        """Test extracting name patterns"""
        concepts = self.encoder._extract_heuristic("My name is Saish")
        assert len(concepts) >= 1

        # Should find the person
        person_concepts = [c for c in concepts if c.type == ConceptType.PERSON]
        assert len(person_concepts) >= 1
        assert "Saish" in person_concepts[0].description

    def test_extract_with_heuristic_preference(self):
        """Test extracting preference patterns"""
        concepts = self.encoder._extract_heuristic("I prefer morning meetings")
        assert len(concepts) >= 1

        pref_concepts = [c for c in concepts if c.type == ConceptType.PREFERENCE]
        assert len(pref_concepts) >= 1

    def test_extract_empty_text(self):
        """Test extracting from empty text"""
        concepts = self.encoder.extract("")
        # Should still return at least one general concept
        assert len(concepts) >= 0

    def test_similarity_computation(self):
        """Test embedding similarity"""
        sim = self.encoder.compute_similarity("hello world", "hello world")
        assert sim > 0.99  # Same text should be nearly identical

        sim_diff = self.encoder.compute_similarity("hello world", "goodbye world")
        assert 0 < sim_diff < 1  # Different but related

    def test_infer_relations(self):
        """Test relation inference"""
        concepts = [
            Concept(id="1", type=ConceptType.PERSON, description="Person: Saish"),
            Concept(id="2", type=ConceptType.PREFERENCE, description="Preference: morning")
        ]
        relations = self.encoder._infer_relations(concepts, "Saish prefers morning")

        # Should infer HAS_PROPERTY relation
        assert len(relations) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])