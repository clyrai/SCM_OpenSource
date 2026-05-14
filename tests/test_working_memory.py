"""
Tests for WorkingMemory
"""
import pytest
from src.core.working_memory import WorkingMemory
from src.core.models import Episode, ImportanceVector, MemoryState
from src.core.time_utils import utc_now


class TestWorkingMemory:
    """Test WorkingMemory functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.wm = WorkingMemory(capacity=3)  # Small capacity for testing

    def test_store_episode(self):
        """Test storing an episode"""
        episode = Episode(
            concept_ids=["c1", "c2"],
            raw_content="My name is Saish",
            importance=ImportanceVector()
        )

        self.wm.store(episode)
        assert self.wm.size() == 1

    def test_capacity_enforcement(self):
        """Test that capacity is enforced"""
        # Add more than capacity
        for i in range(5):
            episode = Episode(
                concept_ids=[f"c{i}"],
                raw_content=f"Episode {i}",
                importance=ImportanceVector()
            )
            self.wm.store(episode)

        # Should be at capacity
        assert self.wm.size() == 3
        assert self.wm.is_full()

    def test_retrieve_returns_recent(self):
        """Test retrieve returns most recent"""
        for i in range(3):
            episode = Episode(
                concept_ids=[f"c{i}"],
                raw_content=f"Episode {i}",
                importance=ImportanceVector(novelty=i/10)
            )
            self.wm.store(episode)

        retrieved = self.wm.retrieve(limit=2)
        assert len(retrieved) == 2

    def test_remove_episode(self):
        """Test removing specific episode"""
        episode = Episode(
            concept_ids=["c1"],
            raw_content="Test",
            importance=ImportanceVector()
        )
        self.wm.store(episode)

        assert self.wm.size() == 1
        result = self.wm.remove_episode(episode.id)
        assert result == True
        assert self.wm.size() == 0

    def test_get_by_state(self):
        """Test filtering by state"""
        episode1 = Episode(
            concept_ids=["c1"],
            raw_content="Active",
            state=MemoryState.ACTIVE
        )
        episode2 = Episode(
            concept_ids=["c2"],
            raw_content="Consolidating",
            state=MemoryState.CONSOLIDATING
        )

        self.wm.store(episode1)
        self.wm.store(episode2)

        active = self.wm.get_by_state(MemoryState.ACTIVE)
        assert len(active) == 1

        consolidating = self.wm.get_by_state(MemoryState.CONSOLIDATING)
        assert len(consolidating) == 1

    def test_clear(self):
        """Test clearing all episodes"""
        for i in range(3):
            episode = Episode(concept_ids=[f"c{i}"], raw_content=f"Episode {i}")
            self.wm.store(episode)

        self.wm.clear()
        assert self.wm.size() == 0

    def test_to_dict_serialization(self):
        """Test serialization to dict"""
        episode = Episode(
            concept_ids=["c1"],
            raw_content="Test episode",
            importance=ImportanceVector(novelty=0.8)
        )
        self.wm.store(episode)

        data = self.wm.to_dict()
        assert 'episodes' in data
        assert len(data['episodes']) == 1

    def test_from_dict_deserialization(self):
        """Test deserialization from dict"""
        data = {
            'episodes': [
                {
                    'id': 'e1',
                    'timestamp': utc_now().isoformat(),
                    'concept_ids': ['c1'],
                    'raw_content': 'Test',
                    'context': {},
                    'importance': {'novelty': 0.5, 'emotional': 0.0, 'task_relevance': 0.5, 'repetition': 0.5},
                    'state': 'active',
                    'source': 'user'
                }
            ]
        }

        wm = WorkingMemory.from_dict(data)
        assert wm.size() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
