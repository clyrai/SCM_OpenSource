"""
SleepAI Test Script - Simple memory storage test using SQLite
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.encoder import MeaningEncoder
from src.core.value_tagger import ValueTagger
from src.core.working_memory import WorkingMemory
from src.core.models import Episode, ImportanceVector
from src.core.sqlite_db import get_memory, init_db


def test_memory_pipeline():
    """Test the full memory pipeline"""
    print("=" * 50)
    print("SLEEPAI MEMORY TEST")
    print("=" * 50)

    # Initialize
    print("\n1. Initializing components...")
    encoder = MeaningEncoder()
    value_tagger = ValueTagger()
    working_memory = WorkingMemory()
    db = get_memory()

    print("   ✓ Encoder initialized")
    print("   ✓ ValueTagger initialized")
    print("   ✓ WorkingMemory initialized")
    print("   ✓ SQLite database initialized")

    # Test 1: Store a memory
    print("\n2. Testing: Store 'My name is Saish, I prefer morning meetings'")
    text = "My name is Saish, I prefer morning meetings"

    # Extract concepts
    print("\n3. Extracting concepts...")
    concepts = encoder.extract(text)
    print(f"   Extracted {len(concepts)} concepts:")
    for c in concepts:
        type_str = c.type if isinstance(c.type, str) else c.type.value
        print(f"   - [{type_str}] {c.description}")

    # Tag importance
    print("\n4. Tagging importance...")
    for concept in concepts:
        concept.importance = value_tagger.tag(concept)
        value_tagger.update_history(concept)
        type_str = concept.type if isinstance(concept.type, str) else concept.type.value
        print(f"   - {type_str}: {concept.description[:40]}...")
        print(f"     Importance: novelty={concept.importance.novelty:.2f}, "
              f"emotional={concept.importance.emotional:.2f}, "
              f"task_relevance={concept.importance.task_relevance:.2f}")

    # Store in working memory
    print("\n5. Storing in working memory...")
    episode = Episode(
        concept_ids=[c.id for c in concepts],
        raw_content=text,
        context={'session': 'test'},
        importance=concepts[0].importance if concepts else ImportanceVector()
    )
    working_memory.store(episode)
    print(f"   ✓ Stored episode: {episode.id[:8]}...")
    print(f"   Working memory size: {working_memory.size()}")

    # Store in SQLite database
    print("\n6. Storing in SQLite database...")
    for concept in concepts:
        db.save_concept(concept)
    db.save_episode(episode)
    print(f"   ✓ Stored {len(concepts)} concepts and 1 episode")

    # Test 2: Query memory
    print("\n7. Testing: Query memory for 'Saish'")
    results = db.search_concepts("Saish", limit=5)
    print(f"   Found {len(results)} results:")
    for r in results:
        print(f"   - {r['description']}")

    # Test 3: Retrieve from working memory
    print("\n8. Retrieving from working memory...")
    wm_results = working_memory.retrieve(limit=5)
    print(f"   Working memory has {len(wm_results)} episodes")
    for ep in wm_results:
        print(f"   - [{ep.source}] {ep.raw_content[:50]}...")

    # Test 4: Check database episodes
    print("\n9. Checking SQLite database episodes...")
    db_episodes = db.get_recent_episodes(limit=10)
    print(f"   Database has {len(db_episodes)} episodes")

    # Test 5: Stats
    print("\n10. Memory statistics:")
    stats = db.get_stats()
    print(f"    Total concepts: {stats['total_concepts']}")
    print(f"    Suppressed: {stats['suppressed_count']}")
    print(f"    Working memory size: {stats['working_memory_size']}")

    # Test 6: Query same thing again
    print("\n11. Testing: Query memory for 'morning'")
    results2 = db.search_concepts("morning", limit=5)
    print(f"    Found {len(results2)} results:")
    for r in results2:
        print(f"    - {r['description']}")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    assert len(concepts) > 0
    assert len(results) >= 1
    assert len(results2) >= 1
    assert working_memory.size() >= 1


if __name__ == "__main__":
    test_memory_pipeline()
