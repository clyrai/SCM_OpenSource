"""
Phase 4 MicroSleep tests.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.core.models import Concept, ConceptType, Episode, ImportanceVector, PredicateType, Relation
from src.sleep.micro_sleep import MicroSleep


class TestMicroSleep(unittest.TestCase):
    def test_micro_sleep_replay_and_light_decay(self):
        micro = MicroSleep(replay_top_k=1, light_decay_factor=0.90)

        replay_target = Concept(
            id="r1",
            type=ConceptType.FACT,
            description="important unstable trace",
            importance=ImportanceVector(novelty=0.9, emotional=0.0, task_relevance=0.8, repetition=0.2),
            strength=1.0,
        )
        replay_target.retention_score = 0.1
        replay_target.prediction_error = 0.9

        weak_trace = Concept(
            id="w1",
            type=ConceptType.FACT,
            description="weak low-value trace",
            importance=ImportanceVector(novelty=0.05, emotional=0.0, task_relevance=0.05, repetition=0.05),
            strength=0.4,
        )
        weak_trace.retention_score = 0.2

        before_weak = weak_trace.strength
        updated_concepts, _, stats = micro.run(
            concepts=[replay_target, weak_trace],
            relations=[],
            episodes=[],
        )
        updated_map = {c.id: c for c in updated_concepts}

        self.assertEqual(stats["replayed"], 1)
        self.assertGreater(updated_map["r1"].strength, 1.0)
        self.assertLess(updated_map["w1"].strength, before_weak)

    def test_micro_sleep_reinforces_and_creates_relations(self):
        micro = MicroSleep(replay_top_k=1, min_pair_repeats=2)

        c1 = Concept(id="c1", type=ConceptType.FACT, description="user likes jazz", importance=ImportanceVector(), strength=1.0)
        c2 = Concept(id="c2", type=ConceptType.FACT, description="user plays piano", importance=ImportanceVector(), strength=1.0)
        c3 = Concept(id="c3", type=ConceptType.FACT, description="user attends concerts", importance=ImportanceVector(), strength=1.0)

        existing = Relation(
            subject_id="c1",
            predicate=PredicateType.RELATED_TO,
            object_id="c2",
            strength=0.5,
            bidirectional=True,
        )

        episodes = [
            Episode(id="e1", concept_ids=["c1", "c2"], raw_content="jazz and piano", importance=ImportanceVector()),
            Episode(id="e2", concept_ids=["c1", "c2"], raw_content="jazz and piano again", importance=ImportanceVector()),
            Episode(id="e3", concept_ids=["c2", "c3"], raw_content="concerts and piano", importance=ImportanceVector()),
            Episode(id="e4", concept_ids=["c2", "c3"], raw_content="concerts and piano again", importance=ImportanceVector()),
        ]

        _, updated_relations, stats = micro.run(
            concepts=[c1, c2, c3],
            relations=[existing],
            episodes=episodes,
        )

        rel_map = {(r.subject_id, r.object_id): r for r in updated_relations}

        self.assertGreaterEqual(stats["relations_reinforced"], 1)
        self.assertGreaterEqual(stats["relations_created"], 1)
        self.assertGreater(rel_map[("c1", "c2")].strength, 0.5)
        has_c2_c3 = ("c2", "c3") in rel_map or ("c3", "c2") in rel_map
        self.assertTrue(has_c2_c3)

    def test_micro_sleep_merges_duplicates_and_rewrites_relations(self):
        micro = MicroSleep(duplicate_similarity=0.75)

        c1 = Concept(
            id="c1",
            type=ConceptType.PREFERENCE,
            description="User likes green tea",
            importance=ImportanceVector(novelty=0.8, emotional=0.0, task_relevance=0.8, repetition=0.6),
            strength=1.2,
        )
        c2 = Concept(
            id="c2",
            type=ConceptType.PREFERENCE,
            description="User likes green tea daily",
            importance=ImportanceVector(novelty=0.6, emotional=0.0, task_relevance=0.6, repetition=0.5),
            strength=0.8,
        )
        c3 = Concept(
            id="c3",
            type=ConceptType.FACT,
            description="Tea is consumed in the evening",
            importance=ImportanceVector(),
            strength=1.0,
        )

        rel = Relation(
            subject_id="c2",
            predicate=PredicateType.RELATED_TO,
            object_id="c3",
            strength=0.6,
            bidirectional=False,
        )

        updated_concepts, updated_relations, stats = micro.run(
            concepts=[c1, c2, c3],
            relations=[rel],
            episodes=[],
        )

        concept_ids = {c.id for c in updated_concepts}
        self.assertEqual(stats["duplicates_merged"], 1)
        self.assertNotIn("c2", concept_ids)

        rewritten = any(r.subject_id == "c1" and r.object_id == "c3" for r in updated_relations)
        self.assertTrue(rewritten)


if __name__ == "__main__":
    unittest.main(verbosity=2)
