"""
Phase 2 Sleep Module Tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.core.models import Concept, Episode, Relation, ConceptType, PredicateType, MemoryState, ImportanceVector
from src.core.time_utils import utc_now
from src.sleep.trigger import SleepTrigger
from src.sleep.nrem import NREMConsolidation
from src.sleep.rem import REMDreaming
from src.sleep.forgetting import ForgettingModule
from src.sleep.sleep_cycle import SleepCycleOrchestrator


class TestSleepTrigger(unittest.TestCase):
    """Test SleepTrigger module"""

    def setUp(self):
        self.trigger = SleepTrigger(
            entropy_threshold=0.7,
            conflict_threshold=0.3,
            max_interval=100
        )

    def test_no_sleep_when_empty(self):
        should_sleep, reason = self.trigger.should_sleep([], [])
        self.assertFalse(should_sleep)
        self.assertIn("No concepts", reason)

    def test_sleep_on_high_entropy(self):
        concepts = []
        for i in range(10):
            imp = ImportanceVector(
                novelty=0.5, emotional=0.0,
                task_relevance=0.5, repetition=0.5
            )
            concepts.append(Concept(
                id=f"c{i}",
                type=ConceptType.FACT,
                description=f"concept {i}",
                importance=imp
            ))

        should_sleep, reason = self.trigger.should_sleep(concepts, [])
        self.assertTrue(should_sleep)
        self.assertIn("entropy", reason.lower())

    def test_sleep_on_time_interval(self):
        concepts = [Concept(
            id="c1", type=ConceptType.FACT,
            description="test", importance=ImportanceVector()
        )]
        self.trigger.record_sleep(utc_now())

        should_sleep, reason = self.trigger.should_sleep(
            concepts, [], time_since_last_sleep=150
        )
        self.assertTrue(should_sleep)
        self.assertIn("interval", reason.lower())

    def test_record_sleep(self):
        now = utc_now()
        self.trigger.record_sleep(now)
        time_since = self.trigger.time_since_last_sleep()
        self.assertIsNotNone(time_since)
        self.assertLess(time_since, 1.0)

    def test_get_trigger_stats(self):
        concepts = [Concept(
            id="c1", type=ConceptType.FACT,
            description="test", importance=ImportanceVector()
        )]
        stats = self.trigger.get_trigger_stats(concepts, [])
        self.assertIn('current_entropy', stats)
        self.assertIn('should_sleep', stats)


class TestNREMConsolidation(unittest.TestCase):
    """Test NREM consolidation module"""

    def setUp(self):
        self.nrem = NREMConsolidation(
            downscale_factor=0.8,
            hebbian_learning_rate=0.1,
            consolidation_threshold=0.4
        )

    def test_empty_consolidation(self):
        concepts, relations, stats = self.nrem.consolidate([], [], [])
        self.assertEqual(len(concepts), 0)
        self.assertEqual(stats['concepts_processed'], 0)

    def test_hebbian_strengthening(self):
        concept = Concept(
            id="c1",
            type=ConceptType.FACT,
            description="test concept",
            importance=ImportanceVector(novelty=0.7, emotional=0.0,
                                       task_relevance=0.6, repetition=0.5),
            strength=1.0
        )

        episode = Episode(
            id="e1",
            concept_ids=["c1"],
            raw_content="test",
            importance=ImportanceVector()
        )

        concepts, relations, stats = self.nrem.consolidate(
            [concept], [episode], []
        )

        self.assertGreater(len(concepts), 0)
        self.assertGreater(stats['concepts_processed'], 0)

    def test_synaptic_downscaling(self):
        concepts = [
            Concept(id=f"c{i}", type=ConceptType.FACT,
                   description=f"concept {i}", importance=ImportanceVector(),
                   strength=1.5)
            for i in range(5)
        ]

        updated, _, _ = self.nrem.consolidate(concepts, [], [])
        for c in updated:
            self.assertLessEqual(c.strength, 1.5)


class TestREMDreaming(unittest.TestCase):
    """Test REM dreaming module"""

    def setUp(self):
        self.rem = REMDreaming(
            dream_count=3,
            novelty_factor=0.3,
            integration_threshold=0.4
        )

    def test_empty_dream(self):
        dreams, relations, stats = self.rem.dream([], [], [])
        self.assertEqual(len(dreams), 0)
        self.assertEqual(stats['dreams_generated'], 0)

    def test_dream_generation(self):
        concepts = [
            Concept(id=f"c{i}", type=ConceptType.FACT,
                   description=f"concept {i}",
                   importance=ImportanceVector(),
                   strength=1.0,
                   embedding=[0.1] * 384)
            for i in range(10)
        ]

        dreams, relations, stats = self.rem.dream(concepts, [], [])

        self.assertGreater(stats['dreams_generated'], 0)
        self.assertGreater(stats['concepts_in_dreams'], 0)

    def test_activation_spread(self):
        c1 = Concept(id="c1", type=ConceptType.FACT, description="test1",
                    importance=ImportanceVector(), strength=1.0,
                    embedding=[0.5] * 384)
        c2 = Concept(id="c2", type=ConceptType.FACT, description="test2",
                    importance=ImportanceVector(), strength=1.0,
                    embedding=[0.5] * 384)

        activation = self.rem._compute_activation_spread(c1, c2)
        self.assertGreaterEqual(activation, 0.0)
        self.assertLessEqual(activation, 1.0)


class TestForgettingModule(unittest.TestCase):
    """Test forgetting module"""

    def setUp(self):
        # These tests exercise legacy aggressive forgetting mechanics. The
        # production default in v0.7.1+ is protect_salience=0.5 which would
        # block forgetting of any default-salience concept; opt back in to
        # legacy behavior here so the assertions about specific concepts
        # being forgotten still hold.
        self.forgetting = ForgettingModule(
            importance_threshold=0.3,
            decay_rate=0.05,
            protect_salience=0.0,
            min_rehearsal_before_archive=0,
            freshness_floor_hours=0.0,
        )

    def test_empty_evaluation(self):
        forgotten, preserved, stats = self.forgetting.evaluate_forgetting([])
        self.assertEqual(len(forgotten), 0)
        self.assertEqual(stats['total_evaluated'], 0)

    def test_importance_based_forgetting(self):
        concepts = [
            Concept(id="c1", type=ConceptType.FACT, description="high value",
                   importance=ImportanceVector(novelty=0.9, emotional=0.5,
                                              task_relevance=0.9, repetition=0.9),
                   strength=1.5),
            Concept(id="c2", type=ConceptType.FACT, description="low value",
                   importance=ImportanceVector(novelty=0.05, emotional=0.0,
                                              task_relevance=0.05, repetition=0.05),
                   strength=0.3)
        ]

        forgotten, preserved, stats = self.forgetting.evaluate_forgetting(concepts)

        self.assertIn("c2", forgotten)
        self.assertIn("c1", preserved)

    def test_forgetting_threshold_dynamic(self):
        concepts = [Concept(
            id=f"c{i}", type=ConceptType.FACT,
            description=f"concept {i}", importance=ImportanceVector()
        ) for i in range(50)]

        threshold = self.forgetting.compute_forgetting_threshold(concepts)
        self.assertGreaterEqual(threshold, 0.3)

    def test_get_forgetting_stats(self):
        concepts = [Concept(
            id="c1", type=ConceptType.FACT,
            description="test", importance=ImportanceVector()
        )]
        stats = self.forgetting.get_forgetting_stats(concepts)
        self.assertIn('total_concepts', stats)
        self.assertIn('forgettable', stats)


class TestSleepCycleOrchestrator(unittest.TestCase):
    """Test sleep cycle orchestrator"""

    def setUp(self):
        self.orchestrator = SleepCycleOrchestrator()

    def test_check_should_sleep_empty(self):
        should_sleep, reason, stats = self.orchestrator.check_should_sleep([], [])
        self.assertFalse(should_sleep)

    def test_full_sleep_cycle(self):
        concepts = [
            Concept(
                id=f"c{i}",
                type=ConceptType.FACT,
                description=f"concept {i}",
                importance=ImportanceVector(
                    novelty=min(0.5 + i*0.05, 1.0),
                    emotional=0.0,
                    task_relevance=0.5,
                    repetition=0.5
                ),
                strength=1.0,
                embedding=[0.1] * 384
            )
            for i in range(15)
        ]

        relations = [
            Relation(
                subject_id="c0",
                predicate=PredicateType.RELATED_TO,
                object_id="c1",
                strength=0.8
            )
        ]

        episodes = [
            Episode(
                id=f"e{i}",
                concept_ids=[f"c{i}", f"c{i+1}"],
                raw_content=f"episode {i}",
                importance=ImportanceVector()
            )
            for i in range(5)
        ]

        success, cycle, stats = self.orchestrator.begin_sleep_cycle(
            concepts, relations, episodes
        )

        self.assertTrue(success)
        self.assertIsNotNone(cycle)
        self.assertIn('nrem', stats)
        self.assertIn('rem', stats)
        self.assertIn('forgetting', stats)

    def test_get_sleep_readiness(self):
        concepts = [Concept(
            id="c1", type=ConceptType.FACT,
            description="test", importance=ImportanceVector()
        )]
        readiness = self.orchestrator.get_sleep_readiness(concepts, [])
        self.assertIn('should_sleep', readiness)
        self.assertIn('entropy', readiness)
        self.assertIn('suggested_mode', readiness)

    def test_select_sleep_mode(self):
        concepts = [
            Concept(
                id=f"c{i}",
                type=ConceptType.FACT,
                description=f"concept {i}",
                importance=ImportanceVector(novelty=0.8, emotional=0.0, task_relevance=0.8, repetition=0.5),
            )
            for i in range(8)
        ]
        mode, reason, stats = self.orchestrator.select_sleep_mode(concepts, [], turns_since_micro=5, session_turns=3)
        self.assertIn(mode, {"micro", "deep", None})
        self.assertIsInstance(reason, str)
        self.assertIn('selected_mode', stats)


if __name__ == '__main__':
    unittest.main(verbosity=2)
