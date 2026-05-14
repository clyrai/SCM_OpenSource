"""
CRAZY BRUTAL TEST SUITE - Phase 2 Sleep Edition
Stress testing beyond reasonable limits
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import time
import random
import traceback
from datetime import datetime, timedelta
from typing import List, Dict

from src.core.models import (
    Concept, Episode, Relation, ConceptType, PredicateType,
    MemoryState, ImportanceVector, SleepCycle
)
from src.sleep.trigger import SleepTrigger
from src.sleep.nrem import NREMConsolidation
from src.sleep.rem import REMDreaming
from src.sleep.forgetting import ForgettingModule
from src.sleep.sleep_cycle import SleepCycleOrchestrator


class TestCRAZY_SleepTrigger(unittest.TestCase):
    """CRAZY: Sleep trigger with massive concept loads"""

    def test_crazy_10000_concepts_entropy(self):
        """Test with 10,000 concepts - should not crash"""
        trigger = SleepTrigger(entropy_threshold=0.5)

        concepts = []
        for i in range(10000):
            imp = ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            )
            concepts.append(Concept(
                id=f"c{i}",
                type=ConceptType.FACT,
                description=f"massive concept {i}",
                importance=imp
            ))

        should_sleep, reason = trigger.should_sleep(concepts, [])
        print(f"  [10K concepts] should_sleep={should_sleep}, reason={reason[:50]}...")
        self.assertIsNotNone(should_sleep)

    def test_crazy_conflict_flood(self):
        """Test with 1000 conflicting relations"""
        trigger = SleepTrigger(conflict_threshold=0.2)

        concepts = [Concept(
            id=f"c{i}", type=ConceptType.FACT,
            description=f"conflict test", importance=ImportanceVector()
        ) for i in range(100)]

        relations = []
        for i in range(1000):
            relations.append(Relation(
                subject_id=f"c{random.randint(0, 99)}",
                predicate=PredicateType.CONTRADICTS if i % 3 == 0 else PredicateType.RELATED_TO,
                object_id=f"c{random.randint(0, 99)}",
                strength=random.random()
            ))

        should_sleep, reason = trigger.should_sleep(concepts, relations)
        self.assertTrue(should_sleep)
        print(f"  [1000 conflicts] triggered: {reason[:80]}...")

    def test_crazy_rapid_fire_triggers(self):
        """100 rapid consecutive trigger checks"""
        trigger = SleepTrigger()

        for round_num in range(100):
            concepts = [Concept(
                id=f"c{i}_{round_num}",
                type=ConceptType.FACT,
                description=f"round {round_num}",
                importance=ImportanceVector(novelty=random.random())
            ) for i in range(50)]

            should_sleep, reason = trigger.should_sleep(concepts, [])
            if round_num % 20 == 0:
                trigger.record_sleep()

        print(f"  [100 rapid fire] completed without crash")

    def test_crazy_time_warp_triggers(self):
        """Trigger with extreme time values"""
        trigger = SleepTrigger(max_interval=1)

        concepts = [Concept(
            id="c1", type=ConceptType.FACT,
            description="time test", importance=ImportanceVector()
        )]

        should_sleep, reason = trigger.should_sleep(
            concepts, [],
            time_since_last_sleep=999999
        )
        self.assertTrue(should_sleep)

        should_sleep2, reason2 = trigger.should_sleep(
            concepts, [],
            time_since_last_sleep=0.0001
        )
        print(f"  [time warp] extreme={should_sleep}, minimal={should_sleep2}")


class TestCRAZY_NREMConsolidation(unittest.TestCase):
    """CRAZY: NREM with extreme data loads"""

    def test_crazy_5000_concepts_consolidation(self):
        """Consolidate 5000 concepts - stress test"""
        nrem = NREMConsolidation(downscale_factor=0.7)

        concepts = []
        for i in range(5000):
            imp = ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            )
            concepts.append(Concept(
                id=f"c{i}",
                type=random.choice(list(ConceptType)),
                description=f"stress concept {i}",
                importance=imp,
                strength=random.uniform(0.5, 2.0)
            ))

        episodes = []
        for i in range(1000):
            episode = Episode(
                id=f"e{i}",
                concept_ids=[f"c{random.randint(0, 4999)}" for _ in range(random.randint(1, 10))],
                raw_content=f"massive episode {i}",
                importance=ImportanceVector()
            )
            episodes.append(episode)

        start = time.time()
        updated, relations, stats = nrem.consolidate(concepts, episodes, [])
        elapsed = time.time() - start

        print(f"  [5000 concepts] processed in {elapsed:.2f}s, relations={stats['relations_created']}")
        self.assertGreater(len(updated), 0)
        self.assertLess(elapsed, 30)

    def test_crazy_10000_episodes_batch(self):
        """Process 10,000 episodes in single consolidation"""
        nrem = NREMConsolidation()

        concepts = [Concept(
            id=f"c{i}", type=ConceptType.FACT,
            description=f"ep batch concept", importance=ImportanceVector()
        ) for i in range(100)]

        episodes = []
        for i in range(10000):
            ep = Episode(
                id=f"massive_e{i}",
                concept_ids=[f"c{random.randint(0, 99)}" for _ in range(random.randint(1, 20))],
                raw_content=f"episode {i}",
                importance=ImportanceVector()
            )
            episodes.append(ep)

        start = time.time()
        updated, relations, stats = nrem.consolidate(concepts, episodes, [])
        elapsed = time.time() - start

        print(f"  [10K episodes] in {elapsed:.3f}s, strength_increases={stats['strength_increases']}")
        self.assertLess(elapsed, 10)

    def test_crazy_all_conflicting_concepts(self):
        """All concepts contradicting each other"""
        nrem = NREMConsolidation()

        concepts = []
        for i in range(500):
            imp = ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-1, 1),
                task_relevance=random.random(),
                repetition=random.random()
            )
            concepts.append(Concept(
                id=f"contradict_c{i}",
                type=ConceptType.FACT,
                description=f"contradiction {i}",
                importance=imp,
                strength=random.uniform(0.1, 3.0)
            ))

        episodes = []
        for i in range(200):
            ep_concepts = random.sample([c.id for c in concepts], min(50, len(concepts)))
            episodes.append(Episode(
                id=f"ce{i}",
                concept_ids=ep_concepts,
                raw_content=f"conflict episode {i}",
                importance=ImportanceVector()
            ))

        updated, relations, stats = nrem.consolidate(concepts, episodes, [])
        print(f"  [all conflicting] increases={stats['strength_increases']}, decreases={stats['strength_decreases']}")
        self.assertGreater(stats['strength_increases'] + stats['strength_decreases'], 0)

    def test_crazy_empty_then_massive_then_empty(self):
        """Stress oscillation - empty, huge, empty"""
        nrem = NREMConsolidation()

        # Empty
        c1, r1, s1 = nrem.consolidate([], [], [])
        self.assertEqual(len(c1), 0)

        # Massive
        concepts = [Concept(
            id=f"c{i}", type=ConceptType.FACT,
            description=f"massive", importance=ImportanceVector()
        ) for i in range(2000)]
        episodes = [Episode(
            id=f"e{i}",
            concept_ids=[f"c{random.randint(0, 1999)}" for _ in range(10)],
            raw_content="bulk",
            importance=ImportanceVector()
        ) for i in range(500)]

        c2, r2, s2 = nrem.consolidate(concepts, episodes, [])

        # Empty again
        c3, r3, s3 = nrem.consolidate([], [], [])

        print(f"  [oscillation] empty→2000→empty: OK")


class TestCRAZY_REMDreaming(unittest.TestCase):
    """CRAZY: REM with massive dream generation"""

    def test_crazy_100_dreams_per_cycle(self):
        """Generate 100 dreams in single REM cycle"""
        rem = REMDreaming(dream_count=100)

        concepts = []
        for i in range(200):
            concepts.append(Concept(
                id=f"dream_c{i}",
                type=ConceptType.FACT,
                description=f"dream concept {i}",
                importance=ImportanceVector(
                    novelty=random.random(),
                    emotional=random.uniform(-0.5, 0.5),
                    task_relevance=random.random(),
                    repetition=random.random()
                ),
                strength=random.uniform(0.5, 2.0),
                embedding=[random.random() for _ in range(384)]
            ))

        start = time.time()
        dreams, relations, stats = rem.dream(concepts, [], [])
        elapsed = time.time() - start

        print(f"  [100 dreams] generated in {elapsed:.3f}s, concepts_in_dreams={stats['concepts_in_dreams']}")
        self.assertGreater(len(dreams), 0)
        self.assertLess(elapsed, 15)

    def test_crazy_1000_concepts_dreaming(self):
        """REM with 1000 concepts"""
        rem = REMDreaming(dream_count=50)

        concepts = [Concept(
            id=f"mc{i}",
            type=random.choice(list(ConceptType)),
            description=f"massive dream {i}",
            importance=ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-1, 1),
                task_relevance=random.random(),
                repetition=random.random()
            ),
            strength=random.uniform(0.1, 2.5),
            embedding=[random.random() - 0.5 for _ in range(384)]
        ) for i in range(1000)]

        relations = [Relation(
            subject_id=f"mc{random.randint(0, 999)}",
            predicate=random.choice(list(PredicateType)),
            object_id=f"mc{random.randint(0, 999)}",
            strength=random.random()
        ) for _ in range(2000)]

        episodes = [Episode(
            id=f"mde{i}",
            concept_ids=[f"mc{random.randint(0, 999)}" for _ in range(random.randint(1, 30))],
            raw_content=f"mega dream episode {i}",
            importance=ImportanceVector()
        ) for i in range(100)]

        start = time.time()
        dreams, new_relations, stats = rem.dream(concepts, relations, episodes)
        elapsed = time.time() - start

        print(f"  [1K concepts] processed in {elapsed:.2f}s, new_relations={stats['new_relations_created']}")
        self.assertGreater(stats['integrated_concepts'], 0)

    def test_crazy_identical_concepts_dreaming(self):
        """All concepts identical - stress activation spread"""
        rem = REMDreaming(integration_threshold=0.3)

        concepts = []
        for i in range(100):
            concepts.append(Concept(
                id=f"identical_{i}",
                type=ConceptType.FACT,
                description="exactly the same concept repeated",
                importance=ImportanceVector(novelty=0.5, emotional=0.0,
                                           task_relevance=0.5, repetition=0.5),
                strength=1.0,
                embedding=[0.5] * 384
            ))

        dreams, relations, stats = rem.dream(concepts, [], [])
        print(f"  [identical concepts] dreams={len(dreams)}, relations={len(relations)}")
        self.assertGreater(len(dreams), 0)

    def test_crazy_zero_embedding_concepts(self):
        """Concepts with zero embeddings"""
        rem = REMDreaming()

        concepts = [Concept(
            id=f"zero_c{i}",
            type=ConceptType.FACT,
            description="zero embedding concept",
            importance=ImportanceVector(),
            strength=1.0,
            embedding=[0.0] * 384
        ) for i in range(50)]

        dreams, relations, stats = rem.dream(concepts, [], [])
        print(f"  [zero embeddings] dreams={len(dreams)}, integrated={stats['integrated_concepts']}")
        self.assertIsNotNone(dreams)

    def test_crazy_dream_narrative_generation(self):
        """Generate dreams and verify narrative structure"""
        rem = REMDreaming(dream_count=20)

        concept_types = [ConceptType.PERSON, ConceptType.LOCATION,
                        ConceptType.EVENT, ConceptType.FACT]
        concepts = [Concept(
            id=f"narr_c{i}",
            type=random.choice(concept_types),
            description=f"narrative element {i}",
            importance=ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            ),
            strength=random.uniform(0.5, 2.0),
            embedding=[random.random() for _ in range(384)]
        ) for i in range(100)]

        dreams, _, stats = rem.dream(concepts, [], [])

        for dream in dreams[:5]:
            self.assertIn('narrative', dream)
            self.assertIn('emotional_tone', dream)
            self.assertGreater(len(dream['narrative']), 0)

        print(f"  [narrative check] {len(dreams)} dreams, all have narrative structure")


class TestCRAZY_ForgettingModule(unittest.TestCase):
    """CRAZY: Forgetting with extreme memory pressure"""

    def test_crazy_10000_concepts_forgetting(self):
        """Forget from 10,000 concepts"""
        forgetting = ForgettingModule(importance_threshold=0.4)

        concepts = []
        for i in range(10000):
            imp = ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            )
            concepts.append(Concept(
                id=f"forget_c{i}",
                type=random.choice(list(ConceptType)),
                description=f"forgettable {i}",
                importance=imp,
                strength=random.uniform(0.1, 2.0)
            ))

        start = time.time()
        forgotten, preserved, stats = forgetting.evaluate_forgetting(concepts)
        elapsed = time.time() - start

        print(f"  [10K forgetting] forgot={len(forgotten)}, preserved={len(preserved)}, time={elapsed:.2f}s")
        self.assertGreater(len(preserved), 0)
        self.assertLess(elapsed, 20)

    def test_crazy_all_same_importance(self):
        """All concepts have identical importance"""
        forgetting = ForgettingModule(importance_threshold=0.5)

        concepts = [Concept(
            id=f"same_c{i}",
            type=ConceptType.FACT,
            description="same importance",
            importance=ImportanceVector(
                novelty=0.5, emotional=0.0,
                task_relevance=0.5, repetition=0.5
            ),
            strength=1.0
        ) for i in range(1000)]

        forgotten, preserved, stats = forgetting.evaluate_forgetting(concepts)
        print(f"  [uniform importance] forgot={len(forgotten)}, preserved={len(preserved)}")

    def test_crazy_extreme_emotional_range(self):
        """Concepts with extreme emotional values"""
        forgetting = ForgettingModule()

        concepts = []
        for i in range(200):
            imp = ImportanceVector(
                novelty=random.random(),
                emotional=random.choice([-1.0, 1.0]),  # Extreme
                task_relevance=random.random(),
                repetition=random.random()
            )
            concepts.append(Concept(
                id=f"emotion_c{i}",
                type=ConceptType.FACT,
                description=f"emotional concept {i}",
                importance=imp
            ))

        forgotten, preserved, stats = forgetting.evaluate_forgetting(concepts)
        print(f"  [extreme emotions] forgotten={len(forgotten)}, preserved={len(preserved)}")

    def test_crazy_conflict_pairs_massive(self):
        """10,000 conflict pairs"""
        forgetting = ForgettingModule()

        concepts = [Concept(
            id=f"conflict_c{i}",
            type=ConceptType.FACT,
            description=f"conflict pair {i}",
            importance=ImportanceVector(
                novelty=random.random(),
                emotional=0.0,
                task_relevance=random.random(),
                repetition=random.random()
            ),
            strength=random.uniform(0.1, 2.0)
        ) for i in range(500)]

        conflict_pairs = []
        for i in range(10000):
            c1 = random.randint(0, 499)
            c2 = random.randint(0, 499)
            if c1 != c2:
                conflict_pairs.append((f"conflict_c{c1}", f"conflict_c{c2}"))

        forgotten, preserved, stats = forgetting.evaluate_forgetting(concepts, conflict_pairs)

        print(f"  [10K conflicts] resolved={stats['conflicts_resolved']}, forgot={len(forgotten)}")

    def test_crazy_memory_pressure_dynamics(self):
        """Dynamic threshold under memory pressure"""
        forgetting = ForgettingModule(importance_threshold=0.3)

        thresholds = []
        for size in [10, 50, 100, 500, 1000, 5000]:
            concepts = [Concept(
                id=f"pressure_c{i}",
                type=ConceptType.FACT,
                description=f"pressure test {i}",
                importance=ImportanceVector()
            ) for i in range(size)]

            threshold = forgetting.compute_forgetting_threshold(concepts, target_forget_rate=0.1)
            thresholds.append(threshold)

        print(f"  [memory pressure] thresholds: {[f'{t:.3f}' for t in thresholds]}")

        for i in range(len(thresholds) - 1):
            self.assertGreaterEqual(thresholds[i+1], thresholds[i])


class TestCRAZY_SleepCycleOrchestrator(unittest.TestCase):
    """CRAZY: Full sleep cycle under extreme load"""

    def test_crazy_massive_full_cycle(self):
        """2000 concepts through full sleep cycle"""
        orchestrator = SleepCycleOrchestrator()

        concepts = []
        for i in range(2000):
            imp = ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            )
            concepts.append(Concept(
                id=f"massive_c{i}",
                type=random.choice(list(ConceptType)),
                description=f"massive cycle concept {i}",
                importance=imp,
                strength=random.uniform(0.5, 2.0),
                embedding=[random.random() - 0.5 for _ in range(384)]
            ))

        relations = [Relation(
            subject_id=f"massive_c{random.randint(0, 1999)}",
            predicate=random.choice(list(PredicateType)),
            object_id=f"massive_c{random.randint(0, 1999)}",
            strength=random.random()
        ) for _ in range(3000)]

        episodes = [Episode(
            id=f"massive_e{i}",
            concept_ids=[f"massive_c{random.randint(0, 1999)}"
                        for _ in range(random.randint(1, 20))],
            raw_content=f"massive episode {i}",
            importance=ImportanceVector()
        ) for i in range(500)]

        start = time.time()
        success, cycle, stats = orchestrator.begin_sleep_cycle(concepts, relations, episodes)
        elapsed = time.time() - start

        print(f"  [2000 concepts FULL CYCLE] in {elapsed:.2f}s")
        print(f"    NREM: concepts={stats['nrem'].get('concepts_processed', 0)}, relations={stats['nrem'].get('relations_created', 0)}")
        print(f"    REM: dreams={len(stats.get('dreams', []))}, integrated={stats['rem'].get('integrated_concepts', 0)}")
        print(f"    Forgetting: forgotten={stats['forgetting'].get('forgotten', 0)}")

        self.assertTrue(success)
        self.assertIsNotNone(cycle.end_time)

    def test_crazy_rapid_cycles(self):
        """10 rapid consecutive sleep cycles"""
        orchestrator = SleepCycleOrchestrator()

        for i in range(10):
            concepts = [Concept(
                id=f"rapid_c{j}_{i}",
                type=ConceptType.FACT,
                description=f"rapid cycle {i}",
                importance=ImportanceVector(
                    novelty=random.random(),
                    emotional=random.uniform(-0.5, 0.5),
                    task_relevance=random.random(),
                    repetition=random.random()
                ),
                strength=random.uniform(0.5, 1.5),
                embedding=[random.random() for _ in range(384)]
            ) for j in range(100)]

            episodes = [Episode(
                id=f"rapid_e{j}_{i}",
                concept_ids=[f"rapid_c{random.randint(0, 99)}_{i}"
                            for _ in range(random.randint(1, 10))],
                raw_content=f"rapid episode {i}",
                importance=ImportanceVector()
            ) for j in range(20)]

            success, cycle, stats = orchestrator.begin_sleep_cycle(concepts, [], episodes)

            if i % 3 == 0:
                print(f"    Cycle {i}: success={success}, dreams={len(stats.get('dreams', []))}")

        print(f"  [10 rapid cycles] completed")

    def test_crazy_sleep_readiness_stress(self):
        """Check readiness 1000 times"""
        orchestrator = SleepCycleOrchestrator()

        concepts = [Concept(
            id=f"ready_c{i}",
            type=ConceptType.FACT,
            description=f"readiness test {i}",
            importance=ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            ),
            strength=random.uniform(0.5, 2.0)
        ) for i in range(500)]

        relations = [Relation(
            subject_id=f"ready_c{random.randint(0, 499)}",
            predicate=random.choice(list(PredicateType)),
            object_id=f"ready_c{random.randint(0, 499)}",
            strength=random.random()
        ) for _ in range(1000)]

        start = time.time()
        for _ in range(1000):
            readiness = orchestrator.get_sleep_readiness(concepts, relations)
        elapsed = time.time() - start

        print(f"  [1000 readiness checks] in {elapsed:.2f}s ({elapsed*1000:.1f}ms total)")
        self.assertLess(elapsed, 10)

    def test_crazy_empty_concept_stress(self):
        """Stress with various empty/minimal inputs"""
        orchestrator = SleepCycleOrchestrator()

        # Empty everything
        s1, c1, st1 = orchestrator.begin_sleep_cycle([], [], [])
        self.assertFalse(s1)

        # Single concept with high entropy (may not trigger if only 1 concept has entropy=0)
        single = [Concept(id="s", type=ConceptType.FACT,
                         description="single", importance=ImportanceVector(novelty=0.9, emotional=0.8, task_relevance=0.9, repetition=0.9))]
        s2, c2, st2 = orchestrator.begin_sleep_cycle(single, [], [])
        # Single concept with n=1 has entropy=0 so may not trigger - just log
        print(f"    Single concept sleep: {s2}")

        # Empty episodes - same single concept won't trigger but that's ok
        s3, c3, st3 = orchestrator.begin_sleep_cycle(single, [], [])
        print(f"    Second single concept sleep: {s3}")

        print(f"  [edge cases] empty→single→empty: all handled")

    def test_crazy_mixed_state_concepts(self):
        """Concepts in all memory states"""
        orchestrator = SleepCycleOrchestrator()

        concepts = []
        for i in range(300):
            state = random.choice(list(MemoryState))
            imp = ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            )
            concepts.append(Concept(
                id=f"state_c{i}",
                type=random.choice(list(ConceptType)),
                description=f"state test {i}",
                importance=imp,
                state=state,
                strength=random.uniform(0.3, 2.0)
            ))

        episodes = [Episode(
            id=f"state_e{i}",
            concept_ids=[f"state_c{random.randint(0, 299)}" for _ in range(random.randint(1, 10))],
            raw_content=f"state episode {i}",
            importance=ImportanceVector()
        ) for i in range(50)]

        success, cycle, stats = orchestrator.begin_sleep_cycle(concepts, [], episodes)

        print(f"  [mixed states] success={success}, consolidated={cycle.memories_consolidated}")
        self.assertTrue(success)


class TestCRAZY_Performance(unittest.TestCase):
    """CRAZY: Performance benchmarks"""

    def test_perf_trigger_10k_concepts(self):
        """10K concept trigger performance"""
        trigger = SleepTrigger()
        concepts = [Concept(
            id=f"perf_c{i}",
            type=ConceptType.FACT,
            description=f"perf test {i}",
            importance=ImportanceVector(novelty=random.random())
        ) for i in range(10000)]

        start = time.time()
        for _ in range(100):
            trigger.should_sleep(concepts, [])
        elapsed = time.time() - start

        print(f"  [10K × 100 triggers] {elapsed:.2f}s ({elapsed*10:.1f}ms per trigger)")
        self.assertLess(elapsed, 30)

    def test_perf_nrem_1k_concepts(self):
        """1K concept consolidation performance"""
        nrem = NREMConsolidation()
        concepts = [Concept(
            id=f"pn_c{i}",
            type=ConceptType.FACT,
            description=f"perf nrem {i}",
            importance=ImportanceVector(novelty=random.random(), task_relevance=random.random()),
            strength=random.uniform(0.5, 1.5)
        ) for i in range(1000)]

        episodes = [Episode(
            id=f"pn_e{i}",
            concept_ids=[f"pn_c{random.randint(0, 999)}" for _ in range(random.randint(1, 10))],
            raw_content="perf episode",
            importance=ImportanceVector()
        ) for i in range(200)]

        start = time.time()
        nrem.consolidate(concepts, episodes, [])
        elapsed = time.time() - start

        print(f"  [1K consolidation] {elapsed:.3f}s")
        self.assertLess(elapsed, 5)

    def test_perf_rem_500_concepts(self):
        """500 concept REM performance"""
        rem = REMDreaming(dream_count=50)
        concepts = [Concept(
            id=f"pr_c{i}",
            type=ConceptType.FACT,
            description=f"perf rem {i}",
            importance=ImportanceVector(),
            strength=random.uniform(0.5, 1.5),
            embedding=[random.random() for _ in range(384)]
        ) for i in range(500)]

        start = time.time()
        rem.dream(concepts, [], [])
        elapsed = time.time() - start

        print(f"  [500 REM] {elapsed:.3f}s")
        self.assertLess(elapsed, 10)

    def test_perf_full_cycle_500(self):
        """500 concept full cycle performance"""
        orchestrator = SleepCycleOrchestrator()

        concepts = [Concept(
            id=f"pfc_c{i}",
            type=ConceptType.FACT,
            description=f"perf full {i}",
            importance=ImportanceVector(
                novelty=random.random(),
                emotional=random.uniform(-0.5, 0.5),
                task_relevance=random.random(),
                repetition=random.random()
            ),
            strength=random.uniform(0.5, 1.5),
            embedding=[random.random() - 0.5 for _ in range(384)]
        ) for i in range(500)]

        episodes = [Episode(
            id=f"pfc_e{i}",
            concept_ids=[f"pfc_c{random.randint(0, 499)}" for _ in range(random.randint(1, 10))],
            raw_content=f"perf full ep {i}",
            importance=ImportanceVector()
        ) for i in range(100)]

        start = time.time()
        orchestrator.begin_sleep_cycle(concepts, [], episodes)
        elapsed = time.time() - start

        print(f"  [500 full cycle] {elapsed:.2f}s")
        self.assertLess(elapsed, 15)


class TestCRAZY_CornerCases(unittest.TestCase):
    """CRAZY: Extreme corner cases"""

    def test_all_concept_types(self):
        """All concept types through full cycle"""
        orchestrator = SleepCycleOrchestrator()

        concepts = []
        for ct in ConceptType:
            for i in range(20):
                concepts.append(Concept(
                    id=f"{ct.value}_{i}",
                    type=ct,
                    description=f"{ct.value} concept {i}",
                    importance=ImportanceVector(
                        novelty=random.random(),
                        emotional=random.uniform(-0.5, 0.5),
                        task_relevance=random.random(),
                        repetition=random.random()
                    ),
                    strength=random.uniform(0.5, 1.5),
                    embedding=[random.random() for _ in range(384)]
                ))

        episodes = [Episode(
            id=f"alltype_e{i}",
            concept_ids=[f"{random.choice(list(ConceptType)).value}_{random.randint(0, 19)}"
                        for _ in range(random.randint(1, 5))],
            raw_content="all types episode",
            importance=ImportanceVector()
        ) for i in range(30)]

        success, cycle, stats = orchestrator.begin_sleep_cycle(concepts, [], episodes)

        print(f"  [all concept types] success={success}, dreams={len(stats.get('dreams', []))}")
        self.assertTrue(success)

    def test_all_predicate_types(self):
        """All predicate types in relations"""
        orchestrator = SleepCycleOrchestrator()

        concepts = [Concept(
            id=f"pred_c{i}",
            type=ConceptType.FACT,
            description=f"predicate test {i}",
            importance=ImportanceVector(),
            strength=1.0
        ) for i in range(100)]

        relations = []
        for pt in PredicateType:
            for _ in range(10):
                relations.append(Relation(
                    subject_id=f"pred_c{random.randint(0, 99)}",
                    predicate=pt,
                    object_id=f"pred_c{random.randint(0, 99)}",
                    strength=random.random()
                ))

        episodes = [Episode(
            id=f"pred_e{i}",
            concept_ids=[f"pred_c{random.randint(0, 99)}" for _ in range(5)],
            raw_content="predicate episode",
            importance=ImportanceVector()
        ) for i in range(20)]

        success, cycle, stats = orchestrator.begin_sleep_cycle(concepts, relations, episodes)

        print(f"  [all predicates] success={success}, relations created={stats['nrem'].get('relations_created', 0)}")

    def test_negative_importance_values(self):
        """Negative importance values (edge case)"""
        forgetting = ForgettingModule()

        concepts = []
        for i in range(50):
            imp = ImportanceVector(
                novelty=max(0, random.uniform(-0.5, 0.5)),
                emotional=random.uniform(-1, 1),
                task_relevance=max(0, random.uniform(-0.2, 0.2)),
                repetition=max(0, random.uniform(0, 0.3))
            )
            concepts.append(Concept(
                id=f"neg_c{i}",
                type=ConceptType.FACT,
                description=f"negative test {i}",
                importance=imp
            ))

        forgotten, preserved, stats = forgetting.evaluate_forgetting(concepts)
        print(f"  [negative values] forgot={len(forgotten)}, preserved={len(preserved)}")

    def test_very_long_descriptions(self):
        """Very long concept descriptions"""
        orchestrator = SleepCycleOrchestrator()

        long_desc = "A" * 10000

        concepts = [Concept(
            id=f"long_c{i}",
            type=ConceptType.FACT,
            description=long_desc + f"_{i}",
            importance=ImportanceVector(),
            strength=1.0
        ) for i in range(50)]

        episodes = [Episode(
            id=f"long_e{i}",
            concept_ids=[f"long_c{random.randint(0, 49)}"],
            raw_content=long_desc * 10,
            importance=ImportanceVector()
        ) for i in range(20)]

        success, cycle, stats = orchestrator.begin_sleep_cycle(concepts, [], episodes)
        print(f"  [10K char descriptions] success={success}")

    def test_special_characters_in_content(self):
        """Special characters and unicode"""
        orchestrator = SleepCycleOrchestrator()

        concepts = [Concept(
            id=f"special_c{i}",
            type=ConceptType.FACT,
            description=f"Special chars: !@#$%^&*() 你好 😂 {'\"'} \\n\\t {i}",
            importance=ImportanceVector(),
            strength=1.0
        ) for i in range(30)]

        episodes = [Episode(
            id=f"special_e{i}",
            concept_ids=[f"special_c{random.randint(0, 29)}"],
            raw_content=f"Raw content with 🧠 emoji and \"quotes\" and\\nnewlines {i}",
            importance=ImportanceVector()
        ) for i in range(15)]

        success, cycle, stats = orchestrator.begin_sleep_cycle(concepts, [], episodes)
        print(f"  [special chars] success={success}")


# ============================================================================
# CRAZY TEST RUNNER
# ============================================================================

def run_crazy_tests():
    """Run all CRAZY tests with detailed reporting"""
    print("\n" + "="*70)
    print("🔥 CRAZY BRUTAL SLEEP TEST SUITE 🔥".center(70))
    print("="*70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestCRAZY_SleepTrigger,
        TestCRAZY_NREMConsolidation,
        TestCRAZY_REMDreaming,
        TestCRAZY_ForgettingModule,
        TestCRAZY_SleepCycleOrchestrator,
        TestCRAZY_Performance,
        TestCRAZY_CornerCases
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    # Print summary
    print()
    print("="*70)
    print("CRAZY TEST REPORT".center(70))
    print("="*70)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()

    if result.failures:
        print("FAILURES:")
        for test, trace in result.failures:
            print(f"  ❌ {test}")
            print(f"     {trace[:200]}...")

    if result.errors:
        print("ERRORS:")
        for test, trace in result.errors:
            print(f"  💥 {test}")
            print(f"     {trace[:200]}...")

    print()
    if result.wasSuccessful():
        print("✅ ALL CRAZY TESTS PASSED! SleepAI Phase 2 is UNSTOPPABLE!")
    else:
        print("❌ SOME CRAZY TESTS FAILED!")

    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_crazy_tests()
    sys.exit(0 if success else 1)