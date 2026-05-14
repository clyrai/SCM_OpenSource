"""
Phase 4 DeepSleep and sleep-mode routing tests.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone

from src.core.models import Concept, ConceptType, Episode, ImportanceVector, PredicateType, Relation
from src.sleep.deep_sleep import DeepSleep
from src.sleep.rem import REMDreaming
from src.sleep.sleep_cycle import SleepCycleOrchestrator


class TestDeepSleep(unittest.TestCase):
    def test_deep_sleep_downscale_and_forgetting(self):
        # Opt into legacy aggressive forgetting: v0.7.1 raised the salience
        # protection floor to 0.5, which would block this assertion that
        # the low-value concept gets forgotten.
        from src.sleep.forgetting_dynamics import ForgettingDynamics
        legacy_forgetting = ForgettingDynamics(
            protect_salience=0.0,
            min_rehearsal_before_archive=0,
            freshness_floor_hours=0.0,
        )
        deep = DeepSleep(
            global_downscale_factor=0.90,
            enable_synthesis=False,
            forgetting=legacy_forgetting,
        )

        high = Concept(
            id="h1",
            type=ConceptType.FACT,
            description="high-value durable memory",
            importance=ImportanceVector(novelty=0.9, emotional=0.2, task_relevance=0.9, repetition=0.8),
            strength=1.2,
        )
        low = Concept(
            id="l1",
            type=ConceptType.FACT,
            description="low-value noise memory",
            importance=ImportanceVector(novelty=0.05, emotional=0.0, task_relevance=0.05, repetition=0.05),
            strength=0.3,
        )

        before_high = high.strength
        updated_concepts, updated_relations, stats = deep.run(
            concepts=[high, low],
            relations=[],
            episodes=[],
        )

        ids = {c.id for c in updated_concepts}
        high_updated = next(c for c in updated_concepts if c.id == "h1")

        self.assertEqual(stats["mode"], "deep")
        self.assertIn("nrem", stats)
        self.assertIn("forgetting", stats)
        self.assertIn("h1", ids)
        self.assertNotIn("l1", ids)
        self.assertLess(high_updated.strength, before_high)
        self.assertEqual(len(updated_relations), 0)

    def test_deep_sleep_with_synthesis_generates_dreams(self):
        rem = REMDreaming(dream_count=2, novelty_factor=0.0, integration_threshold=0.3)
        deep = DeepSleep(rem=rem, enable_synthesis=True)

        concepts = [
            Concept(
                id=f"c{i}",
                type=ConceptType.FACT,
                description=f"concept {i}",
                importance=ImportanceVector(novelty=0.6, emotional=0.0, task_relevance=0.6, repetition=0.6),
                strength=1.0,
                embedding=[0.1 + i * 0.001] * 384,
            )
            for i in range(8)
        ]
        episodes = [
            Episode(id="e1", concept_ids=["c0", "c1", "c2"], raw_content="episode one", importance=ImportanceVector()),
            Episode(id="e2", concept_ids=["c3", "c4", "c5"], raw_content="episode two", importance=ImportanceVector()),
        ]

        _, _, stats = deep.run(
            concepts=concepts,
            relations=[],
            episodes=episodes,
        )

        self.assertGreaterEqual(stats["rem"].get("dreams_generated", 0), 1)
        self.assertGreaterEqual(len(stats.get("dreams", [])), 1)

    def test_deep_sleep_preserves_metadata_for_consolidated_concepts(self):
        deep = DeepSleep(enable_synthesis=False)

        concept = Concept(
            id="stable_1",
            type=ConceptType.FACT,
            description="stable project memory",
            importance=ImportanceVector(novelty=0.95, emotional=0.2, task_relevance=0.95, repetition=0.9),
            strength=1.4,
            retention_score=0.8,
            rehearsal_count=3,
            activation_count=2,
            association_density=0.45,
            context_tags={"channel": "paper"},
        )

        updated_concepts, _, _ = deep.run(
            concepts=[concept],
            relations=[],
            episodes=[],
        )

        updated = next(c for c in updated_concepts if c.id == "stable_1")
        self.assertGreaterEqual(updated.rehearsal_count, 3)
        self.assertGreaterEqual(updated.activation_count, 2)
        self.assertEqual(updated.context_tags.get("channel"), "paper")
        self.assertIn("consolidation_score", updated.context_tags)

    def test_deep_sleep_suppresses_low_importance_hubs(self):
        # See note in test_deep_sleep_downscale_and_forgetting — opt into
        # legacy forgetting because v0.7.1's protection floor would keep
        # the low-importance hub alive and break the assertion.
        from src.sleep.forgetting_dynamics import ForgettingDynamics
        legacy_forgetting = ForgettingDynamics(
            protect_salience=0.0,
            min_rehearsal_before_archive=0,
            freshness_floor_hours=0.0,
        )
        deep = DeepSleep(enable_synthesis=False, forgetting=legacy_forgetting)

        hub = Concept(
            id="hub",
            type=ConceptType.FACT,
            description="generic bridge memory",
            importance=ImportanceVector(novelty=0.2, emotional=0.0, task_relevance=0.25, repetition=0.2),
            strength=1.1,
        )
        keeper = Concept(
            id="keeper",
            type=ConceptType.FACT,
            description="specific durable memory",
            importance=ImportanceVector(novelty=0.95, emotional=0.1, task_relevance=0.95, repetition=0.9),
            strength=1.2,
        )
        leaves = [
            Concept(
                id=f"leaf_{i}",
                type=ConceptType.FACT,
                description=f"leaf memory {i}",
                importance=ImportanceVector(novelty=0.6, emotional=0.0, task_relevance=0.55, repetition=0.4),
                strength=0.9,
            )
            for i in range(12)
        ]
        episodes = [
            Episode(
                id=f"hub_ep_{i}",
                concept_ids=["hub", "keeper", leaves[i].id],
                raw_content=f"hub episode {i}",
                importance=ImportanceVector(),
            )
            for i in range(12)
        ]

        updated_concepts, _, _ = deep.run(
            concepts=[hub, keeper, *leaves],
            relations=[],
            episodes=episodes,
        )

        updated_ids = {c.id for c in updated_concepts}
        self.assertIn("keeper", updated_ids)
        self.assertNotIn("hub", updated_ids)

    def test_rem_dreaming_is_deterministic_for_same_inputs(self):
        rem = REMDreaming(dream_count=2, novelty_factor=0.3, integration_threshold=0.3)

        concepts = [
            Concept(
                id=f"c{i}",
                type=ConceptType.FACT,
                description=f"concept {i}",
                importance=ImportanceVector(novelty=0.6, emotional=0.0, task_relevance=0.6, repetition=0.6),
                strength=1.0,
                embedding=[0.1 + i * 0.001] * 384,
            )
            for i in range(8)
        ]
        episodes = [
            Episode(id="e1", concept_ids=["c0", "c1", "c2"], raw_content="episode one", importance=ImportanceVector()),
            Episode(id="e2", concept_ids=["c3", "c4", "c5"], raw_content="episode two", importance=ImportanceVector()),
        ]

        dreams1, relations1, stats1 = rem.dream(
            concepts=concepts,
            relations=[],
            recent_episodes=episodes,
        )
        dreams2, relations2, stats2 = rem.dream(
            concepts=concepts,
            relations=[],
            recent_episodes=episodes,
        )

        self.assertEqual([d["sequence"] for d in dreams1], [d["sequence"] for d in dreams2])
        self.assertEqual(
            [(r.subject_id, str(r.predicate), r.object_id, round(r.strength, 4)) for r in relations1],
            [(r.subject_id, str(r.predicate), r.object_id, round(r.strength, 4)) for r in relations2],
        )
        self.assertEqual(stats1["dreams_generated"], stats2["dreams_generated"])

    def test_replay_window_covers_full_episode_range(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        episodes = [
            Episode(
                id=f"e{i}",
                timestamp=base + timedelta(minutes=i),
                concept_ids=[f"c{i}"],
                raw_content=f"episode {i}",
                importance=ImportanceVector(),
            )
            for i in range(60)
        ]

        selected = DeepSleep._select_replay_episodes(list(reversed(episodes)))

        self.assertEqual(len(selected), 30)
        self.assertEqual(selected[0].id, "e0")
        self.assertEqual(selected[-1].id, "e59")


class TestSleepModeRouting(unittest.TestCase):
    def test_orchestrator_runs_micro_and_deep_modes(self):
        orchestrator = SleepCycleOrchestrator()
        concepts = [
            Concept(
                id=f"c{i}",
                type=ConceptType.FACT,
                description=f"routing concept {i}",
                importance=ImportanceVector(novelty=0.7, emotional=0.0, task_relevance=0.7, repetition=0.5),
                strength=1.0,
                embedding=[0.1] * 384,
            )
            for i in range(6)
        ]
        relations = [
            Relation(
                subject_id="c0",
                predicate=PredicateType.RELATED_TO,
                object_id="c1",
                strength=0.7,
            )
        ]
        episodes = [
            Episode(id="e1", concept_ids=["c0", "c1", "c2"], raw_content="routing episode", importance=ImportanceVector()),
            Episode(id="e2", concept_ids=["c1", "c2", "c3"], raw_content="routing episode two", importance=ImportanceVector()),
        ]

        ok_micro, _, stats_micro = orchestrator.begin_sleep_cycle(
            concepts=concepts,
            relations=relations,
            episodes=episodes,
            force=True,
            mode="micro",
        )
        self.assertTrue(ok_micro)
        self.assertEqual(stats_micro.get("mode"), "micro")

        ok_deep, _, stats_deep = orchestrator.begin_sleep_cycle(
            concepts=concepts,
            relations=relations,
            episodes=episodes,
            force=True,
            mode="deep",
        )
        self.assertTrue(ok_deep)
        self.assertEqual(stats_deep.get("mode"), "deep")
        self.assertIn("nrem", stats_deep)
        self.assertIn("rem", stats_deep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
