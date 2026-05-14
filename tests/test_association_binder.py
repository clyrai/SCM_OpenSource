"""Tests for AssociationBinder."""

import pytest

from src.core.association_binder import AssociationBinder
from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, EventSchema, ImportanceVector, PredicateType


def _concept(
    desc: str,
    ctype: ConceptType = ConceptType.FACT,
    salience: float = 0.8,
    grasp: float = 0.8,
    embedding=None,
) -> Concept:
    return Concept(
        type=ctype,
        description=desc,
        importance=ImportanceVector(novelty=0.7, task_relevance=0.8),
        salience_score=salience,
        grasp_score=grasp,
        embedding=embedding,
    )


class TestAssociationBinder:
    def setup_method(self):
        self.binder = AssociationBinder(
            learning_rate=0.4,
            semantic_threshold=0.7,
            max_edges_per_concept=4,
            min_edge_strength=0.1,
            aging_decay=0.9,
            aging_stale_steps=2,
            aging_threshold=0.4,
        )
        self.ltm = LongTermMemory()

    def test_binds_intra_event_concepts(self):
        c1 = _concept("Alice", ConceptType.PERSON)
        c2 = _concept("Seattle", ConceptType.LOCATION)
        c3 = _concept("works at Google", ConceptType.FACT)

        for c in (c1, c2, c3):
            self.ltm.add_concept(c)

        event = EventSchema(
            who="Alice",
            what="Alice works at Google in Seattle",
            where="Seattle",
            salience=0.85,
            grasp=0.78,
            certainty=0.9,
        )

        stats = self.binder.bind_event(event, [c1, c2, c3], self.ltm)

        assert stats["edges_created"] >= 2
        assert stats["coverage"] > 0
        assert self.ltm.graph.number_of_edges() >= 2

    def test_updates_existing_edge_strength(self):
        c1 = _concept("Alice")
        c2 = _concept("Alice works at Google")
        self.ltm.add_concept(c1)
        self.ltm.add_concept(c2)

        event = EventSchema(
            who="Alice",
            what="Alice works at Google",
            salience=0.8,
            grasp=0.8,
            certainty=0.9,
        )

        self.binder.bind_event(event, [c1, c2], self.ltm)
        first_strength = self.ltm.graph[c1.id][c2.id]["strength"]

        stats = self.binder.bind_event(event, [c1, c2], self.ltm)
        second_strength = self.ltm.graph[c1.id][c2.id]["strength"]

        assert stats["edges_updated"] >= 1
        assert second_strength >= first_strength

    def test_infers_contradiction_predicate(self):
        c1 = _concept("I prefer morning meetings")
        c2 = _concept("I prefer evening meetings")
        self.ltm.add_concept(c1)
        self.ltm.add_concept(c2)

        event = EventSchema(
            who="user",
            what="Actually no, I changed my mind and now prefer evenings",
            salience=0.9,
            grasp=0.8,
            certainty=0.9,
            is_contradiction=True,
        )

        self.binder.bind_event(event, [c1, c2], self.ltm)
        predicate = self.ltm.graph[c1.id][c2.id]["predicate"]
        assert predicate == PredicateType.CONTRADICTS.value

    def test_ages_and_prunes_stale_weak_edges(self):
        c1 = _concept("A")
        c2 = _concept("B")
        self.ltm.add_concept(c1)
        self.ltm.add_concept(c2)

        # Create an intentionally weak edge.
        self.ltm.graph.add_edge(
            c1.id,
            c2.id,
            predicate=PredicateType.RELATED_TO.value,
            strength=0.12,
            id="edge-1",
            age_steps=0,
        )

        pruned_1 = self.binder.age_relations(self.ltm, touched_edges=set())
        pruned_2 = self.binder.age_relations(self.ltm, touched_edges=set())
        pruned_3 = self.binder.age_relations(self.ltm, touched_edges=set())

        assert pruned_1 == 0
        assert pruned_2 == 0
        assert pruned_3 >= 1
        assert not self.ltm.graph.has_edge(c1.id, c2.id)

    def test_enforces_edge_cap_per_concept(self):
        anchor = _concept("Anchor concept")
        self.ltm.add_concept(anchor)

        peers = []
        for i in range(8):
            peer = _concept(f"Peer {i}")
            self.ltm.add_concept(peer)
            peers.append(peer)

        event = EventSchema(
            who="user",
            what="Anchor linked to many peers",
            salience=0.85,
            grasp=0.85,
            certainty=0.9,
        )

        self.binder.bind_event(event, [anchor] + peers, self.ltm)
        out_degree = self.ltm.graph.out_degree(anchor.id)
        assert out_degree <= self.binder.max_edges_per_concept

    def test_updates_association_density(self):
        c1 = _concept("Alice")
        c2 = _concept("Seattle")
        self.ltm.add_concept(c1)
        self.ltm.add_concept(c2)

        event = EventSchema(
            who="Alice",
            what="Alice moved to Seattle",
            salience=0.8,
            grasp=0.8,
            certainty=0.9,
        )

        self.binder.bind_event(event, [c1, c2], self.ltm)

        assert c1.association_density > 0
        assert c2.association_density > 0
        assert c1.activation_count >= 1
        assert c2.activation_count >= 1
