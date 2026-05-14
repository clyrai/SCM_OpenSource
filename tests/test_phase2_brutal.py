"""Brutal tests for Phase 2 (EventCompiler + AssociationBinder)."""

from __future__ import annotations

import random
import time
import uuid
from typing import List

import src.chat.engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.association_binder import AssociationBinder
from src.core.event_compiler import EventCompiler
from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, EventSchema, ImportanceVector


def _embedding(seed: int) -> List[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _concept(
    description: str,
    ctype: ConceptType,
    seed: int,
    salience: float = 0.82,
    grasp: float = 0.8,
) -> Concept:
    return Concept(
        type=ctype,
        description=description,
        embedding=_embedding(seed),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.8, repetition=0.2),
        salience_score=salience,
        grasp_score=grasp,
    )


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    return ltm


class _DummyLLM:
    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        return "ok"


class _StubEncoder:
    def extract(self, text: str) -> List[Concept]:
        idx = sum(ord(c) for c in text) % 37
        city = f"City{idx % 9}"
        project = f"Project{idx % 11}"
        return [
            _concept(f"Person: User{idx % 5}", ConceptType.PERSON, idx),
            _concept(f"Location: {city}", ConceptType.LOCATION, idx + 1),
            _concept(f"Work: {project}", ConceptType.FACT, idx + 2),
        ]

    def _get_embedding(self, text: str):
        idx = sum(ord(c) for c in text) % 37
        return _embedding(idx)


class TestPhase2Brutal:
    def setup_method(self):
        random.seed(1337)

    def test_event_compiler_throughput_and_field_extraction(self):
        compiler = EventCompiler()
        episodes = []

        for i in range(5000):
            city = f"City{i % 50}"
            episodes.append(
                chat_engine_module.Episode(
                    raw_content=(
                        f"I moved to {city} today because project {i % 17} needed support "
                        f"at 09:3{i % 10} am."
                    ),
                    source="user",
                    salience_score=0.83,
                    grasp_score=0.79,
                )
            )

        start = time.perf_counter()
        events = compiler.compile_batch(episodes, interlocutor="user", task_context="stress")
        elapsed = time.perf_counter() - start

        assert len(events) == len(episodes)
        assert elapsed < 5.0

        when_count = sum(1 for e in events if e.when)
        where_count = sum(1 for e in events if e.where)
        why_count = sum(1 for e in events if e.why)

        assert when_count >= int(len(events) * 0.95)
        assert where_count >= int(len(events) * 0.95)
        assert why_count >= int(len(events) * 0.95)

    def test_association_coverage_target_on_synthetic_events(self):
        ltm = _fast_ltm()
        binder = AssociationBinder(max_edges_per_concept=10)

        groups = []
        for g in range(40):
            group = [
                _concept(f"Person G{g}", ConceptType.PERSON, g * 10 + 1),
                _concept(f"Location G{g}", ConceptType.LOCATION, g * 10 + 2),
                _concept(f"Preference G{g}", ConceptType.PREFERENCE, g * 10 + 3),
                _concept(f"Task G{g}", ConceptType.FACT, g * 10 + 4),
            ]
            groups.append(group)
            for concept in group:
                ltm.add_concept(concept)

        expected_pairs = 0
        bound_pairs = 0

        for i in range(300):
            group = groups[i % len(groups)]
            event_concepts = [group[0], group[1], group[2]]
            event = EventSchema(
                who="user",
                what=f"Group event {i}",
                where=f"Zone{i % 7}",
                salience=0.86,
                grasp=0.8,
                certainty=0.9,
            )

            stats = binder.bind_event(
                event=event,
                event_concepts=event_concepts,
                long_term_memory=ltm,
                candidate_pool=event_concepts,
            )

            expected_pairs += 3
            bound_pairs += int(stats["pairs_bound"])

        coverage = bound_pairs / expected_pairs
        assert coverage >= 0.80

    def test_duplicate_event_inflation_below_ten_percent(self):
        compiler = EventCompiler()

        unique_count = 120
        history = []
        accepted = 0

        for i in range(unique_count):
            city = f"City{i % 30}"
            base = f"I work in {city} today because planning {i % 13} is pending."
            variants = [
                base,
                f"Today I work in {city} because planning {i % 13} is pending.",
                f"I work in {city} today because planning {i % 13} is pending.",
            ]

            for text in variants:
                event = compiler.compile_episode(
                    chat_engine_module.Episode(
                        raw_content=text,
                        source="user",
                        salience_score=0.75,
                        grasp_score=0.72,
                    ),
                    interlocutor="user",
                )
                is_dup = compiler.is_duplicate(event, history)
                if not is_dup:
                    history.append(event)
                    accepted += 1

        inflation = max(0, accepted - unique_count) / unique_count
        assert inflation <= 0.10

    def test_hotspot_edge_explosion_is_capped(self):
        ltm = _fast_ltm()
        binder = AssociationBinder(max_edges_per_concept=12)

        anchor = _concept("Anchor user memory", ConceptType.FACT, 1)
        ltm.add_concept(anchor)

        for i in range(180):
            peer = _concept(f"Peer concept {i}", ConceptType.FACT, i + 2)
            helper = _concept(f"Helper concept {i % 17}", ConceptType.FACT, i + 500)
            ltm.add_concept(peer)
            ltm.add_concept(helper)

            event = EventSchema(
                who="user",
                what=f"Anchor and peer link {i}",
                salience=0.84,
                grasp=0.8,
                certainty=0.88,
            )
            binder.bind_event(event, [anchor, peer, helper], ltm)

        assert ltm.graph.out_degree(anchor.id) <= binder.max_edges_per_concept

    def test_phase2_pipeline_stress_hme_enabled(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoder(),
            enable_auto_sleep=False,
            session_id=f"phase2_brutal_{uuid.uuid4().hex}",
        )

        engine.long_term_memory._persist_concept = lambda concept: None
        engine.long_term_memory._persist_relation = lambda relation: None

        for i in range(160):
            engine._extract_and_store(
                f"I moved to City{i % 20} today because project {i % 11} changed.",
                source="user",
            )

        engine._extract_and_store(
            "I moved to City5 today because project 2 changed.",
            source="user",
        )
        engine._extract_and_store(
            "I moved to City5 today because project 2 changed.",
            source="user",
        )

        wm_recent = engine.working_memory.get_all()
        user_eps = [ep for ep in wm_recent if ep.source == "user"]
        assert user_eps
        assert any("event_key" in ep.context for ep in user_eps)
        assert any(ep.context.get("is_duplicate_event") for ep in user_eps)

        ltm_stats = engine.long_term_memory.get_stats()
        assert ltm_stats["total_relations"] > 0
        assert len(engine._event_history) > 0

    def test_association_binder_large_scale_performance(self):
        ltm = _fast_ltm()
        binder = AssociationBinder(max_edges_per_concept=16)

        concept_pool = []
        for i in range(1200):
            c = _concept(f"Concept {i}", ConceptType.FACT, i)
            concept_pool.append(c)
            ltm.add_concept(c)

        events = []
        for i in range(1000):
            events.append(
                (
                    EventSchema(
                        who="user",
                        what=f"Event {i}",
                        where=f"Zone{i % 25}",
                        salience=0.83,
                        grasp=0.79,
                        certainty=0.9,
                    ),
                    [
                        concept_pool[(i * 3) % 1200],
                        concept_pool[(i * 3 + 1) % 1200],
                        concept_pool[(i * 3 + 2) % 1200],
                    ],
                )
            )

        start = time.perf_counter()
        total_bound = 0
        for event, concepts in events:
            stats = binder.bind_event(
                event=event,
                event_concepts=concepts,
                long_term_memory=ltm,
                candidate_pool=concepts,
            )
            total_bound += int(stats["pairs_bound"])
        elapsed = time.perf_counter() - start

        assert total_bound >= 2500
        assert elapsed < 12.0

    def test_association_aging_removes_stale_links(self):
        ltm = _fast_ltm()
        binder = AssociationBinder(
            min_edge_strength=0.1,
            aging_decay=0.9,
            aging_stale_steps=2,
            aging_threshold=0.5,
        )

        concepts = [_concept(f"Node {i}", ConceptType.FACT, i) for i in range(90)]
        for c in concepts:
            ltm.add_concept(c)

        # Add many weak links manually.
        for i in range(0, 88):
            ltm.graph.add_edge(
                concepts[i].id,
                concepts[i + 1].id,
                predicate="related_to",
                strength=0.11,
                id=f"weak-{i}",
                age_steps=0,
            )

        initial_edges = ltm.graph.number_of_edges()
        removed_total = 0
        for _ in range(6):
            removed_total += binder.age_relations(ltm, touched_edges=set())

        final_edges = ltm.graph.number_of_edges()

        assert removed_total > 0
        assert final_edges < initial_edges
