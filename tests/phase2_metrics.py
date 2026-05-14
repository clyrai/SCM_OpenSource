"""Generate machine-readable Phase 2 benchmark metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

# Ensure repo root is on import path when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.chat.engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.association_binder import AssociationBinder
from src.core.event_compiler import EventCompiler
from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, Episode, EventSchema, ImportanceVector


def _embedding(seed: int):
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _concept(description: str, ctype: ConceptType, seed: int, salience: float = 0.82, grasp: float = 0.8) -> Concept:
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
    def extract(self, text: str):
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


def compute_phase2_metrics() -> dict:
    metrics: dict = {}

    # 1) Event compiler throughput + extraction quality
    compiler = EventCompiler()
    episodes = []
    for i in range(5000):
        city = f"City{i % 50}"
        episodes.append(
            Episode(
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
    compile_secs = time.perf_counter() - start

    metrics["event_compiler"] = {
        "episodes": len(events),
        "seconds": round(compile_secs, 6),
        "events_per_second": round(len(events) / compile_secs, 2),
        "field_extraction_rate": {
            "when": round(sum(1 for e in events if e.when) / len(events), 4),
            "where": round(sum(1 for e in events if e.where) / len(events), 4),
            "why": round(sum(1 for e in events if e.why) / len(events), 4),
        },
    }

    # 2) Association coverage target on synthetic events
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

    metrics["association_coverage"] = {
        "expected_pairs": expected_pairs,
        "bound_pairs": bound_pairs,
        "coverage": round(bound_pairs / expected_pairs, 4),
        "target": 0.80,
    }

    # 3) Duplicate event inflation
    compiler2 = EventCompiler()
    unique_count = 120
    history = []
    accepted = 0
    for i in range(unique_count):
        city = f"City{i % 30}"
        variants = [
            f"I work in {city} today because planning {i % 13} is pending.",
            f"Today I work in {city} because planning {i % 13} is pending.",
            f"I work in {city} today because planning {i % 13} is pending.",
        ]
        for text in variants:
            event = compiler2.compile_episode(
                Episode(raw_content=text, source="user", salience_score=0.75, grasp_score=0.72),
                interlocutor="user",
            )
            if not compiler2.is_duplicate(event, history):
                history.append(event)
                accepted += 1

    inflation = max(0, accepted - unique_count) / unique_count
    metrics["duplicate_event_inflation"] = {
        "unique_baseline": unique_count,
        "accepted_after_dedup": accepted,
        "inflation": round(inflation, 4),
        "target_max": 0.10,
    }

    # 4) Hotspot edge cap pressure
    ltm_hot = _fast_ltm()
    binder_hot = AssociationBinder(max_edges_per_concept=12)
    anchor = _concept("Anchor user memory", ConceptType.FACT, 1)
    ltm_hot.add_concept(anchor)

    for i in range(180):
        peer = _concept(f"Peer concept {i}", ConceptType.FACT, i + 2)
        helper = _concept(f"Helper concept {i % 17}", ConceptType.FACT, i + 500)
        ltm_hot.add_concept(peer)
        ltm_hot.add_concept(helper)
        event = EventSchema(
            who="user",
            what=f"Anchor and peer link {i}",
            salience=0.84,
            grasp=0.8,
            certainty=0.88,
        )
        binder_hot.bind_event(event, [anchor, peer, helper], ltm_hot)

    metrics["edge_cap_pressure"] = {
        "anchor_out_degree": ltm_hot.graph.out_degree(anchor.id),
        "cap": binder_hot.max_edges_per_concept,
    }

    # 5) Large-scale binder throughput
    ltm_large = _fast_ltm()
    binder_large = AssociationBinder(max_edges_per_concept=16)
    concept_pool = []
    for i in range(1200):
        c = _concept(f"Concept {i}", ConceptType.FACT, i)
        concept_pool.append(c)
        ltm_large.add_concept(c)

    total_bound = 0
    start = time.perf_counter()
    for i in range(1000):
        event = EventSchema(
            who="user",
            what=f"Event {i}",
            where=f"Zone{i % 25}",
            salience=0.83,
            grasp=0.79,
            certainty=0.9,
        )
        trio = [
            concept_pool[(i * 3) % 1200],
            concept_pool[(i * 3 + 1) % 1200],
            concept_pool[(i * 3 + 2) % 1200],
        ]
        stats = binder_large.bind_event(
            event=event,
            event_concepts=trio,
            long_term_memory=ltm_large,
            candidate_pool=trio,
        )
        total_bound += int(stats["pairs_bound"])
    bind_secs = time.perf_counter() - start

    metrics["association_large_scale"] = {
        "events": 1000,
        "seconds": round(bind_secs, 6),
        "events_per_second": round(1000 / bind_secs, 2),
        "pairs_bound": total_bound,
    }

    # 6) Relation aging/removal effectiveness
    ltm_age = _fast_ltm()
    binder_age = AssociationBinder(
        min_edge_strength=0.1,
        aging_decay=0.9,
        aging_stale_steps=2,
        aging_threshold=0.5,
    )
    nodes = [_concept(f"Node {i}", ConceptType.FACT, i) for i in range(90)]
    for node in nodes:
        ltm_age.add_concept(node)
    for i in range(88):
        ltm_age.graph.add_edge(
            nodes[i].id,
            nodes[i + 1].id,
            predicate="related_to",
            strength=0.11,
            id=f"weak-{i}",
            age_steps=0,
        )

    initial_edges = ltm_age.graph.number_of_edges()
    removed_total = 0
    for _ in range(6):
        removed_total += binder_age.age_relations(ltm_age, touched_edges=set())
    final_edges = ltm_age.graph.number_of_edges()

    metrics["relation_aging"] = {
        "initial_edges": initial_edges,
        "removed": removed_total,
        "final_edges": final_edges,
    }

    # 7) Pipeline stress with HME enabled
    chat_engine_module.HME_ENABLED = True
    engine = ChatEngine(
        llm=_DummyLLM(),
        encoder=_StubEncoder(),
        enable_auto_sleep=False,
        session_id=f"phase2_metrics_{uuid.uuid4().hex}",
    )
    engine.long_term_memory._persist_concept = lambda concept: None
    engine.long_term_memory._persist_relation = lambda relation: None

    for i in range(160):
        engine._extract_and_store(
            f"I moved to City{i % 20} today because project {i % 11} changed.",
            source="user",
        )
    engine._extract_and_store("I moved to City5 today because project 2 changed.", source="user")
    engine._extract_and_store("I moved to City5 today because project 2 changed.", source="user")

    user_eps = [ep for ep in engine.working_memory.get_all() if ep.source == "user"]
    metrics["pipeline_stress"] = {
        "working_memory_user_episodes": len(user_eps),
        "episodes_with_event_key": sum(1 for ep in user_eps if "event_key" in ep.context),
        "episodes_duplicate_flagged": sum(1 for ep in user_eps if ep.context.get("is_duplicate_event")),
        "ltm_relations": engine.long_term_memory.get_stats().get("total_relations", 0),
        "event_history": len(engine._event_history),
    }

    return metrics


def build_report() -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    metrics = compute_phase2_metrics()

    report = {
        "benchmark": "phase2_hme",
        "timestamp_utc": timestamp,
        "platform": "darwin",
        "python": "3.14",
        "targets": {
            "association_coverage_min": 0.80,
            "duplicate_inflation_max": 0.10,
        },
        "metrics": metrics,
        "status": {
            "association_coverage_pass": metrics["association_coverage"]["coverage"] >= 0.80,
            "duplicate_inflation_pass": metrics["duplicate_event_inflation"]["inflation"] <= 0.10,
        },
    }
    report["status"]["overall_pass"] = all(report["status"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 benchmark metrics JSON")
    parser.add_argument(
        "--output",
        default="research/metrics/phase2_metrics_latest.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    report = build_report()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote metrics to {output_path}")
    print(json.dumps(report["status"], indent=2))


if __name__ == "__main__":
    main()
