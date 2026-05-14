"""Generate machine-readable Phase 3 brutal benchmark metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import uuid

# Ensure repo root is on import path when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.chat.engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.long_term_memory import LongTermMemory
from src.core.models import (
    Concept,
    ConceptType,
    ImportanceVector,
    PredicateType,
    Relation,
)
from src.core.working_memory import WorkingMemory
from src.retrieval.hypothesis_ranker import HypothesisRanker
from src.retrieval.spreading_activation import SpreadingActivationRetriever
from src.core import config as config_module


BRUTAL_TEST_TARGETS = [
    "tests/test_brutal.py",
    "tests/test_crazy_brutal.py",
    "tests/test_phase2_brutal.py",
    "tests/test_spreading_activation.py::TestSpreadingActivationBrutal",
    "tests/test_hypothesis_ranker.py::TestHypothesisRankerBrutal",
    "tests/test_phase3_pipeline.py::TestPhase3PipelineStress",
]


def _embedding(seed: int):
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _concept(
    description: str,
    ctype: ConceptType,
    seed: int,
    salience: float = 0.8,
    grasp: float = 0.8,
    context_tags: dict | None = None,
) -> Concept:
    concept = Concept(
        type=ctype,
        description=description,
        embedding=_embedding(seed),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.8, repetition=0.2),
        salience_score=salience,
        grasp_score=grasp,
    )
    if context_tags:
        concept.context_tags = context_tags
    return concept


def _add_relation(ltm: LongTermMemory, from_id: str, to_id: str, strength: float):
    rel = Relation(
        subject_id=from_id,
        object_id=to_id,
        predicate=PredicateType.RELATED_TO,
        strength=strength,
    )
    ltm.add_relation(rel)
    if ltm.graph.has_edge(from_id, to_id):
        ltm.graph[from_id][to_id]["strength"] = strength


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False
    return ltm


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * p
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _latency_stats_ms(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(latencies_ms),
        "mean": round(statistics.fmean(latencies_ms), 4),
        "p50": round(_percentile(latencies_ms, 0.50), 4),
        "p95": round(_percentile(latencies_ms, 0.95), 4),
        "p99": round(_percentile(latencies_ms, 0.99), 4),
        "min": round(min(latencies_ms), 4),
        "max": round(max(latencies_ms), 4),
    }


def benchmark_spreading_activation(
    rounds: int,
    queries_per_round: int,
    concept_count: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    ltm = _fast_ltm()

    topics = 80
    concepts = []
    for i in range(concept_count):
        topic = i % topics
        concept = _concept(
            description=f"Topic {topic} concept {i}",
            ctype=ConceptType.FACT,
            seed=i,
            salience=0.6 + (i % 7) * 0.04,
            grasp=0.6 + (i % 5) * 0.05,
            context_tags={
                "session_id": f"sess_{topic % 5}",
                "person": f"user_{topic % 11}",
            },
        )
        ltm.add_concept(concept)
        concepts.append(concept)

    for i in range(concept_count):
        _add_relation(
            ltm,
            concepts[i].id,
            concepts[(i + 1) % concept_count].id,
            strength=0.4 + (i % 9) * 0.05,
        )
        for _ in range(2):
            j = rng.randrange(concept_count)
            if j != i:
                _add_relation(
                    ltm,
                    concepts[i].id,
                    concepts[j].id,
                    strength=0.2 + rng.random() * 0.6,
                )

    retriever = SpreadingActivationRetriever(
        working_memory=WorkingMemory(),
        long_term_memory=ltm,
        spreading_steps=4,
        spreading_decay=0.45,
        activation_threshold=0.02,
        max_candidates=80,
    )

    latencies_ms: list[float] = []
    activated_counts: list[int] = []
    seeds_counts: list[int] = []

    start = time.perf_counter()
    for _round in range(rounds):
        for _ in range(queries_per_round):
            topic = rng.randrange(topics)
            query = f"recall topic {topic} concept {rng.randrange(concept_count)}"
            context = {
                "session_id": f"sess_{topic % 5}",
                "person": f"user_{topic % 11}",
            }
            config_module.current_time = time.time()
            t0 = time.perf_counter()
            activated, stats = retriever.retrieve(
                query=query,
                context_tags=context,
                max_seed_concepts=10,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            activated_counts.append(len(activated))
            seeds_counts.append(stats.get("seeds", 0))
    total_sec = time.perf_counter() - start

    total_queries = rounds * queries_per_round
    return {
        "rounds": rounds,
        "queries_per_round": queries_per_round,
        "total_queries": total_queries,
        "graph": {
            "concepts": concept_count,
            "edges": int(ltm.graph.number_of_edges()),
        },
        "throughput_qps": round(total_queries / total_sec, 2) if total_sec > 0 else 0.0,
        "latency_ms": _latency_stats_ms(latencies_ms),
        "avg_activated": round(statistics.fmean(activated_counts), 3),
        "avg_seeds": round(statistics.fmean(seeds_counts), 3),
    }


def benchmark_hypothesis_ranker(
    rounds: int,
    calls_per_round: int,
    concepts_per_call: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    ranker = HypothesisRanker(max_hypotheses=12)

    pool = []
    for i in range(500):
        c = _concept(
            description=f"Hypothesis concept {i} topic {i % 35}",
            ctype=ConceptType.FACT,
            seed=9000 + i,
            salience=0.25 + (i % 11) * 0.06,
            grasp=0.3 + (i % 9) * 0.06,
        )
        c.rehearsal_count = i % 12
        c.association_density = min(1.0, 0.2 + (i % 8) * 0.1)
        if i % 29 == 0:
            c.version_parent = f"legacy_{i}"
        pool.append(c)

    latencies_ms: list[float] = []
    hypothesis_counts: list[int] = []
    ensemble_scores: list[float] = []

    start = time.perf_counter()
    for _round in range(rounds):
        for _ in range(calls_per_round):
            selected = rng.sample(pool, k=concepts_per_call)
            activation_map = {
                c.id: max(0.01, 1.0 - idx * 0.01)
                for idx, c in enumerate(selected)
            }
            config_module.current_time = time.time()
            t0 = time.perf_counter()
            hypothesis_set = ranker.rank(
                activated_concepts=selected,
                activation_map=activation_map,
                context_tags={"session_id": "bench"},
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            hypothesis_counts.append(len(hypothesis_set.hypotheses))
            ensemble_scores.append(hypothesis_set.ensemble_score)
    total_sec = time.perf_counter() - start

    total_calls = rounds * calls_per_round
    return {
        "rounds": rounds,
        "calls_per_round": calls_per_round,
        "total_calls": total_calls,
        "concepts_per_call": concepts_per_call,
        "throughput_calls_per_sec": round(total_calls / total_sec, 2) if total_sec > 0 else 0.0,
        "latency_ms": _latency_stats_ms(latencies_ms),
        "avg_hypotheses": round(statistics.fmean(hypothesis_counts), 3),
        "avg_ensemble_score": round(statistics.fmean(ensemble_scores), 4),
    }


class _DummyLLM:
    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        return "ok"


class _StubEncoder:
    def extract(self, text: str):
        idx = sum(ord(c) for c in text) % 997
        city = f"City{idx % 40}"
        project = f"Project{idx % 25}"
        return [
            _concept(f"User{idx % 12}", ConceptType.PERSON, idx + 1, salience=0.82, grasp=0.81),
            _concept(city, ConceptType.LOCATION, idx + 2, salience=0.8, grasp=0.79),
            _concept(f"{project} shifted timeline", ConceptType.FACT, idx + 3, salience=0.78, grasp=0.77),
        ]

    def _get_embedding(self, text: str):
        seed = sum(ord(c) for c in text) % 997
        return _embedding(seed)


def benchmark_pipeline_retrieval(
    preload_messages: int,
    retrieval_calls: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    chat_engine_module.HME_ENABLED = True

    ltm = _fast_ltm()
    engine = ChatEngine(
        llm=_DummyLLM(),
        encoder=_StubEncoder(),
        working_memory=WorkingMemory(capacity=64),
        long_term_memory=ltm,
        enable_auto_sleep=False,
        session_id=f"phase3_metrics_{uuid.uuid4().hex}",
    )

    for i in range(preload_messages):
        msg = (
            f"I moved to City{i % 40} today because project {i % 25} changed "
            f"after meeting {i % 11}."
        )
        engine._extract_and_store(msg, source="user")

    duplicate_message = "I moved to City5 today because project 2 changed after meeting 1."
    engine._extract_and_store(duplicate_message, source="user")
    engine._extract_and_store(duplicate_message, source="user")

    latencies_ms: list[float] = []
    activated_counts: list[int] = []
    confidence_histogram = {"high": 0, "medium": 0, "low": 0, "none": 0}

    start = time.perf_counter()
    for i in range(retrieval_calls):
        topic = rng.randrange(40)
        query = f"what do you remember about city {topic} project {i % 25}?"
        config_module.current_time = time.time()
        t0 = time.perf_counter()
        _ctx, stats = engine._retrieve_hme(query=query, query_embedding=None)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        activated_counts.append(stats.get("total_concepts_activated", 0))
        conf = stats.get("hypothesis_confidence", "none")
        confidence_histogram[conf] = confidence_histogram.get(conf, 0) + 1
    total_sec = time.perf_counter() - start

    user_eps = [ep for ep in engine.working_memory.get_all() if ep.source == "user"]
    duplicate_count = sum(1 for ep in user_eps if ep.context.get("is_duplicate_event"))
    ltm_stats = engine.long_term_memory.get_stats()

    return {
        "preload_messages": preload_messages,
        "retrieval_calls": retrieval_calls,
        "throughput_qps": round(retrieval_calls / total_sec, 2) if total_sec > 0 else 0.0,
        "latency_ms": _latency_stats_ms(latencies_ms),
        "avg_activated": round(statistics.fmean(activated_counts), 3),
        "confidence_histogram": confidence_histogram,
        "working_memory_user_episodes": len(user_eps),
        "duplicate_events_flagged": duplicate_count,
        "ltm_concepts": ltm_stats.get("total_concepts", 0),
        "ltm_relations": ltm_stats.get("total_relations", 0),
        "event_history": len(engine._event_history),
    }


def benchmark_pytest_brutal_repeats(rounds: int, timeout_seconds: int, seed: int) -> dict:
    rng = random.Random(seed)
    durations_sec: list[float] = []
    pass_rounds = 0
    fail_rounds = 0
    round_details = []

    for round_idx in range(1, rounds + 1):
        hash_seed = rng.randint(1, 2_000_000_000)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *BRUTAL_TEST_TARGETS,
            "-q",
            "-W",
            "ignore",
        ]

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(hash_seed)

        t0 = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            returncode = proc.returncode
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            output = (exc.stdout or "") + "\n" + (exc.stderr or "")

        duration = time.perf_counter() - t0
        durations_sec.append(duration)

        passed = (returncode == 0) and (not timed_out)
        if passed:
            pass_rounds += 1
        else:
            fail_rounds += 1

        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        passed_tests = int(passed_match.group(1)) if passed_match else 0
        failed_tests = int(failed_match.group(1)) if failed_match else 0

        tail = "\n".join([line for line in output.splitlines() if line.strip()][-8:])
        round_details.append(
            {
                "round": round_idx,
                "pythonhashseed": hash_seed,
                "duration_seconds": round(duration, 4),
                "returncode": returncode,
                "timed_out": timed_out,
                "passed": passed,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "output_tail": tail,
            }
        )

    return {
        "rounds": rounds,
        "targets": BRUTAL_TEST_TARGETS,
        "timeout_seconds": timeout_seconds,
        "pass_rounds": pass_rounds,
        "fail_rounds": fail_rounds,
        "all_passed": fail_rounds == 0,
        "durations_seconds": [round(d, 4) for d in durations_sec],
        "avg_seconds": round(statistics.fmean(durations_sec), 4) if durations_sec else 0.0,
        "p95_seconds": round(_percentile(durations_sec, 0.95), 4) if durations_sec else 0.0,
        "max_seconds": round(max(durations_sec), 4) if durations_sec else 0.0,
        "min_seconds": round(min(durations_sec), 4) if durations_sec else 0.0,
        "round_details": round_details,
    }


def build_report(args: argparse.Namespace) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    seed = args.seed

    spreading = benchmark_spreading_activation(
        rounds=args.spreading_rounds,
        queries_per_round=args.spreading_queries,
        concept_count=args.spreading_concepts,
        seed=seed + 11,
    )
    ranker = benchmark_hypothesis_ranker(
        rounds=args.ranker_rounds,
        calls_per_round=args.ranker_calls,
        concepts_per_call=args.ranker_concepts_per_call,
        seed=seed + 29,
    )
    pipeline = benchmark_pipeline_retrieval(
        preload_messages=args.pipeline_preload_messages,
        retrieval_calls=args.pipeline_retrieval_calls,
        seed=seed + 47,
    )
    pytest_repeats = benchmark_pytest_brutal_repeats(
        rounds=args.pytest_rounds,
        timeout_seconds=args.pytest_timeout_seconds,
        seed=seed + 71,
    )

    report = {
        "benchmark": "phase3_hme_brutal",
        "timestamp_utc": timestamp,
        "platform": platform.system().lower(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "seed": seed,
        "targets": {
            "retrieval_p95_ms_max": 150.0,
            "ranker_p95_ms_max": 50.0,
            "pytest_all_rounds_pass": True,
        },
        "inputs": {
            "spreading_rounds": args.spreading_rounds,
            "spreading_queries": args.spreading_queries,
            "spreading_concepts": args.spreading_concepts,
            "ranker_rounds": args.ranker_rounds,
            "ranker_calls": args.ranker_calls,
            "ranker_concepts_per_call": args.ranker_concepts_per_call,
            "pipeline_preload_messages": args.pipeline_preload_messages,
            "pipeline_retrieval_calls": args.pipeline_retrieval_calls,
            "pytest_rounds": args.pytest_rounds,
            "pytest_timeout_seconds": args.pytest_timeout_seconds,
        },
        "metrics": {
            "spreading_activation": spreading,
            "hypothesis_ranker": ranker,
            "pipeline_retrieval": pipeline,
            "pytest_brutal_repeats": pytest_repeats,
        },
    }

    status = {
        "spreading_p95_pass": spreading["latency_ms"]["p95"] <= report["targets"]["retrieval_p95_ms_max"],
        "pipeline_p95_pass": pipeline["latency_ms"]["p95"] <= report["targets"]["retrieval_p95_ms_max"],
        "ranker_p95_pass": ranker["latency_ms"]["p95"] <= report["targets"]["ranker_p95_ms_max"],
        "pytest_repeats_pass": pytest_repeats["all_passed"] == report["targets"]["pytest_all_rounds_pass"],
    }
    status["overall_pass"] = all(status.values())
    report["status"] = status

    return report


def _write_report(report: dict, output: Path, write_history: bool, history_dir: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    history_path = None
    if write_history:
        history_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        history_path = history_dir / f"phase3_brutal_{ts_safe}.json"
        history_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return history_path


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 3 brutal benchmark metrics JSON")
    parser.add_argument(
        "--output",
        default="research/metrics/phase3_brutal_latest.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--spreading-rounds", type=int, default=6)
    parser.add_argument("--spreading-queries", type=int, default=250)
    parser.add_argument("--spreading-concepts", type=int, default=1500)

    parser.add_argument("--ranker-rounds", type=int, default=8)
    parser.add_argument("--ranker-calls", type=int, default=400)
    parser.add_argument("--ranker-concepts-per-call", type=int, default=80)

    parser.add_argument("--pipeline-preload-messages", type=int, default=300)
    parser.add_argument("--pipeline-retrieval-calls", type=int, default=400)

    parser.add_argument("--pytest-rounds", type=int, default=5)
    parser.add_argument("--pytest-timeout-seconds", type=int, default=180)

    parser.add_argument(
        "--write-history",
        action="store_true",
        help="Also write a timestamped metrics snapshot",
    )
    parser.add_argument(
        "--history-dir",
        default="research/metrics/history/phase3",
        help="Directory for timestamped snapshots",
    )

    args = parser.parse_args()
    report = build_report(args)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    history_dir = Path(args.history_dir)
    if not history_dir.is_absolute():
        history_dir = REPO_ROOT / history_dir

    history_path = _write_report(
        report=report,
        output=output_path,
        write_history=args.write_history,
        history_dir=history_dir,
    )

    print(f"Wrote metrics to {output_path}")
    if history_path:
        print(f"Wrote history snapshot to {history_path}")
    print(json.dumps(report["status"], indent=2))


if __name__ == "__main__":
    main()
