"""Generate machine-readable Phase 4 (MicroSleep + DeepSleep) benchmark metrics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

# Ensure repo root is on import path when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, Episode, ImportanceVector, Relation
from src.core.working_memory import WorkingMemory
from src.retrieval.spreading_activation import SpreadingActivationRetriever
from src.sleep.sleep_cycle import SleepCycleOrchestrator
from src.sleep.trigger import SleepTrigger


def _embedding(seed: int) -> list[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 11) * 0.0001) for i in range(384)]


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False
    return ltm


def _build_phase4_dataset(pair_count: int, seed: int) -> tuple[list[Concept], list[Relation], list[Episode], dict]:
    rng = random.Random(seed)
    concepts: list[Concept] = []
    relations: list[Relation] = []
    episodes: list[Episode] = []
    key_ids: dict[int, str] = {}
    noise_ids: dict[int, str] = {}

    # Build pairs: one low-value distractor inserted first, then true key memory.
    for i in range(pair_count):
        token = f"token{i}"
        description = f"project {token} record"

        distractor = Concept(
            type=ConceptType.FACT,
            description=description,
            embedding=_embedding(10_000 + i),
            importance=ImportanceVector(novelty=0.08, emotional=0.0, task_relevance=0.08, repetition=0.10),
            # Explicit low salience: distractors are noise by design.
            # In v0.7.1+ the default protect_salience floor is 0.5, so
            # noise concepts must declare their low salience to remain
            # eligible for forgetting.
            salience_score=0.1,
            strength=1.7,
        )
        distractor.context_tags = {"session_id": "phase4", "person": "user"}
        distractor.retention_score = 0.20
        distractor.prediction_error = 0.10

        key = Concept(
            type=ConceptType.FACT,
            description=description,
            embedding=_embedding(i),
            importance=ImportanceVector(novelty=0.88, emotional=0.1, task_relevance=0.92, repetition=0.75),
            strength=1.0,
        )
        key.context_tags = {"session_id": "phase4", "person": "user"}
        key.retention_score = 0.75
        key.prediction_error = 0.25

        # Distractor first to intentionally hurt baseline top-1 retrieval on tie.
        concepts.append(distractor)
        concepts.append(key)
        key_ids[i] = key.id
        noise_ids[i] = distractor.id

    # Add a few bridge concepts to make propagation non-trivial.
    bridge_ids: list[str] = []
    for i in range(max(4, pair_count // 6)):
        bridge = Concept(
            type=ConceptType.ABSTRACT,
            description=f"bridge concept {i}",
            embedding=_embedding(30_000 + i),
            importance=ImportanceVector(novelty=0.5, emotional=0.0, task_relevance=0.55, repetition=0.4),
            strength=0.8 + (i % 3) * 0.1,
        )
        bridge.context_tags = {"session_id": "phase4", "person": "user"}
        concepts.append(bridge)
        bridge_ids.append(bridge.id)

    # Link keys into a sparse chain and to bridges.
    for i in range(pair_count - 1):
        relations.append(
            Relation(
                subject_id=key_ids[i],
                object_id=key_ids[i + 1],
                predicate="related_to",
                strength=0.65 + (i % 5) * 0.04,
            )
        )
    for i, bridge_id in enumerate(bridge_ids):
        kidx = i % pair_count
        relations.append(
            Relation(
                subject_id=key_ids[kidx],
                object_id=bridge_id,
                predicate="related_to",
                strength=0.55 + (i % 4) * 0.05,
            )
        )

    # Episodes strongly replay keys and pair-neighbors (helps micro relation reinforcement).
    for i in range(pair_count):
        neighbors = [key_ids[i]]
        if i > 0:
            neighbors.append(key_ids[i - 1])
        if i < pair_count - 1:
            neighbors.append(key_ids[i + 1])
        if bridge_ids:
            neighbors.append(bridge_ids[i % len(bridge_ids)])
        episodes.append(
            Episode(
                id=f"phase4_ep_{i}",
                concept_ids=neighbors,
                raw_content=f"episode for token{i}",
                importance=ImportanceVector(
                    novelty=0.75,
                    emotional=0.0,
                    task_relevance=0.8,
                    repetition=0.7,
                ),
                source="user",
            )
        )

    # Shuffle only episodes (not concepts) for variety.
    rng.shuffle(episodes)
    metadata = {
        "pair_count": pair_count,
        "key_ids": key_ids,
        "noise_ids": noise_ids,
    }
    return concepts, relations, episodes, metadata


def _build_retriever(concepts: list[Concept], relations: list[Relation]) -> SpreadingActivationRetriever:
    ltm = _fast_ltm()
    for concept in concepts:
        ltm.add_concept(concept)
    for relation in relations:
        ltm.add_relation(relation)
    return SpreadingActivationRetriever(
        working_memory=WorkingMemory(capacity=64),
        long_term_memory=ltm,
        spreading_steps=3,
        spreading_decay=0.45,
        activation_threshold=0.02,
        max_candidates=20,
    )


def _top1_recall(
    concepts: list[Concept],
    relations: list[Relation],
    key_ids: dict[int, str],
) -> float:
    retriever = _build_retriever(concepts, relations)
    hits = 0
    total = len(key_ids)

    for idx, key_id in key_ids.items():
        query = f"what about project token{idx} record?"
        activated, _stats = retriever.retrieve(
            query=query,
            context_tags={"session_id": "phase4", "person": "user"},
            max_seed_concepts=8,
        )
        if activated and activated[0].id == key_id:
            hits += 1

    return hits / total if total else 0.0


def _disambiguation_recall(
    concepts: list[Concept],
    relations: list[Relation],
    key_ids: dict[int, str],
    noise_ids: dict[int, str],
) -> float:
    """
    For each query, score success when the true memory ranks above its distractor.
    """
    retriever = _build_retriever(concepts, relations)
    wins = 0
    total = len(key_ids)

    for idx, key_id in key_ids.items():
        noise_id = noise_ids[idx]
        query = f"what about project token{idx} record?"
        activated, _stats = retriever.retrieve(
            query=query,
            context_tags={"session_id": "phase4", "person": "user"},
            max_seed_concepts=8,
        )
        ranks = {concept.id: rank for rank, concept in enumerate(activated)}
        key_rank = ranks.get(key_id, 9999)
        noise_rank = ranks.get(noise_id, 9999)
        if key_rank < noise_rank:
            wins += 1

    return wins / total if total else 0.0


def _entropy(concepts: list[Concept]) -> float:
    trigger = SleepTrigger()
    return trigger._compute_entropy(concepts)


def _memory_pressure(concepts: list[Concept], key_ids: dict[int, str], noise_ids: dict[int, str]) -> dict:
    concept_ids = {c.id for c in concepts}
    key_kept = sum(1 for cid in key_ids.values() if cid in concept_ids)
    noise_kept = sum(1 for cid in noise_ids.values() if cid in concept_ids)
    total = len(concepts)
    return {
        "total_concepts": total,
        "key_kept": key_kept,
        "noise_kept": noise_kept,
        "key_retention": (key_kept / len(key_ids)) if key_ids else 0.0,
        "noise_retention": (noise_kept / len(noise_ids)) if noise_ids else 0.0,
    }


def _run_mode(
    mode: str,
    concepts: list[Concept],
    relations: list[Relation],
    episodes: list[Episode],
) -> tuple[list[Concept], list[Relation], dict]:
    orch = SleepCycleOrchestrator()
    ok, cycle, stats = orch.begin_sleep_cycle(
        concepts=deepcopy(concepts),
        relations=deepcopy(relations),
        episodes=deepcopy(episodes),
        force=True,
        mode=mode,
    )
    if not ok:
        raise RuntimeError(f"{mode} sleep did not run: {stats}")
    return stats.get("updated_concepts", []), stats.get("updated_relations", []), stats


def benchmark_sleep_gain(pair_count: int, seed: int) -> dict:
    concepts, relations, episodes, meta = _build_phase4_dataset(pair_count=pair_count, seed=seed)
    key_ids = meta["key_ids"]
    noise_ids = meta["noise_ids"]

    baseline_top1 = _top1_recall(concepts, relations, key_ids)
    baseline_disambiguation = _disambiguation_recall(concepts, relations, key_ids, noise_ids)
    baseline_entropy = _entropy(concepts)
    baseline_pressure = _memory_pressure(concepts, key_ids, noise_ids)

    micro_concepts, micro_relations, micro_stats = _run_mode("micro", concepts, relations, episodes)
    micro_top1 = _top1_recall(micro_concepts, micro_relations, key_ids)
    micro_disambiguation = _disambiguation_recall(micro_concepts, micro_relations, key_ids, noise_ids)
    micro_entropy = _entropy(micro_concepts)
    micro_pressure = _memory_pressure(micro_concepts, key_ids, noise_ids)

    deep_concepts, deep_relations, deep_stats = _run_mode("deep", concepts, relations, episodes)
    deep_top1 = _top1_recall(deep_concepts, deep_relations, key_ids)
    deep_disambiguation = _disambiguation_recall(deep_concepts, deep_relations, key_ids, noise_ids)
    deep_entropy = _entropy(deep_concepts)
    deep_pressure = _memory_pressure(deep_concepts, key_ids, noise_ids)

    return {
        "dataset": {
            "pair_count": pair_count,
            "concepts_initial": len(concepts),
            "relations_initial": len(relations),
            "episodes": len(episodes),
        },
      "baseline": {
            "top1_recall": round(baseline_top1, 4),
            "disambiguation_recall": round(baseline_disambiguation, 4),
            "entropy": round(baseline_entropy, 4),
            "pressure": baseline_pressure,
        },
        "micro_sleep": {
            "top1_recall": round(micro_top1, 4),
            "top1_gain_abs": round(micro_top1 - baseline_top1, 4),
            "disambiguation_recall": round(micro_disambiguation, 4),
            "disambiguation_gain_abs": round(micro_disambiguation - baseline_disambiguation, 4),
            "entropy": round(micro_entropy, 4),
            "entropy_delta": round(micro_entropy - baseline_entropy, 4),
            "pressure": micro_pressure,
            "stats": {
                "replayed": micro_stats.get("micro", {}).get("replayed", 0),
                "duplicates_merged": micro_stats.get("micro", {}).get("duplicates_merged", 0),
                "relations_reinforced": micro_stats.get("micro", {}).get("relations_reinforced", 0),
            },
        },
        "deep_sleep": {
            "top1_recall": round(deep_top1, 4),
            "top1_gain_abs": round(deep_top1 - baseline_top1, 4),
            "disambiguation_recall": round(deep_disambiguation, 4),
            "disambiguation_gain_abs": round(deep_disambiguation - baseline_disambiguation, 4),
            "entropy": round(deep_entropy, 4),
            "entropy_delta": round(deep_entropy - baseline_entropy, 4),
            "pressure": deep_pressure,
            "stats": {
                "consolidated_to_ltm": deep_stats.get("nrem", {}).get("consolidated_to_ltm", 0),
                "forgotten": deep_stats.get("forgetting", {}).get("forgotten", 0),
                "dreams_generated": len(deep_stats.get("dreams", [])),
            },
        },
    }


def benchmark_threshold_sweep(
    pair_count: int,
    seed: int,
    micro_turns_grid: list[int],
    deep_pressure_grid: list[float],
) -> dict:
    concepts, relations, episodes, _meta = _build_phase4_dataset(pair_count=pair_count, seed=seed)

    rows = []
    for micro_turns in micro_turns_grid:
        for deep_pressure in deep_pressure_grid:
            orch = SleepCycleOrchestrator()
            orch.trigger.micro_interval_turns = micro_turns
            orch.trigger.deep_pressure_threshold = deep_pressure
            mode, reason, _stats = orch.select_sleep_mode(
                concepts=deepcopy(concepts),
                relations=deepcopy(relations),
                turns_since_micro=micro_turns,
                session_turns=max(6, micro_turns - 1),
            )
            rows.append(
                {
                    "micro_interval_turns": micro_turns,
                    "deep_pressure_threshold": round(deep_pressure, 3),
                    "selected_mode": mode,
                    "reason": reason,
                }
            )

    mode_counts = {
        "micro": sum(1 for r in rows if r["selected_mode"] == "micro"),
        "deep": sum(1 for r in rows if r["selected_mode"] == "deep"),
        "none": sum(1 for r in rows if r["selected_mode"] is None),
    }
    return {"rows": rows, "mode_counts": mode_counts}


def build_report(args: argparse.Namespace) -> dict:
    metrics = benchmark_sleep_gain(pair_count=args.pair_count, seed=args.seed + 7)
    sweep = benchmark_threshold_sweep(
        pair_count=args.pair_count,
        seed=args.seed + 17,
        micro_turns_grid=args.micro_turns_grid,
        deep_pressure_grid=args.deep_pressure_grid,
    )

    micro_disambig_gain = metrics["micro_sleep"]["disambiguation_gain_abs"]
    deep_disambig_gain = metrics["deep_sleep"]["disambiguation_gain_abs"]
    deep_noise_retention = metrics["deep_sleep"]["pressure"]["noise_retention"]

    report = {
        "benchmark": "phase4_sleepkernel_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "seed": args.seed,
        "targets": {
            "micro_disambiguation_gain_min_abs": 0.20,
            "deep_disambiguation_gain_min_abs": 0.20,
            "deep_noise_retention_max": 0.40,
        },
        "inputs": {
            "pair_count": args.pair_count,
            "micro_turns_grid": args.micro_turns_grid,
            "deep_pressure_grid": args.deep_pressure_grid,
        },
        "metrics": {
            "sleep_gain": metrics,
            "threshold_sweep": sweep,
        },
        "status": {
            "micro_gain_pass": bool(micro_disambig_gain >= 0.20),
            "deep_gain_pass": bool(deep_disambig_gain >= 0.20),
            "deep_noise_pass": bool(deep_noise_retention <= 0.40),
        },
    }
    report["status"]["overall_pass"] = all(report["status"].values())
    return report


def _write_report(report: dict, output: Path, write_history: bool, history_dir: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    history_path = None
    if write_history:
        history_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        history_path = history_dir / f"phase4_metrics_{ts_safe}.json"
        history_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return history_path


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 4 benchmark metrics JSON")
    parser.add_argument(
        "--output",
        default="research/metrics/phase4_micro_deep_latest.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--pair-count", type=int, default=36)
    parser.add_argument(
        "--micro-turns-grid",
        type=_parse_int_list,
        default=_parse_int_list("3,4,5,6"),
    )
    parser.add_argument(
        "--deep-pressure-grid",
        type=_parse_float_list,
        default=_parse_float_list("0.86,0.90,0.93,0.96"),
    )
    parser.add_argument(
        "--write-history",
        action="store_true",
        help="Also write a timestamped metrics snapshot",
    )
    parser.add_argument(
        "--history-dir",
        default="research/metrics/history/phase4",
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
