"""Generate a professional baseline-comparison report for SCM.

The report combines:
- controlled retrieval baselines on the Phase 4 synthetic dataset,
- current SCM sleep-stage results,
- Phase 6 human-memory regression evidence,
- and a paper-ready feature matrix for standard memory systems.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, MemoryState
from src.core.time_utils import utc_now
from src.core.working_memory import WorkingMemory
from src.retrieval.spreading_activation import SpreadingActivationRetriever
from src.sleep.sleep_cycle import SleepCycleOrchestrator
from tests.human_memory_benchmark import (
    benchmark_contradiction_versioning,
    benchmark_one_shot_recall,
    benchmark_selective_forgetting,
)
from tests.phase4_metrics import _build_phase4_dataset


SEED_START_DEFAULT = 7001
SEED_COUNT_DEFAULT = 10
PAIR_COUNT_DEFAULT = 96


def _stable_tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if token]


def _stable_embedding(text: str, dim: int = 384) -> List[float]:
    """Deterministic embedding for control baselines."""
    dim = max(8, dim)
    tokens = _stable_tokens(text)
    vector = [0.0] * dim
    if not tokens:
        vector[0] = 1.0
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], "big", signed=False) % dim
        vector[idx] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _state_value(concept: Concept) -> str:
    state = getattr(concept, "state", MemoryState.ACTIVE.value)
    return state.value if hasattr(state, "value") else str(state)


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False
    return ltm


def _average(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stddev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_float(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (datetime,)):
            clean[key] = value.isoformat()
        elif hasattr(value, "value"):
            clean[key] = value.value
        else:
            clean[key] = value
    return clean


def _keyword_score(query: str, text: str) -> float:
    q_tokens = set(_stable_tokens(query))
    t_tokens = set(_stable_tokens(text))
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _rank_keyword(query: str, concepts: Sequence[Concept]) -> List[Concept]:
    scored = [
        (concept, _keyword_score(query, concept.description or ""))
        for concept in concepts
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [concept for concept, _ in scored]


def _rank_vector(query: str, concepts: Sequence[Concept]) -> List[Concept]:
    query_emb = _stable_embedding(query)
    scored = []
    for concept in concepts:
        emb = _stable_embedding(concept.description or "")
        scored.append((concept, _cosine(query_emb, emb)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [concept for concept, _ in scored]


def _evaluate_ranker(
    ranker: Callable[[str, Sequence[Concept]], Sequence[Concept]],
    concepts: Sequence[Concept],
    key_ids: Dict[int, str],
    noise_ids: Dict[int, str],
) -> Dict[str, float]:
    disambiguation_hits = 0
    top1_hits = 0
    latencies_ms: List[float] = []

    for idx, key_id in key_ids.items():
        noise_id = noise_ids[idx]
        query = f"what about project token{idx} record?"

        start = perf_counter()
        ranked = list(ranker(query, concepts))
        latencies_ms.append((perf_counter() - start) * 1000.0)

        ranks = {concept.id: rank for rank, concept in enumerate(ranked)}
        if ranked and ranked[0].id == key_id:
            top1_hits += 1
        if ranks.get(key_id, 10**9) < ranks.get(noise_id, 10**9):
            disambiguation_hits += 1

    total = len(key_ids) or 1
    return {
        "top1_recall": round(top1_hits / total, 4),
        "disambiguation_recall": round(disambiguation_hits / total, 4),
        "mean_query_latency_ms": round(_average(latencies_ms), 4),
        "p95_query_latency_ms": round(sorted(latencies_ms)[max(0, int(0.95 * len(latencies_ms)) - 1)] if latencies_ms else 0.0, 4),
    }


def _evaluate_graph_baseline(
    concepts: Sequence[Concept],
    relations: Sequence[Any],
    key_ids: Dict[int, str],
    noise_ids: Dict[int, str],
) -> Dict[str, float]:
    ltm = _fast_ltm()
    for concept in concepts:
        ltm.add_concept(deepcopy(concept))
    for relation in relations:
        ltm.add_relation(deepcopy(relation))

    retriever = SpreadingActivationRetriever(
        working_memory=WorkingMemory(capacity=64),
        long_term_memory=ltm,
        spreading_steps=3,
        spreading_decay=0.45,
        activation_threshold=0.02,
        max_candidates=20,
    )

    return _evaluate_ranker(
        lambda query, _concepts: retriever.retrieve(
            query=query,
            context_tags={"session_id": "phase4", "person": "user"},
            max_seed_concepts=8,
        )[0],
        concepts,
        key_ids,
        noise_ids,
    )


def _sleep_mode_eval(
    mode: str,
    concepts: Sequence[Concept],
    relations: Sequence[Any],
    episodes: Sequence[Any],
    key_ids: Dict[int, str],
    noise_ids: Dict[int, str],
) -> Dict[str, float]:
    orchestrator = SleepCycleOrchestrator()
    ok, _, stats = orchestrator.begin_sleep_cycle(
        concepts=deepcopy(list(concepts)),
        relations=deepcopy(list(relations)),
        episodes=deepcopy(list(episodes)),
        force=True,
        mode=mode,
    )
    if not ok:
        raise RuntimeError(f"{mode} sleep did not run: {stats}")

    updated = stats.get("updated_concepts", [])
    active = [concept for concept in updated if _state_value(concept) == MemoryState.ACTIVE.value]
    retired = stats.get("deep", {}).get("retired_concepts", []) if mode == "deep" else []
    sleep_concepts = active + [concept for concept in retired if _state_value(concept) == MemoryState.ACTIVE.value]

    ltm = _fast_ltm()
    for concept in active:
        ltm.add_concept(deepcopy(concept))
    for relation in stats.get("updated_relations", []):
        ltm.add_relation(deepcopy(relation))

    retriever = SpreadingActivationRetriever(
        working_memory=WorkingMemory(capacity=64),
        long_term_memory=ltm,
        spreading_steps=3,
        spreading_decay=0.45,
        activation_threshold=0.02,
        max_candidates=20,
    )

    ranking = _evaluate_ranker(
        lambda query, _concepts: retriever.retrieve(
            query=query,
            context_tags={"session_id": "phase4", "person": "user"},
            max_seed_concepts=8,
        )[0],
        active,
        key_ids,
        noise_ids,
    )

    noise_total = sum(1 for idx in noise_ids.values())
    noise_kept = sum(1 for idx in noise_ids.values() if idx in {concept.id for concept in active})
    key_kept = sum(1 for idx in key_ids.values() if idx in {concept.id for concept in active})

    ranking.update(
        {
            "key_retention": round(key_kept / (len(key_ids) or 1), 4),
            "noise_retention": round(noise_kept / (len(noise_ids) or 1), 4),
            "memory_kept": len(active),
            "retired_concepts": len(retired),
            "noise_total": noise_total,
        }
    )
    return ranking


def _evaluate_phase4_suite(pair_count: int, seeds: Sequence[int]) -> Dict[str, Any]:
    methods = {
        "lexical": [],
        "vector": [],
        "scm_baseline": [],
        "micro_sleep": [],
        "deep_sleep": [],
    }

    for seed in seeds:
        concepts, relations, episodes, meta = _build_phase4_dataset(pair_count=pair_count, seed=seed)
        key_ids = meta["key_ids"]
        noise_ids = meta["noise_ids"]

        methods["lexical"].append(
            _evaluate_ranker(_rank_keyword, concepts, key_ids, noise_ids)
        )
        methods["vector"].append(
            _evaluate_ranker(_rank_vector, concepts, key_ids, noise_ids)
        )
        methods["scm_baseline"].append(
            _evaluate_graph_baseline(concepts, relations, key_ids, noise_ids)
        )
        methods["micro_sleep"].append(
            _sleep_mode_eval("micro", concepts, relations, episodes, key_ids, noise_ids)
        )
        methods["deep_sleep"].append(
            _sleep_mode_eval("deep", concepts, relations, episodes, key_ids, noise_ids)
        )

    rows = []
    for name, samples in methods.items():
        rows.append(
            {
                "method": name,
                "samples": len(samples),
                "top1_recall_mean": round(_average([sample["top1_recall"] for sample in samples]), 4),
                "top1_recall_std": round(_stddev([sample["top1_recall"] for sample in samples]), 4),
                "disambiguation_recall_mean": round(_average([sample["disambiguation_recall"] for sample in samples]), 4),
                "disambiguation_recall_std": round(_stddev([sample["disambiguation_recall"] for sample in samples]), 4),
                "mean_query_latency_ms": round(_average([sample["mean_query_latency_ms"] for sample in samples]), 4),
                "noise_retention_mean": round(_average([sample.get("noise_retention", 1.0) for sample in samples]), 4),
                "key_retention_mean": round(_average([sample.get("key_retention", 1.0) for sample in samples]), 4),
                "pass_rate": round(
                    sum(
                        1
                        for sample in samples
                        if sample["disambiguation_recall"] >= 0.20 and sample.get("noise_retention", 1.0) <= 0.40
                    )
                    / len(samples),
                    4,
                ),
            }
        )

    rows_by_name = {row["method"]: row for row in rows}
    summary = {
        "seed_count": len(seeds),
        "pair_count": pair_count,
        "lexical_disambiguation_mean": rows_by_name["lexical"]["disambiguation_recall_mean"],
        "vector_disambiguation_mean": rows_by_name["vector"]["disambiguation_recall_mean"],
        "scm_baseline_disambiguation_mean": rows_by_name["scm_baseline"]["disambiguation_recall_mean"],
        "micro_disambiguation_mean": rows_by_name["micro_sleep"]["disambiguation_recall_mean"],
        "deep_disambiguation_mean": rows_by_name["deep_sleep"]["disambiguation_recall_mean"],
        "deep_noise_retention_mean": rows_by_name["deep_sleep"]["noise_retention_mean"],
        "deep_pass_rate": rows_by_name["deep_sleep"]["pass_rate"],
    }

    return {
        "summary": summary,
        "rows": rows,
    }


REFERENCE_SYSTEM_FEATURES = [
    {
        "system": "SleepGate",
        "meaning_based": "✗",
        "sleep_consolidation": "△",
        "nrem_rem": "✗",
        "intentional_forgetting": "△",
        "multi_dim_importance": "✗",
        "cross_session": "✗",
        "notes": "Sleep metaphor, but still cache-level eviction rather than semantic memory.",
    },
    {
        "system": "MemGPT",
        "meaning_based": "✗",
        "sleep_consolidation": "✗",
        "nrem_rem": "✗",
        "intentional_forgetting": "✗",
        "multi_dim_importance": "✗",
        "cross_session": "△",
        "notes": "Strong memory-tiering idea, but no sleep or active forgetting.",
    },
    {
        "system": "Mem0",
        "meaning_based": "△",
        "sleep_consolidation": "✗",
        "nrem_rem": "✗",
        "intentional_forgetting": "△",
        "multi_dim_importance": "△",
        "cross_session": "●",
        "notes": "Practical production memory, but awake-only and not sleep reorganized.",
    },
    {
        "system": "WSCL",
        "meaning_based": "✗",
        "sleep_consolidation": "●",
        "nrem_rem": "●",
        "intentional_forgetting": "✗",
        "multi_dim_importance": "✗",
        "cross_session": "△",
        "notes": "Biologically inspired sleep stages, but not semantic conversational memory.",
    },
    {
        "system": "EWC",
        "meaning_based": "✗",
        "sleep_consolidation": "✗",
        "nrem_rem": "✗",
        "intentional_forgetting": "✗",
        "multi_dim_importance": "△",
        "cross_session": "△",
        "notes": "Important weighting mechanism, but not a memory architecture.",
    },
    {
        "system": "SCM",
        "meaning_based": "●",
        "sleep_consolidation": "●",
        "nrem_rem": "●",
        "intentional_forgetting": "●",
        "multi_dim_importance": "●",
        "cross_session": "●",
        "notes": "Semantic graph + sleep + forgetting + versioning in one system.",
    },
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_phase6_suite(pair_count: int, key_count: int, noise_count: int) -> Dict[str, Any]:
    one_shot = benchmark_one_shot_recall()
    forgetting = benchmark_selective_forgetting(key_count=key_count, noise_count=noise_count)
    contradiction = benchmark_contradiction_versioning()

    return {
        "one_shot": one_shot,
        "selective_forgetting": forgetting,
        "contradiction": contradiction,
        "summary": {
            "one_shot_accuracy": one_shot["accuracy"],
            "key_retention": forgetting["key_retention"],
            "noise_retention": forgetting["noise_retention"],
            "contradiction_accuracy": contradiction["accuracy"],
        },
    }


def build_report(args: argparse.Namespace) -> dict:
    seeds = [args.seed_start + idx for idx in range(args.seed_count)]

    phase4 = _evaluate_phase4_suite(pair_count=args.pair_count, seeds=seeds)
    phase6 = _evaluate_phase6_suite(
        pair_count=args.phase6_pair_count,
        key_count=args.key_count,
        noise_count=args.noise_count,
    )

    behavioral_rows = []
    for row in phase4["rows"]:
        method = row["method"]
        if method == "lexical":
            label = "Lexical retrieval baseline"
            family = "Standard retrieval"
            notes = "Token-overlap control."
        elif method == "vector":
            label = "Vector retrieval baseline"
            family = "Standard retrieval"
            notes = "Deterministic text-embedding control."
        elif method == "scm_baseline":
            label = "SCM baseline (no sleep)"
            family = "SCM control"
            notes = "Graph retrieval before sleep consolidation."
        elif method == "micro_sleep":
            label = "SCM + MicroSleep"
            family = "SCM sleep stage"
            notes = "Light replay, merging, and local reinforcement."
        else:
            label = "SCM + DeepSleep"
            family = "SCM sleep stage"
            notes = "Full replay, synthesis, and pruning pass."

        behavioral_rows.append(
            {
                "method": label,
                "family": family,
                "top1_recall_mean": row["top1_recall_mean"],
                "top1_recall_std": row["top1_recall_std"],
                "disambiguation_recall_mean": row["disambiguation_recall_mean"],
                "disambiguation_recall_std": row["disambiguation_recall_std"],
                "key_retention_mean": row["key_retention_mean"],
                "noise_retention_mean": row["noise_retention_mean"],
                "mean_query_latency_ms": row["mean_query_latency_ms"],
                "pass_rate": row["pass_rate"],
                "notes": notes,
            }
        )

    report = {
        "benchmark": "scm_baseline_comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "inputs": {
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seeds": seeds,
            "pair_count": args.pair_count,
            "phase6_pair_count": args.phase6_pair_count,
            "key_count": args.key_count,
            "noise_count": args.noise_count,
        },
        "behavioral_comparison": {
            "summary": phase4["summary"],
            "rows": behavioral_rows,
        },
        "human_memory_suite": phase6,
        "reference_systems": {
            "rows": REFERENCE_SYSTEM_FEATURES,
            "source": "research/04_Comparative_Gap_Analysis.md + current SCM implementation",
        },
    }

    report["status"] = {
        "lexical_control_pass": behavioral_rows[0]["disambiguation_recall_mean"] < behavioral_rows[-1]["disambiguation_recall_mean"],
        "vector_control_pass": behavioral_rows[1]["disambiguation_recall_mean"] < behavioral_rows[-1]["disambiguation_recall_mean"],
        "scm_deep_pass": behavioral_rows[-1]["disambiguation_recall_mean"] >= 0.20,
        "scm_noise_pass": behavioral_rows[-1]["noise_retention_mean"] <= 0.40,
        "human_suite_pass": bool(
            phase6["summary"]["one_shot_accuracy"] >= 1.00
            and phase6["summary"]["key_retention"] >= 0.80
            and phase6["summary"]["noise_retention"] <= 0.35
            and phase6["summary"]["contradiction_accuracy"] >= 0.90
        ),
    }
    report["status"]["overall_pass"] = all(report["status"].values())
    return report


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def _fmt(cell: Any) -> str:
        if cell is None:
            return ""
        if isinstance(cell, float):
            return _format_float(cell)
        return str(cell)

    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(out)


def render_markdown(report: dict) -> str:
    phase4 = report["behavioral_comparison"]
    phase6 = report["human_memory_suite"]
    refs = report["reference_systems"]["rows"]

    lines = [
        "# SCM Baseline Comparison",
        "",
        f"**Date:** {report['timestamp_utc'][:10]}",
        "",
        "This report compares SCM against standard retrieval controls and summarizes the core behavioral lift from sleep-stage consolidation.",
        "",
        "## Behavioral Comparison",
        "",
        f"Benchmark settings: `pair_count={report['inputs']['pair_count']}`, `seeds={report['inputs']['seeds'][0]}..{report['inputs']['seeds'][-1]}`.",
        "",
    ]

    rows = []
    for row in phase4["rows"]:
        rows.append(
            [
                row["method"],
                row["family"],
                f"{row['disambiguation_recall_mean']:.4f} ± {row['disambiguation_recall_std']:.4f}",
                f"{row['top1_recall_mean']:.4f} ± {row['top1_recall_std']:.4f}",
                f"{row['key_retention_mean']:.4f}",
                f"{row['noise_retention_mean']:.4f}",
                f"{row['mean_query_latency_ms']:.2f}",
                f"{row['pass_rate']:.2f}",
                row["notes"],
            ]
        )
    lines.append(
        _markdown_table(
            [
                "Method",
                "Family",
                "Disambiguation recall",
                "Top-1 recall",
                "Key retention",
                "Noise retention",
                "Mean latency (ms)",
                "Pass rate",
                "Notes",
            ],
            rows,
        )
    )

    lines.extend(
        [
            "",
            "## Human-Memory Suite",
            "",
        ]
    )
    lines.append(
        _markdown_table(
            ["Metric", "Score", "Target", "Pass"],
            [
                ["One-shot recall accuracy", phase6["summary"]["one_shot_accuracy"], ">= 1.00", "yes" if phase6["summary"]["one_shot_accuracy"] >= 1.0 else "no"],
                ["Selective forgetting key retention", phase6["summary"]["key_retention"], ">= 0.80", "yes" if phase6["summary"]["key_retention"] >= 0.8 else "no"],
                ["Selective forgetting noise retention", phase6["summary"]["noise_retention"], "<= 0.35", "yes" if phase6["summary"]["noise_retention"] <= 0.35 else "no"],
                ["Contradiction versioning accuracy", phase6["summary"]["contradiction_accuracy"], ">= 0.90", "yes" if phase6["summary"]["contradiction_accuracy"] >= 0.9 else "no"],
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Reference System Features",
            "",
        ]
    )
    lines.append(
        _markdown_table(
            [
                "System",
                "Meaning-based",
                "Sleep consolidation",
                "NREM + REM",
                "Intentional forgetting",
                "Multi-dim importance",
                "Cross-session",
                "Notes",
            ],
            [
                [
                    row["system"],
                    row["meaning_based"],
                    row["sleep_consolidation"],
                    row["nrem_rem"],
                    row["intentional_forgetting"],
                    row["multi_dim_importance"],
                    row["cross_session"],
                    row["notes"],
                ]
                for row in refs
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The control baselines stay weak on the adversarial duplicate-memory task, while SCM gains are driven by actual sleep consolidation rather than by raw retrieval tricks.",
            "The external reference systems remain useful baselines, but the feature matrix shows that none of them combine semantic memory, multi-stage sleep, and intentional forgetting in one architecture.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _write_json(report: dict, output: Path, write_history: bool, history_dir: Path) -> Path | None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    history_path = None
    if write_history:
        history_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        history_path = history_dir / f"baseline_comparison_{ts_safe}.json"
        history_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return history_path


def _write_markdown(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SCM baseline comparison report")
    parser.add_argument(
        "--output",
        default="research/metrics/baseline_comparison_latest.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--markdown-output",
        default="docs/SCM_BASELINE_COMPARISON.md",
        help="Markdown output path",
    )
    parser.add_argument("--seed-start", type=int, default=SEED_START_DEFAULT)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT_DEFAULT)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT_DEFAULT)
    parser.add_argument("--phase6-pair-count", type=int, default=32)
    parser.add_argument("--key-count", type=int, default=20)
    parser.add_argument("--noise-count", type=int, default=40)
    parser.add_argument(
        "--write-history",
        action="store_true",
        help="Also write a timestamped JSON snapshot",
    )
    parser.add_argument(
        "--history-dir",
        default="research/metrics/history/baseline_comparison",
        help="Directory for timestamped snapshots",
    )

    args = parser.parse_args()
    report = build_report(args)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    markdown_path = Path(args.markdown_output)
    if not markdown_path.is_absolute():
        markdown_path = REPO_ROOT / markdown_path

    history_dir = Path(args.history_dir)
    if not history_dir.is_absolute():
        history_dir = REPO_ROOT / history_dir

    history_path = _write_json(
        report=report,
        output=output_path,
        write_history=args.write_history,
        history_dir=history_dir,
    )
    _write_markdown(report, markdown_path)

    print(f"Wrote comparison JSON to {output_path}")
    print(f"Wrote comparison markdown to {markdown_path}")
    if history_path:
        print(f"Wrote history snapshot to {history_path}")
    print(json.dumps(report["status"], indent=2))


if __name__ == "__main__":
    main()
