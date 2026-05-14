"""Generate machine-readable Phase 6 human-memory benchmark metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import uuid

# Ensure repo root is on import path when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.chat.engine as chat_engine_module
import src.core.config as config_module
from src.chat.engine import ChatEngine
from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, ImportanceVector, MemoryState
from src.core.time_utils import utc_now
from src.sleep.forgetting_dynamics import ForgettingDynamics
from tests.phase4_metrics import benchmark_sleep_gain


def _embedding(seed: int) -> list[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _normalize_preference_text(text: str) -> str:
    return re.sub(
        r"\s+(?:right now|for now|currently|at the moment|now)\s*$",
        "",
        (text or "").strip(),
        flags=re.IGNORECASE,
    ).strip(" .,!?:;")


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False
    return ltm


class _FailLLM:
    """Forces ChatEngine to use deterministic fallback response path."""

    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        raise RuntimeError("offline fallback benchmark")


class _ProfileEncoder:
    def extract(self, text: str):
        lower = text.lower()
        concepts: list[Concept] = []

        name_match = re.search(r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z'-]*)", text, flags=re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            concepts.append(
                Concept(
                    type=ConceptType.PERSON,
                    description=f"Person: {name}",
                    embedding=self._get_embedding(f"name:{name}"),
                    importance=ImportanceVector(novelty=0.85, task_relevance=0.9, repetition=0.5),
                    salience_score=0.85,
                    grasp_score=0.9,
                )
            )

        location_match = re.search(r"(?:i live in|live in)\s+([A-Za-z][A-Za-z\s'-]+)", text, flags=re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            concepts.append(
                Concept(
                    type=ConceptType.LOCATION,
                    description=f"I live in {location}",
                    embedding=self._get_embedding(f"location:{location}"),
                    importance=ImportanceVector(novelty=0.8, task_relevance=0.85, repetition=0.4),
                    salience_score=0.8,
                    grasp_score=0.85,
                )
            )

        if "prefer" in lower:
            pref_match = re.search(r"(?:i prefer)\s+([^.!?]+)", text, flags=re.IGNORECASE)
            if pref_match:
                pref = _normalize_preference_text(pref_match.group(1))
                concepts.append(
                    Concept(
                        type=ConceptType.PREFERENCE,
                        description=f"I prefer {pref}",
                        embedding=self._get_embedding(f"pref:{pref}"),
                        importance=ImportanceVector(novelty=0.78, task_relevance=0.88, repetition=0.35),
                        salience_score=0.78,
                        grasp_score=0.82,
                    )
                )

        if not concepts and not any(token in lower for token in ("what is my", "where do i", "what do i")):
            concepts.append(
                Concept(
                    type=ConceptType.FACT,
                    description=text.strip(),
                    embedding=self._get_embedding(text),
                    importance=ImportanceVector(novelty=0.6, task_relevance=0.6, repetition=0.3),
                    salience_score=0.6,
                    grasp_score=0.6,
                )
            )

        return concepts

    def _get_embedding(self, text: str):
        seed = sum(ord(ch) for ch in text) % 4096
        return _embedding(seed)


def benchmark_one_shot_recall() -> dict:
    config_module.ENABLE_SESSION_PERSISTENCE = False
    chat_engine_module.HME_ENABLED = True

    engine = ChatEngine(
        llm=_FailLLM(),
        encoder=_ProfileEncoder(),
        long_term_memory=_fast_ltm(),
        enable_auto_sleep=False,
        session_id=f"phase6_human_{uuid.uuid4().hex}",
    )

    engine.chat("My name is Alice.")
    engine.chat("I live in Seattle.")

    name_response, _ = engine.chat("What is my name?")
    location_response, _ = engine.chat("Where do I live?")

    name_hit = "alice" in name_response.lower()
    location_hit = "seattle" in location_response.lower()
    accuracy = (float(name_hit) + float(location_hit)) / 2.0

    return {
        "name_response": name_response,
        "location_response": location_response,
        "name_hit": name_hit,
        "location_hit": location_hit,
        "accuracy": round(accuracy, 4),
    }


def benchmark_selective_forgetting(key_count: int, noise_count: int) -> dict:
    now = utc_now()
    concepts: list[Concept] = []

    for i in range(key_count):
        c = Concept(
            id=f"key_{i}",
            type=ConceptType.FACT,
            description=f"critical memory {i}",
            embedding=_embedding(i),
            importance=ImportanceVector(novelty=0.86, emotional=0.1, task_relevance=0.92, repetition=0.7),
            salience_score=0.88,
            grasp_score=0.9,
            rehearsal_count=4 + (i % 3),
            association_density=0.7,
            strength=1.0,
        )
        c.last_accessed = now - timedelta(hours=1 + (i % 4))
        concepts.append(c)

    for i in range(noise_count):
        c = Concept(
            id=f"noise_{i}",
            type=ConceptType.FACT,
            description=f"noisy trace {i}",
            embedding=_embedding(20_000 + i),
            importance=ImportanceVector(novelty=0.08, emotional=0.0, task_relevance=0.1, repetition=0.05),
            salience_score=0.1,
            grasp_score=0.08,
            rehearsal_count=0,
            association_density=0.03,
            strength=0.45,
        )
        c.last_accessed = now - timedelta(hours=72 + (i % 12))
        concepts.append(c)

    dynamics = ForgettingDynamics(suppress_threshold=0.32, archive_threshold=0.16)
    updated_first, first_stats = dynamics.apply(concepts, now=now)
    updated_second, second_stats = dynamics.apply(updated_first, now=now + timedelta(hours=24))

    by_id = {concept.id: concept for concept in updated_second}
    key_active = sum(
        1
        for i in range(key_count)
        if (by_id[f"key_{i}"].state.value if hasattr(by_id[f"key_{i}"].state, "value") else by_id[f"key_{i}"].state)
        == MemoryState.ACTIVE.value
    )
    noise_active = sum(
        1
        for i in range(noise_count)
        if (by_id[f"noise_{i}"].state.value if hasattr(by_id[f"noise_{i}"].state, "value") else by_id[f"noise_{i}"].state)
        == MemoryState.ACTIVE.value
    )

    key_retention = key_active / key_count if key_count else 0.0
    noise_retention = noise_active / noise_count if noise_count else 0.0

    return {
        "inputs": {"key_count": key_count, "noise_count": noise_count},
        "first_cycle": first_stats,
        "second_cycle": second_stats,
        "key_retention": round(key_retention, 4),
        "noise_retention": round(noise_retention, 4),
    }


def benchmark_contradiction_versioning() -> dict:
    ltm = _fast_ltm()
    tags = {"session_id": "phase6", "person": "user", "task": "conversation"}

    old = Concept(
        id="pref_old",
        type=ConceptType.PREFERENCE,
        description="I prefer morning meetings",
        embedding=_embedding(501),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.9, repetition=0.7),
        context_tags=dict(tags),
    )
    ltm.add_concept(old, context_tags=tags, allow_versioning=False)

    new = Concept(
        id="pref_new",
        type=ConceptType.PREFERENCE,
        description="I prefer evening meetings",
        embedding=_embedding(501),
        importance=ImportanceVector(novelty=0.88, task_relevance=0.95, repetition=0.6),
        context_tags=dict(tags),
    )
    stored = ltm.add_concept(new, context_tags=tags, allow_versioning=True)

    current_ids = {c.id for c in ltm.get_all_concepts()}
    history_ids = {c.id for c in ltm.get_all_concepts(include_superseded=True)}
    retrieved = ltm.search_by_embedding(stored.embedding, limit=3)

    chain_ok = stored.version_parent == old.id and stored.version_root == old.id
    old_hidden = old.id not in current_ids and old.id in history_ids
    retrieval_prefers_new = bool(retrieved) and retrieved[0].id == stored.id

    checks = [chain_ok, old_hidden, retrieval_prefers_new]
    accuracy = sum(1 for check in checks if check) / len(checks)

    return {
        "chain_ok": chain_ok,
        "old_hidden": old_hidden,
        "retrieval_prefers_new": retrieval_prefers_new,
        "accuracy": round(accuracy, 4),
    }


def build_report(args: argparse.Namespace) -> dict:
    one_shot = benchmark_one_shot_recall()
    sleep_gain = benchmark_sleep_gain(pair_count=args.pair_count, seed=args.seed + 11)
    forgetting = benchmark_selective_forgetting(
        key_count=args.key_count,
        noise_count=args.noise_count,
    )
    contradiction = benchmark_contradiction_versioning()

    micro_gain = sleep_gain["micro_sleep"]["disambiguation_gain_abs"]
    deep_gain = sleep_gain["deep_sleep"]["disambiguation_gain_abs"]
    deep_noise_retention = sleep_gain["deep_sleep"]["pressure"]["noise_retention"]

    report = {
        "benchmark": "phase6_human_memory_behavior",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "seed": args.seed,
        "targets": {
            "one_shot_accuracy_min": 1.00,
            "micro_disambiguation_gain_min_abs": 0.20,
            "deep_disambiguation_gain_min_abs": 0.20,
            "deep_noise_retention_max": 0.40,
            "key_retention_min": 0.80,
            "noise_retention_max": 0.35,
            "contradiction_accuracy_min": 0.90,
        },
        "inputs": {
            "pair_count": args.pair_count,
            "key_count": args.key_count,
            "noise_count": args.noise_count,
        },
        "metrics": {
            "one_shot_recall": one_shot,
            "sleep_gain": sleep_gain,
            "selective_forgetting": forgetting,
            "contradiction_versioning": contradiction,
        },
    }

    report["status"] = {
        "one_shot_pass": one_shot["accuracy"] >= report["targets"]["one_shot_accuracy_min"],
        "micro_gain_pass": micro_gain >= report["targets"]["micro_disambiguation_gain_min_abs"],
        "deep_gain_pass": deep_gain >= report["targets"]["deep_disambiguation_gain_min_abs"],
        "deep_noise_pass": deep_noise_retention <= report["targets"]["deep_noise_retention_max"],
        "key_retention_pass": forgetting["key_retention"] >= report["targets"]["key_retention_min"],
        "noise_retention_pass": forgetting["noise_retention"] <= report["targets"]["noise_retention_max"],
        "contradiction_pass": contradiction["accuracy"] >= report["targets"]["contradiction_accuracy_min"],
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
        history_path = history_dir / f"phase6_human_memory_{ts_safe}.json"
        history_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return history_path


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 6 human-memory benchmark metrics JSON")
    parser.add_argument(
        "--output",
        default="research/metrics/phase6_human_memory_latest.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--pair-count", type=int, default=32)
    parser.add_argument("--key-count", type=int, default=20)
    parser.add_argument("--noise-count", type=int, default=40)
    parser.add_argument(
        "--write-history",
        action="store_true",
        help="Also write a timestamped metrics snapshot",
    )
    parser.add_argument(
        "--history-dir",
        default="research/metrics/history/phase6",
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
