"""Generate a long-horizon SCM memory benchmark report.

This benchmark simulates a multi-day memory history with repeated interference,
sleep reactivation, and a late contradiction update. It compares an awake-only
control with the sleep-enabled SCM path so we can measure retention, noise
suppression, and correction stability over time.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, Episode, ImportanceVector, MemoryState, PredicateType, Relation
from src.core.working_memory import WorkingMemory
from src.retrieval.spreading_activation import SpreadingActivationRetriever
from src.sleep.sleep_cycle import SleepCycleOrchestrator
from tests.phase4_metrics import _build_phase4_dataset


SESSION_ID = "long_horizon"
DEFAULT_ANCHOR_UPDATE_DAY = 4
DEFAULT_DEEP_DAYS = (3, 6, 7)


def _stable_tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if token]


def _stable_embedding(text: str, dim: int = 384) -> List[float]:
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


def _text_seed(text: str) -> int:
    return sum(ord(ch) for ch in (text or "")) % 4096


def _embedding_from_text(text: str) -> List[float]:
    base = (_text_seed(text) + 1) / 1000.0
    return [base + ((i % 11) * 0.0001) for i in range(384)]


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory()
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False
    return ltm


def _state_value(concept: Concept) -> str:
    state = getattr(concept, "state", MemoryState.ACTIVE.value)
    return state.value if hasattr(state, "value") else str(state)


def _average(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stddev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _format_float(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def _best_rank(ranks: Dict[str, int], candidate_ids: Sequence[str]) -> int:
    return min((ranks.get(candidate_id, 10**9) for candidate_id in candidate_ids), default=10**9)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _build_retriever(ltm: LongTermMemory) -> SpreadingActivationRetriever:
    return SpreadingActivationRetriever(
        working_memory=WorkingMemory(capacity=64),
        long_term_memory=ltm,
        spreading_steps=3,
        spreading_decay=0.45,
        activation_threshold=0.02,
        max_candidates=20,
    )


def _augment_day_batch(
    day_index: int,
    concepts: List[Concept],
    relations: List[Relation],
    episodes: List[Episode],
    extra_noise_count: int,
) -> tuple[list[Concept], list[Relation], list[Episode], dict]:
    """Attach long-horizon metadata and add a little extra pure noise."""
    rng = random.Random(day_index * 97 + extra_noise_count)
    day_tag = f"day{day_index}"
    stray_noise_ids: List[str] = []

    for concept in concepts:
        concept.context_tags = concept.context_tags or {}
        concept.context_tags.update(
            {
                "session_id": SESSION_ID,
                "person": "user",
                "task": "long_horizon",
                "day": day_tag,
            }
        )

    for episode in episodes:
        episode.context = episode.context or {}
        episode.context.update(
            {
                "session_id": SESSION_ID,
                "person": "user",
                "task": "long_horizon",
                "day": day_tag,
            }
        )

    for noise_idx in range(extra_noise_count):
        desc = f"{day_tag} stray note {noise_idx}"
        noise = Concept(
            type=ConceptType.FACT,
            description=desc,
            embedding=_embedding_from_text(desc),
            importance=ImportanceVector(
                novelty=0.05,
                emotional=0.0,
                task_relevance=0.06,
                repetition=0.03,
            ),
            salience_score=0.08,
            grasp_score=0.07,
            strength=0.35,
            rehearsal_count=0,
            association_density=0.01,
        )
        noise.context_tags = {
            "session_id": SESSION_ID,
            "person": "user",
            "task": "long_horizon",
            "day": day_tag,
            "role": "stray_noise",
        }
        concepts.append(noise)
        stray_noise_ids.append(noise.id)
        episodes.append(
            Episode(
                concept_ids=[noise.id],
                raw_content=f"{day_tag} stray note {noise_idx}",
                importance=ImportanceVector(
                    novelty=0.04,
                    emotional=0.0,
                    task_relevance=0.05,
                    repetition=0.02,
                ),
                source="user",
            )
        )

    rng.shuffle(episodes)
    metadata = {"stray_noise_ids": stray_noise_ids, "day_tag": day_tag}
    return concepts, relations, episodes, metadata


def _make_anchor_concept(owner: str) -> Concept:
    desc = f"project atlas owner is {owner}"
    concept = Concept(
        type=ConceptType.FACT,
        description=desc,
        embedding=_embedding_from_text(desc),
        importance=ImportanceVector(
            novelty=0.78,
            emotional=0.05,
            task_relevance=0.96,
            repetition=0.70,
        ),
        salience_score=0.88,
        grasp_score=0.92,
        prediction_error=0.08,
        strength=1.0,
        rehearsal_count=4,
        association_density=0.45,
    )
    concept.context_tags = {
        "session_id": SESSION_ID,
        "person": "user",
        "task": "long_horizon",
        "anchor": "atlas",
        "anchor_type": "versioned_fact",
        "owner": owner.lower(),
    }
    return concept


def _make_anchor_episode(concept_id: str, owner: str, day_index: int) -> Episode:
    desc = f"project atlas owner is {owner}"
    episode = Episode(
        concept_ids=[concept_id],
        raw_content=desc,
        importance=ImportanceVector(
            novelty=0.25,
            emotional=0.02,
            task_relevance=0.95,
            repetition=0.80,
        ),
        source="user",
    )
    episode.context = {
        "session_id": SESSION_ID,
        "person": "user",
        "task": "long_horizon",
        "day": f"day{day_index}",
        "anchor": "atlas",
    }
    return episode


def _make_review_episodes(day1_key_ids: Dict[int, str], day_index: int, review_count: int) -> List[Episode]:
    episodes: List[Episode] = []
    for idx in list(day1_key_ids.keys())[:review_count]:
        episodes.append(
            Episode(
                concept_ids=[day1_key_ids[idx]],
                raw_content=f"review project token{idx} record",
                importance=ImportanceVector(
                    novelty=0.12,
                    emotional=0.0,
                    task_relevance=0.92,
                    repetition=0.82,
                ),
                source="user",
            )
        )
        episodes[-1].context = {
            "session_id": SESSION_ID,
            "person": "user",
            "task": "long_horizon",
            "day": f"day{day_index}",
            "kind": "replay",
        }
    return episodes


def _add_batch_to_ltm(
    ltm: LongTermMemory,
    concepts: Sequence[Concept],
    relations: Sequence[Relation],
) -> None:
    for concept in concepts:
        ltm.add_concept(concept, context_tags=concept.context_tags)
    for relation in relations:
        ltm.add_relation(relation)


def _apply_sleep_updates(ltm: LongTermMemory, prior_concepts: Sequence[Concept], stats: Dict[str, Any]) -> Dict[str, int]:
    updated_concepts = stats.get("updated_concepts")
    updated_relations = stats.get("updated_relations")
    retired_concepts = stats.get("retired_concepts") or stats.get("deep", {}).get("retired_concepts", [])

    if updated_concepts is None:
        return {"synced_concepts": 0, "removed_concepts": 0, "synced_relations": 0}

    prior_ids = {concept.id for concept in prior_concepts}
    surviving_ids = set()
    for concept in list(updated_concepts) + list(retired_concepts):
        surviving_ids.add(concept.id)
        ltm.add_concept(concept)

    removed_ids = prior_ids - surviving_ids
    for concept_id in removed_ids:
        if concept_id in ltm._concept_cache:
            del ltm._concept_cache[concept_id]
        if concept_id in ltm.graph:
            ltm.graph.remove_node(concept_id)

    synced_relations = 0
    if updated_relations is not None:
        for relation in updated_relations:
            if relation.subject_id not in ltm.graph:
                continue
            if relation.object_id not in ltm.graph:
                continue
            ltm.add_relation(relation)
            synced_relations += 1

    return {
        "synced_concepts": len(updated_concepts),
        "removed_concepts": len(removed_ids),
        "synced_relations": synced_relations,
    }


def _evaluate_holdout(
    ltm: LongTermMemory,
    key_ids: Dict[int, str],
    noise_ids: Dict[int, str],
    session_id: str,
    key_family_ids: Dict[int, Sequence[str]] | None = None,
    noise_family_ids: Dict[int, Sequence[str]] | None = None,
) -> Dict[str, Any]:
    retriever = _build_retriever(ltm)
    context_tags = {"session_id": session_id, "person": "user", "task": "long_horizon"}
    strict_top1_hits = 0
    strict_disambiguation_hits = 0
    family_top1_hits = 0
    family_disambiguation_hits = 0
    latencies_ms: List[float] = []

    for idx, key_id in key_ids.items():
        noise_id = noise_ids[idx]
        query = f"project token{idx} record"
        key_family = list(key_family_ids.get(idx, [key_id])) if key_family_ids else [key_id]
        noise_family = list(noise_family_ids.get(idx, [noise_id])) if noise_family_ids else [noise_id]

        start = perf_counter()
        activated, _stats = retriever.retrieve(
            query=query,
            context_tags=context_tags,
            max_seed_concepts=8,
        )
        latencies_ms.append((perf_counter() - start) * 1000.0)

        ranks = {concept.id: rank for rank, concept in enumerate(activated)}
        if activated and activated[0].id == key_id:
            strict_top1_hits += 1
        if ranks.get(key_id, 10**9) < ranks.get(noise_id, 10**9):
            strict_disambiguation_hits += 1
        if activated and activated[0].id in set(key_family):
            family_top1_hits += 1
        if _best_rank(ranks, key_family) < _best_rank(ranks, noise_family):
            family_disambiguation_hits += 1

    total = len(key_ids) or 1
    active_ids = {concept.id for concept in ltm.get_all_concepts(include_suppressed=False)}
    key_retention = sum(1 for cid in key_ids.values() if cid in active_ids) / total
    noise_retention = sum(1 for cid in noise_ids.values() if cid in active_ids) / total

    return {
        "top1_recall": round(family_top1_hits / total, 4),
        "disambiguation_recall": round(family_disambiguation_hits / total, 4),
        "strict_top1_recall": round(strict_top1_hits / total, 4),
        "strict_disambiguation_recall": round(strict_disambiguation_hits / total, 4),
        "mean_latency_ms": round(_average(latencies_ms), 4),
        "p95_latency_ms": round(sorted(latencies_ms)[max(0, int(0.95 * len(latencies_ms)) - 1)] if latencies_ms else 0.0, 4),
        "key_retention": round(key_retention, 4),
        "noise_retention": round(noise_retention, 4),
        "active_concepts": len(active_ids),
    }


def _evaluate_anchor(
    ltm: LongTermMemory,
    expected_id: str,
    session_id: str,
) -> Dict[str, Any]:
    retriever = _build_retriever(ltm)
    query = "project atlas owner"
    activated, _stats = retriever.retrieve(
        query=query,
        context_tags={"session_id": session_id, "person": "user", "task": "long_horizon"},
        max_seed_concepts=6,
    )

    top_id = activated[0].id if activated else None
    expected = ltm.get_concept(expected_id)

    return {
        "query": query,
        "expected_id": expected_id,
        "top_id": top_id,
        "top_hit": top_id == expected_id,
        "expected_is_current": bool(expected and expected.is_current_version),
        "expected_is_active": bool(expected and _state_value(expected) == MemoryState.ACTIVE.value),
    }


def _sleep_schedule(day_index: int, horizon_days: int) -> str:
    if day_index == horizon_days:
        return "deep"
    if day_index in DEFAULT_DEEP_DAYS:
        return "deep"
    return "micro"


def _simulate_mode(
    *,
    mode_name: str,
    seed: int,
    horizon_days: int,
    daily_pair_count: int,
    extra_noise_count: int,
    review_count: int,
    anchor_update_day: int,
    sleep_enabled: bool,
) -> Dict[str, Any]:
    ltm = _fast_ltm()
    sleep_orchestrator = SleepCycleOrchestrator()
    day1_key_ids: Dict[int, str] = {}
    day1_noise_ids: Dict[int, str] = {}
    key_family_ids: Dict[int, List[str]] = {}
    noise_family_ids: Dict[int, List[str]] = {}
    anchor_initial_id: str | None = None
    anchor_update_id: str | None = None
    curves: List[Dict[str, Any]] = []
    strict_curves: List[Dict[str, Any]] = []
    sleep_history: List[Dict[str, Any]] = []
    all_noise_ids: List[str] = []
    all_key_ids: List[str] = []

    for day_index in range(1, horizon_days + 1):
        day_seed = seed + day_index * 1009
        concepts, relations, episodes, meta = _build_phase4_dataset(
            pair_count=daily_pair_count,
            seed=day_seed,
        )
        concepts, relations, episodes, augment_meta = _augment_day_batch(
            day_index=day_index,
            concepts=concepts,
            relations=relations,
            episodes=episodes,
            extra_noise_count=extra_noise_count,
        )

        if day_index == 1:
            day1_key_ids = dict(meta["key_ids"])
            day1_noise_ids = dict(meta["noise_ids"])

        for idx, key_id in meta["key_ids"].items():
            key_family_ids.setdefault(idx, []).append(key_id)
        for idx, noise_id in meta["noise_ids"].items():
            noise_family_ids.setdefault(idx, []).append(noise_id)

        all_key_ids.extend(meta["key_ids"].values())
        all_noise_ids.extend(meta["noise_ids"].values())
        all_noise_ids.extend(augment_meta["stray_noise_ids"])

        anchor_episodes: List[Episode] = []
        if day_index == 1:
            initial_anchor = _make_anchor_concept("Alice")
            initial_anchor = ltm.add_concept(
                initial_anchor,
                context_tags=initial_anchor.context_tags,
            )
            anchor_initial_id = initial_anchor.id
            anchor_episodes.append(_make_anchor_episode(anchor_initial_id, "Alice", day_index))
        if day_index == anchor_update_day:
            update_anchor = _make_anchor_concept("Bob")
            update_anchor = ltm.add_concept(
                update_anchor,
                context_tags=update_anchor.context_tags,
                allow_versioning=True,
            )
            anchor_update_id = update_anchor.id
            anchor_episodes.append(_make_anchor_episode(anchor_update_id, "Bob", day_index))

        if sleep_enabled and day_index > 1 and day1_key_ids:
            episodes.extend(_make_review_episodes(day1_key_ids, day_index, review_count))

        _add_batch_to_ltm(ltm, concepts, relations)
        for episode in anchor_episodes:
            episodes.append(episode)

        if sleep_enabled:
            prior_concepts = ltm.get_all_concepts(include_suppressed=False, include_superseded=False)
            current_relations = ltm.get_all_relations(include_history=False)
            sleep_mode = _sleep_schedule(day_index, horizon_days)
            ok, cycle, stats = sleep_orchestrator.begin_sleep_cycle(
                concepts=deepcopy(prior_concepts),
                relations=deepcopy(current_relations),
                episodes=deepcopy(episodes),
                force=True,
                mode=sleep_mode,
                session_turns=day_index,
                turns_since_micro=day_index,
            )
            if not ok:
                raise RuntimeError(f"Sleep cycle failed on day {day_index}: {stats}")

            sync_stats = _apply_sleep_updates(ltm, prior_concepts, stats)
            sleep_history.append(
                {
                    "day": day_index,
                    "mode": sleep_mode,
                    "consolidated": cycle.memories_consolidated,
                    "forgotten": cycle.memories_forgotten,
                    "dreams": len(cycle.dreams_generated),
                    "synced_concepts": sync_stats["synced_concepts"],
                    "removed_concepts": sync_stats["removed_concepts"],
                    "synced_relations": sync_stats["synced_relations"],
                }
            )

        current_expected_id = anchor_update_id if anchor_update_id and day_index >= anchor_update_day else anchor_initial_id
        snapshot = _evaluate_holdout(
            ltm=ltm,
            key_ids=day1_key_ids,
            noise_ids=day1_noise_ids,
            session_id=SESSION_ID,
            key_family_ids=key_family_ids,
            noise_family_ids=noise_family_ids,
        )
        if current_expected_id:
            anchor_snapshot = _evaluate_anchor(
                ltm=ltm,
                expected_id=current_expected_id,
                session_id=SESSION_ID,
            )
        else:
            anchor_snapshot = {
                "query": "project atlas owner",
                "expected_id": None,
                "top_id": None,
                "top_hit": False,
                "expected_is_current": False,
                "expected_is_active": False,
            }

        strict_curves.append(
            {
                "day": day_index,
                "strict_top1_recall": snapshot["strict_top1_recall"],
                "strict_disambiguation_recall": snapshot["strict_disambiguation_recall"],
            }
        )
        snapshot.update(
            {
                "day": day_index,
                "sleep_mode": _sleep_schedule(day_index, horizon_days) if sleep_enabled else "awake",
                "anchor": anchor_snapshot,
                "active_noise_count": sum(
                    1 for cid in all_noise_ids if cid in {c.id for c in ltm.get_all_concepts(include_suppressed=False)}
                ),
            }
        )
        curves.append(snapshot)

    final = curves[-1]
    final_active_ids = {concept.id for concept in ltm.get_all_concepts(include_suppressed=False)}
    final_noise_retention = sum(1 for cid in all_noise_ids if cid in final_active_ids) / (len(all_noise_ids) or 1)
    final_key_retention = sum(1 for cid in all_key_ids if cid in final_active_ids) / (len(all_key_ids) or 1)

    anchor_hits = [snapshot["anchor"]["top_hit"] for snapshot in curves if snapshot["anchor"]["expected_id"]]
    anchor_currents = [snapshot["anchor"]["expected_is_current"] for snapshot in curves if snapshot["anchor"]["expected_id"]]
    anchor_actives = [snapshot["anchor"]["expected_is_active"] for snapshot in curves if snapshot["anchor"]["expected_id"]]
    anchor_hit_rates = [
        1.0 if snapshot["anchor"]["top_hit"] else 0.0
        for snapshot in curves
        if snapshot["anchor"]["expected_id"]
    ]
    anchor_accuracy = (
        sum(
            1
            for snapshot in curves
            if snapshot["anchor"]["expected_id"]
            and snapshot["anchor"]["top_hit"]
            and snapshot["anchor"]["expected_is_current"]
            and snapshot["anchor"]["expected_is_active"]
        )
        / len([snapshot for snapshot in curves if snapshot["anchor"]["expected_id"]])
        if any(snapshot["anchor"]["expected_id"] for snapshot in curves)
        else 0.0
    )

    return {
        "mode": mode_name,
        "seed": seed,
        "inputs": {
            "horizon_days": horizon_days,
            "daily_pair_count": daily_pair_count,
            "extra_noise_count": extra_noise_count,
            "review_count": review_count,
            "anchor_update_day": anchor_update_day,
            "sleep_enabled": sleep_enabled,
            "deep_days": list(DEFAULT_DEEP_DAYS),
        },
        "curve": curves,
        "summary": {
            "final_disambiguation_recall": final["disambiguation_recall"],
            "final_top1_recall": final["top1_recall"],
            "final_strict_disambiguation_recall": final["strict_disambiguation_recall"],
            "final_strict_top1_recall": final["strict_top1_recall"],
            "final_anchor_hit_rate": round(anchor_hit_rates[-1], 4) if anchor_hit_rates else 0.0,
            "final_noise_retention": round(final_noise_retention, 4),
            "final_key_retention": round(final_key_retention, 4),
            "mean_disambiguation_recall": round(_average([snapshot["disambiguation_recall"] for snapshot in curves]), 4),
            "mean_anchor_hit_rate": round(_average(anchor_hit_rates), 4),
            "mean_noise_retention": round(_average([snapshot["noise_retention"] for snapshot in curves]), 4),
            "mean_key_retention": round(_average([snapshot["key_retention"] for snapshot in curves]), 4),
            "mean_latency_ms": round(_average([snapshot["mean_latency_ms"] for snapshot in curves]), 4),
            "mean_strict_disambiguation_recall": round(_average([snapshot["strict_disambiguation_recall"] for snapshot in curves]), 4),
            "mean_strict_top1_recall": round(_average([snapshot["strict_top1_recall"] for snapshot in curves]), 4),
            "anchor_accuracy": round(anchor_accuracy, 4),
            "anchor_top_hit_rate": round(_average(anchor_hit_rates), 4) if anchor_hit_rates else 0.0,
            "sleep_cycles": len(sleep_history),
            "sleep_history": sleep_history,
        },
        "strict_curve": strict_curves,
    }


def build_report(args: argparse.Namespace) -> dict:
    per_seed: List[Dict[str, Any]] = []
    for seed_offset in range(args.seed_count):
        seed = args.seed_start + seed_offset
        per_seed.append(
            {
                "seed": seed,
                "awake_only": _simulate_mode(
                    mode_name="awake_only",
                    seed=seed,
                    horizon_days=args.horizon_days,
                    daily_pair_count=args.daily_pair_count,
                    extra_noise_count=args.extra_noise_count,
                    review_count=args.review_count,
                    anchor_update_day=args.anchor_update_day,
                    sleep_enabled=False,
                ),
                "sleep_enabled": _simulate_mode(
                    mode_name="sleep_enabled",
                    seed=seed,
                    horizon_days=args.horizon_days,
                    daily_pair_count=args.daily_pair_count,
                    extra_noise_count=args.extra_noise_count,
                    review_count=args.review_count,
                    anchor_update_day=args.anchor_update_day,
                    sleep_enabled=True,
                ),
            }
        )

    def _summary_for(mode: str, field: str) -> float:
        return round(
            _average([seed_row[mode]["summary"][field] for seed_row in per_seed]),
            4,
        )

    awake_summary = {
        "final_disambiguation_recall": _summary_for("awake_only", "final_disambiguation_recall"),
        "final_top1_recall": _summary_for("awake_only", "final_top1_recall"),
        "final_strict_disambiguation_recall": _summary_for("awake_only", "final_strict_disambiguation_recall"),
        "final_strict_top1_recall": _summary_for("awake_only", "final_strict_top1_recall"),
        "final_anchor_hit_rate": _summary_for("awake_only", "final_anchor_hit_rate"),
        "final_noise_retention": _summary_for("awake_only", "final_noise_retention"),
        "final_key_retention": _summary_for("awake_only", "final_key_retention"),
        "mean_disambiguation_recall": _summary_for("awake_only", "mean_disambiguation_recall"),
        "mean_strict_disambiguation_recall": _summary_for("awake_only", "mean_strict_disambiguation_recall"),
        "mean_strict_top1_recall": _summary_for("awake_only", "mean_strict_top1_recall"),
        "mean_anchor_hit_rate": _summary_for("awake_only", "mean_anchor_hit_rate"),
        "mean_noise_retention": _summary_for("awake_only", "mean_noise_retention"),
        "mean_key_retention": _summary_for("awake_only", "mean_key_retention"),
        "mean_latency_ms": _summary_for("awake_only", "mean_latency_ms"),
        "anchor_accuracy": _summary_for("awake_only", "anchor_accuracy"),
        "sleep_cycles": round(_average([seed_row["awake_only"]["summary"]["sleep_cycles"] for seed_row in per_seed]), 4),
    }
    sleep_summary = {
        "final_disambiguation_recall": _summary_for("sleep_enabled", "final_disambiguation_recall"),
        "final_top1_recall": _summary_for("sleep_enabled", "final_top1_recall"),
        "final_strict_disambiguation_recall": _summary_for("sleep_enabled", "final_strict_disambiguation_recall"),
        "final_strict_top1_recall": _summary_for("sleep_enabled", "final_strict_top1_recall"),
        "final_anchor_hit_rate": _summary_for("sleep_enabled", "final_anchor_hit_rate"),
        "final_noise_retention": _summary_for("sleep_enabled", "final_noise_retention"),
        "final_key_retention": _summary_for("sleep_enabled", "final_key_retention"),
        "mean_disambiguation_recall": _summary_for("sleep_enabled", "mean_disambiguation_recall"),
        "mean_strict_disambiguation_recall": _summary_for("sleep_enabled", "mean_strict_disambiguation_recall"),
        "mean_strict_top1_recall": _summary_for("sleep_enabled", "mean_strict_top1_recall"),
        "mean_anchor_hit_rate": _summary_for("sleep_enabled", "mean_anchor_hit_rate"),
        "mean_noise_retention": _summary_for("sleep_enabled", "mean_noise_retention"),
        "mean_key_retention": _summary_for("sleep_enabled", "mean_key_retention"),
        "mean_latency_ms": _summary_for("sleep_enabled", "mean_latency_ms"),
        "anchor_accuracy": _summary_for("sleep_enabled", "anchor_accuracy"),
        "sleep_cycles": round(_average([seed_row["sleep_enabled"]["summary"]["sleep_cycles"] for seed_row in per_seed]), 4),
    }

    curve_summary = []
    strict_curve_summary = []
    for day_index in range(1, args.horizon_days + 1):
        awake_day = [seed_row["awake_only"]["curve"][day_index - 1] for seed_row in per_seed]
        sleep_day = [seed_row["sleep_enabled"]["curve"][day_index - 1] for seed_row in per_seed]
        curve_summary.append(
            {
                "day": day_index,
                "awake_only": {
                    "sleep_mode": "awake",
                    "disambiguation_recall": round(_average([row["disambiguation_recall"] for row in awake_day]), 4),
                    "noise_retention": round(_average([row["noise_retention"] for row in awake_day]), 4),
                    "anchor_hit_rate": round(_average([1.0 if row["anchor"]["top_hit"] else 0.0 for row in awake_day]), 4),
                },
                "sleep_enabled": {
                    "sleep_mode": _sleep_schedule(day_index, args.horizon_days),
                    "disambiguation_recall": round(_average([row["disambiguation_recall"] for row in sleep_day]), 4),
                    "noise_retention": round(_average([row["noise_retention"] for row in sleep_day]), 4),
                    "anchor_hit_rate": round(_average([1.0 if row["anchor"]["top_hit"] else 0.0 for row in sleep_day]), 4),
                },
            }
        )
        awake_strict_day = [seed_row["awake_only"]["strict_curve"][day_index - 1] for seed_row in per_seed]
        sleep_strict_day = [seed_row["sleep_enabled"]["strict_curve"][day_index - 1] for seed_row in per_seed]
        strict_curve_summary.append(
            {
                "day": day_index,
                "awake_only": {
                    "strict_top1_recall": round(_average([row["strict_top1_recall"] for row in awake_strict_day]), 4),
                    "strict_disambiguation_recall": round(
                        _average([row["strict_disambiguation_recall"] for row in awake_strict_day]),
                        4,
                    ),
                },
                "sleep_enabled": {
                    "strict_top1_recall": round(_average([row["strict_top1_recall"] for row in sleep_strict_day]), 4),
                    "strict_disambiguation_recall": round(
                        _average([row["strict_disambiguation_recall"] for row in sleep_strict_day]),
                        4,
                    ),
                },
            }
        )

    report = {
        "benchmark": "scm_long_horizon_memory",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "inputs": {
            "horizon_days": args.horizon_days,
            "daily_pair_count": args.daily_pair_count,
            "extra_noise_count": args.extra_noise_count,
            "review_count": args.review_count,
            "anchor_update_day": args.anchor_update_day,
            "deep_days": list(DEFAULT_DEEP_DAYS),
        },
        "modes": {
            "awake_only": awake_summary,
            "sleep_enabled": sleep_summary,
        },
        "daily_curve": curve_summary,
        "strict_curve": strict_curve_summary,
        "per_seed": per_seed,
    }

    report["comparative"] = {
        "disambiguation_lift": round(
            sleep_summary["final_disambiguation_recall"] - awake_summary["final_disambiguation_recall"],
            4,
        ),
        "strict_disambiguation_lift": round(
            sleep_summary["final_strict_disambiguation_recall"] - awake_summary["final_strict_disambiguation_recall"],
            4,
        ),
        "noise_reduction": round(
            awake_summary["final_noise_retention"] - sleep_summary["final_noise_retention"],
            4,
        ),
        "anchor_lift": round(
            sleep_summary["anchor_accuracy"] - awake_summary["anchor_accuracy"],
            4,
        ),
    }

    status = {
        "sleep_final_recall_pass": sleep_summary["final_disambiguation_recall"] >= 0.90,
        "sleep_final_noise_pass": sleep_summary["final_noise_retention"] <= 0.40,
        "sleep_anchor_pass": sleep_summary["anchor_accuracy"] >= 0.90,
        "sleep_curve_pass": sleep_summary["mean_disambiguation_recall"] >= 0.90,
    }
    status["overall_pass"] = all(status.values())
    report["status"] = status
    return report


def render_markdown(report: dict) -> str:
    awake = report["modes"]["awake_only"]
    sleep = report["modes"]["sleep_enabled"]
    comparative = report["comparative"]

    summary_rows = [
        [
            "Final stable-fact hit rate",
            _format_float(awake["final_anchor_hit_rate"]),
            _format_float(sleep["final_anchor_hit_rate"]),
            _format_float(sleep["final_anchor_hit_rate"] - awake["final_anchor_hit_rate"]),
        ],
        [
            "Final duplicate-family recall",
            _format_float(awake["final_disambiguation_recall"]),
            _format_float(sleep["final_disambiguation_recall"]),
            _format_float(sleep["final_disambiguation_recall"] - awake["final_disambiguation_recall"]),
        ],
        [
            "Final noise retention",
            _format_float(awake["final_noise_retention"]),
            _format_float(sleep["final_noise_retention"]),
            _format_float(comparative["noise_reduction"]),
        ],
        [
            "Mean stable-fact hit rate",
            _format_float(awake["mean_anchor_hit_rate"]),
            _format_float(sleep["mean_anchor_hit_rate"]),
            _format_float(sleep["mean_anchor_hit_rate"] - awake["mean_anchor_hit_rate"]),
        ],
        [
            "Mean duplicate-family recall",
            _format_float(awake["mean_disambiguation_recall"]),
            _format_float(sleep["mean_disambiguation_recall"]),
            _format_float(sleep["mean_disambiguation_recall"] - awake["mean_disambiguation_recall"]),
        ],
        [
            "Anchor update accuracy",
            _format_float(awake["anchor_accuracy"]),
            _format_float(sleep["anchor_accuracy"]),
            _format_float(comparative["anchor_lift"]),
        ],
        [
            "Mean latency (ms)",
            _format_float(awake["mean_latency_ms"]),
            _format_float(sleep["mean_latency_ms"]),
            _format_float(sleep["mean_latency_ms"] - awake["mean_latency_ms"]),
        ],
    ]

    curve_rows = []
    for day in report["daily_curve"]:
        curve_rows.append(
            [
                day["day"],
                day["awake_only"]["sleep_mode"],
                _format_float(day["awake_only"]["disambiguation_recall"]),
                _format_float(day["awake_only"]["noise_retention"]),
                day["sleep_enabled"]["sleep_mode"],
                _format_float(day["sleep_enabled"]["disambiguation_recall"]),
                _format_float(day["sleep_enabled"]["noise_retention"]),
            ]
        )

    strict_rows = []
    for day in report.get("strict_curve", []):
        strict_rows.append(
            [
                day["day"],
                _format_float(day["awake_only"]["strict_disambiguation_recall"]),
                _format_float(day["sleep_enabled"]["strict_disambiguation_recall"]),
            ]
        )

    notes = [
        "This benchmark simulates repeated day-by-day interference, then checks whether the original memory still wins at the end of the horizon.",
        "The sleep-enabled path is expected to keep the target family stronger while pruning low-value noise more aggressively than the awake-only control.",
        "A late correction is also injected so we can verify versioning survives a longer memory history.",
        "Strict day-1 duplicate-pair recall is retained as a stress signal in the JSON appendix; the family-aware signal is the release gate.",
    ]

    return "\n".join(
        [
            "# SCM Long-Horizon Memory",
            "",
        "A reproducible multi-day retention benchmark for SCM.",
        "",
        "## Summary",
        "",
        "_Stable-fact hit rate and family-aware duplicate recall are the main release gates; strict duplicate-pair disambiguation remains a stress signal in the JSON artifact._",
        "",
        _markdown_table(
            ["Metric", "Awake-only", "Sleep-enabled", "Lift"],
            summary_rows,
        ),
            "",
            "## Day-by-Day Curve",
            "",
            _markdown_table(
                [
                    "Day",
                    "Awake mode",
                    "Awake recall",
                    "Awake noise",
                    "Sleep mode",
                    "Sleep recall",
                    "Sleep noise",
                ],
                curve_rows,
            ),
            "",
            "## Duplicate Stress Signal",
            "",
            _markdown_table(
                [
                    "Day",
                    "Awake strict recall",
                    "Sleep strict recall",
                ],
                strict_rows,
            ),
            "",
            "## Interpretation",
            "",
            *[f"- {note}" for note in notes],
            "",
            "## Acceptance Gate",
            "",
            f"- Overall pass: {'yes' if report['status']['overall_pass'] else 'no'}",
            f"- Sleep final recall pass: {'yes' if report['status']['sleep_final_recall_pass'] else 'no'}",
            f"- Sleep final noise pass: {'yes' if report['status']['sleep_final_noise_pass'] else 'no'}",
            f"- Sleep anchor pass: {'yes' if report['status']['sleep_anchor_pass'] else 'no'}",
            f"- Sleep curve pass: {'yes' if report['status']['sleep_curve_pass'] else 'no'}",
        ]
    )


def _write_report(report: dict, output: Path, markdown_output: Path, write_history: bool, history_dir: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report) + "\n", encoding="utf-8")

    history_path = None
    if write_history:
        history_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        history_path = history_dir / f"long_horizon_{ts_safe}.json"
        history_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return history_path


def main():
    parser = argparse.ArgumentParser(description="Generate SCM long-horizon memory evidence")
    parser.add_argument(
        "--output",
        default="research/metrics/long_horizon_latest.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--markdown",
        default="docs/SCM_LONG_HORIZON.md",
        help="Output markdown report path",
    )
    parser.add_argument("--seed-start", type=int, default=8801)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--daily-pair-count", type=int, default=16)
    parser.add_argument("--extra-noise-count", type=int, default=8)
    parser.add_argument("--review-count", type=int, default=3)
    parser.add_argument("--anchor-update-day", type=int, default=DEFAULT_ANCHOR_UPDATE_DAY)
    parser.add_argument("--history-dir", default="research/metrics/history/long_horizon")
    parser.add_argument("--write-history", action="store_true", default=True)
    parser.add_argument("--no-history", action="store_false", dest="write_history")

    args = parser.parse_args()
    report = build_report(args)
    history_path = _write_report(
        report=report,
        output=Path(args.output),
        markdown_output=Path(args.markdown),
        write_history=args.write_history,
        history_dir=Path(args.history_dir),
    )

    print(json.dumps(report["status"], indent=2))
    print(f"JSON report: {Path(args.output).resolve()}")
    print(f"Markdown report: {Path(args.markdown).resolve()}")
    if history_path:
        print(f"History report: {history_path.resolve()}")


if __name__ == "__main__":
    main()
