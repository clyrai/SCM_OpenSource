"""
LOCOMO++ benchmark runner.

Evaluates four memory architectures on the perturbed LOCOMO conversations
produced by `locomo_plus_perturbations.py`. All systems are offline (no LLM
spend), so this benchmark can be re-run unlimited times.

Systems:
  1. lexical    — BM25-style token overlap over raw turns
  2. vector     — cosine similarity over MiniLM-L6-v2 embeddings of raw turns
                   (architectural equivalent of Mem0's retrieval; minus its
                    LLM-rewrite step)
  3. scm_phase4 — direct LongTermMemory + SleepCycleOrchestrator
                   (heuristic concept extraction)
  4. scm_hme    — full ChatEngine with HME_ENABLED=True
                   (Phase 1-5 stack, heuristic concept extraction)

Metrics (each tied to an architectural property):
  - original_recall_clean        baseline LOCOMO score on UN-perturbed conv
  - original_recall_perturbed    LOCOMO score after noise + contradictions
  - contradiction_current        % CURRENT-value questions answered correctly
  - contradiction_old            % OLD-value questions answered correctly
                                 (penalises systems that overwrite vs. version)
  - entity_disambig              % entity-disambig questions answered correctly
  - noise_rejection              1 - rate of noise-substring leakage in answers

Output: research/metrics/locomo_plus_<seed>_<conv>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.encoder import HashEmbeddingModel, HybridEncoder, MeaningEncoder
from src.core.long_term_memory import LongTermMemory
from src.core.models import Concept, ConceptType, Episode, ImportanceVector
from src.core.value_tagger import ValueTagger
from src.core.working_memory import WorkingMemory
from src.core.time_utils import utc_now
from src.sleep.sleep_cycle import SleepCycleOrchestrator

from tests.locomo_plus_perturbations import (
    ANCHOR_FACTS,
    LocomoPlusPerturber,
    PerturbationConfig,
)


# ─── Stub LLM so SCM/MeaningEncoder fall back to heuristic mode ─────────────


class _NoLLM:
    """Disables LLM calls so MeaningEncoder uses regex heuristics only."""

    def extract_concepts(self, text):
        return []

    def _chat(self, *args, **kwargs):
        return ""


# ─── Embedding model (shared by vector + SCM systems for fair comparison) ──


def _make_embedder():
    """Try MiniLM, fall back to deterministic hash. Same model SCM uses."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=False)
    except Exception:
        return HashEmbeddingModel(384)


_EMBEDDER = None


def _embed(text: str) -> np.ndarray:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _make_embedder()
    out = _EMBEDDER.encode(text)
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 0:
        return np.zeros(384, dtype=np.float32)
    return arr


# ─── Memory representation: (text, embedding) tuples ────────────────────────


def _flatten_turns(conv: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of (speaker, text) in chronological order."""
    conv_data = conv["conversation"]
    session_keys = sorted(
        [k for k in conv_data.keys()
         if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    out: List[Tuple[str, str]] = []
    for sk in session_keys:
        for turn in conv_data[sk]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if text.strip():
                out.append((speaker, text))
    return out


# ─── Lexical retrieval system ──────────────────────────────────────────────


class LexicalSystem:
    """Token-overlap (BM25-lite) retrieval over raw turns."""

    name = "lexical"

    def __init__(self):
        self.docs: List[Tuple[str, str]] = []  # (speaker, text)

    def ingest(self, conv: Dict[str, Any]) -> Dict[str, int]:
        self.docs = _flatten_turns(conv)
        return {"docs": len(self.docs)}

    def answer(self, question: str, top_k: int = 5) -> str:
        if not self.docs:
            return ""
        q_tokens = set(re.findall(r"\b\w{2,}\b", question.lower()))
        if not q_tokens:
            return ""
        scored: List[Tuple[float, str]] = []
        for speaker, text in self.docs:
            d_tokens = set(re.findall(r"\b\w{2,}\b", text.lower()))
            if not d_tokens:
                continue
            overlap = len(q_tokens & d_tokens)
            if overlap == 0:
                continue
            # Simple TF-IDF lite — recency tiebreaker via doc index baked in
            score = overlap / (len(d_tokens) ** 0.5)
            scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return " | ".join(t for _, t in scored[:top_k])


# ─── Vector retrieval system (Mem0-architectural equivalent) ───────────────


class VectorSystem:
    """Cosine retrieval over MiniLM embeddings of raw turns."""

    name = "vector"

    def __init__(self):
        self.docs: List[Tuple[str, str]] = []
        self.embeddings: np.ndarray = np.zeros((0, 384), dtype=np.float32)

    def ingest(self, conv: Dict[str, Any]) -> Dict[str, int]:
        self.docs = _flatten_turns(conv)
        if not self.docs:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            return {"docs": 0}
        self.embeddings = np.stack([_embed(t) for _, t in self.docs])
        return {"docs": len(self.docs)}

    def answer(self, question: str, top_k: int = 5) -> str:
        if not self.docs:
            return ""
        q = _embed(question)
        if np.linalg.norm(q) == 0:
            return ""
        q = q / (np.linalg.norm(q) + 1e-9)
        norms = np.linalg.norm(self.embeddings, axis=1) + 1e-9
        sims = self.embeddings @ q / norms
        top_idx = np.argsort(-sims)[:top_k]
        return " | ".join(self.docs[i][1] for i in top_idx)


# ─── SCM Phase-4-only system (LongTermMemory + sleep, no HME) ──────────────


class SCMPhase4System:
    """Direct LongTermMemory + sleep cycles with heuristic concept extraction."""

    name = "scm_phase4"

    def __init__(self, sleep_between_sessions: bool = True):
        self.sleep_between_sessions = sleep_between_sessions
        self.ltm: LongTermMemory = None
        self.encoder: MeaningEncoder = None

    def ingest(self, conv: Dict[str, Any]) -> Dict[str, int]:
        self.ltm = LongTermMemory(persist=False)
        self.encoder = MeaningEncoder(llm=None)  # heuristic regex
        tagger = ValueTagger()
        wm = WorkingMemory(capacity=7)
        sleep_orch = SleepCycleOrchestrator()

        conv_data = conv["conversation"]
        session_keys = sorted(
            [k for k in conv_data.keys()
             if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )

        stats = {"turns": 0, "concepts": 0, "sleep_cycles": 0}

        for sk in session_keys:
            for turn in conv_data[sk]:
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                if not text.strip():
                    continue
                stats["turns"] += 1
                full_text = f"{speaker}: {text}"
                concepts = self.encoder.extract(full_text)
                for c in concepts:
                    c.importance = tagger.tag(c)
                    if not c.embedding:
                        c.embedding = _embed(c.description).tolist()
                    self.ltm.add_concept(c)
                if concepts:
                    ep = Episode(
                        concept_ids=[c.id for c in concepts],
                        raw_content=full_text,
                        importance=concepts[0].importance,
                        source=speaker.lower(),
                    )
                    wm.store(ep)
                    stats["concepts"] += len(concepts)

            if self.sleep_between_sessions and wm.size() >= 3:
                concepts_list = self.ltm.get_all_concepts(include_suppressed=False)
                try:
                    success, cycle, cycle_stats = sleep_orch.begin_sleep_cycle(
                        concepts=concepts_list,
                        relations=[],
                        episodes=wm.get_all(),
                        force=True,
                    )
                    if success:
                        stats["sleep_cycles"] += 1
                        if "updated_concepts" in cycle_stats:
                            updated = cycle_stats["updated_concepts"]
                            surviving = {c.id for c in updated}
                            # Sync paraphrased / consolidated concepts back to LTM
                            # so retrieval sees the post-sleep representations.
                            for c in updated:
                                self.ltm._concept_cache[c.id] = c
                                if c.id in self.ltm.graph:
                                    self.ltm.graph.nodes[c.id].update(
                                        c.model_dump(exclude={"id"})
                                    )
                            forgotten = set(self.ltm._concept_cache.keys()) - surviving
                            for fid in forgotten:
                                if fid in self.ltm._concept_cache:
                                    del self.ltm._concept_cache[fid]
                                if fid in self.ltm.graph:
                                    self.ltm.graph.remove_node(fid)
                        wm.clear()
                except Exception:
                    pass

        return stats

    def answer(self, question: str, top_k: int = 5) -> str:
        if self.ltm is None:
            return ""
        all_concepts = list(self.ltm._concept_cache.values())
        if not all_concepts:
            return ""
        q_emb = _embed(question)
        q_norm = np.linalg.norm(q_emb) + 1e-9
        q_tokens = set(re.findall(r"\b\w{2,}\b", question.lower()))

        scored: List[Tuple[float, str]] = []
        for c in all_concepts:
            if c.embedding:
                c_emb = np.array(c.embedding, dtype=np.float32)
                c_norm = np.linalg.norm(c_emb) + 1e-9
                sim = float(np.dot(c_emb, q_emb) / (c_norm * q_norm))
            else:
                sim = 0.0
            d_tokens = set(re.findall(r"\b\w{2,}\b", c.description.lower()))
            kw_overlap = (len(q_tokens & d_tokens) / len(q_tokens)) if q_tokens else 0.0
            importance = c.importance.overall if c.importance else 0.5
            score = 0.5 * sim + 0.3 * kw_overlap + 0.2 * importance
            scored.append((score, c.description))
        scored.sort(key=lambda x: x[0], reverse=True)
        return " | ".join(t for _, t in scored[:top_k])


# ─── SCM HME-full system (full ChatEngine with HME pipeline) ──────────────


class SCMHMESystem:
    """Full ChatEngine with HME_ENABLED=True (heuristic encoder for cost reasons)."""

    name = "scm_hme"

    def __init__(self):
        self.engine = None

    def ingest(self, conv: Dict[str, Any]) -> Dict[str, int]:
        from src.chat import engine as chat_engine_module
        from src.chat.engine import ChatEngine
        chat_engine_module.HME_ENABLED = True
        self.engine = ChatEngine(
            llm=_NoLLM(),
            encoder=MeaningEncoder(llm=None),
            session_id=f"locomo_plus_{uuid.uuid4().hex[:8]}",
            profile="research",
            sandbox_mode=True,
            enable_persistence=False,
            enable_auto_sleep=True,
            sleep_check_interval=4,
        )

        conv_data = conv["conversation"]
        session_keys = sorted(
            [k for k in conv_data.keys()
             if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )
        stats = {"turns": 0, "auto_sleeps": 0, "errors": 0}
        for sk in session_keys:
            for turn in conv_data[sk]:
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                if not text.strip():
                    continue
                stats["turns"] += 1
                tagged = f"{speaker} says: {text}"
                try:
                    self.engine._extract_and_store(tagged, source="user")
                    self.engine._message_count += 1
                    self.engine._turns_since_micro_sleep += 1
                    if (
                        self.engine.enable_auto_sleep
                        and self.engine._message_count % self.engine.sleep_check_interval == 0
                    ):
                        if self.engine._check_and_trigger_sleep() is not None:
                            stats["auto_sleeps"] += 1
                except Exception:
                    stats["errors"] += 1

        # Final deep-sleep flush
        try:
            self.engine.force_sleep(mode="deep")
            stats["auto_sleeps"] += 1
        except Exception:
            pass
        return stats

    def answer(self, question: str, top_k: int = 5) -> str:
        if self.engine is None or self.engine._spreading_activation is None:
            return ""
        try:
            activated, _ = self.engine._spreading_activation.retrieve(
                query=question,
                context_tags={"session_id": self.engine.session_id},
            )
        except Exception:
            return ""
        if not activated:
            return ""
        if self.engine._hypothesis_ranker is not None:
            try:
                activation_map = {c.id: 1.0 - i * 0.02 for i, c in enumerate(activated)}
                ranked = self.engine._hypothesis_ranker.rank(
                    activated_concepts=activated,
                    activation_map=activation_map,
                    context_tags={"session_id": self.engine.session_id},
                )
                if ranked.hypotheses:
                    return " | ".join(h.concept.description for h in ranked.hypotheses[:top_k])
            except Exception:
                pass
        return " | ".join(c.description for c in activated[:top_k])


SYSTEMS = [LexicalSystem, VectorSystem, SCMPhase4System, SCMHMESystem]


# ─── Scoring functions ─────────────────────────────────────────────────────


def _token_f1(predicted: str, ground_truth: str) -> Tuple[float, float]:
    pred_tokens = set(re.findall(r"\b\w{2,}\b", predicted.lower()))
    gt_tokens = set(re.findall(r"\b\w{2,}\b", ground_truth.lower()))
    if not gt_tokens:
        return 0.0, 0.0
    overlap = pred_tokens & gt_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, recall


def _score_locomo_qa(predicted, ground_truth, category: int) -> float:
    """Same scoring as locomo_benchmark.py for fair comparison."""
    ground_truth = "" if ground_truth is None else str(ground_truth)
    predicted = "" if predicted is None else str(predicted)
    if not predicted or predicted.lower().strip() in (
        "i don't know", "unknown", "not sure", "n/a", "",
    ):
        return 0.0
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()
    if pred_lower == gt_lower:
        return 1.0
    f1, recall = _token_f1(predicted, ground_truth)
    recall_threshold = {1: 0.5, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.5}.get(category, 0.5)
    cat_multiplier = {2: 0.8, 3: 0.9}.get(category, 1.0)
    if recall >= recall_threshold:
        return 1.0
    return f1 * cat_multiplier


def _score_contradiction_current(predicted: str, expected_new: str, incorrect_old: str) -> float:
    """1.0 if NEW is mentioned and OLD is NOT. 0.0 if OLD is mentioned. 0.5 partial."""
    p = predicted.lower()
    new_hit = expected_new.lower() in p
    old_hit = incorrect_old.lower() in p
    if new_hit and not old_hit:
        return 1.0
    if old_hit and not new_hit:
        return 0.0  # actively wrong
    if new_hit and old_hit:
        return 0.5  # leaked old fact (versioning failure)
    return 0.0  # didn't surface anything


def _score_contradiction_old(predicted: str, expected_old: str, incorrect_new: str) -> float:
    """Symmetrical: 1.0 if OLD mentioned and NEW absent."""
    p = predicted.lower()
    old_hit = expected_old.lower() in p
    new_hit = incorrect_new.lower() in p
    if old_hit and not new_hit:
        return 1.0
    if new_hit and not old_hit:
        return 0.0
    if old_hit and new_hit:
        return 0.5
    return 0.0


def _score_entity_disambig(predicted: str, expected: str, others: List[str]) -> float:
    """
    1.0 if expected name appears AND no other distractor names appear.
    0.5 if expected appears alongside at least one other (partial — system
        retrieved relevant turn but didn't disambiguate).
    0.0 if expected doesn't appear.
    """
    p = predicted.lower()
    expected_hit = expected.lower() in p
    if not expected_hit:
        return 0.0
    other_hits = sum(1 for o in others if o.lower() in p)
    if other_hits == 0:
        return 1.0
    return 0.5


def _score_noise_rejection(predicted: str, noise_substring: str) -> float:
    """1.0 if noise substring is NOT in predicted (good = system filtered it)."""
    return 0.0 if noise_substring.lower() in predicted.lower() else 1.0


# ─── Run a single (system, conversation, seed) configuration ──────────────


def run_config(
    system_cls,
    conv_clean: Dict[str, Any],
    conv_perturbed: Dict[str, Any],
    augmented_qa: List[Dict[str, Any]],
    perturbation_stats,
) -> Dict[str, Any]:
    sys_inst = system_cls()
    metrics = {
        "system": sys_inst.name,
        "ingest": {},
        "scores": {},
        "ingest_time_s": 0.0,
        "qa_time_s": 0.0,
    }

    # Phase A: clean conversation, original QA only
    t0 = time.time()
    sys_inst_clean = system_cls()
    metrics["ingest"]["clean"] = sys_inst_clean.ingest(conv_clean)
    metrics["ingest_time_s"] += time.time() - t0

    t0 = time.time()
    clean_scores = []
    for qa in conv_clean.get("qa", []):
        if qa.get("category") == 5:
            continue
        predicted = sys_inst_clean.answer(qa["question"])
        clean_scores.append(_score_locomo_qa(predicted, qa.get("answer", ""), qa.get("category", 0)))
    metrics["qa_time_s"] += time.time() - t0
    metrics["scores"]["original_recall_clean"] = (
        sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
    )
    metrics["scores"]["original_n_clean"] = len(clean_scores)

    # Phase B: perturbed conversation
    t0 = time.time()
    sys_inst_pert = system_cls()
    metrics["ingest"]["perturbed"] = sys_inst_pert.ingest(conv_perturbed)
    metrics["ingest_time_s"] += time.time() - t0

    t0 = time.time()

    # Original LOCOMO QA on perturbed conversation
    pert_scores = []
    for qa in conv_perturbed.get("qa", []):
        if qa.get("category") == 5:
            continue
        predicted = sys_inst_pert.answer(qa["question"])
        pert_scores.append(_score_locomo_qa(predicted, qa.get("answer", ""), qa.get("category", 0)))
    metrics["scores"]["original_recall_perturbed"] = (
        sum(pert_scores) / len(pert_scores) if pert_scores else 0.0
    )

    # Augmented QA
    cur_scores, old_scores = [], []
    disambig_scores = []
    noise_scores = []
    for qa in augmented_qa:
        predicted = sys_inst_pert.answer(qa["question"])
        qtype = qa.get("_qtype")
        if qtype == "contradiction_current":
            cur_scores.append(_score_contradiction_current(
                predicted,
                qa["evidence"]["expected"],
                qa["evidence"]["incorrect_value"],
            ))
        elif qtype == "contradiction_old":
            old_scores.append(_score_contradiction_old(
                predicted,
                qa["evidence"]["expected"],
                qa["evidence"]["incorrect_value"],
            ))
        elif qtype == "entity_disambig_distractor":
            disambig_scores.append(_score_entity_disambig(
                predicted,
                qa["evidence"]["expected_entity"],
                qa["evidence"]["other_entities"],
            ))
        elif qtype == "noise_reject":
            noise_scores.append(_score_noise_rejection(predicted, qa["_negative_substring"]))
    metrics["qa_time_s"] += time.time() - t0

    metrics["scores"]["contradiction_current"] = (
        sum(cur_scores) / len(cur_scores) if cur_scores else 0.0
    )
    metrics["scores"]["contradiction_old"] = (
        sum(old_scores) / len(old_scores) if old_scores else 0.0
    )
    metrics["scores"]["entity_disambig"] = (
        sum(disambig_scores) / len(disambig_scores) if disambig_scores else 0.0
    )
    metrics["scores"]["noise_rejection"] = (
        sum(noise_scores) / len(noise_scores) if noise_scores else 0.0
    )
    metrics["scores"]["counts"] = {
        "n_contradiction_current": len(cur_scores),
        "n_contradiction_old": len(old_scores),
        "n_entity_disambig": len(disambig_scores),
        "n_noise_reject": len(noise_scores),
    }
    return metrics


# ─── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LOCOMO++ benchmark")
    parser.add_argument("--data", default="data/locomo/locomo10.json")
    parser.add_argument("--max-conv", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    parser.add_argument("--noise-density", type=float, default=0.30)
    parser.add_argument("--contradictions", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=3)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["lexical", "vector", "scm_phase4", "scm_hme"],
        choices=["lexical", "vector", "scm_phase4", "scm_hme"],
    )
    parser.add_argument("--output-name", default="locomo_plus_latest.json")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, args.data)
    out_path = os.path.join(repo_root, "research", "metrics", args.output_name)

    with open(data_path) as f:
        all_convs = json.load(f)
    convs = all_convs[: args.max_conv]

    sys_map = {s.name: s for s in SYSTEMS}
    chosen_systems = [sys_map[n] for n in args.systems]

    print(f"\n{'#'*60}")
    print(f"# LOCOMO++ BENCHMARK")
    print(f"# Conversations: {len(convs)}  Seeds: {args.seeds}")
    print(f"# Systems: {[s.name for s in chosen_systems]}")
    print(f"{'#'*60}")

    grand_start = time.time()
    all_results: List[Dict[str, Any]] = []
    for conv_idx, conv in enumerate(convs):
        speakers = (
            f"{conv['conversation'].get('speaker_a','?')}"
            f"/{conv['conversation'].get('speaker_b','?')}"
        )
        for seed in args.seeds:
            print(f"\n=== Conv {conv_idx} ({speakers}) seed={seed} ===")
            cfg = PerturbationConfig(
                seed=seed,
                noise_density=args.noise_density,
                contradiction_count=args.contradictions,
                distractor_entity_count=args.distractors,
            )
            perturber = LocomoPlusPerturber(cfg)
            conv_perturbed, augmented_qa, pert_stats = perturber.perturb(conv)

            print(
                f"  perturbations: noise={pert_stats.noise_turns_added} "
                f"contra={pert_stats.contradiction_turns_added} "
                f"distract={pert_stats.distractor_turns_added} "
                f"aug_qa={len(augmented_qa)}"
            )

            for system_cls in chosen_systems:
                t0 = time.time()
                m = run_config(system_cls, conv, conv_perturbed, augmented_qa, pert_stats)
                m["conv_idx"] = conv_idx
                m["seed"] = seed
                m["perturbations"] = {
                    "noise_turns_added": pert_stats.noise_turns_added,
                    "contradiction_turns_added": pert_stats.contradiction_turns_added,
                    "distractor_turns_added": pert_stats.distractor_turns_added,
                }
                elapsed = time.time() - t0
                print(
                    f"  [{m['system']:<10}] "
                    f"clean={m['scores']['original_recall_clean']:.3f}  "
                    f"perturbed={m['scores']['original_recall_perturbed']:.3f}  "
                    f"cur={m['scores']['contradiction_current']:.3f}  "
                    f"old={m['scores']['contradiction_old']:.3f}  "
                    f"disambig={m['scores']['entity_disambig']:.3f}  "
                    f"noise_rej={m['scores']['noise_rejection']:.3f}  "
                    f"({elapsed:.1f}s)"
                )
                all_results.append(m)

    # Aggregate by system
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        for metric_name, value in r["scores"].items():
            if isinstance(value, (int, float)) and metric_name != "original_n_clean":
                agg[r["system"]][metric_name].append(float(value))

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sys_name, metric_lists in agg.items():
        summary[sys_name] = {}
        for metric, values in metric_lists.items():
            summary[sys_name][metric] = {
                "mean": round(sum(values) / len(values), 4) if values else 0.0,
                "std": round(_std(values), 4),
                "n": len(values),
            }

    report = {
        "timestamp": utc_now().isoformat(),
        "benchmark": "LOCOMO++ (workload sensitivity)",
        "config": vars(args),
        "n_convs": len(convs),
        "n_seeds": len(args.seeds),
        "total_runs": len(all_results),
        "total_time_s": round(time.time() - grand_start, 1),
        "summary_by_system": summary,
        "per_run": all_results,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'#'*60}\n# SUMMARY\n{'#'*60}")
    print(
        f"{'System':<14}{'clean':>8}{'perturbed':>11}"
        f"{'cur':>8}{'old':>8}{'disambig':>10}{'noise_rej':>11}"
    )
    for sys_name in args.systems:
        s = summary.get(sys_name, {})
        print(
            f"{sys_name:<14}"
            f"{s.get('original_recall_clean',{}).get('mean',0):>8.3f}"
            f"{s.get('original_recall_perturbed',{}).get('mean',0):>11.3f}"
            f"{s.get('contradiction_current',{}).get('mean',0):>8.3f}"
            f"{s.get('contradiction_old',{}).get('mean',0):>8.3f}"
            f"{s.get('entity_disambig',{}).get('mean',0):>10.3f}"
            f"{s.get('noise_rejection',{}).get('mean',0):>11.3f}"
        )

    print(f"\nSaved to: {out_path}")
    print(f"Total runtime: {report['total_time_s']}s")


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


if __name__ == "__main__":
    main()
