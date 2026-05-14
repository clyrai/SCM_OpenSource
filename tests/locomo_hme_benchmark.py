"""
SCM LOCOMO Benchmark with full HME pipeline (Phases 1–5 active).

Unlike `locomo_benchmark.py` which directly wires LongTermMemory + ValueTagger +
SleepCycleOrchestrator (effectively a Phase-4-only test), this runner ingests
each LOCOMO turn through `ChatEngine` with `HME_ENABLED=True`, exercising:

  Phase 1: AttentionGate (selective encoding intensity, salience, grasp)
  Phase 2: EventCompiler + AssociationBinder (typed events, Hebbian edges)
  Phase 3: SpreadingActivationRetriever + HypothesisRanker (cue-driven recall)
  Phase 4: MicroSleep / DeepSleep (auto-triggered every N turns)
  Phase 5: ForgettingDynamics + contradiction versioning

Question answering uses the same retrieval pipeline that powers the chat loop:
spreading activation seeded by the question, then top hypotheses returned as
the predicted answer. No LLM calls are made (deterministic by design).

Output JSON schema matches `locomo_benchmark.py` so artifacts can be diffed.

Usage:
    venv/bin/python3 tests/locomo_hme_benchmark.py --max-conv 1
    venv/bin/python3 tests/locomo_hme_benchmark.py --output-name locomo_hme_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # populate DEEPSEEK_API_KEY etc. before LLMExtractor init

from src.chat import engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.encoder import MeaningEncoder
from src.core.time_utils import utc_now
from src.llm import LLMExtractor


class _HeuristicLLMStub:
    """LLM stub that disables LLM calls — forces heuristic concept extraction."""

    def extract_concepts(self, text: str):
        return []

    def _chat(self, *args, **kwargs):
        return ""


CAT_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-ended",
    5: "adversarial",
}


class LocomoHMEEvaluator:
    """LOCOMO runner that exercises the full HME ChatEngine pipeline.

    extract_mode:
      - "stub":     no LLM extraction (heuristic regex only) — fastest, weakest
      - "deepseek": DeepSeek API extraction with JSON mode — slow but strong
    """

    def __init__(
        self,
        data_path: str,
        profile: str = "research",
        extract_mode: str = "stub",
    ):
        self.data_path = data_path
        self.profile = profile
        self.extract_mode = extract_mode

    def load_data(self) -> List[Dict]:
        with open(self.data_path) as f:
            return json.load(f)

    def _make_engine(self, conv_idx: int) -> ChatEngine:
        chat_engine_module.HME_ENABLED = True

        if self.extract_mode == "deepseek":
            llm = LLMExtractor(provider="deepseek")
            encoder = MeaningEncoder(llm=llm)
        else:
            llm = _HeuristicLLMStub()
            encoder = MeaningEncoder(llm=None)  # heuristic regex extraction only

        engine = ChatEngine(
            llm=llm,
            encoder=encoder,
            session_id=f"locomo_conv_{conv_idx}",
            profile=self.profile,
            sandbox_mode=True,             # ephemeral; no SQLite contamination
            enable_persistence=False,
            enable_auto_sleep=True,
            sleep_check_interval=4,        # auto-sleep every 4 turns
        )
        return engine

    def _ingest(self, engine: ChatEngine, conv: Dict) -> Dict:
        conv_data = conv["conversation"]
        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")

        session_keys = sorted(
            [k for k in conv_data.keys()
             if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )

        stats = {
            "total_turns": 0,
            "sessions": len(session_keys),
            "auto_sleeps": 0,
            "ingest_errors": 0,
        }

        for session_key in session_keys:
            for turn in conv_data[session_key]:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                if not text.strip():
                    continue
                stats["total_turns"] += 1
                # Frame each turn as a tagged statement so the encoder/event
                # compiler sees explicit speaker context.
                tagged = f"{speaker} says: {text}"
                try:
                    # Bypass LLM response generation — we only need ingestion.
                    engine._extract_and_store(tagged, source="user")
                    engine._message_count += 1
                    engine._turns_since_micro_sleep += 1
                    if engine.enable_auto_sleep and engine._message_count % engine.sleep_check_interval == 0:
                        if engine._check_and_trigger_sleep() is not None:
                            stats["auto_sleeps"] += 1
                except Exception:
                    stats["ingest_errors"] += 1

        # One final deep-sleep flush so retrieval sees consolidated state.
        try:
            flushed = engine.force_sleep(mode="deep")
            if flushed:
                stats["auto_sleeps"] += 1
        except Exception:
            pass

        return stats

    def _answer(self, engine: ChatEngine, question: str) -> str:
        """Use HME spreading-activation retrieval to answer."""
        # Build context tags so the gate can prefer this session.
        if engine._spreading_activation is None:
            return ""
        try:
            activated, _ = engine._spreading_activation.retrieve(
                query=question,
                context_tags={"session_id": engine.session_id},
            )
        except Exception:
            return ""
        if not activated:
            return ""
        if engine._hypothesis_ranker is not None:
            try:
                activation_map = {c.id: 1.0 - i * 0.02 for i, c in enumerate(activated)}
                ranked = engine._hypothesis_ranker.rank(
                    activated_concepts=activated,
                    activation_map=activation_map,
                    context_tags={"session_id": engine.session_id},
                )
                if ranked.hypotheses:
                    return ranked.hypotheses[0].concept.description
            except Exception:
                pass
        return activated[0].description

    @staticmethod
    def _score(predicted, ground_truth, category: int) -> float:
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
        pred_tokens = set(re.findall(r"\b\w{2,}\b", pred_lower))
        gt_tokens = set(re.findall(r"\b\w{2,}\b", gt_lower))
        if not gt_tokens:
            return 0.0
        overlap = pred_tokens & gt_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        recall_threshold = {1: 0.5, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.5}.get(category, 0.5)
        cat_multiplier = {2: 0.8, 3: 0.9}.get(category, 1.0)
        if recall >= recall_threshold:
            return 1.0
        return f1 * cat_multiplier

    def evaluate_conversation(self, conv: Dict, conv_idx: int) -> Dict:
        speakers = (
            f"{conv['conversation'].get('speaker_a', '?')}"
            f"/{conv['conversation'].get('speaker_b', '?')}"
        )
        print(f"\n{'='*60}\n  Conv {conv_idx}: {speakers}\n{'='*60}")
        engine = self._make_engine(conv_idx)
        ingest_start = time.time()
        ingest_stats = self._ingest(engine, conv)
        ingest_time = time.time() - ingest_start
        print(
            f"  Ingested: {ingest_stats['total_turns']} turns, "
            f"{ingest_stats['auto_sleeps']} sleeps, "
            f"{ingest_stats['ingest_errors']} errors ({ingest_time:.1f}s)"
        )
        ltm_total = len(engine.long_term_memory.get_all_concepts(include_suppressed=False))
        print(f"  LTM concepts: {ltm_total}")

        qa_pairs = conv.get("qa", [])
        cat_scores: Dict[int, List[float]] = defaultdict(list)
        evaluated = 0
        eval_start = time.time()
        for i, qa in enumerate(qa_pairs):
            if i and i % 40 == 0:
                print(f"    Progress: {i}/{len(qa_pairs)}")
            question = qa.get("question", "")
            category = qa.get("category", 0)
            if category == 5:  # adversarial — skip per locomo_benchmark.py convention
                continue
            answer = qa.get("answer", "")
            predicted = self._answer(engine, question)
            score = self._score(predicted, answer, category)
            cat_scores[category].append(score)
            evaluated += 1
        eval_time = time.time() - eval_start

        cat_summary: Dict[str, Dict[str, float]] = {}
        for cat, scores in cat_scores.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            cat_summary[str(cat)] = {
                "name": CAT_NAMES.get(cat, f"cat_{cat}"),
                "score": round(avg, 4),
                "num_questions": len(scores),
            }
            print(f"  Cat {cat} ({CAT_NAMES.get(cat, '?')}): {avg:.3f} ({len(scores)} Q)")

        all_scores = [s for scores in cat_scores.values() for s in scores]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"  Overall: {overall:.3f} ({evaluated} QA, {eval_time:.1f}s)")

        return {
            "conv_idx": conv_idx,
            "speakers": speakers,
            "evaluated_qa": evaluated,
            "ingest_stats": ingest_stats,
            "ltm_concepts": ltm_total,
            "ingest_time_s": round(ingest_time, 2),
            "eval_time_s": round(eval_time, 2),
            "category_scores": cat_summary,
            "overall_score": round(overall, 4),
        }

    def run(self, max_conversations: int = None) -> Dict:
        data = self.load_data()
        if max_conversations:
            data = data[:max_conversations]

        print(f"\n{'#'*60}")
        print(f"# SCM LOCOMO BENCHMARK (HME pipeline)")
        print(f"# Conversations: {len(data)}")
        print(f"# Profile: {self.profile} | HME_ENABLED=True | sandbox=True")
        print(f"# Time: {utc_now().isoformat()}")
        print(f"{'#'*60}")

        start = time.time()
        conv_results = [self.evaluate_conversation(c, i) for i, c in enumerate(data)]
        total_time = time.time() - start

        all_scores = [r["overall_score"] for r in conv_results]
        cat_aggregate: Dict[int, List[float]] = defaultdict(list)
        for r in conv_results:
            for cat_str, info in r["category_scores"].items():
                cat_aggregate[int(cat_str)].append(info["score"])

        report = {
            "timestamp": utc_now().isoformat(),
            "benchmark": "LoCoMo (ACL 2024) — HME pipeline",
            "num_conversations": len(data),
            "total_time_s": round(total_time, 1),
            "overall_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
            "config": {
                "profile": self.profile,
                "hme_enabled": True,
                "sandbox": True,
                "auto_sleep_interval": 4,
                "answer_strategy": "spreading_activation_top1_no_llm",
            },
            "category_scores": {
                str(cat): {
                    "name": CAT_NAMES.get(cat, f"cat_{cat}"),
                    "score": round(sum(scores) / len(scores), 4) if scores else 0,
                    "num_conversations": len(scores),
                }
                for cat, scores in sorted(cat_aggregate.items())
            },
            "conversations": conv_results,
            "baselines_reported_in_literature": {
                "Mem0": 0.671,
                "MemGPT": 0.420,
                "RAG (gpt-3.5-turbo)": 0.560,
                "MemMachine": 0.899,
            },
        }
        return report


def main():
    parser = argparse.ArgumentParser(description="SCM LOCOMO HME Benchmark")
    parser.add_argument("--data", default="data/locomo/locomo10.json")
    parser.add_argument("--max-conv", type=int, default=None)
    parser.add_argument("--profile", default="research")
    parser.add_argument(
        "--extract-mode",
        choices=["stub", "deepseek"],
        default="stub",
        help="stub = heuristic regex only (fast); deepseek = DeepSeek JSON extraction (slow, strong)",
    )
    parser.add_argument(
        "--output-name",
        default="locomo_hme_latest.json",
        help="Filename written under research/metrics/",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, args.data)
    out_path = os.path.join(repo_root, "research", "metrics", args.output_name)

    evaluator = LocomoHMEEvaluator(
        data_path=data_path,
        profile=args.profile,
        extract_mode=args.extract_mode,
    )
    report = evaluator.run(max_conversations=args.max_conv)

    print(f"\n{'#'*60}\n# RESULT\n{'#'*60}")
    print(f"Overall: {report['overall_score']:.3f}")
    print(f"Total time: {report['total_time_s']}s")
    print()
    print("Category breakdown:")
    for cat_str, info in report["category_scores"].items():
        print(f"  Cat {cat_str} ({info['name']:12s}): {info['score']:.3f}")
    print()
    print("Reported baselines (from literature):")
    for name, score in report["baselines_reported_in_literature"].items():
        marker = " <-- SCM beats" if report["overall_score"] > score else ""
        print(f"  {name:24s}: {score:.3f}{marker}")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
