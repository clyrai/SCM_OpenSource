"""
LOCOMO benchmark using Mem0 as the memory layer.

Provides an apples-to-apples comparison against `tests/locomo_benchmark.py` and
`tests/locomo_hme_benchmark.py`: same conversations, same QA pairs, same scoring
function — only the memory architecture differs.

Backend: DeepSeek (OpenAI-compatible) for both Mem0's LLM and SCM, so the only
variable is the memory architecture itself.

Usage:
    venv/bin/python3 tests/locomo_mem0_baseline.py --max-conv 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # pull DEEPSEEK_API_KEY etc. from .env before mem0 init

from src.core.time_utils import utc_now


CAT_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-ended",
    5: "adversarial",
}


def make_mem0_client():
    """Build a Mem0 instance configured for DeepSeek + 384-dim MiniLM."""
    from mem0 import Memory

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing — set it in .env")

    config = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "temperature": 0.1,
                "max_tokens": 512,
                "api_key": api_key,
                "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"scm_locomo_{uuid.uuid4().hex[:8]}",
                "embedding_model_dims": 384,
                "on_disk": False,
            },
        },
    }
    return Memory.from_config(config)


def score_answer(predicted, ground_truth, category: int) -> float:
    """Identical scoring to locomo_benchmark.py for fair comparison."""
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


def ingest(memory, conv: Dict, user_id: str) -> Dict:
    conv_data = conv["conversation"]
    session_keys = sorted(
        [k for k in conv_data.keys()
         if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    stats = {"turns": 0, "add_calls": 0, "errors": 0}
    for sk in session_keys:
        for turn in conv_data[sk]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if not text.strip():
                continue
            stats["turns"] += 1
            try:
                memory.add(f"{speaker} says: {text}", user_id=user_id)
                stats["add_calls"] += 1
            except Exception:
                stats["errors"] += 1
    return stats


def answer(memory, question: str, user_id: str, top_k: int = 5) -> str:
    # Mem0 v2.x requires filters dict instead of top-level user_id.
    try:
        results = memory.search(
            query=question,
            filters={"user_id": user_id},
            limit=top_k,
        )
    except TypeError:
        # Fall back for older mem0 versions.
        try:
            results = memory.search(query=question, user_id=user_id, limit=top_k)
        except Exception:
            return ""
    except Exception:
        return ""
    rows = results.get("results", []) if isinstance(results, dict) else (results or [])
    if not rows:
        return ""
    return " | ".join(r.get("memory", "") for r in rows[:top_k] if r.get("memory"))


def evaluate_conversation(memory, conv: Dict, conv_idx: int) -> Dict:
    speakers = (
        f"{conv['conversation'].get('speaker_a', '?')}"
        f"/{conv['conversation'].get('speaker_b', '?')}"
    )
    print(f"\n{'='*60}\n  Mem0 conv {conv_idx}: {speakers}\n{'='*60}")
    user_id = f"locomo_{conv_idx}"
    t0 = time.time()
    ingest_stats = ingest(memory, conv, user_id)
    ingest_time = time.time() - t0
    print(
        f"  Ingested: {ingest_stats['turns']} turns, "
        f"{ingest_stats['add_calls']} add() calls, "
        f"{ingest_stats['errors']} errors ({ingest_time:.1f}s)"
    )

    qa_pairs = conv.get("qa", [])
    cat_scores: Dict[int, List[float]] = defaultdict(list)
    evaluated = 0
    eval_start = time.time()
    for i, qa in enumerate(qa_pairs):
        if i and i % 40 == 0:
            print(f"    Progress: {i}/{len(qa_pairs)}")
        category = qa.get("category", 0)
        if category == 5:
            continue
        question = qa.get("question", "")
        gt = qa.get("answer", "")
        predicted = answer(memory, question, user_id)
        score = score_answer(predicted, gt, category)
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
        "ingest_time_s": round(ingest_time, 2),
        "eval_time_s": round(eval_time, 2),
        "category_scores": cat_summary,
        "overall_score": round(overall, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Mem0 LOCOMO Baseline")
    parser.add_argument("--data", default="data/locomo/locomo10.json")
    parser.add_argument("--max-conv", type=int, default=None)
    parser.add_argument("--output-name", default="locomo_mem0_latest.json")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, args.data)
    out_path = os.path.join(repo_root, "research", "metrics", args.output_name)

    with open(data_path) as f:
        data = json.load(f)
    if args.max_conv:
        data = data[: args.max_conv]

    print(f"\n{'#'*60}")
    print(f"# Mem0 LOCOMO BASELINE")
    print(f"# Conversations: {len(data)}")
    print(f"# LLM: DeepSeek (deepseek-chat)")
    print(f"# Time: {utc_now().isoformat()}")
    print(f"{'#'*60}")

    start = time.time()
    conv_results = []
    for i, conv in enumerate(data):
        memory = make_mem0_client()  # fresh collection per conversation
        conv_results.append(evaluate_conversation(memory, conv, i))
    total_time = time.time() - start

    all_scores = [r["overall_score"] for r in conv_results]
    cat_aggregate: Dict[int, List[float]] = defaultdict(list)
    for r in conv_results:
        for cat_str, info in r["category_scores"].items():
            cat_aggregate[int(cat_str)].append(info["score"])

    report = {
        "timestamp": utc_now().isoformat(),
        "benchmark": "LoCoMo (ACL 2024) — Mem0 baseline",
        "num_conversations": len(data),
        "total_time_s": round(total_time, 1),
        "overall_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
        "config": {
            "memory_provider": "mem0",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_store": "qdrant (in-memory)",
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
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'#'*60}\n# RESULT (Mem0)\n{'#'*60}")
    print(f"Overall: {report['overall_score']:.3f}")
    print(f"Total time: {report['total_time_s']}s")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
