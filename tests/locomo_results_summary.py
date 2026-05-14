"""
Aggregate LOCOMO benchmark artifacts into a single comparison table.

Reads every JSON file under research/metrics/locomo_*.json and emits both a
human-readable Markdown table and the raw rows so the paper's Table can be
filled in by re-running this script after any of the underlying benchmarks.

Usage:
    venv/bin/python3 tests/locomo_results_summary.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(REPO_ROOT, "research", "metrics")


def load_artifacts() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "locomo_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN] could not parse {path}: {exc}", file=sys.stderr)
            continue
        rows.append({
            "name": os.path.basename(path).replace(".json", ""),
            "benchmark": data.get("benchmark", "?"),
            "overall": data.get("overall_score", 0.0),
            "num_conv": data.get("num_conversations", 0),
            "total_time_s": data.get("total_time_s", 0),
            "config": data.get("config", {}),
            "category_scores": data.get("category_scores", {}),
        })
    return rows


def system_label(row: Dict[str, Any]) -> str:
    name = row["name"]
    if "mem0" in name:
        backend = row["config"].get("llm_provider", "?")
        return f"Mem0 ({backend})"
    if "phase4_only" in name or "phase4_no_llm" in name:
        return "SCM Phase-4-only (heuristic)"
    if "phase4_deepseek" in name:
        return "SCM Phase-4-only (deepseek)"
    if "hme_pilot" in name:
        return "SCM HME-full (heuristic)"
    if "hme_deepseek" in name or "hme_full_deepseek" in name:
        return "SCM HME-full (deepseek)"
    if "hme" in name:
        return f"SCM HME-full ({row['config'].get('answer_strategy', 'unknown')})"
    return name


def main():
    rows = load_artifacts()
    if not rows:
        print("No LOCOMO artifacts found in research/metrics/")
        return

    rows.sort(key=lambda r: r["overall"], reverse=True)

    print()
    print(f"{'System':<40} {'Overall':>8} {'#Conv':>6} {'Time(s)':>8}")
    print("-" * 64)
    for row in rows:
        print(
            f"{system_label(row):<40} "
            f"{row['overall']:>8.3f} "
            f"{row['num_conv']:>6} "
            f"{row['total_time_s']:>8}"
        )

    print()
    print("Per-category breakdown:")
    print(f"{'System':<40} {'1':>6} {'2':>6} {'3':>6} {'4':>6}")
    print("-" * 64)
    for row in rows:
        cells = []
        for cat in ("1", "2", "3", "4"):
            cell = row["category_scores"].get(cat, {})
            cells.append(f"{cell.get('score', 0.0):.3f}")
        print(f"{system_label(row):<40} {' '.join(f'{c:>6}' for c in cells)}")

    # Markdown for paper inclusion
    print("\n--- Markdown table (drop into paper):\n")
    print("| System | Overall | Cat 1 | Cat 2 | Cat 3 | Cat 4 |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        cat = row["category_scores"]
        c1 = cat.get("1", {}).get("score", 0.0)
        c2 = cat.get("2", {}).get("score", 0.0)
        c3 = cat.get("3", {}).get("score", 0.0)
        c4 = cat.get("4", {}).get("score", 0.0)
        print(
            f"| {system_label(row)} | {row['overall']:.3f} | "
            f"{c1:.3f} | {c2:.3f} | {c3:.3f} | {c4:.3f} |"
        )


if __name__ == "__main__":
    main()
