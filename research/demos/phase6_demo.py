"""Phase 6 demo scenarios for product + research storytelling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure repo root is on import path when run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.human_memory_benchmark import (
    benchmark_contradiction_versioning,
    benchmark_one_shot_recall,
    benchmark_selective_forgetting,
)
from tests.phase4_metrics import benchmark_sleep_gain


def run_demo(args: argparse.Namespace) -> dict:
    one_shot = benchmark_one_shot_recall()
    sleep_gain = benchmark_sleep_gain(pair_count=args.pair_count, seed=args.seed + 5)
    forgetting = benchmark_selective_forgetting(
        key_count=args.key_count,
        noise_count=args.noise_count,
    )
    contradiction = benchmark_contradiction_versioning()

    micro_gain = sleep_gain["micro_sleep"]["disambiguation_gain_abs"]
    deep_gain = sleep_gain["deep_sleep"]["disambiguation_gain_abs"]
    deep_noise_retention = sleep_gain["deep_sleep"]["pressure"]["noise_retention"]

    story = {
        "one_shot_memory": {
            "accuracy": one_shot["accuracy"],
            "example_name_response": one_shot["name_response"],
            "example_location_response": one_shot["location_response"],
        },
        "sleep_benefit": {
            "baseline_disambiguation_recall": sleep_gain["baseline"]["disambiguation_recall"],
            "micro_disambiguation_gain_abs": micro_gain,
            "deep_disambiguation_gain_abs": deep_gain,
            "deep_noise_retention": deep_noise_retention,
        },
        "selective_forgetting": {
            "key_retention": forgetting["key_retention"],
            "noise_retention": forgetting["noise_retention"],
            "first_cycle_forgotten": forgetting["first_cycle"]["forgotten"],
            "second_cycle_forgotten": forgetting["second_cycle"]["forgotten"],
        },
        "contradiction_safety": {
            "chain_ok": contradiction["chain_ok"],
            "old_hidden": contradiction["old_hidden"],
            "retrieval_prefers_new": contradiction["retrieval_prefers_new"],
            "accuracy": contradiction["accuracy"],
        },
    }

    return {
        "demo": "phase6_story_pack",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "inputs": {
            "pair_count": args.pair_count,
            "key_count": args.key_count,
            "noise_count": args.noise_count,
        },
        "story": story,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 6 product/research demo scenarios")
    parser.add_argument(
        "--output",
        default="research/metrics/phase6_demo_latest.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--pair-count", type=int, default=24)
    parser.add_argument("--key-count", type=int, default=16)
    parser.add_argument("--noise-count", type=int, default=32)
    args = parser.parse_args()

    report = run_demo(args)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote demo report to {output_path}")
    print(json.dumps(report["story"], indent=2))


if __name__ == "__main__":
    main()
