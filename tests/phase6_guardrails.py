"""Phase 6 hardening guardrails: warnings, regressions, and benchmark aggregation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Iterable

# Ensure repo root is on import path when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.human_memory_benchmark import build_report as build_human_report
from tests.phase2_metrics import build_report as build_phase2_report
from tests.phase4_metrics import build_report as build_phase4_report


DEPRECATED_PATTERNS = {
    "datetime.utcnow(": "Use timezone-aware utc_now()/datetime.now(timezone.utc).",
    ".dict(": "Use Pydantic v2 model_dump() instead of dict().",
}

PYTEST_TARGETS = [
    "tests/test_phase2_pipeline.py",
    "tests/test_phase3_pipeline.py",
    "tests/test_micro_sleep.py",
    "tests/test_deep_sleep.py",
    "tests/test_forgetting_dynamics.py",
    "tests/test_contradiction_versioning.py",
    "tests/test_human_memory_behavior.py",
    "tests/test_memory_pipeline.py",
]


def _iter_python_files(root_paths: Iterable[Path]) -> Iterable[Path]:
    for root in root_paths:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if ".pytest_cache" in path.parts:
                continue
            if "venv" in path.parts:
                continue
            yield path


def scan_deprecated_patterns() -> dict:
    hits = []
    files_scanned = 0

    for path in _iter_python_files([REPO_ROOT / "src", REPO_ROOT / "tests"]):
        files_scanned += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, guidance in DEPRECATED_PATTERNS.items():
                if pattern in line:
                    if f'"{pattern}"' in line or f"'{pattern}'" in line:
                        continue
                    hits.append(
                        {
                            "file": str(path.relative_to(REPO_ROOT)),
                            "line": lineno,
                            "pattern": pattern,
                            "guidance": guidance,
                            "snippet": line.strip(),
                        }
                    )
    return {
        "files_scanned": files_scanned,
        "patterns": list(DEPRECATED_PATTERNS.keys()),
        "hit_count": len(hits),
        "hits": hits,
    }


def run_pytest_guardrail() -> dict:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *PYTEST_TARGETS,
        "-q",
        "-W",
        "default::DeprecationWarning",
        "-W",
        "default::PendingDeprecationWarning",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    passed_match = re.search(r"(\d+)\s+passed", output)
    failed_match = re.search(r"(\d+)\s+failed", output)
    warning_match = re.search(r"(\d+)\s+warnings?", output)

    return {
        "command": cmd,
        "returncode": proc.returncode,
        "passed_tests": int(passed_match.group(1)) if passed_match else 0,
        "failed_tests": int(failed_match.group(1)) if failed_match else 0,
        "warnings_count": int(warning_match.group(1)) if warning_match else 0,
        "output_tail": "\n".join([line for line in output.splitlines() if line.strip()][-12:]),
    }


def build_report(args: argparse.Namespace) -> dict:
    phase2 = build_phase2_report()

    phase4_args = argparse.Namespace(
        seed=args.seed + 101,
        pair_count=args.phase4_pair_count,
        micro_turns_grid=args.phase4_micro_turns_grid,
        deep_pressure_grid=args.phase4_deep_pressure_grid,
    )
    phase4 = build_phase4_report(phase4_args)

    human_args = argparse.Namespace(
        seed=args.seed + 211,
        pair_count=args.human_pair_count,
        key_count=args.human_key_count,
        noise_count=args.human_noise_count,
    )
    human = build_human_report(human_args)

    deprecations = scan_deprecated_patterns()
    pytest_guard = run_pytest_guardrail()

    report = {
        "benchmark": "phase6_full_stack_guardrails",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system().lower(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "seed": args.seed,
        "targets": {
            "phase2_overall_pass": True,
            "phase4_overall_pass": True,
            "human_overall_pass": True,
            "pytest_returncode_zero": True,
            "pytest_warnings_zero": True,
            "deprecated_patterns_zero": True,
        },
        "inputs": {
            "phase4_pair_count": args.phase4_pair_count,
            "phase4_micro_turns_grid": args.phase4_micro_turns_grid,
            "phase4_deep_pressure_grid": args.phase4_deep_pressure_grid,
            "human_pair_count": args.human_pair_count,
            "human_key_count": args.human_key_count,
            "human_noise_count": args.human_noise_count,
            "pytest_targets": PYTEST_TARGETS,
        },
        "metrics": {
            "phase2": phase2,
            "phase4": phase4,
            "human_memory": human,
            "deprecation_scan": deprecations,
            "pytest_guardrail": pytest_guard,
        },
    }

    status = {
        "phase2_pass": bool(phase2.get("status", {}).get("overall_pass", False)),
        "phase4_pass": bool(phase4.get("status", {}).get("overall_pass", False)),
        "human_pass": bool(human.get("status", {}).get("overall_pass", False)),
        "pytest_pass": pytest_guard["returncode"] == 0,
        "warnings_pass": pytest_guard["warnings_count"] == 0,
        "deprecations_pass": deprecations["hit_count"] == 0,
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
        history_path = history_dir / f"phase6_guardrails_{ts_safe}.json"
        history_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return history_path


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 6 full-stack guardrails JSON")
    parser.add_argument(
        "--output",
        default="research/metrics/phase6_guardrails_latest.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=9090)

    parser.add_argument("--phase4-pair-count", type=int, default=24)
    parser.add_argument(
        "--phase4-micro-turns-grid",
        type=_parse_int_list,
        default=_parse_int_list("3,4,5"),
    )
    parser.add_argument(
        "--phase4-deep-pressure-grid",
        type=_parse_float_list,
        default=_parse_float_list("0.88,0.92,0.95"),
    )

    parser.add_argument("--human-pair-count", type=int, default=24)
    parser.add_argument("--human-key-count", type=int, default=16)
    parser.add_argument("--human-noise-count", type=int, default=32)

    parser.add_argument(
        "--write-history",
        action="store_true",
        help="Also write a timestamped guardrail snapshot",
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
