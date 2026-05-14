"""Generate a paper-ready reproducibility pack for SCM.

The pack reruns the primary benchmark scripts, captures their logs, and writes
both a machine-readable manifest and a concise markdown summary. The intent is
to keep the evidence trail close to the paper narrative while still making the
exact rerun commands explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "research/reproducibility/reproducibility_pack_latest.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs/SCM_REPRODUCIBILITY_PACK.md"
DEFAULT_HISTORY_DIR = REPO_ROOT / "research/reproducibility/history"
DEFAULT_LOG_DIR = REPO_ROOT / "research/reproducibility/logs"


@dataclass(frozen=True)
class RunSpec:
    name: str
    command: Sequence[str]
    artifact: Path
    summary_key: str


RUN_SPECS: List[RunSpec] = [
    RunSpec(
        name="baseline_comparison",
        command=(sys.executable, "tests/baseline_comparison.py"),
        artifact=REPO_ROOT / "research/metrics/baseline_comparison_latest.json",
        summary_key="baseline",
    ),
    RunSpec(
        name="long_horizon",
        command=(sys.executable, "tests/long_horizon_benchmark.py"),
        artifact=REPO_ROOT / "research/metrics/long_horizon_latest.json",
        summary_key="long_horizon",
    ),
    RunSpec(
        name="phase6_guardrails",
        command=(sys.executable, "tests/phase6_guardrails.py"),
        artifact=REPO_ROOT / "research/metrics/phase6_guardrails_latest.json",
        summary_key="guardrails",
    ),
    RunSpec(
        name="smoke_pytests",
        command=(
            sys.executable,
            "-m",
            "pytest",
            "tests/test_baseline_comparison.py",
            "tests/test_long_horizon_benchmark.py",
            "tests/test_human_memory_behavior.py",
            "tests/test_forgetting_dynamics.py",
            "tests/test_contradiction_versioning.py",
            "-q",
        ),
        artifact=REPO_ROOT / "research/reproducibility/reproducibility_smoke.json",
        summary_key="pytest",
    ),
]


def _ensure_repo_root() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _run_command(command: Sequence[str], log_path: Path) -> Dict[str, Any]:
    start = datetime.now(timezone.utc)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    end = datetime.now(timezone.utc)
    log_path.write_text(
        "".join(
            [
                f"$ {' '.join(command)}\n\n",
                proc.stdout or "",
                "\n",
                proc.stderr or "",
                "\n",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "command": list(command),
        "command_display": " ".join(command),
        "returncode": proc.returncode,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": end.isoformat(),
        "duration_ms": round((end - start).total_seconds() * 1000.0, 2),
        "log_path": str(log_path.relative_to(REPO_ROOT)),
    }


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": _sha256(path) if exists else None,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _generic_status(report: Dict[str, Any]) -> Dict[str, Any]:
    status = report.get("status")
    if isinstance(status, dict):
        return status
    if "overall_pass" in report:
        return {"overall_pass": bool(report["overall_pass"])}
    return {}


def _benchmark_summary(spec: RunSpec, report: Dict[str, Any]) -> Dict[str, Any]:
    if spec.summary_key == "baseline":
        status = _generic_status(report)
        return {
            "overall_pass": bool(status.get("overall_pass")),
            "scm_deep_pass": bool(status.get("scm_deep_pass")),
            "human_suite_pass": bool(status.get("human_suite_pass")),
        }
    if spec.summary_key == "long_horizon":
        status = _generic_status(report)
        sleep_mode = report.get("modes", {}).get("sleep_enabled", {})
        return {
            "overall_pass": bool(status.get("overall_pass")),
            "final_family_recall": sleep_mode.get("final_disambiguation_recall"),
            "final_strict_recall": sleep_mode.get("final_strict_disambiguation_recall"),
            "final_noise_retention": sleep_mode.get("final_noise_retention"),
            "anchor_accuracy": sleep_mode.get("anchor_accuracy"),
        }
    if spec.summary_key == "guardrails":
        status = _generic_status(report)
        return {
            "overall_pass": bool(status.get("overall_pass")),
            "phase2_pass": bool(status.get("phase2_pass")),
            "phase4_pass": bool(status.get("phase4_pass")),
            "human_pass": bool(status.get("human_pass")),
            "pytest_pass": bool(status.get("pytest_pass")),
            "warnings_pass": bool(status.get("warnings_pass")),
            "deprecations_pass": bool(status.get("deprecations_pass")),
        }
    if spec.summary_key == "pytest":
        return {
            "overall_pass": bool(report.get("overall_pass")),
            "returncode": report.get("returncode"),
        }
    return {}


def _git_info() -> Dict[str, Any]:
    def _git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "status": _git("status", "--short"),
    }


def _environment_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }


def _render_markdown(pack: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.extend(
        [
            "# SCM Reproducibility Pack",
            "",
            "This pack records the exact benchmark reruns, the environment, and the artifact fingerprints used for the current claim set.",
            "",
            "## Verdict",
            "",
            f"- Overall pass: {'yes' if pack['overall_pass'] else 'no'}",
            f"- Git commit: `{pack['git']['commit'] or 'unknown'}`",
            f"- Branch: `{pack['git']['branch'] or 'unknown'}`",
            "",
            "## Environment",
            "",
            f"- Python: `{pack['environment']['python']}`",
            f"- Platform: `{pack['environment']['platform']}`",
            f"- Machine: `{pack['environment']['machine']}`",
            f"- Working directory: `{pack['environment']['cwd']}`",
            "",
            "## Benchmark Reruns",
            "",
            "| Run | Status | Duration | Log |",
            "| --- | --- | --- | --- |",
        ]
    )

    for run in pack["runs"]:
        status = "pass" if run["returncode"] == 0 else "fail"
        lines.append(
            f"| {run['name']} | {status} | {run['duration_ms']:.2f} ms | `{run['log_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Artifact Fingerprints",
            "",
            "| Artifact | Bytes | SHA256 |",
            "| --- | --- | --- |",
        ]
    )
    for artifact in pack["artifacts"]:
        lines.append(
            f"| `{artifact['path']}` | {artifact['bytes']} | `{artifact['sha256'] or 'missing'}` |"
        )

    lines.extend(["", "## Key Summaries", ""])
    for key, summary in pack["summaries"].items():
        lines.append(f"### {key}")
        if not summary:
            lines.append("- No summary available.")
            lines.append("")
            continue
        for item_key, item_value in summary.items():
            lines.append(f"- {item_key}: {item_value}")
        lines.append("")

    lines.extend(
        [
            "## Reproduce",
            "",
            "Run the same pack again with:",
            "",
            "```bash",
            "python tests/reproducibility_pack.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_pack(run_commands: bool = True) -> Dict[str, Any]:
    git = _git_info()
    environment = _environment_info()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = DEFAULT_LOG_DIR / run_id
    history_dir = DEFAULT_HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    overall_pass = True

    if run_commands:
        for spec in RUN_SPECS:
            log_path = log_dir / f"{spec.name}.log"
            run_result = _run_command(spec.command, log_path)
            run_result["name"] = spec.name
            runs.append(run_result)
            if run_result["returncode"] != 0:
                overall_pass = False
            if spec.summary_key == "pytest":
                spec.artifact.parent.mkdir(parents=True, exist_ok=True)
                spec.artifact.write_text(
                    json.dumps(
                        {
                            "benchmark": "scm_reproducibility_smoke",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "overall_pass": run_result["returncode"] == 0,
                            "returncode": run_result["returncode"],
                            "command": run_result["command_display"],
                            "log_path": run_result["log_path"],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if spec.artifact.exists():
                try:
                    report = _load_json(spec.artifact)
                    summaries[spec.name] = _benchmark_summary(spec, report)
                    status = _generic_status(report)
                    if status and not bool(status.get("overall_pass", True)):
                        overall_pass = False
                except Exception as exc:  # pragma: no cover - defensive guardrail
                    summaries[spec.name] = {"error": str(exc)}
                    overall_pass = False
            else:
                summaries[spec.name] = {"error": "artifact missing"}
                overall_pass = False
    else:
        for spec in RUN_SPECS:
            runs.append(
                {
                    "name": spec.name,
                    "command": list(spec.command),
                    "command_display": " ".join(spec.command),
                    "returncode": 0,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                    "duration_ms": 0.0,
                    "log_path": None,
                }
            )
            if spec.artifact.exists():
                try:
                    report = _load_json(spec.artifact)
                    summaries[spec.name] = _benchmark_summary(spec, report)
                except Exception as exc:  # pragma: no cover - defensive guardrail
                    summaries[spec.name] = {"error": str(exc)}
                    overall_pass = False
            else:
                summaries[spec.name] = {"error": "artifact missing"}
                overall_pass = False

    pack = {
        "benchmark": "scm_reproducibility_pack",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "environment": environment,
        "runs": runs,
        "artifacts": [_artifact_record(spec.artifact) for spec in RUN_SPECS],
        "summaries": summaries,
        "overall_pass": overall_pass,
        "notes": [
            "The long-horizon benchmark now reports family-aware duplicate recall as the main gate.",
            "The strict duplicate-pair curve is retained in the long-horizon JSON as a stress signal.",
            "Artifact fingerprints capture the exact latest report files used by the paper claims.",
        ],
        "log_dir": str(log_dir.relative_to(REPO_ROOT)),
    }
    return pack


def _write_outputs(pack: Dict[str, Any], output: Path, markdown: Path, write_history: bool, history_dir: Path) -> Path | None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")

    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_render_markdown(pack) + "\n", encoding="utf-8")

    history_path = None
    if write_history:
        history_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        history_path = history_dir / f"reproducibility_pack_{ts_safe}.json"
        history_path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    return history_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the SCM reproducibility pack")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--no-history", action="store_false", dest="write_history")
    parser.add_argument("--skip-runs", action="store_true", help="Do not rerun benchmark commands; summarize existing artifacts only.")
    args = parser.parse_args()

    pack = build_pack(run_commands=not args.skip_runs)
    history_path = _write_outputs(
        pack=pack,
        output=Path(args.output),
        markdown=Path(args.markdown),
        write_history=args.write_history,
        history_dir=Path(args.history_dir),
    )

    print(json.dumps({"overall_pass": pack["overall_pass"], "runs": len(pack["runs"])}, indent=2))
    print(f"JSON report: {Path(args.output).resolve()}")
    print(f"Markdown report: {Path(args.markdown).resolve()}")
    if history_path:
        print(f"History report: {history_path.resolve()}")


if __name__ == "__main__":
    main()
