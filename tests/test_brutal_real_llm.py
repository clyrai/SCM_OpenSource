"""
Real-LLM brutal tests.

Two backends:
  - Ollama (local, free, slow ~2-3s/call) — runs unbounded
  - DeepSeek (cloud, fast, costs ~$0.001-0.003 per call) — runs ONCE per
    pytest invocation, with a hard cost cap and pre-flight budget gate.

Both tests use the compact `developer_short.json` persona to bound runtime
and cost. They produce JSON artifacts under research/metrics/ for honest
comparison against the stub-LLM baseline.

Skip behavior:
  - Ollama test skips if `http://localhost:11434` is unreachable.
  - DeepSeek test skips if `DEEPSEEK_API_KEY` is missing in env (.env is
    auto-loaded). To force-run the DeepSeek test, set
    SCM_BRUTAL_RUN_DEEPSEEK=1 in env. Default is to skip even if key is
    present, because each run consumes API budget.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # type: ignore

load_dotenv(Path(__file__).parent.parent / ".env")

from tests.brutal.harness import BrutalReport, run_brutal


PERSONAS_DIR = Path(__file__).parent / "brutal" / "personas"
METRICS_DIR = Path(__file__).parent.parent / "research" / "metrics"


# ─── Pre-flight checks ─────────────────────────────────────────────────────


def _ollama_reachable(timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=timeout_s
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _deepseek_configured() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _deepseek_run_authorized() -> bool:
    """Belt-and-braces: only run DeepSeek brutal if user explicitly opts in
    via SCM_BRUTAL_RUN_DEEPSEEK=1. Default is to skip even when key is set,
    because each run costs real money."""
    return os.getenv("SCM_BRUTAL_RUN_DEEPSEEK") == "1"


# ─── Common assertions for real-LLM runs ──────────────────────────────────


def _assert_minimum_brutal_health(report: BrutalReport, backend_label: str) -> None:
    """
    Health floor for any real-LLM brutal run. We use a more lenient bar than
    the stub tests because LLM extraction is non-deterministic and the
    persona is short.
    """
    assert report.turns_ingested >= 6, (
        f"[{backend_label}] not enough turns ingested: {report.turns_ingested}"
    )
    assert report.sleep_cycles >= 1, (
        f"[{backend_label}] no sleep cycle fired"
    )
    # Wake summary should at least render the welcome line
    assert "Welcome back" in report.final_narrative, (
        f"[{backend_label}] narrative malformed: {report.final_narrative!r}"
    )
    # At least ONE recall scenario should pass — if the LLM extractor produces
    # garbage, this is what catches it.
    recall_passed = sum(
        1 for s in report.scenarios
        if s.category in ("long_horizon_recall", "contradiction_current") and s.passed
    )
    assert recall_passed >= 1, (
        f"[{backend_label}] zero recall scenarios passed; "
        f"detail: {[(s.name, s.passed, s.detail[:120]) for s in report.scenarios]}"
    )


def _save_report(report: BrutalReport, output_name: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / output_name
    payload = {
        "persona": report.persona_name,
        "started_at": report.started_at.isoformat(),
        "ended_at": report.ended_at.isoformat(),
        "turns_ingested": report.turns_ingested,
        "sleep_cycles": report.sleep_cycles,
        "schemas_in_final_summary": report.schemas_formed_in_final_summary,
        "daemon_blocked_count": report.daemon_blocked_count,
        "pass_rate": report.pass_rate,
        "passed": report.passed,
        "failed": report.failed,
        "by_category": report.by_category(),
        "scenarios": [
            {
                "name": s.name, "category": s.category,
                "passed": s.passed, "description": s.description, "detail": s.detail,
            }
            for s in report.scenarios
        ],
        "final_narrative": report.final_narrative,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


# ─── Ollama brutal ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable on localhost:11434")
def test_brutal_real_llm_ollama_developer_short():
    """
    Run the developer-short persona through the FULL Phase 7 stack with
    the local Ollama LLM doing real concept extraction. This is the
    no-cost, slow-but-real proof that the system works with a non-stub LLM.

    Wall-clock: ~60-180 seconds depending on Ollama latency.
    """
    persona_path = PERSONAS_DIR / "developer_short.json"
    assert persona_path.exists(), f"Persona missing: {persona_path}"

    # Curiosity dictionary covers the entities the persona mentions, so the
    # static source can fill gaps without an LLM call. The LLMSource is
    # added too as a fallback — but it'll cost only if the dict misses.
    glossary = {
        "Kafka": "Kafka is an open-source distributed event-streaming platform.",
        "PostgreSQL": "PostgreSQL is an open-source relational database.",
        "VSCode": "VSCode is a popular code editor from Microsoft.",
        "FinTech": "FinTech is the persona's current employer in this scenario.",
    }
    t0 = time.time()
    report = run_brutal(
        persona_path=persona_path,
        enable_hme=True,
        enable_schemas=True,
        enable_curiosity=True,
        curiosity_dictionary=glossary,
        enable_paraphrase=True,
        llm_backend="ollama",
        add_llm_curiosity_source=False,  # static dict only — keep it free
    )
    elapsed = time.time() - t0
    print(f"\n[ollama] elapsed={elapsed:.1f}s")
    report.print()
    out = _save_report(report, "brutal_real_llm_ollama_latest.json")
    print(f"[ollama] report saved: {out}")
    _assert_minimum_brutal_health(report, "ollama")


# ─── DeepSeek brutal (gated: real money) ───────────────────────────────────


@pytest.mark.skipif(
    not (_deepseek_configured() and _deepseek_run_authorized()),
    reason=(
        "DeepSeek brutal test costs API tokens. To run, set "
        "SCM_BRUTAL_RUN_DEEPSEEK=1 and ensure DEEPSEEK_API_KEY is in .env."
    ),
)
def test_brutal_real_llm_deepseek_developer_short():
    """
    Run the developer-short persona with DeepSeek as the LLM extractor.

    Cost guard: this test is gated behind SCM_BRUTAL_RUN_DEEPSEEK=1 because
    each invocation costs real money (~$0.05-0.10 with 8 turns + 2-3 sleep
    cycles + curiosity LLM source). Run sparingly.

    Wall-clock: ~30-60 seconds.
    """
    persona_path = PERSONAS_DIR / "developer_short.json"
    assert persona_path.exists(), f"Persona missing: {persona_path}"

    print("\n[deepseek] WARNING: this run will consume DeepSeek API budget.")
    print("[deepseek] Persona: developer_short (8 turns, 2 days)")
    print("[deepseek] Estimated cost: ~$0.05-0.10")

    glossary = {
        "Kafka": "Kafka is an open-source distributed event-streaming platform.",
        "PostgreSQL": "PostgreSQL is an open-source relational database.",
    }
    t0 = time.time()
    report = run_brutal(
        persona_path=persona_path,
        enable_hme=True,
        enable_schemas=True,
        enable_curiosity=True,
        curiosity_dictionary=glossary,
        enable_paraphrase=True,
        llm_backend="deepseek",
        # LLMSource opt-in: lets DeepSeek itself fill gaps the static dict misses.
        # This is the real autonomous-knowledge experience.
        add_llm_curiosity_source=True,
    )
    elapsed = time.time() - t0
    print(f"\n[deepseek] elapsed={elapsed:.1f}s")
    report.print()
    out = _save_report(report, "brutal_real_llm_deepseek_latest.json")
    print(f"[deepseek] report saved: {out}")
    _assert_minimum_brutal_health(report, "deepseek")
