"""
Brutal multi-agent harness runner.

Spins up the SCM /v1 server, builds three specialist agents (Researcher,
Coder, Reviewer) sharing a single user namespace + each owning a private
namespace. Drives them through 7 tiers of multi-agent scenarios.

Cost-conscious: uses DeepSeek v4 Flash (cheapest) and the HybridEncoder
(via SCM defaults) so only ~30-40% of turns escalate to LLM extraction.
A full run is roughly $0.10-$0.30 in DeepSeek API spend.

Usage:
    DEEPSEEK_API_KEY=sk-... python -m tests.brutal_multiagent.runner
    DEEPSEEK_API_KEY=sk-... python -m tests.brutal_multiagent.runner --tiers 1 3
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Load .env so DEEPSEEK_API_KEY is picked up.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import uvicorn

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


# ─── Configure SCM for the brutal multi-agent harness ────────────────────


# Short idle threshold so we can observe sleep cycles in seconds
os.environ.setdefault("SCM_IDLE_THRESHOLD_SEC", "10")
os.environ.setdefault("SCM_MCP_SWEEP_INTERVAL_SEC", "3")
os.environ.setdefault("SCM_LOG_LEVEL", "WARNING")

# Use DeepSeek for SCM's concept extraction too (matches the agent's LLM).
os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Sentence-transformers default for embeddings (Ollama optional).
os.environ.setdefault("SCM_EMBEDDING_BACKEND", "sentence_transformers")

# Per-test temp data dir (sandbox mode means this isn't actually used,
# but env var helps anything that DOES persist).
_TMPDIR = tempfile.mkdtemp(prefix="scm_brutal_multiagent_")
os.environ.setdefault("SCM_DATA_DIR", _TMPDIR)


logging.basicConfig(level="WARNING", format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("scm.mcp").setLevel(logging.INFO)


# ─── Server lifecycle (same shape as the LangChain brutal harness) ────────


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class SCMServer:
    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None):
        self.host = host
        self.port = port or _free_port()
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        from src.api.main import app
        config = uvicorn.Config(
            app, host=self.host, port=self.port, log_level="warning",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)

        def _run():
            try:
                self._server.run()
            except Exception as e:
                logging.exception(f"SCM server died: {e}")

        self._thread = threading.Thread(target=_run, name="scm-server", daemon=True)
        self._thread.start()
        import requests
        for _ in range(300):
            try:
                requests.get(
                    f"http://{self.host}:{self.port}/v1/health", timeout=1
                ).raise_for_status()
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError(f"SCM server did not come up at {self.base_url} after 60s")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True


# ─── Test scaffolding ─────────────────────────────────────────────────────


@dataclass
class Scenario:
    name: str
    tier: int
    passed: bool
    detail: str
    elapsed_seconds: float = 0.0


@dataclass
class BrutalReport:
    scenarios: List[Scenario] = field(default_factory=list)

    def by_tier(self):
        out = {}
        for s in self.scenarios:
            d = out.setdefault(s.tier, [0, 0])
            if s.passed:
                d[0] += 1
            else:
                d[1] += 1
        return out

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scenarios if s.passed)

    @property
    def total(self) -> int:
        return len(self.scenarios)

    def print(self) -> None:
        print("\n" + "=" * 76)
        print("BRUTAL MULTI-AGENT REPORT  (DeepSeek v4 + 3 LangChain agents)")
        print("=" * 76)
        print(f"Pass rate: {self.passed}/{self.total} "
              f"({100 * self.passed / max(1, self.total):.0f}%)")
        print(f"\nBy tier:")
        for tier, (p, f) in sorted(self.by_tier().items()):
            print(f"  Tier {tier}: {p}/{p + f}")
        print(f"\nDetails:")
        for s in self.scenarios:
            mark = "✓" if s.passed else "✗"
            print(f"  [T{s.tier}][{mark}] {s.name:55s} ({s.elapsed_seconds:.1f}s)")
            for line in s.detail.splitlines()[:4]:
                print(f"        {line[:120]}")
        print("=" * 76 + "\n")


TestFn = Callable[[str], List[Scenario]]


def run_brutal(test_fns: List[TestFn]) -> BrutalReport:
    server = SCMServer()
    print(f"[brutal-multiagent] starting SCM server on {server.base_url}…")
    server.start()
    print(f"[brutal-multiagent] up. data dir: {_TMPDIR}")

    report = BrutalReport()
    try:
        for fn in test_fns:
            print(f"[brutal-multiagent] running {fn.__name__}…")
            t0 = time.perf_counter()
            try:
                results = fn(server.base_url)
            except Exception as e:
                import traceback
                results = [Scenario(
                    name=fn.__name__, tier=0, passed=False,
                    detail=f"runner crashed: {e!r}\n{traceback.format_exc()[:500]}",
                )]
            elapsed = time.perf_counter() - t0
            for s in results:
                if s.elapsed_seconds == 0.0:
                    s.elapsed_seconds = elapsed / max(1, len(results))
            report.scenarios.extend(results)
    finally:
        server.stop()
        time.sleep(0.5)
    return report


# ─── Entry point ──────────────────────────────────────────────────────────


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("error: DEEPSEEK_API_KEY not set. Add to .env or export it.",
              file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=int, nargs="*", default=None,
                    help="run only these tier numbers (default: all)")
    args = ap.parse_args()

    from .scenarios import (
        tier1_per_agent_specialty,
        tier2_shared_user_handoff,
        tier3_cross_agent_contradiction,
        tier4_per_agent_wake_summary,
        tier5_collaborative_task,
        tier6_strict_isolation,
        tier7_deepseek_extraction_depth,
    )

    all_tests = [
        (1, tier1_per_agent_specialty),
        (2, tier2_shared_user_handoff),
        (3, tier3_cross_agent_contradiction),
        (4, tier4_per_agent_wake_summary),
        (5, tier5_collaborative_task),
        (6, tier6_strict_isolation),
        (7, tier7_deepseek_extraction_depth),
    ]
    if args.tiers:
        tiers_set = set(args.tiers)
        all_tests = [(t, fn) for t, fn in all_tests if t in tiers_set]

    report = run_brutal([fn for _, fn in all_tests])
    report.print()
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
