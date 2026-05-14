"""
Brutal harness runner.

Spins up the SCM /v1 API server in a background thread, sets up
deliberately-short idle thresholds (so we can observe sleep cycles in
under a minute), constructs LangChain agents per user, and drives them
through a tier-based test plan. Reports pass/fail per scenario.

Usage:

    python tests/brutal_langchain/runner.py            # run all tiers
    python tests/brutal_langchain/runner.py --tiers 1 3 5

Requires:
    - Ollama running with llama3.2:latest pulled
    - venv with langchain + langchain-ollama installed

Cost: $0. Everything runs locally.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket

# Enable INFO logging for the SCM MCP sweeper so we can verify auto-sleep
# fires during the brutal harness. Other loggers stay at WARNING.
logging.basicConfig(level="WARNING", format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("scm.mcp").setLevel(logging.INFO)
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import uvicorn

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


# ─── Configure SCM for brutal-harness use ─────────────────────────────────


# Short idle threshold so we can observe sleep cycles in seconds, not minutes
os.environ.setdefault("SCM_IDLE_THRESHOLD_SEC", "8")
# Sweep faster than default 30s so 12s waits in tests can catch sweeper firing
os.environ.setdefault("SCM_MCP_SWEEP_INTERVAL_SEC", "3")
# Never break things silently — surface every issue
os.environ.setdefault("SCM_LOG_LEVEL", "WARNING")
# Phase 6 protection: don't archive freshly-ingested user facts on the
# first sleep cycle. Without this floor, a single user statement like
# "I'm allergic to seafood" can be archived during the very first
# deep_sleep, defeating the point of long-term memory entirely.
os.environ.setdefault("FORGETTING_PROTECT_SALIENCE", "0.5")
# Don't archive concepts on their first sleep cycle — give them at least
# one rehearsal cycle to demonstrate importance.
os.environ.setdefault("FORGETTING_MIN_REHEARSAL_BEFORE_ARCHIVE", "1")
# In-memory SQLite scoped to a temp dir so runs are isolated
import tempfile
_TMPDIR = tempfile.mkdtemp(prefix="scm_brutal_")
os.environ.setdefault("SCM_DATA_DIR", _TMPDIR)

# Ensure the brutal-harness uses Ollama for both LLM extraction AND embeddings
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "llama3.2:latest")
os.environ.setdefault("SCM_EMBEDDING_BACKEND", "ollama")
os.environ.setdefault("SCM_EMBEDDING_MODEL", "nomic-embed-text")


# ─── Server lifecycle ─────────────────────────────────────────────────────


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class SCMServer:
    """Run the SCM FastAPI app in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None):
        self.host = host
        self.port = port or _free_port()
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        # Import the app lazily so env vars are honored at module import.
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
        # Wait for server to come up. FastAPI lifespan + sentence-transformer
        # load can take 15-30s on first start, so be patient.
        import requests
        for _ in range(300):  # up to 60s
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
        print("\n" + "=" * 72)
        print(f"BRUTAL LANGCHAIN-AGENT REPORT")
        print("=" * 72)
        print(f"Pass rate: {self.passed}/{self.total} "
              f"({100 * self.passed / max(1, self.total):.0f}%)")
        print(f"\nBy tier:")
        for tier, (p, f) in sorted(self.by_tier().items()):
            print(f"  Tier {tier}: {p}/{p + f}")
        print(f"\nDetails:")
        for s in self.scenarios:
            mark = "✓" if s.passed else "✗"
            print(f"  [T{s.tier}][{mark}] {s.name:48s} ({s.elapsed_seconds:.1f}s)")
            for line in s.detail.splitlines()[:3]:
                print(f"        {line[:100]}")
        print("=" * 72 + "\n")


# Test runner takes a (server_url) and returns Scenario list
TestFn = Callable[[str], List[Scenario]]


def run_brutal(test_fns: List[TestFn]) -> BrutalReport:
    """Spin up the server, run all test functions, tear down."""
    server = SCMServer()
    print(f"[brutal] starting SCM server on {server.base_url}…")
    server.start()
    print(f"[brutal] up. data dir: {_TMPDIR}")

    report = BrutalReport()
    try:
        for fn in test_fns:
            print(f"[brutal] running {fn.__name__}…")
            t0 = time.perf_counter()
            try:
                results = fn(server.base_url)
            except Exception as e:
                results = [Scenario(
                    name=fn.__name__, tier=0, passed=False,
                    detail=f"runner crashed: {e!r}",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=int, nargs="*", default=None,
                    help="run only these tier numbers (default: all)")
    args = ap.parse_args()

    from .scenarios import (
        tier1_multi_day_recall,
        tier2_contradiction_handling,
        tier3_idle_fired_wake_summary,
        tier4_cross_session_synthesis,
        tier5_adversarial_storm,
        tier6_multi_user_isolation,
        tier7_failure_mode,
    )

    all_tests = [
        (1, tier1_multi_day_recall),
        (2, tier2_contradiction_handling),
        (3, tier3_idle_fired_wake_summary),
        (4, tier4_cross_session_synthesis),
        (5, tier5_adversarial_storm),
        (6, tier6_multi_user_isolation),
        (7, tier7_failure_mode),
    ]
    if args.tiers:
        tiers_set = set(args.tiers)
        all_tests = [(t, fn) for t, fn in all_tests if t in tiers_set]

    report = run_brutal([fn for _, fn in all_tests])
    report.print()
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
