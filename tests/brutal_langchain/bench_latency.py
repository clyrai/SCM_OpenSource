"""
Focused latency benchmark for the async ingest path (v0.7.2).

Measures the user-facing latency of add_memory in two configurations:
    - sync=true (forces synchronous LLM extraction; blocks the caller)
    - default async (returns immediately after queueing)

This is the real product metric: how long does the agent wait between
the user sending a message and the next thing the agent does?

For interactive chat, anything > 1s is bad. Anything > 3s feels broken.
Mem0 is sub-second per add. SCM was 1-3s synchronously; with async,
p50 should be ~50ms.
"""
from __future__ import annotations

import os

# IMPORTANT: kill any LLM/Ollama config BEFORE any imports so the server
# threads built during import don't pick it up. We want this benchmark
# to measure SCM-internal cost (heuristic encoder + sentence-transformers
# embeddings), not Ollama HTTP latency.
for k in ("LLM_PROVIDER", "LLM_MODEL", "DEEPSEEK_MODEL"):
    os.environ.pop(k, None)
os.environ["SCM_AUTO_SLEEP_DISABLE"] = "1"
os.environ["SCM_EMBEDDING_BACKEND"] = "sentence_transformers"
os.environ.setdefault("SCM_LOG_LEVEL", "ERROR")

import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.brutal_langchain.runner import SCMServer
from src.integrations.langchain_adapter import SCMClient


SAMPLE_TEXTS = [
    "I'm Saish, lead engineer at Northstar Robotics in Bangalore.",
    "I run every Tuesday morning before standup.",
    "I'm allergic to peanuts; pretty severe.",
    "I prefer Python over JavaScript for everything.",
    "Friday dinner with Mara at the noodle place is our standing tradition.",
]


def measure(client: SCMClient, texts: list[str], sync: bool) -> dict:
    """Time each add_memory call, return p50/p95/total stats."""
    latencies = []
    t_total = time.perf_counter()
    for text in texts:
        t0 = time.perf_counter()
        body = {"text": text, "user_id": client.user_id}
        if sync:
            body["sync"] = True
        # Use the raw HTTP client so we can pass `sync=true` reliably.
        import requests
        r = requests.post(
            f"{client.base_url}/memories",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        r.raise_for_status()
        latencies.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_total
    return {
        "n": len(latencies),
        "total_s": total,
        "min_ms": min(latencies) * 1000,
        "p50_ms": statistics.median(latencies) * 1000,
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies))] * 1000,
        "max_ms": max(latencies) * 1000,
    }


def main() -> int:
    import os
    # Use heuristic encoder (no LLM_PROVIDER) so the contrast between
    # sync and async is dominated by SCM-internal work, not Ollama latency.
    os.environ.pop("LLM_PROVIDER", None)
    os.environ["SCM_AUTO_SLEEP_DISABLE"] = "1"
    os.environ["SCM_EMBEDDING_BACKEND"] = "sentence_transformers"
    # Pretty logs
    import logging
    logging.basicConfig(level="WARNING")

    server = SCMServer()
    print(f"[bench] starting SCM server on {server.base_url}…")
    server.start()
    print(f"[bench] up.")

    try:
        sync_client = SCMClient(user_id="bench_sync", base_url=server.base_url, timeout=60)
        async_client = SCMClient(user_id="bench_async", base_url=server.base_url, timeout=60)

        # Warm up — first call has model-load overhead we shouldn't include.
        sync_client.add_memory("warmup")
        async_client.add_memory("warmup")
        time.sleep(0.5)

        print("\n[bench] measuring SYNC add_memory (blocks on extraction)…")
        sync_stats = measure(sync_client, SAMPLE_TEXTS, sync=True)
        print(f"  sync: {sync_stats}")

        print("\n[bench] measuring ASYNC add_memory (queue-only)…")
        async_stats = measure(async_client, SAMPLE_TEXTS, sync=False)
        print(f"  async: {async_stats}")

        # Headline ratio
        if async_stats["p50_ms"] > 0:
            speedup = sync_stats["p50_ms"] / async_stats["p50_ms"]
            print(f"\n[bench] p50 speedup: {speedup:.0f}x faster (sync {sync_stats['p50_ms']:.0f}ms → async {async_stats['p50_ms']:.0f}ms)")
        if async_stats["p95_ms"] > 0:
            speedup_p95 = sync_stats["p95_ms"] / async_stats["p95_ms"]
            print(f"[bench] p95 speedup: {speedup_p95:.0f}x faster (sync {sync_stats['p95_ms']:.0f}ms → async {async_stats['p95_ms']:.0f}ms)")
        total_speedup = sync_stats["total_s"] / max(0.001, async_stats["total_s"])
        print(f"[bench] total wall-time speedup ({len(SAMPLE_TEXTS)} adds): {total_speedup:.1f}x")
    finally:
        server.stop()
        time.sleep(0.5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
