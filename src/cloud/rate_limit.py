"""Per-API-key rate limiter (in-process token bucket).

Why token bucket: smooth across small bursts (a real user hits the API
several times in a row at the start of a chat turn) while bounding
sustained throughput. Pure-Python, no Redis, fine for one server. When
SCM Cloud needs multi-machine, swap this for a Redis-backed limiter
with the same `acquire(key_id, capacity)` signature.

Memory layout: dict[key_id] -> (tokens_left, last_refill_at). Cleaned
opportunistically when keys haven't been seen in 1h.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Tuple


class TokenBucket:
    """One-server token bucket. Thread-safe, O(1) per request."""

    def __init__(self, cleanup_interval_sec: float = 3600.0):
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval_sec
        self._last_cleanup = time.time()

    def acquire(self, key_id: str, capacity_per_min: int) -> bool:
        """Take 1 token from this key's bucket. Return True on success,
        False if the bucket is empty (caller should respond 429)."""
        if capacity_per_min <= 0:
            return True  # disabled / unlimited tier
        now = time.time()
        refill_per_sec = capacity_per_min / 60.0
        with self._lock:
            tokens, last = self._buckets.get(key_id, (capacity_per_min, now))
            elapsed = now - last
            tokens = min(capacity_per_min, tokens + elapsed * refill_per_sec)
            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[key_id] = (tokens, now)
                self._maybe_cleanup(now)
                return True
            self._buckets[key_id] = (tokens, now)
            return False

    def _maybe_cleanup(self, now: float) -> None:
        """Drop buckets unseen for >cleanup_interval. Called inside lock."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        cutoff = now - self._cleanup_interval
        stale = [k for k, (_, t) in self._buckets.items() if t < cutoff]
        for k in stale:
            self._buckets.pop(k, None)
        self._last_cleanup = now


_default_bucket: TokenBucket = TokenBucket()


def acquire(key_id: str, capacity_per_min: int) -> bool:
    return _default_bucket.acquire(key_id, capacity_per_min)
