"""Quick benchmark: ANN vector index vs legacy linear cosine scan.

Measures search latency and memory at increasing index sizes. Not part of
pytest CI — run manually:

    python tests/bench_vector_index.py
"""
import time
import numpy as np

from src.retrieval.vector_index import InMemoryVectorIndex


def _legacy_linear_search(query, items, top_k=10):
    """Reproduces the pre-vector-index O(n) cosine scan inside SA._embedding_seeds."""
    def cos(a, b):
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        na = float(np.linalg.norm(a_arr))
        nb = float(np.linalg.norm(b_arr))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (na * nb))

    scored = []
    for cid, emb in items:
        scored.append((cid, cos(query, emb)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def _bench(n: int, dim: int = 384, top_k: int = 10, iters: int = 50):
    rng = np.random.default_rng(42)
    items = [(f"c{i}", rng.normal(size=dim).tolist()) for i in range(n)]
    query = rng.normal(size=dim).tolist()

    # Build the index (one-time cost).
    idx = InMemoryVectorIndex()
    t0 = time.perf_counter()
    idx.rebuild(items)
    rebuild_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup.
    idx.search(query, top_k=top_k)
    _legacy_linear_search(query, items, top_k=top_k)

    # ANN search.
    t0 = time.perf_counter()
    for _ in range(iters):
        idx.search(query, top_k=top_k)
    ann_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Legacy scan.
    t0 = time.perf_counter()
    for _ in range(iters):
        _legacy_linear_search(query, items, top_k=top_k)
    legacy_ms = (time.perf_counter() - t0) * 1000.0 / iters

    speedup = legacy_ms / ann_ms if ann_ms > 0 else float("inf")

    print(
        f"n={n:>7,}  rebuild={rebuild_ms:>7.1f}ms  "
        f"ANN={ann_ms:>7.3f}ms  legacy={legacy_ms:>9.2f}ms  "
        f"speedup={speedup:>7.1f}×"
    )


if __name__ == "__main__":
    print(f"{'n':>9}  {'rebuild':>14}  {'ANN':>13}  {'legacy':>15}  {'speedup':>10}")
    for n in [100, 1_000, 10_000, 100_000]:
        _bench(n=n, dim=384, top_k=10, iters=20 if n >= 10_000 else 50)
