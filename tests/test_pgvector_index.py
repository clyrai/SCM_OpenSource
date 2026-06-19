"""Tests for PgvectorIndex — the production ANN backend.

Skips cleanly when no Postgres+pgvector instance is reachable. To run
locally:

    docker run --rm -p 5433:5432 \\
      -e POSTGRES_PASSWORD=test -e POSTGRES_DB=scm_test \\
      pgvector/pgvector:pg16

    POSTGRES_TEST_DSN="postgresql://postgres:test@localhost:5433/scm_test" \\
    pytest tests/test_pgvector_index.py
"""
from __future__ import annotations

import math
import os
import uuid

import pytest


# ── Skip mechanism ──────────────────────────────────────────────────────


def _try_connect():
    """Return (psycopg2, dsn) if Postgres+pgvector is reachable, else None."""
    dsn = os.environ.get("POSTGRES_TEST_DSN")
    if not dsn:
        return None
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector  # noqa: F401
    except ImportError:
        return None
    try:
        conn = psycopg2.connect(dsn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        conn.close()
        return (psycopg2, dsn)
    except Exception:
        return None


_AVAILABLE = _try_connect()
pytestmark = pytest.mark.skipif(
    _AVAILABLE is None,
    reason="POSTGRES_TEST_DSN unset or Postgres+pgvector unreachable",
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def index():
    """Fresh PgvectorIndex per test, isolated by a unique table name so
    tests can run in parallel without trampling each other."""
    from src.retrieval.pgvector_index import PgvectorIndex
    table = f"pgvec_test_{uuid.uuid4().hex[:12]}"
    idx = PgvectorIndex(
        dim=4,
        dsn=_AVAILABLE[1],
        table_name=table,
        ensure_extension=False,
    )
    yield idx
    # Clean up after the test.
    psycopg2, dsn = _AVAILABLE
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.close()
    idx.close()


# ── Tests ───────────────────────────────────────────────────────────────


def test_empty_index_returns_no_hits(index):
    assert index.search([1.0, 0.0, 0.0, 0.0]) == []
    assert index.size() == 0


def test_add_and_search_returns_top_k_by_cosine(index):
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.add("b", [0.0, 1.0, 0.0, 0.0])
    index.add("c", [0.71, 0.71, 0.0, 0.0])
    hits = index.search([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert [h.concept_id for h in hits] == ["a", "c", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits[2].score == pytest.approx(0.0, abs=1e-3)


def test_top_k_truncates(index):
    for i in range(20):
        index.add(f"c{i}", [1.0, 0.0, 0.0, float(i) / 20.0])
    assert len(index.search([1.0, 0.1, 0.0, 0.0], top_k=5)) == 5


def test_min_score_filters(index):
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.add("b", [-1.0, 0.0, 0.0, 0.0])  # cosine = -1 with the query
    hits = index.search([1.0, 0.0, 0.0, 0.0], top_k=10, min_score=0.5)
    assert [h.concept_id for h in hits] == ["a"]


def test_add_replaces_existing_id(index):
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.add("a", [0.0, 1.0, 0.0, 0.0])
    assert index.size() == 1
    hits = index.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert hits[0].concept_id == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)


def test_remove(index):
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.add("b", [0.0, 1.0, 0.0, 0.0])
    index.add("c", [0.71, 0.71, 0.0, 0.0])
    index.remove("b")
    assert index.size() == 2
    hits = index.search([0.0, 1.0, 0.0, 0.0], top_k=3)
    assert {h.concept_id for h in hits} == {"a", "c"}


def test_remove_unknown_is_noop(index):
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.remove("nonexistent")  # must not raise
    assert index.size() == 1


def test_rebuild_replaces_everything(index):
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.add("b", [0.0, 1.0, 0.0, 0.0])
    index.rebuild([
        ("x", [1.0, 1.0, 0.0, 0.0]),
        ("y", [1.0, -1.0, 0.0, 0.0]),
    ])
    assert index.size() == 2
    hits = index.search([1.0, 1.0, 0.0, 0.0], top_k=2)
    assert hits[0].concept_id == "x"


def test_dim_mismatch_silently_skipped(index):
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.add("b", [0.5, 0.5, 0.5])  # 3-dim, mismatched — skipped
    assert index.size() == 1


def test_reconnect_after_drop():
    """A reused PgvectorIndex must survive a Postgres restart / idle
    timeout. Simulated by closing the underlying connection and issuing
    another call."""
    from src.retrieval.pgvector_index import PgvectorIndex
    table = f"pgvec_test_reconn_{uuid.uuid4().hex[:8]}"
    idx = PgvectorIndex(
        dim=4, dsn=_AVAILABLE[1], table_name=table, ensure_extension=False,
    )
    try:
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        # Murder the connection from under the index.
        idx._conn.close()
        # The next call must reconnect transparently.
        hits = idx.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert hits[0].concept_id == "a"
    finally:
        psycopg2, dsn = _AVAILABLE
        conn = psycopg2.connect(dsn); conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.close()
        idx.close()


def test_satisfies_vector_index_interface():
    """The class must remain interchangeable with InMemoryVectorIndex.
    If someone deletes a method by accident, this catches it."""
    from src.retrieval.pgvector_index import PgvectorIndex
    from src.retrieval.vector_index import VectorIndex
    assert issubclass(PgvectorIndex, VectorIndex)
    for name in ("add", "remove", "search", "rebuild", "size"):
        assert hasattr(PgvectorIndex, name), f"missing {name}"
