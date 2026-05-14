"""Postgres + pgvector backend for the VectorIndex interface.

Drop-in replacement for ``InMemoryVectorIndex`` when you've outgrown
single-server numpy. The interface is identical (`add`, `remove`,
`search`, `rebuild`, `size`); only the storage moves to Postgres.

When to use which backend:

  • InMemoryVectorIndex: ≤100K concepts, single server, dev/laptop.
  • PgvectorIndex: ≥100K concepts, multi-server, multi-tenant cloud.

Indexing strategy:

  • Default = HNSW (hnsw_m=16, hnsw_ef_construction=64) on cosine distance.
    Sub-linear search. Build cost is amortized over inserts.
  • For very small tables (<10K rows) pgvector's planner may pick a seq
    scan anyway, which is fine — HNSW becomes essential past ~50K rows.

Cross-tenancy: this index does NOT enforce account isolation. The caller
(LongTermMemory + the cloud auth middleware) namespaces concept_ids
under the account before they reach this layer. We intentionally don't
add account_id to the index because the index is a pure id→vector store;
namespacing belongs at the layer that owns identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .vector_index import VectorHit, VectorIndex


def _import_pgvector():
    """Defer the pgvector + psycopg2 import. Both are optional deps."""
    try:
        import psycopg2  # type: ignore
        from pgvector.psycopg2 import register_vector  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PgvectorIndex requires the 'postgres' optional dependency. "
            "Install with: pip install scm-memory[postgres]"
        ) from e
    return psycopg2, register_vector


class PgvectorIndex(VectorIndex):
    """ANN backed by Postgres + pgvector. Thread-safe; uses one cursor
    per call (acceptable at the rates a single SCM server hits this).

    Args:
        dsn:        Postgres connection string. Read from `POSTGRES_DSN` /
                    `DATABASE_URL` env if None.
        dim:        Embedding dimension. Required at first table-create.
                    Mismatched dim later is a hard error (would corrupt
                    the index).
        table_name: Override the default `scm_concept_vectors`. Useful
                    when running multiple SCM instances against one DB.
        ensure_extension: Auto-`CREATE EXTENSION IF NOT EXISTS vector`
                    on connect. Disable in production where the role
                    doesn't have CREATE EXTENSION privs (run it manually
                    once with a superuser).
    """

    def __init__(
        self,
        dim: int,
        dsn: Optional[str] = None,
        table_name: str = "scm_concept_vectors",
        ensure_extension: bool = True,
    ):
        self._psycopg2, self._register_vector = _import_pgvector()
        if dsn is None:
            import os
            dsn = os.environ.get("POSTGRES_DSN") or os.environ.get("DATABASE_URL")
            if not dsn:
                from ..core.config import DATABASE_URL
                dsn = DATABASE_URL
        self._dsn = dsn
        self._dim = int(dim)
        self._table = table_name
        self._ensure_extension = bool(ensure_extension)
        self._conn = None
        self._connect_and_init()

    # ── connection lifecycle ─────────────────────────────────────────────

    def _connect_and_init(self) -> None:
        self._conn = self._psycopg2.connect(self._dsn)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            if self._ensure_extension:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    concept_id  TEXT PRIMARY KEY,
                    embedding   vector({self._dim}) NOT NULL,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # HNSW with cosine distance — pgvector calls it `vector_cosine_ops`.
            # IF NOT EXISTS prevents the cost-prohibitive rebuild on every
            # process start. Drop + rebuild manually if you change m/ef.
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table}_hnsw_cos
                ON {self._table} USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        # Register the pgvector type adapter on this connection so we can
        # pass numpy arrays / lists straight in.
        self._register_vector(self._conn)

    def _ensure_conn(self):
        """Reconnect if the underlying socket has gone away (Postgres
        idle-timeout, restart, etc.)."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._connect_and_init()

    # ── VectorIndex API ──────────────────────────────────────────────────

    def add(self, concept_id: str, embedding: Sequence[float]) -> None:
        if embedding is None or len(embedding) != self._dim:
            return  # silently skip — same contract as InMemoryVectorIndex
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table} (concept_id, embedding, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (concept_id) DO UPDATE SET
                    embedding  = EXCLUDED.embedding,
                    updated_at = now()
                """,
                (concept_id, list(embedding)),
            )

    def remove(self, concept_id: str) -> None:
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE concept_id = %s",
                (concept_id,),
            )

    def search(
        self,
        embedding: Sequence[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[VectorHit]:
        if embedding is None or len(embedding) != self._dim:
            return []
        self._ensure_conn()
        # pgvector's <=> is cosine *distance* (0 = identical, 2 = opposite).
        # Cosine *similarity* = 1 - distance. We filter by min_score (sim).
        max_distance = 1.0 - max(-1.0, min(1.0, float(min_score)))
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT concept_id, 1 - (embedding <=> %s) AS sim
                FROM {self._table}
                WHERE embedding <=> %s <= %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (list(embedding), list(embedding), max_distance,
                 list(embedding), int(top_k)),
            )
            rows = cur.fetchall()
        return [VectorHit(concept_id=r[0], score=float(r[1])) for r in rows]

    def rebuild(
        self,
        items: Iterable[Tuple[str, Sequence[float]]],
    ) -> None:
        """Bulk replace. Used on engine startup when restoring concepts
        from the primary store. Faster than n incremental inserts because
        we batch into one transaction."""
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE {self._table}")
            batch = []
            for cid, emb in items:
                if emb is None or len(emb) != self._dim:
                    continue
                batch.append((cid, list(emb)))
                if len(batch) >= 1000:
                    cur.executemany(
                        f"""
                        INSERT INTO {self._table} (concept_id, embedding)
                        VALUES (%s, %s)
                        """,
                        batch,
                    )
                    batch = []
            if batch:
                cur.executemany(
                    f"""
                    INSERT INTO {self._table} (concept_id, embedding)
                    VALUES (%s, %s)
                    """,
                    batch,
                )

    def size(self) -> int:
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
