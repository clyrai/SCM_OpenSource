# Postgres + pgvector for production SCM

When to switch from the default in-memory vector index to Postgres-backed ANN, and how to do it.

---

## When to switch

| Concept count | Backend | Rationale |
|---|---|---|
| **≤ 100K** | `InMemoryVectorIndex` (default) | numpy matrix multiplication is fast at this scale (~0.5 ms per query at 10K, ~7.5 ms at 100K) |
| **> 100K** | `PgvectorIndex` | numpy gets cache-thrashy past 100K; HNSW indexes stay sub-linear |
| **Multi-server** | `PgvectorIndex` | InMemory is per-process; multiple SCM instances can't share an in-memory index |
| **Multi-tenant cloud** | `PgvectorIndex` | Per-account namespacing in shared Postgres; tenancy enforced by middleware (see [API_AUTH.md](API_AUTH.md)) |

The interface is identical (`VectorIndex` ABC in [`src/retrieval/vector_index.py`](../src/retrieval/vector_index.py)); only the storage moves.

---

## Quick start (5 minutes)

### 1. Run a Postgres + pgvector container

The official `pgvector/pgvector` image has the extension preinstalled:

```bash
docker run -d --name scm-pg \
    -p 5433:5432 \
    -e POSTGRES_PASSWORD=changeme \
    -e POSTGRES_DB=scm \
    -v scm_pg_data:/var/lib/postgresql/data \
    pgvector/pgvector:pg16
```

### 2. Install the optional dep

```bash
pip install scm-memory[postgres]
# Adds psycopg2-binary + pgvector
```

### 3. Point SCM at the database

```bash
export POSTGRES_DSN="postgresql://postgres:changeme@localhost:5433/scm"
export SCM_VECTOR_BACKEND=pgvector
export SCM_EMBEDDING_DIM=768   # 384 for MiniLM, 768 for nomic-embed-text
scm serve --port 8000
```

On startup you should see:

```
[LTM] vector backend = PgvectorIndex (dsn=postgresql://...:5433/scm, dim=768)
SleepAI ready!
```

If pgvector isn't available, SCM logs a warning and silently falls back to `InMemoryVectorIndex`:

```
[LTM] SCM_VECTOR_BACKEND=pgvector requested but unavailable
      (ImportError: ...); falling back to InMemoryVectorIndex
```

That's intentional — a misconfigured deploy degrades to working ANN rather than crashing.

### 4. Verify

```bash
psql "$POSTGRES_DSN" -c "\d scm_concept_vectors"
# Should show the table with concept_id, embedding (vector), updated_at
# and an HNSW index on the embedding column.

psql "$POSTGRES_DSN" -c "SELECT count(*) FROM scm_concept_vectors;"
# After a few /v1/memories POSTs, this should be > 0.
```

---

## What the index actually does

Per-server: when SCM ingests a memory, the embedding goes into both the existing concept storage AND `scm_concept_vectors`. When a `/v1/memories/search` call needs ANN seeds for spreading activation, it queries pgvector instead of doing an O(n) cosine scan over all concepts.

Schema:

```sql
CREATE TABLE scm_concept_vectors (
    concept_id  TEXT PRIMARY KEY,
    embedding   vector(768) NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON scm_concept_vectors USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

The HNSW index is what makes search sub-linear. Build cost is amortized over inserts; once built, queries are <10 ms even at millions of rows.

### Distance metric

We use **cosine distance** (`vector_cosine_ops`). Embeddings are normalized to unit length implicitly by SCM's encoder, so cosine and inner-product are equivalent — pick whichever your operations team prefers, but stick with one.

To change to L2:

```sql
DROP INDEX scm_concept_vectors_hnsw_cos;
CREATE INDEX ON scm_concept_vectors USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);
```

Then update `PgvectorIndex` to use `<->` (L2) instead of `<=>` (cosine).

---

## Tuning knobs

| Knob | Default | When to change |
|---|---|---|
| `m` (HNSW connectivity) | 16 | Bump to 32 for higher recall on >10M concepts (more memory) |
| `ef_construction` | 64 | 128 for slightly better index quality at build time (slower inserts) |
| `ef_search` (per-query) | uses index default | `SET LOCAL hnsw.ef_search = 200;` before a query for higher recall |
| Connection pooling | none in PgvectorIndex | Wrap with PgBouncer in front of Postgres for multi-process SCM |

The defaults are calibrated for the typical SCM Cloud workload (1K-10M concepts, p95 query latency target <50 ms). Don't touch unless you've measured a problem.

---

## Migration: SQLite → Postgres

If you've been running self-hosted SCM with SQLite and want to switch:

```bash
# 1. Stop SCM
kill $(cat ~/.scm/server.pid)

# 2. Export concepts from SQLite using SCM's built-in export
scm export --output ./scm-snapshot.json
# (writes concepts + relations + episodes as JSON)

# 3. Start the new SCM with Postgres backend
export POSTGRES_DSN="..." SCM_VECTOR_BACKEND=pgvector
scm serve --port 8000 &

# 4. Import the snapshot into the new instance
scm import --input ./scm-snapshot.json
```

The import re-runs the encoder so embeddings end up in pgvector automatically. Sleep history is preserved; idle thresholds reset.

---

## Self-hosted vs cloud differences

| | Self-hosted (default) | SCM Cloud |
|---|---|---|
| Vector backend | `InMemoryVectorIndex` | `PgvectorIndex` |
| Storage | SQLite at `SCM_DATA_DIR` | Postgres |
| Multi-tenancy | Single user_id namespace | Account-level isolation enforced server-side |
| Auth | None (open `/v1/*`) | Bearer API keys via `SCM_CLOUD_AUTH=1` |
| Scale | One process | Multi-instance behind a load balancer |

You can mix: run SCM Cloud auth (`SCM_CLOUD_AUTH=1`) without pgvector if you have <100K concepts total. Or run pgvector without cloud auth if you're a single-tenant operator who just wants the scale.

---

## Operator runbook

### Backups

`pg_dump` works as expected. The `scm_concept_vectors` table is small relative to the embeddings themselves (sentence-transformers concept descriptions are usually larger than the 768-float vector). Daily incremental + weekly full is overkill for most deployments; weekly full is fine.

### Index rebuild

Rebuilding an HNSW index isn't free. If you must (e.g., changing `m`):

```sql
BEGIN;
DROP INDEX scm_concept_vectors_hnsw_cos;
-- DDL is fast; the cost is in the next CREATE
CREATE INDEX scm_concept_vectors_hnsw_cos
    ON scm_concept_vectors USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 128);
COMMIT;
```

For online rebuild, use `CREATE INDEX CONCURRENTLY` (slower, no table-level lock) and drop the old index after.

### Monitoring

- `pg_stat_user_indexes`: confirm the HNSW index is being scanned (`idx_scan` > 0)
- `pg_stat_database`: connection count, query throughput
- Long-tail queries: `pg_stat_statements` with `query LIKE '%scm_concept_vectors%'`

### Disk usage

Each row is ~3 KB at dim=768 (the embedding alone is 768 × 4 bytes = 3 KB plus index overhead). Plan ~5 KB per concept. 1M concepts ≈ 5 GB. SSD recommended.

---

## Tests

The pgvector tests skip cleanly when no Postgres is reachable. To run them:

```bash
docker run --rm -d --name scm-pg-test \
    -p 5433:5432 -e POSTGRES_PASSWORD=test -e POSTGRES_DB=scm_test \
    pgvector/pgvector:pg16

POSTGRES_TEST_DSN="postgresql://postgres:test@localhost:5433/scm_test" \
    pytest tests/test_pgvector_index.py -v

docker rm -f scm-pg-test
```

11 tests cover: add/search/remove/rebuild/size, top-k truncation, min_score filtering, idempotent add, dim-mismatch handling, reconnect after socket drop, interface contract.

CI integration: add `pgvector/pgvector:pg16` as a service in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) and set `POSTGRES_TEST_DSN` in the job env. The skip mechanism means existing developer machines without Postgres don't break.
