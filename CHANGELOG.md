# SCM Changelog

Format: each release lists what shipped, why it shipped, what tests verified it, and what broke.

---

## Strategic decisions

### 2026-05-04 — Paper held until product ready

The paper push to arXiv is **deferred** until the product-ready checklist in [`docs/ROADMAP.md`](docs/ROADMAP.md) is fully green (hosted demo + PyPI + npm publish + demo video + tutorial + lighthouse users + repo public). Reasoning: papers without products fade; products with papers compound. The arXiv submission bundle stays staged at `research/arxiv_submission/` ready for fast turnaround once the gate opens. See [`research/arxiv_submission/DO_NOT_SUBMIT.md`](research/arxiv_submission/DO_NOT_SUBMIT.md).

This inverts the original committed sequence (Build → Paper → Publish → Productize) into (Build → Paper → Productize → Publish). The paper still exists — it just waits for the moment it can convert readers into adopters.

---

## v0.7.8 — 2026-05-06

**Theme: SCM Cloud foundations — multi-tenant auth, BYOK, pgvector backend, marketing site, supervisor multi-agent integration.**

The infrastructure layer that turns the open-source self-hosted SCM into a hostable SaaS, while preserving 100% of the self-hosted shape for users who don't want any of it.

### Phase 1: Cloud auth (multi-tenant)

- **New `accounts` + `api_keys` SQLite tables** ([src/core/sqlite_db.py](src/core/sqlite_db.py)). Account holds email + tier + encrypted BYOK LLM config. Keys hold sha256 hash, prefix, scopes, rate limit, last-used timestamp, optional revocation timestamp.
- **`src/cloud/accounts.py`** — account CRUD, API key issuance/revocation/validation, BYOK config with at-rest XOR encryption keyed off `SCM_CLOUD_SECRET_KEY`, tenant-namespacing helper (`account_id::user_id`).
- **`src/cloud/auth_middleware.py`** — FastAPI middleware that gates `/v1/memories/*`, `/v1/wake-summary`, `/v1/users/*`, `/v1/cloud/me/*` behind `Authorization: Bearer scm_live_<keyid>_<secret>`. Stamps `request.state.scm_account` for downstream handlers. Public exceptions: `/v1/health`, `/v1/tools`, `/v1/openapi.json`, `/v1/cloud/accounts*`. Off by default; enabled with `SCM_CLOUD_AUTH=1`.
- **`src/cloud/rate_limit.py`** — in-process token-bucket limiter, per-key, configurable on issuance (default 60/min). Returns 429 on exhaustion.
- **`src/cloud/cloud_api.py`** — `/v1/cloud/*` endpoints:
  - Public: `POST /v1/cloud/accounts`, `POST /v1/cloud/accounts/{id}/keys/initial` (gated by `SCM_CLOUD_SIGNUP_TOKEN` if set)
  - Account-scoped: `GET /v1/cloud/me`, `GET/POST /v1/cloud/me/keys`, `DELETE /v1/cloud/me/keys/{id}`, `POST/DELETE /v1/cloud/me/byok`
- **Tenant namespacing** ([src/integrations/memories_api.py](src/integrations/memories_api.py)) — `_namespace_for_account` rewrites `user_id` → `account_id::user_id` server-side before any storage write. Cross-tenant reads are impossible even when callers guess each other's user_ids.
- **11 new tests** ([tests/test_cloud_auth.py](tests/test_cloud_auth.py)) — signup → mint → use, missing/invalid/revoked → 401, rate limit → 429, BYOK encryption round-trip, **cross-tenant isolation verified**, signup gate via env var, open-mode (auth off) preserved.
- **[docs/API_AUTH.md](docs/API_AUTH.md)** — full operator + integrator guide.

### Phase 2: Production vector backend (pgvector)

- **New module [src/retrieval/pgvector_index.py](src/retrieval/pgvector_index.py)** — `PgvectorIndex` implementing the same `VectorIndex` interface as `InMemoryVectorIndex`. Stores embeddings in Postgres with HNSW cosine indexing (`m=16, ef_construction=64`). Sub-linear ANN at any scale.
- **Auto-detect** ([src/core/long_term_memory.py](src/core/long_term_memory.py)) — `LongTermMemory` resolves the vector backend at init time:
  1. caller-provided argument → use it
  2. `SCM_VECTOR_BACKEND=pgvector` → try PgvectorIndex, fall back to InMemory with a warning if Postgres/pgvector/optional deps unavailable
  3. default → InMemoryVectorIndex
- **Reconnection handling** — `_ensure_conn()` survives Postgres restarts and idle timeouts transparently.
- **Bulk rebuild path** — `rebuild()` truncates + batches into 1K-row inserts so engine startup with N concepts is one transaction, not N.
- **New optional dep** `scm-memory[postgres]` — adds `psycopg2-binary` + `pgvector`.
- **11 new tests** ([tests/test_pgvector_index.py](tests/test_pgvector_index.py)) — full interface coverage, dim-mismatch handling, idempotent add, reconnection after socket drop, interface-contract subclass check. Skip cleanly when `POSTGRES_TEST_DSN` is unset.
- **[docs/POSTGRES_PGVECTOR.md](docs/POSTGRES_PGVECTOR.md)** — operator runbook: when to switch, 5-minute Docker quickstart, schema, tuning knobs, migration from SQLite, monitoring, backups.

### Backwards compatibility

- **Self-hosted users get zero behavior change.** `SCM_CLOUD_AUTH` defaults to `0`, the middleware is a no-op, `/v1/memories/*` accepts unauthenticated requests exactly like v0.7.7. Verified by `test_open_mode_routes_work_without_token`.
- **Default vector backend is unchanged** (`InMemoryVectorIndex`). Postgres is opt-in.
- **No breaking schema changes** — new tables are additive.

### What this enables

| Capability | Before v0.7.8 | After v0.7.8 |
|---|---|---|
| Multi-tenant SaaS deployment | engineer-glue | `SCM_CLOUD_AUTH=1` |
| Per-account API key rotation | impossible | `POST /v1/cloud/me/keys` + `DELETE /v1/cloud/me/keys/{id}` |
| BYOK LLM | environment-variable hack | per-account encrypted config |
| Vector ANN at >100K concepts | RAM-bound, single server | pgvector + HNSW, multi-server |
| Cross-tenant data isolation | manual user_id discipline | enforced server-side by middleware |
| Public unauthenticated mode | only mode | still default for self-hosted |

The four things that gate "SCM Cloud as a paid product" are now plumbed: tenancy, key management, BYOK, and production-scale ANN. Phases 3-4 (Stripe billing + landing page + dashboard) build on top without rewiring.

### Phase 3: LangChain agent-with-tools + supervisor multi-agent

- **New `src/integrations/langchain_tools.py`** — `make_scm_tools(scm_client)` returns 5 `@tool`-decorated callables (`search_memory`, `add_memory`, `consolidate`, `wake_summary`, `get_user_profile`). Drop into `create_agent` or LangGraph nodes; the agent decides when to recall and when to remember.
- **`get_user_profile()` tool** — handles meta queries ("what do you know about me?") that single `search_memory` calls fail on. Internally runs 6 dimension-targeted searches (name, location, profession, preferences, health, hobbies), dedupes, returns a clean bullet list. No prompt-engineering required.
- **Cleaner `search_memory` output** — strips internal diagnostic noise (`[Rank 1] (medium) [salience=0.58]…`) before returning to the LLM. The LLM sees concept descriptions only.
- **Tested patterns** ([tests/agent_with_tools/](tests/agent_with_tools/)):
  - `test_tool_calling_agent.py` — `create_agent` with SCM tools, 5/5 pass after the `get_user_profile` addition
  - `test_multiagent_langgraph.py` — 3-agent LangGraph team sharing one SCM namespace, 4/4 pass + cross-tenant isolation verified
  - `test_supervisor_team.py` — LangChain 1.x supervisor-via-tools pattern: 1 supervisor + 3 specialist workers (profile / task / recall), each with focused SCM tools. 4/4 pass.

### Phase 4: Marketing site + dashboard

- **New landing page** ([src/api/static/landing.html](src/api/static/landing.html)) served at `/` — Anthropic-flavored typography (Instrument Serif + Inter), hero / thesis / how-it-works / code tabs / use cases / paper / pricing / footer.
- **New dashboard** ([src/api/static/app.html](src/api/static/app.html)) served at `/app` — signup, login, API key management, BYOK config, quickstart code. Same warm aesthetic as the landing.
- **`/docs` and `/research` mounted as static** so all internal links from the landing resolve (LangChain guide, API reference, the 35-page paper PDF).
- **Legacy debug UIs moved** to `src/api/static/_legacy/` — no longer served. v0.7.8's `/static/index.html` returns 404.

### Verification

- 117 regression tests passing (cloud auth + circadian + sleep-config + vector index + baseline + deep sleep + forgetting + spreading activation + wake summary + sleep + idle learner)
- 11 pgvector tests pass when Postgres+pgvector is reachable; skip cleanly otherwise
- Brutal LangChain harness still 16/16 against the existing codebase (auth defaulted off, behavior unchanged)
- Tool-calling agent test 5/5 (after `get_user_profile`)
- LangGraph multi-agent test 4/4
- LangGraph supervisor + 3 workers test 4/4
- Fresh-venv smoke test passes

---

## v0.7.7 — 2026-05-05

**Theme: circadian sleep model — replaces the fixed-idle-timer with timezone-aware nightly consolidation.**

Sleep now fires once per night at the user's configured local bedtime — like human sleep — instead of after a fixed idle gap. The previous polling-timer model (`SCM_IDLE_THRESHOLD_SEC`, default 5min) contradicted the product pitch: humans don't consolidate every 5 minutes of inactivity, they consolidate at night. This release matches behavior to the pitch.

### Shipped

- **New module [`src/lifecycle/circadian.py`](src/lifecycle/circadian.py)** — single source of truth for "should this user sleep right now?". Used by both the public `/v1/*` API and the `/demo` UI. `parse_hhmm`, `is_in_window` (handles wrap-around midnight), `resolve_tz` (silent UTC fallback for unknown names), `should_fire(config, now_utc)` (window check + once-per-night guard).
- **New SQLite table `user_sleep_config`** ([src/core/sqlite_db.py](src/core/sqlite_db.py)) — one row per user with `timezone`, `sleep_start`, `sleep_end`, `enabled`, `last_sleep_at`. Defaults to `(UTC, 23:00→07:00, enabled)` when no row exists. New DAO methods: `get_user_sleep_config`, `save_user_sleep_config`, `mark_user_slept`, `list_user_sleep_configs`.
- **New endpoints under `/v1/users/{user_id}/sleep-config`** ([src/integrations/memories_api.py](src/integrations/memories_api.py)) — GET reads the schedule (with `is_default` flag), POST persists partial updates. Validates IANA timezone names and `HH:MM` format with 400 on bad input.
- **`UserEnginePool._sweep_once` rewritten** ([src/integrations/mcp_server.py](src/integrations/mcp_server.py)) — now calls `should_fire(cfg)` for users with explicit configs, falls back to the legacy idle-timer for users with none (preserves backwards compat for deployments running v0.7.6 with `SCM_IDLE_THRESHOLD_SEC` set).
- **New env var `SCM_LEGACY_IDLE_SLEEP=1` (default)** — explicit toggle to keep the legacy idle-timer fallback. Set to `0` to enforce circadian-only on a fresh deployment.
- **Demo router unchanged externally** ([src/api/demo_router.py](src/api/demo_router.py)) — the `/demo` page already used the circadian model since v0.7.6 build; this release brings the public API in line with what the demo was already doing.

### Tests

- 15 new tests in [tests/test_circadian.py](tests/test_circadian.py): `parse_hhmm`, wrap-around windows, once-per-night guard, timezone-aware firing, unknown-tz fallback.
- 6 new integration tests in [tests/test_sleep_config_api.py](tests/test_sleep_config_api.py): default-on-unknown-user, round-trip, partial updates, invalid timezone/HH:MM rejection, disabled means no fire.
- All previously passing regression tests still green.

### Migration notes

- **No action required for existing deployments.** Users who never POST a config keep the legacy idle-timer behavior. The first POST migrates them to circadian.
- **Recommended:** call `POST /v1/users/{your_user_id}/sleep-config` with the user's actual timezone + bedtime once you have that signal (e.g., from browser `Intl.DateTimeFormat().resolvedOptions().timeZone`).
- **`SCM_IDLE_THRESHOLD_SEC` is deprecated** but still honored. Will be removed in v0.8.0.

### Why this matters

The pitch is "memory that works like yours." Fixed-timer sleep is a developer convenience, not a biological metaphor. v0.7.6 already shipped the right model in the demo UI; v0.7.7 brings it to the public API so what's published matches what's pitched.

---

## v0.7.6 — 2026-05-04

**Theme: native vector index — SCM is now its own retrieval store, no third-party memory layer required.**

The concept graph already stored embeddings; spreading activation already used them as a fallback. What was missing was a real ANN index — at 10K+ concepts the legacy O(n) cosine scan eats itself. Now built in.

### Shipped

- **New module [`src/retrieval/vector_index.py`](src/retrieval/vector_index.py)** — pluggable `VectorIndex` interface + `InMemoryVectorIndex` (numpy-only, zero new dependencies). Cosine similarity, unit-normalized, top-k via `argpartition`, swap-with-last-row deletion to keep the matrix contiguous.
- **`LongTermMemory.vector_index`** — every LTM owns one. `_sync_concept` adds incrementally, `remove_concept(soft=False)` removes. Auto-instantiated; pass `vector_index=` to override.
- **`SpreadingActivationRetriever` uses ANN for seed selection** — when an index is wired (the default in the standard ChatEngine), the embedding-fallback path skips the O(n) cosine loop and queries the index directly. Falls through to the legacy scan if the index errors.
- **`ChatEngine._load_session` rebuilds the index from restored embeddings** — single bulk `rebuild()` instead of n incremental adds during cold start.
- **13 new tests** in [`tests/test_vector_index.py`](tests/test_vector_index.py): add/remove/search/rebuild, idempotent add, dim-mismatch handling, top-k truncation, min_score filtering, swap-with-last-row delete reuse.
- **Benchmark** in [`tests/bench_vector_index.py`](tests/bench_vector_index.py).

### Measured impact

| Concepts | Legacy linear cosine | ANN (this release) | Speedup |
|---|---|---|---|
| 100 | 2.93 ms | 0.037 ms | **79×** |
| 1,000 | 26.83 ms | 0.054 ms | **498×** |
| 10,000 | 264.78 ms | 0.504 ms | **526×** |
| 100,000 | 3,638.64 ms | 7.513 ms | **484×** |

(384-dim, top-k=10, 50-iter mean. M-series Mac.)

The 100K-concept query goes from 3.6 s (broken UX) to 7.5 ms (instant). At realistic deployment scale, retrieval is no longer the bottleneck.

### What this means for positioning

SCM no longer needs to be paired with an external retrieval store. The vector index is part of the memory layer, owned by `LongTermMemory`, persisted as concept embeddings in SQLite, and rebuilt on engine startup. Drop SCM in, get vector retrieval AND wake/sleep cycles in one library.

### Verification

- 72 regression tests pass (spreading_activation, deep_sleep, baseline_comparison, forgetting_dynamics, sleep, wake_summary, idle_learner)
- 13 new vector_index tests pass
- E2E demo flow ([`tests/test_wake_summary_e2e.py`](tests/test_wake_summary_e2e.py)) still surfaces user facts on cold restart, now with `"10 vectors indexed"` confirming the index restored

---

## v0.7.5 — 2026-05-04

**Theme: the demo flow that the marketing pitch promises actually works on cold restart.**

End-to-end test caught that v0.7.4 — and every prior version — could not produce a meaningful wake-summary after a process restart. The exact scenario the demo video shows ("close the app, come back tomorrow, see what I learned about you") returned generic "Welcome back" boilerplate because four pieces of the persistence pipeline were broken in cascading ways.

### Shipped — ten fixes for cross-session persistence

1. **`SCM_DATA_DIR` env var is now honored** ([src/core/config.py](src/core/config.py)). Was hardcoded to `PROJECT_ROOT/data` despite being documented in CLI/MCP. PyPI-installed packages were writing inside `site-packages`.
2. **`_load_session()` rehydrates whenever data exists** ([src/chat/engine.py](src/chat/engine.py)) — was gated on `session_meta` being present, but `session_meta` was never auto-saved.
3. **`session_meta` auto-saved every turn** ([src/chat/engine.py](src/chat/engine.py)) — closes the loop on bug #2.
4. **Sleep cycle history persists across restart** — added [`save_sleep_record`](src/core/sqlite_db.py) + [`get_sleep_records_since`](src/core/sqlite_db.py) and wire-up in `_load_session`. The `sleep_cycles` SQLite table existed but nothing wrote to it.
5. **`HME_ENABLED` defaults to `true`** ([src/core/config.py](src/core/config.py)) — the spreading-activation-based "smart retrieval" was OFF by default, so all queries fell back to substring match. Question-form queries ("where do I work") never matched declarative storage ("works at startup Filtrum").
6. **Freshness floor in [ForgettingDynamics](src/sleep/forgetting_dynamics.py)** — substantive concepts (importance ≥ 0.20) younger than 1 hour stay ACTIVE no matter what. Without this, fresh user facts got forgotten on the first sleep cycle. Configurable via `freshness_floor_hours` and `freshness_importance_min`.
7. **SelfModel boilerplate tagged `_internal=True`** ([src/consciousness/self_model.py](src/consciousness/self_model.py)) — the 11 capability concepts ("can remember conversations", etc.) and the per-cycle sleep-log concept were polluting user retrieval and outranking actual user facts.
8. **Search filters `_internal` concepts** ([src/integrations/tools.py](src/integrations/tools.py)) — `_search_memory_handler._add()` skips system-tagged concepts.
9. **`_retrieve_hme()` filters `_internal`** ([src/chat/engine.py](src/chat/engine.py)) — same filter applied to the formatted memory_context that gets injected into LLM prompts.
10. **WakeSummaryBuilder context-refresh fallback** ([src/lifecycle/wake_summary.py](src/lifecycle/wake_summary.py)) — when no schemas/curiosity entries exist (typical for first sleep cycle on a small dataset), surfaces top 4 user-attributable facts as "Here's what I have on you". Means the wake-summary moment lands even before the schema extractor has enough data.

### Verification

`tests/test_wake_summary_e2e.py` — end-to-end cold-restart test:

Before v0.7.5:
```
--- WAKE SUMMARY ---
Welcome back. I'm ready when you are.
(7 words, mentions nothing)
```

After v0.7.5:
```
--- WAKE SUMMARY ---
Welcome back. While you were away (0 seconds), I ran 1 sleep cycle.
I consolidated 16 memories, generated 5 associative dreams.
Here's what I have on you:
  • favorite coffee shop is Hello Kristof
  • building data pipelines at Filtrum
  • works at startup Filtrum
  • Alex is a backend engineer
I'm ready when you are.
(56 words, surfaces 4 substantive user facts)
```

Plus regression suite passes: 30 tests across `test_deep_sleep`, `test_sleep`, `test_baseline_comparison`, `test_forgetting_dynamics`. Two existing legacy-behavior tests updated to opt out of the new freshness floor with `freshness_floor_hours=0.0`.

### Lesson

`twine check` validates packaging metadata. Fresh-venv smoke test (added v0.7.4) validates imports. Neither catches "the product's main demo doesn't work end-to-end." Added end-to-end test [`tests/test_wake_summary_e2e.py`](tests/test_wake_summary_e2e.py) to the standard suite — it must pass before any future release that touches persistence, retrieval, or sleep.

---

## v0.7.4 — 2026-05-04

**Theme: First public PyPI release — install-time correctness.**

Hotfix for [`v0.7.3`](#v073--2026-05-04) which was the first PyPI publish but had three install-time bugs caught by fresh-venv testing immediately after upload.

### Shipped

- **Three required dependencies added** to [`pyproject.toml`](pyproject.toml):
  - `sqlalchemy>=2.0` — used unconditionally in `src/core/database.py`
  - `rich>=13.0` — used unconditionally in `src/chat/cli.py`
  - `ollama>=0.1.0` — used unconditionally in `src/llm/__init__.py` (moved from optional `[llm]` extra to required)
- **`scm version` fix** ([`src/cli/main.py:297`](src/cli/main.py)) — was querying `importlib.metadata.version("scm")` but the published name is `scm-memory`, so it always fell through to `(unknown)`

### Why it shipped

`pip install scm-memory` succeeded on v0.7.3 but `from src.chat.engine import ChatEngine` immediately raised `ModuleNotFoundError: No module named 'sqlalchemy'`. Caught in fresh-venv smoke test ~2 minutes after the v0.7.3 upload landed. Bumped, fixed, republished within the same session.

### Verification

- Fresh `python -m venv` + `pip install scm-memory==0.7.4` + `from src.chat.engine import ChatEngine` works
- `scm version` prints `0.7.4`

### Lesson

`twine check` only validates packaging metadata, not runtime imports. Add a "fresh-venv import smoke test" to the pre-publish checklist in [`docs/PUBLISH.md`](docs/PUBLISH.md). Consider a CI job that does `pip install dist/*.whl` and runs `python -c "from src.chat.engine import ChatEngine"` before any tag-triggered upload.

---

## v0.7.3 — 2026-05-04

**Theme: RAM efficiency for multi-tenant deployments.**

### Shipped

- **Process-singleton embedding models** ([src/core/encoder.py](src/core/encoder.py))
  - `SentenceTransformer` cached once per process (~414 MB) instead of per-engine
  - `OllamaEmbeddingModel` cached per `(model, base_url)` pair
  - `OpenAICompatibleEmbeddingModel` cached per `(model, base_url, api_key_suffix)` pair
  - Thread-safe via `_singleton_lock`

### Why it shipped

The brutal multi-agent harness OOMed an 8 GB MacBook Air. Diagnosis: each `ChatEngine` was loading its own copy of sentence-transformers + torch (~414 MB each). 6 engines = 2.5 GB wasted. At 100-user scale: ~40 GB wasted.

### Verification

| Engines built | RSS before | RSS after | Saved |
|---|---|---|---|
| 1 | 414 MB | 414 MB | 0 |
| **5** | **2 GB** | **414 MB** | **~1.6 GB** |
| 10 | 4 GB | 414 MB | ~3.6 GB |
| 100 | 41 GB | 414 MB | ~40 GB |

143/143 focused regression tests still passing.

### Known follow-ups

- Per-user `ChatEngine` still holds its own NetworkX graph + LTM cache. At very large concept counts these add up; future work may add an LRU eviction layer.

---

## v0.7.2 — 2026-05-04

**Theme: User-facing latency.**

### Shipped

- **Async ingest queue** ([src/integrations/mcp_server.py](src/integrations/mcp_server.py))
  - `add_memory` returns 202 in <100 ms; LLM extraction runs in a per-user background worker
  - Opt-in `sync=true` body field forces synchronous behavior (for tests / write-confirmation needs)
  - `wait_for_pending=true` on `search_memory` for read-your-writes consistency
  - `pool.wait_for_pending(user_id, timeout)` public method
  - `pool.fire_sleep_now()` always drains the ingest queue first
- **Embedding auto-detect** ([src/core/encoder.py](src/core/encoder.py))
  - Probes Ollama in 0.5 s; if `nomic-embed-text` / `mxbai-embed-large` / `bge-large` is pulled, prefers it
  - Falls back to sentence-transformers MiniLM if Ollama unreachable
  - Honors `SCM_EMBEDDING_BACKEND` env var when set (no auto-detect override)
- **JS SDK** ([sdk/js/src/index.js](sdk/js/src/index.js)) wires `wait_for_pending` through

### Why it shipped

The brutal multi-agent harness exposed that synchronous LLM extraction blocked every `add_memory` call for 1-3 s (heuristic) or 5-15 s (LLM). For interactive chat this is the difference between "feels broken" and "feels normal."

### Verification

[`tests/brutal_langchain/bench_latency.py`](tests/brutal_langchain/bench_latency.py) measured user-facing latency on 5 sequential adds:

| Metric | Sync (blocking) | Async (queued) | Speedup |
|---|---|---|---|
| p50 | 13,567 ms | **2.4 ms** | **5,561×** |
| p95 | 37,690 ms | **6.2 ms** | **6,097×** |
| Total wall (5 adds) | 95.4 s | **0.016 s** | **5,803×** |

Brutal LangChain harness: 16/16 still pass (with `wait_for_pending=True` on search to preserve write-then-read consistency the test design needs). Wall time roughly the same since each turn awaits prior writes; the win is at the API surface for interactive use, not batch throughput.

### Known follow-ups

- Multi-namespace single-call search (collapse private + shared search calls into one)
- Aggressive hybrid encoder (further reduce LLM extraction frequency)

---

## v0.7.1 — 2026-05-02

**Theme: Forgetting safety + retrieval bug fixes.**

### Shipped

- **`FORGETTING_PROTECT_SALIENCE` default `0.0` → `0.5`** ([src/core/config.py](src/core/config.py))
  - Concepts with `salience_score >= 0.5` are protected from forgetting on the first sleep cycle
  - Prevents user-stated facts (e.g., "I'm allergic to seafood") from being archived seconds after ingestion
  - Legacy aggressive-forgetting behavior available via `FORGETTING_PROTECT_SALIENCE=0.0`
- **`FORGETTING_MIN_REHEARSAL_BEFORE_ARCHIVE` default `0` → `1`**
  - Concepts must survive at least one sleep cycle before they can be archived
- **`_context_gate` None-safety** ([src/retrieval/spreading_activation.py](src/retrieval/spreading_activation.py:382))
  - Was: `context_tags['person'].lower()` — crashed when working memory was empty post-consolidation
  - Now: `isinstance(...)` guarded; spreading activation no longer breaks after deep-sleep
- **Cue-match-dominant ranking** in spreading activation
  - Was: 85% consolidation + 15% cue-match — well-rehearsed background concepts beat the actual answer
  - Now: 70% cue-match + 30% consolidation
- **`SCMClient` method names** ([src/integrations/langchain_adapter.py](src/integrations/langchain_adapter.py))
  - Renamed `add` → `add_memory`, `search` → `search_memory` (aliases preserved)
  - Critical bug: prior versions never actually called SCM in the brutal LangChain harness due to method-name mismatch — the LLM's history was doing the work
- **`UserEnginePool` engines now use `sandbox_mode=True`**
  - Multi-user isolation: per-engine in-memory graph; no cross-user leak via shared SQLite
- **3 new regression tests** for `_context_gate` None-safety in [tests/test_spreading_activation.py](tests/test_spreading_activation.py)
- **5 forgetting tests updated** to opt in to legacy aggressive behavior via `protect_salience=0.0`
- **phase4 dataset distractors** now have explicit `salience_score=0.1` instead of relying on default

### Why it shipped

The first multi-day brutal harness run on a real LangChain agent (16/16 passing) hid four real bugs because of the SCMClient method-name mismatch. Each fix is documented in [docs/BUG_LOG.md](docs/BUG_LOG.md).

### Verification

322/322 full regression passing.

---

## v0.7.0 — 2026-05-01

**Theme: Product surface (MCP + REST API + tool definitions + JS SDK).**

### Shipped

- **MCP server** ([src/integrations/mcp_server.py](src/integrations/mcp_server.py))
  - Five tools: `add_memory`, `search_memory`, `consolidate`, `wake_summary`, `forget`
  - Stdio transport (Claude Desktop, Cursor) + HTTP transport
  - `UserEnginePool` with per-user `ChatEngine` + idle-aware sleep sweeper
- **`/v1` REST API** ([src/integrations/memories_api.py](src/integrations/memories_api.py))
  - REST endpoints mounted on the existing FastAPI app
  - Tool definitions exported in OpenAI / Anthropic / Gemini / OpenAPI 3.1 formats
  - `/v1/openapi.json` for ChatGPT Custom GPT Actions integration
- **Tool definitions module** ([src/integrations/tools.py](src/integrations/tools.py))
  - Single source of truth for the 5 SCM tools, exported in 4 formats
- **LangChain adapter** ([src/integrations/langchain_adapter.py](src/integrations/langchain_adapter.py))
  - `SCMClient` (HTTP wrapper, langchain-free) + `SCMMemory` (BaseChatMemory adapter)
- **JavaScript SDK** ([sdk/js/](sdk/js/))
  - Full TypeScript types
  - Works in Node 18+, Bun, browsers, Cloudflare Workers, Vercel Edge
- **Examples directory** ([examples/](examples/))
  - 01_quickstart.py, 02_wake_summary.py, 03_with_ollama.py
- **CLI tool** ([src/cli/](src/cli/))
  - `scm chat / sleep / wake-summary / status / serve / mcp / version / config` — single entry point installed via pip

### Why it shipped

Pivoted from "research code with web UI" to "memory layer behind any agent." Previous v0.6 was research-grade only; v0.7.0 is the first version a third-party developer could integrate.

### Verification

129/129 focused regression passing. Multi-format tool exports verified. CLI smoke tested.

---

## v0.6.x — 2026-04-01 to 2026-05-01

**Theme: Phase 7 autonomous-learning architecture (M1-M6).**

### Shipped (across the v0.6 series)

- **M1: IdleLearner daemon** ([src/lifecycle/idle_learner.py](src/lifecycle/idle_learner.py))
  - Threaded background daemon firing sleep cycles on idle
  - Policy-gated; crash-safe state via atomic JSON file
- **M2: Cross-session memory pool** ([src/core/cross_session_pool.py](src/core/cross_session_pool.py))
  - Borrows episodes from prior sessions for multi-day consolidation
  - `include_current_session` flag for single-user multi-day deployments (brutal-fix)
- **M3: Schema extractor** ([src/sleep/schema_extractor.py](src/sleep/schema_extractor.py))
  - 4-pass detection: REPETITION, COOCCUR, TEMPORAL_CADENCE, TRAJECTORY
  - SHA-1 stable schema concept IDs (brutal-fix to prevent unbounded duplicate growth)
- **M4: Wake-summary builder** ([src/lifecycle/wake_summary.py](src/lifecycle/wake_summary.py))
  - Template-driven narrative from sleep cycle outputs
- **M5: Curiosity engine** ([src/lifecycle/curiosity.py](src/lifecycle/curiosity.py))
  - Pluggable sources: StaticDictionarySource, LocalDocsSource, LLMSource
- **M6: Lifecycle policy** ([src/lifecycle/lifecycle_policy.py](src/lifecycle/lifecycle_policy.py))
  - BatteryPolicy, CPULoadPolicy, CompositePolicy via psutil
- **De-hardcoding via LinguisticResources** ([src/core/linguistic_resources.py](src/core/linguistic_resources.py))
  - All locale-specific keywords / regex / templates moved to [src/core/locales/en.json](src/core/locales/en.json)
- **ALB benchmark spec + framework** ([research/benchmarks/alb/](research/benchmarks/alb/))
  - 7 metrics, pre-registered hypotheses, statistical methodology
  - SCM adapter + 2 pilot personas
- **Brutal testing harness** ([tests/brutal/](tests/brutal/))
  - Persona-driven multi-day simulation; uncovered 4 real bugs that 170+ unit tests missed
- **35-page paper** ([research/SCM_Final_Paper.tex](research/SCM_Final_Paper.tex))
  - Title: "SCM: Autonomous Lifelong Learning for Language Agents via Sleep-Stage Memory Consolidation"
  - 5 figures (architecture, Phase 7 layer, sleep timeline, sleep-gain, NIAL bar chart)
  - Reframed around the wake/sleep two-phase thesis

### Verification

172/172 unit tests passing. Brutal harness runs to completion across 4 tiers.

---

## v0.5 and earlier

**Theme: Phases 1-6 wake-time substrate.**

Phases 1-5 (selective encoding, event binding, spreading-activation retrieval, dual-mode sleep, contradiction versioning) plus Phase 6 architectural fixes (forgetting floor, hybrid encoder, sleep-time paraphrase). See [docs/PHASE1_COMPLETE_DOCUMENTATION.md](docs/PHASE1_COMPLETE_DOCUMENTATION.md) and the LoCoMo / LoCoMo++ benchmark results in the paper for details.

---

## Versioning policy

- **Patch (0.7.x)**: bug fixes, defaults changes, internal refactors. No API breaks.
- **Minor (0.x)**: new features, new modules. May change defaults; preserves API.
- **Major (1.0+)**: not yet released. Will signal API stability commitment.
