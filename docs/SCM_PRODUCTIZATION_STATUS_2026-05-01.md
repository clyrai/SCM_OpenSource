# SCM Productization Status (May 1, 2026)

## What We Implemented In This Pass

1. Runtime profiles (`chatbot`, `agent`, `research`) with explicit memory/sleep tuning:
- Added in [src/core/profiles.py](/Users/saish/Downloads/SleepAI/src/core/profiles.py)
- Exposed via API: `GET /chat/profiles`

2. Sandbox mode (ephemeral memory, no persistence):
- `ChatEngine` now supports `sandbox_mode` and persistence control.
- `LongTermMemory` now supports runtime persistence toggle.
- Session bootstrap endpoint: `POST /chat/session` with `sandbox=true`.

3. Memory export/import for backup/migration/cold-start:
- Core graph serialization in `LongTermMemory.export_memory()` / `import_memory()`.
- API endpoints:
  - `GET /chat/memory-export/{session_id}`
  - `POST /chat/memory-import/{session_id}`

4. Observability for production:
- Structured JSON request/event logging.
- Prometheus-compatible `/metrics` endpoint.
- Chat/sleep/session counters and gauges (when `prometheus_client` is available).

5. Python SDK package scaffold:
- New `scm` package with `SCMEngine` wrapper and profile listing.
- Packaging metadata added via [pyproject.toml](/Users/saish/Downloads/SleepAI/pyproject.toml).

6. Regression tests:
- [tests/test_product_runtime_api.py](/Users/saish/Downloads/SleepAI/tests/test_product_runtime_api.py)
- [tests/test_scm_sdk.py](/Users/saish/Downloads/SleepAI/tests/test_scm_sdk.py)

## What This Enables For Product

- Single-memory runtime configurable per workload type.
- Fully ephemeral session mode for safe experiments and demos.
- Import/export memory snapshots between environments.
- Observable API behavior for ops dashboards and SLO tracking.
- SDK-first embedding into chatbots, frameworks, and agent runtimes.

## Remaining Launch Gaps

1. Session-scoped persistence partitioning:
- Current SQL schema is global for concepts/relations.
- Next: add explicit `session_id` partitioning for safer multi-tenant hosting.

2. Authn/authz + multi-tenant API hardening:
- Add API keys/JWT and tenant isolation on memory endpoints.

3. Rate-limits + abuse protection:
- Add middleware-based throttling and endpoint budgets.

4. Release packaging + CI publish:
- Build/publish flow for PyPI package (`scm`) with semantic versioning and changelog gates.

5. Operator dashboard:
- Add Grafana-ready dashboards and SLO alert templates around `/metrics`.
