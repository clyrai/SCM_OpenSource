# SleepAI Project Status

**Last updated:** May 1, 2026  
**Current state:** Phases 1-5 are complete and validated. Phase 6 hardening is in progress with guardrails, human-memory benchmarks, demo scripts, human-style conversational polish, the professional sleep-consolidation ranking path, and paper-ready baseline, long-horizon, and reproducibility packs now added.

This document is the live source of truth for what we have completed, what is stable, and what is still left.

## Completed

### Baseline Stability

- Added an offline-safe embedding path in `src/core/encoder.py` with a deterministic hash fallback when transformer models are unavailable.
- Added a chat-generation fallback in `src/chat/engine.py` so responses still work when the LLM backend is down.
- Fixed session wiring in `src/api/chat.py` so `session_id` is passed into `ChatEngine`.
- Added SQLite relation persistence in `src/core/sqlite_db.py` and integrated it through `src/core/long_term_memory.py`.
- Restored session state more fully on startup, including relation edges and working-memory context.

### HME Phase 1

- Added selective encoding and fast-grasp behavior.
- Added `AttentionGate`, `EncodeIntensity`, salience scoring, grasp scoring, and prediction error tracking.
- Extended `Concept` and `Episode` with HME metadata and context tags.
- Wired HME Phase 1 into the chat ingestion pipeline.

### HME Phase 2

- Added `EventCompiler` for structured episode extraction.
- Added `AssociationBinder` for event-to-memory binding.
- Stored event metadata on episodes and reinforced relations during ingestion.
- Populated `context_tags` during ingestion so retrieval can use person/session/task context.

### HME Phase 3

- Added `SpreadingActivationRetriever` for cue-driven associative recall.
- Added `HypothesisRanker` for scoring and formatting retrieval candidates.
- Wired HME retrieval into `src/chat/engine.py` behind the `HME_ENABLED` flag.
- Added context-gated retrieval using session and person tags.

### HME Phase 4

- Added `MicroSleep` for lightweight replay, duplicate merging, and reinforcement.
- Added `DeepSleep` for full consolidation, downscaling, synthesis, and pruning.
- Extended `SleepTrigger` with micro-sleep and deep-sleep decision logic.
- Refactored `SleepCycleOrchestrator` to route by mode while preserving legacy behavior.
- Wired mode-aware sleep handling into `ChatEngine` and the sleep APIs.
- Added Phase 4 metrics and benchmarks in `tests/phase4_metrics.py`.

### HME Phase 5

- Added `ForgettingDynamics` for adaptive retention scoring and trace-specific decay.
- Added contradiction-safe versioning with validity windows and lineage markers.
- Updated retrieval, sleep, and persistence paths to respect active versus archived versions.

### HME Phase 6 (In Progress)

- Added full-stack Phase 6 guardrail harness in `tests/phase6_guardrails.py`.
- Added human-memory benchmark harness in `tests/human_memory_benchmark.py`.
- Added regression tests for human-memory behaviors in `tests/test_human_memory_behavior.py`.
- Added product/research demo scenario runner in `research/demos/phase6_demo.py`.
- Added guided in-product user trial panel in `src/api/static/index.html` (`Human-Like Demo`).
- Added live user-trial runbook in `docs/HUMAN_LIKE_USER_TRY.md`.
- Added human-style fallback response polish in `src/chat/engine.py` so memory answers read more like a person, including updated preference parsing for `I prefer ...`.
- Added a rolling paper-ready implementation log in `docs/SCM_RESEARCH_LOG.md`.
- Added a paper-ready baseline comparison report in `docs/SCM_BASELINE_COMPARISON.md` with the machine-readable artifact `research/metrics/baseline_comparison_latest.json`.
- Added a paper-ready long-horizon evidence pack in `docs/SCM_LONG_HORIZON.md` with the machine-readable artifact `research/metrics/long_horizon_latest.json`.
- Updated long-horizon reporting to use family-aware duplicate recall as the release gate while preserving strict duplicate-pair recall as a stress appendix.
- Hardened deep-sleep replay sampling in `src/sleep/deep_sleep.py` so it replays a time-ordered coverage slice instead of only the shuffled tail.
- Made REM dream sequencing deterministic in `src/sleep/rem.py` so repeated runs with the same state stay reproducible.
- Added shared sleep-centric consolidation scoring in `src/core/memory_scoring.py` and routed retrieval ranking through it so pre-sleep distractors stay neutral while replayed memories gain priority.
- Tightened selective forgetting in `src/sleep/forgetting_dynamics.py` so sparse unrehearsed traces can be archived instead of lingering as strong distractors.
- Added a reproducibility runner in `tests/reproducibility_pack.py` that executes the baseline comparison, long-horizon benchmark, and Phase 6 guardrails with artifact fingerprints and per-run logs.
- Added reproducibility report docs in `docs/SCM_REPRODUCIBILITY_PACK.md` with machine-readable output at `research/reproducibility/reproducibility_pack_latest.json`.
- Added product diagnostics endpoint in `src/api/chat.py`: `GET /chat/product/{session_id}`.
- Added product report endpoint in `src/api/chat.py`: `GET /chat/product-report/{session_id}` (runtime + benchmark signals + readiness score).
- Added backend executable demo endpoint in `src/api/chat.py`: `POST /chat/product-demo/{session_id}` (with `/chat/demo/{session_id}` retained for compatibility).
- Added one-call backend smoke endpoint in `src/api/chat.py`: `POST /chat/backend-smoke/{session_id}` (demo + memory/report checks + overall pass/fail).
- Preserved contradiction version lineage through deep sleep by forwarding retired concepts into the sleep sync payload, so the product demo now keeps versioning visible after consolidation.
- Set API chat sessions to run with HME pipeline enabled by default for product experience.
- Fixed warning source in `tests/test_memory_pipeline.py` (`PytestReturnNotNoneWarning`).

### Validation

- Phase 3, Phase 4, and Phase 5 targeted tests pass.
- Phase 4 metrics report is generated at `research/metrics/phase4_micro_deep_latest.json`.
- New Phase 6 artifacts now generate at:
  - `research/metrics/phase6_guardrails_latest.json`
  - `research/metrics/phase6_human_memory_latest.json`
  - `research/metrics/phase6_demo_latest.json`
- Baseline comparison artifact now generates at `research/metrics/baseline_comparison_latest.json`.
- Long-horizon artifact now generates at `research/metrics/long_horizon_latest.json`.
- Reproducibility artifact now generates at `research/reproducibility/reproducibility_pack_latest.json`.
- Research traceability notes are now tracked in `docs/SCM_RESEARCH_LOG.md`.
- The 10-seed brutal Phase 4 sweep on `pair_count=96` now passes cleanly:
  - baseline disambiguation recall: `0.0`
  - micro-sleep disambiguation recall: `1.0`
  - deep-sleep disambiguation recall: `0.9021`
  - deep-sleep pass rate: `1.0`
  - deep-noise retention: `0.0`
  - refreshed artifact: `research/metrics/phase4_seed_sweep_brutal_latest.json`

## What Is Left

### Hardening

- Complete full brute-force matrix runs through the new Phase 6 guardrail harness.
- Finalize release notes and operation guide for release-candidate signoff.

### Longer Roadmap

- Continuous existence / background processing.
- Multi-modal memory for vision and audio.
- Predictive self-model improvements.
- Neuromorphic hardware bridge work.

## Best Next Step

Run the small user pilot with scripted tasks and collect qualitative evidence now that the reproducibility pack is green.
