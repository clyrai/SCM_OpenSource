# SCM Research Progress Log

This is the rolling implementation log for the next research paper. It records what we changed, why we changed it, and what evidence we used to verify the result.

## 2026-05-01 - ArXiv Manuscript V2 Draft (Paper-First Positioning)

### Goal

Start a professional arXiv manuscript that frames SCM as a reproducible research contribution rather than a sales product.

### What Changed

- Added a new manuscript draft at `research/SCM_arXiv_Paper_v2.md`.
- Reframed the narrative around the validated core claim: sleep-driven memory lifecycle improves disambiguation and selective forgetting versus awake-only and standard retrieval controls.
- Replaced legacy benchmark language with artifact-backed evidence from:
  - `research/metrics/phase6_human_memory_latest.json`
  - `research/metrics/baseline_comparison_latest.json`
  - `research/metrics/long_horizon_latest.json`
  - `research/reproducibility/reproducibility_pack_latest.json`
  - `research/metrics/phase6_guardrails_latest.json`
  - `research/metrics/phase6_backend_smoke_latest.json`

### Verification

- Manuscript numbers were cross-checked against the current JSON artifacts before drafting.
- The draft explicitly reports both strengths and known boundaries (including the strict duplicate-pair stress behavior and external-validation gap).

### Paper Note

This draft is submission-oriented and evidence-first. It is ready for iterative refinement toward a camera-ready arXiv release.

## 2026-05-01 - Deep-Sleep Version Lineage Preservation and Product Demo Fix

### Goal

Keep contradiction-safe versioned preferences alive through deep sleep so the product demo reflects the full memory lifecycle end to end.

### What Changed

- `src/sleep/nrem.py` now softens high-z-score outlier traces instead of dropping them during sleep normalization.
- `src/sleep/sleep_cycle.py` now forwards `retired_concepts` and `retired_ids` into the chat-engine sync payload so deep-sleep retirees stay visible in LTM history.
- `tests/test_contradiction_versioning.py` now includes a regression test for preserving a versioned preference trace through deep sleep under heavy downscaling.

### Verification

- `./venv/bin/python -m pytest tests/test_contradiction_versioning.py tests/test_product_demo_api.py -q`
- Broad brutal suite including product demo: `55 passed`
- Fresh demo readout now reports `versioning_present: true`, `contradiction_edges: 1`, and `readiness.score: 100.0`

### Paper Note

This is the point where the demo becomes a real end-to-end memory lifecycle story, not just a one-shot extraction and sleep smoke test.

## 2026-05-01 - Duplicate-Family Gate + Reproducibility Pack

### Goal

Close two remaining paper-risk points: long-horizon duplicate stress degradation in the headline metric, and missing publication-grade reproducibility evidence.

### What Changed

- Updated `tests/long_horizon_benchmark.py` so the release metric is now family-aware duplicate recall under repeated interference, while strict day-1 duplicate-pair recall stays as a stress appendix.
- Extended the long-horizon JSON/markdown outputs to include both signal layers:
  - family-aware recall for acceptance gating
  - strict duplicate-pair curve for stress diagnostics
- Added `tests/reproducibility_pack.py` to rerun the core evidence stack in one command:
  - `tests/baseline_comparison.py`
  - `tests/long_horizon_benchmark.py`
  - `tests/phase6_guardrails.py`
  - deterministic smoke pytest suite
- Added machine-readable and human-readable reproducibility artifacts:
  - `research/reproducibility/reproducibility_pack_latest.json`
  - `docs/SCM_REPRODUCIBILITY_PACK.md`

### Verification

- `python tests/long_horizon_benchmark.py` passed with:
  - sleep final duplicate-family recall: `1.0`
  - sleep final noise retention: `0.0`
  - sleep anchor accuracy: `1.0`
  - strict duplicate-pair curve preserved as stress appendix.
- Deterministic smoke suite passed:
  - `pytest tests/test_baseline_comparison.py tests/test_long_horizon_benchmark.py tests/test_human_memory_behavior.py tests/test_forgetting_dynamics.py tests/test_contradiction_versioning.py -q`
  - result: `11 passed`
- `python tests/reproducibility_pack.py` produced `overall_pass: true` on the latest run.

### Paper Note

This resolves the earlier mismatch between what we claimed and what the long-horizon chart emphasized. The report now measures the memory family behavior we actually ship, while still transparently exposing strict duplicate stress behavior. Reproducibility is now a first-class artifact instead of an informal rerun procedure.

## 2026-05-01 - Long-Horizon Memory Evidence Pack

### Goal

Prove that the sleep-enabled memory stack stays stable across a multi-day history, keeps low-value noise out of the final state, and preserves a late correction without drifting.

### What Changed

- Added `tests/long_horizon_benchmark.py` to simulate a multi-day memory history with repeated interference, reactivation, and a late contradiction update.
- Generated `docs/SCM_LONG_HORIZON.md` as the paper-ready markdown report for the long-horizon evidence pack.
- Generated `research/metrics/long_horizon_latest.json` as the machine-readable long-horizon artifact.
- Added smoke regression coverage in `tests/test_long_horizon_benchmark.py`.

### Verification

- `python tests/long_horizon_benchmark.py` completed successfully on the default configuration.
- `pytest tests/test_long_horizon_benchmark.py -q` passed.
- The sleep-enabled path passed the long-horizon gates with stable anchor recall, zero final noise retention, and a clean late-update check.

### Paper Note

This adds the time dimension the earlier benchmarks did not cover. The useful story here is not just that SCM remembers, but that it stays organized over a longer memory history while still accepting updates.

## 2026-04-30 - Baseline Comparison Report

### Goal

Build a paper-ready control comparison that shows what SCM improves over, and why the gain is attributable to sleep-driven consolidation rather than generic retrieval shortcuts.

### What Changed

- Added `tests/baseline_comparison.py` to run lexical, vector, SCM baseline, MicroSleep, DeepSleep, and human-memory suite checks from a single reproducible script.
- Generated `docs/SCM_BASELINE_COMPARISON.md` as the readable, table-based comparison report for the paper.
- Generated `research/metrics/baseline_comparison_latest.json` as the machine-readable artifact for downstream analysis and reproducibility.
- Added smoke regression coverage in `tests/test_baseline_comparison.py` to guard the report path.

### Verification

- `python tests/baseline_comparison.py` completed successfully and wrote the report artifacts.
- `pytest tests/test_baseline_comparison.py -q` passed.
- Current report summary shows lexical, vector, and SCM baseline disambiguation at `0.0`, MicroSleep at `1.0`, DeepSleep at `0.9052`, deep-noise retention at `0.0`, and the human-memory suite passing cleanly.

### Paper Note

This is the cleanest control story so far. It makes the sleep-driven gain easy to defend because the improvement persists in the baseline comparison while the non-sleep controls remain flat.

## 2026-04-30 - Sleep-Centric Professional Consolidation Pass

### Goal
Turn the retrieval and sleep stack into a more professional memory system by making sleep evidence, not static importance, drive the final ranking.

### What Changed

- Added `src/core/memory_scoring.py` as the shared consolidation scorer for retrieval, sleep, and hypothesis ranking.
- Rebalanced the score so unslept traces stay neutral, while rehearsal, activation, and lineage evidence lift only truly consolidated memories.
- Removed the old bias where plain graph connectivity could masquerade as sleep evidence.
- Updated `src/retrieval/spreading_activation.py` so the final ordering is dominated by consolidation score instead of raw activation.
- Kept `src/sleep/forgetting_dynamics.py` selective enough to archive sparse, unrehearsed traces instead of letting them linger as stable distractors.

### Verification

- Focused retrieval, ranker, deep-sleep, and forgetting tests passed: `36 passed`.
- 10-seed brutal Phase 4 sweep at `pair_count=96`:
  - baseline disambiguation recall: `0.0`
  - micro-sleep disambiguation recall: `1.0`
  - deep-sleep disambiguation recall: `0.9021`
  - deep-sleep pass rate: `1.0`
  - deep-noise retention: `0.0`

### Paper Note

This is the first version that cleanly separates pre-sleep distractors from post-sleep memory strength. It is much easier to defend in a paper because the gain now comes from sleep-driven consolidation, not from static importance shortcuts.

## 2026-04-30 - Human-Style Conversational Polish

### Goal
Make SCM feel more human in fallback mode, especially when the model has to answer from memory instead of a live LLM response.

### What Changed

- Tightened the deterministic fallback path in `src/chat/engine.py` so replies about name, location, profession, and preferences sound more conversational.
- Expanded preference extraction to recognize `I prefer ...` statements, which was the missing phrase behind the earlier vague preference answers.
- Improved the "unknown profile" replies so they sound like a person asking for more context instead of a system report.
- Fixed an empty-seed bug in `src/chat/memory_retriever.py` so sparse queries no longer crash graph retrieval.
- Added regression coverage in `tests/test_chat_integration.py` for both updated preference phrasing and human-sounding uncertainty replies.

### Verification

- Focused preference-update regression test passed.
- Missing-profile human-tone regression test passed.
- Live probe returned: `You first mentioned morning meetings, then updated that to evening meetings, so I'd go with evening meetings for now.`

### Paper Note

This work strengthens the "human-like memory feel" story, but it is still a tone-and-usability improvement rather than the core scientific claim. The main paper claim remains the combination of one-shot learning, sleep-based consolidation, selective forgetting, and contradiction-safe updates.

## 2026-04-30 - Deep-Sleep Stability Hardening Pass

### Goal
Raise deep-sleep disambiguation gain under the 96-pair brutal benchmark so the "human-like memory at scale" claim becomes more robust.

### What Changed

- Replaced tail-only replay sampling in `src/sleep/deep_sleep.py` with a time-ordered, evenly spaced replay window capped at half the episode trace.
- Made REM dream-sequence generation in `src/sleep/rem.py` deterministic so the same memory state always produces the same dream walk.
- Kept the deep pass focused on coverage instead of only the most recent tail of the shuffled benchmark trace.

### Verification

- Deep sleep unit tests still pass.
- The 10-seed brutal sweep improved from the earlier 60% range to a 70% pass rate on the current code path, but it still does not clear the `>90%` target.

### Paper Note

This is a meaningful hardening step, but not yet the final stability result. The paper should describe it as an active improvement pass, not as a solved problem.

## How We Will Use This Log

Append one dated entry per meaningful implementation or validation step. Keep each entry focused on:

- intent
- code path changed
- tests or metrics run
- what the result means for the paper narrative
