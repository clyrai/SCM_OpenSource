# SCM-HME Phase Plan
## Detailed Phases Document (Built on SCM)

**Version**: 1.0  
**Date**: April 2026  
**Status**: Historical execution plan (live status in `docs/PROJECT_STATUS.md`)  
**Base**: SCM + `research/10_SCM_Human_Memory_Blueprint.md`

---

## Implementation Update (April 29, 2026)

This document is the original phase plan. Current implementation has progressed through Phase 5 and into Phase 6 hardening.

- Implemented: Phases 1-5 core system behavior.
- In progress: Phase 6 hardening with full-stack guardrails and human-memory benchmark scripts.
- Key artifacts:
  - `tests/phase6_guardrails.py`
  - `tests/human_memory_benchmark.py`
  - `tests/test_human_memory_behavior.py`
  - `research/demos/phase6_demo.py`
  - `research/metrics/phase6_guardrails_latest.json`

Use `docs/PROJECT_STATUS.md` as the live source of truth for current execution status.

---

## 1. Objective

This document converts the SCM-HME blueprint into an executable phase plan.

Target outcome:

- Move from "memory storage" behavior to "human-memory-like" behavior.
- Keep SCM as the production base while incrementally enabling HME features.
- Deliver measurable gains per phase with strict acceptance gates.

---

## 2. Delivery Strategy

### 2.1 Principles

1. Ship in slices; each phase must run on current codebase.
2. Keep feature flags for all new behavior.
3. Measure every phase against baseline SCM before moving forward.
4. Optimize for M1 (8GB RAM, local Ollama, CPU-first path).

### 2.2 Estimated Timeline

- Total: 12-14 weeks
- Cadence: weekly milestones, phase gates every 2-3 weeks

---

## 3. Phase Map

| Phase | Duration | Theme | Gate Outcome |
|------|----------|-------|--------------|
| Phase 0 | Week 1 | Baseline freeze and instrumentation | Stable baseline metrics |
| Phase 1 | Weeks 2-3 | Selective encoding + fast grasp | One-shot salient recall working |
| Phase 2 | Weeks 4-5 | Event model + association binding | Associative graph quality improved |
| Phase 3 | Weeks 6-7 | Human-like retrieval | Cue-based recall beats baseline |
| Phase 4 | Weeks 8-10 | SleepKernelV2 (micro + deep) | Sleep gain demonstrated |
| Phase 5 | Weeks 11-12 | Forgetting dynamics + contradiction versioning | Noise drop with factual integrity |
| Phase 6 | Weeks 13-14 | Hardening, benchmark pack, demo readiness | Release candidate |

---

## 4. Phase 0 - Baseline Freeze and Instrumentation

### Objective

Lock SCM baseline and add observability needed for HME evaluation.

### Scope

- Freeze current benchmark outputs.
- Add profiling hooks for encode/retrieve/sleep stages.
- Add feature flags scaffolding for future phases.

### Files

- `src/core/config.py`
- `src/chat/engine.py`
- `tests/benchmark.py`
- `tests/benchmark_report.json` (baseline artifact)

### Deliverables

1. Baseline report (recall, noise retention, latency p95).
2. Runtime counters for: encoded traces, dropped traces, sleep duration, forgetting actions.
3. Feature flags:
   - `HME_ENABLE_ATTENTION_GATE`
   - `HME_ENABLE_ASSOCIATIVE_RETRIEVAL`
   - `HME_ENABLE_SLEEP_V2`
   - `HME_ENABLE_FORGETTING_DYNAMICS`

### Exit Criteria

- Existing SCM tests pass unchanged.
- Baseline metrics stored and reproducible.

---

## 5. Phase 1 - Selective Encoding and Fast Grasp

### Objective

Implement human-like selective memory formation at encode time.

### Scope

- `AttentionGate`: salience scoring and encoding intensity.
- `GraspScore`: one-shot encoding strength from salience + schema overlap.
- Weak encode path for low-value inputs.

### New/Updated Files

- `src/core/attention_gate.py` (new)
- `src/core/prediction_error.py` (new)
- `src/core/value_tagger.py` (extend)
- `src/chat/engine.py` (pipeline hook)
- `src/core/models.py` (trace fields: salience, grasp)

### Work Packages

1. Implement salience formula and thresholds.
2. Compute prediction-error feature.
3. Integrate `AttentionGate` before durable storage.
4. Persist salience and grasp metadata.

### Tests

- `tests/test_attention_gate.py` (new)
- `tests/test_grasp_score.py` (new)
- Extend `tests/test_memory_pipeline.py`

### Metrics Target

- Salient one-shot recall >= 0.85
- Noise durable-encode rate <= 0.60 of baseline

### Exit Criteria

- High-salience facts survive one-shot scenarios.
- Low-salience noise significantly reduced.

---

## 6. Phase 2 - Event Compiler and Association Binder

### Objective

Store meaningful events and instantly bind them to existing memory structures.

### Scope

- Event frame generation (`who/what/when/where/why`).
- Association edge creation using semantic + contextual rules.
- Initial contradiction tagging support.

### New/Updated Files

- `src/core/event_compiler.py` (new)
- `src/core/association_binder.py` (new)
- `src/core/encoder.py` (event output hook)
- `src/core/long_term_memory.py` (association updates)

### Work Packages

1. Build event schema and conversion pipeline.
2. Add association-strength update rule.
3. Create relation aging policy for stale weak links.

### Tests

- `tests/test_event_compiler.py` (new)
- `tests/test_association_binder.py` (new)

### Metrics Target

- Association coverage (relevant edges created) >= 0.80 on synthetic scenarios.
- Duplicate event inflation <= 10 percent.

### Exit Criteria

- Memory graph captures event structure and useful links.

---

## 7. Phase 3 - Human-Like Retrieval

### Objective

Replace pure ranking retrieval with cue-driven associative recall.

### Scope

- Spreading activation retriever.
- Context gate (session, task, person, recency).
- Hypothesis ranking with confidence.

### New/Updated Files

- `src/retrieval/spreading_activation.py` (new)
- `src/retrieval/hypothesis_ranker.py` (new)
- `src/chat/memory_retriever.py` (extend)
- `src/chat/engine.py` (routing logic)

### Work Packages

1. Add seed selection from query cues.
2. Implement bounded activation propagation.
3. Add confidence score and fallback behavior.

### Tests

- `tests/test_spreading_activation.py` (new)
- `tests/test_retrieval_context_gate.py` (new)

### Metrics Target

- Associative recall gain >= 25 percent over baseline retrieval.
- False memory rate <= 0.08 in controlled tests.
- Retrieval p95 <= 150 ms (M1 local).

### Exit Criteria

- Cue-based recall clearly outperforms baseline search.

---

## 8. Phase 4 - SleepKernelV2 (Micro + Deep)

### Objective

Implement continuous sleep behavior that improves delayed recall and memory quality.

### Scope

- `MicroSleep`: frequent lightweight replay and cleanup.
- `DeepSleep`: full replay, downscale, synthesis, pruning.
- Scheduler and triggers for both modes.

### New/Updated Files

- `src/sleep/micro_sleep.py` (new)
- `src/sleep/deep_sleep.py` (new)
- `src/sleep/sleep_cycle.py` (refactor/route)
- `src/sleep/trigger.py` (new thresholds)

### Work Packages

1. Add micro-sleep trigger by turns/entropy.
2. Add deep-sleep trigger by idle/pressure thresholds.
3. Implement global downscale in deep mode.
4. Add replay policy and optional synthesis pass.

### Tests

- `tests/test_micro_sleep.py` (new)
- `tests/test_deep_sleep.py` (new)
- Extend `tests/test_sleep.py`

### Metrics Target

- Post-sleep recall gain >= 0.15 absolute over no-sleep control.
- Graph entropy reduced after deep-sleep cycles.

### Exit Criteria

- Sleep shows measurable recall benefit and stability gains.

---

## 9. Phase 5 - Forgetting Dynamics and Contradiction Versioning

### Objective

Make forgetting adaptive and contradiction-safe.

### Scope

- Retention score dynamics (salience, rehearsal, association density, interference).
- Trace decay with variable lambdas.
- Versioned contradiction handling with validity windows.

### New/Updated Files

- `src/sleep/forgetting_dynamics.py` (new)
- `src/core/long_term_memory.py` (version graph support)
- `src/core/models.py` (version metadata)
- `src/chat/engine.py` (contradiction ingestion path)

### Work Packages

1. Implement retention score formula.
2. Add state transitions (`active/suppressed/archived`).
3. Add contradiction version resolution policy.

### Tests

- `tests/test_forgetting_dynamics.py` (new)
- `tests/test_contradiction_versioning.py` (new)

### Metrics Target

- Noise retention <= 0.20
- Contradiction resolution accuracy >= 0.90
- False overwrite incidents near zero

### Exit Criteria

- System forgets aggressively where safe, preserves critical memory integrity.

---

## 10. Phase 6 - Hardening and Release Candidate

### Objective

Package HME as a stable, measurable, demo-ready system.

### Scope

- Build benchmark harness for human-memory behaviors.
- Add diagnostics endpoints.
- Performance tuning and regression guardrails.
- Prepare release report and demo scripts.

### New/Updated Files

- `tests/human_memory_benchmark.py` (new)
- `tests/test_human_memory_behavior.py` (new)
- `src/api/main.py` (diagnostic endpoints)
- `docs/` release notes and operation guide

### Deliverables

1. Final benchmark report (all target metrics).
2. Regression suite integrated into normal test runs.
3. Demo scenario scripts (sleep/no-sleep comparison, selective memory, contradiction handling).

### Exit Criteria

- All phase metrics met.
- Runtime stable in long sessions.
- Release candidate approved.

---

## 11. Cross-Phase Dependencies

| Dependency | Needed By | Notes |
|-----------|-----------|-------|
| Feature flags (Phase 0) | All phases | Safe rollout and A/B testing |
| Trace metadata (Phase 1) | Phases 2-5 | Required for retention, retrieval, sleep policies |
| Association graph (Phase 2) | Phases 3-5 | Needed for spreading activation and replay |
| Retrieval confidence (Phase 3) | Phase 6 | Needed for false-memory metrics |
| Sleep scheduler (Phase 4) | Phase 5 | Drives decay and consolidation timing |

---

## 12. Risks by Phase

### Phase 1 Risk

- Over-filtering important input.
- Mitigation: conservative defaults + trace audit logs.

### Phase 2 Risk

- Graph link explosion.
- Mitigation: edge caps + relation aging + confidence thresholds.

### Phase 3 Risk

- Retrieval latency drift.
- Mitigation: bounded activation depth and candidate limits.

### Phase 4 Risk

- Sleep overhead on interactive sessions.
- Mitigation: micro-sleep budget and deferred deep-sleep.

### Phase 5 Risk

- Incorrect version preference under ambiguity.
- Mitigation: temporal validity + confidence and recency arbitration.

### Phase 6 Risk

- Benchmark mismatch with product reality.
- Mitigation: include both synthetic and real-session scenarios.

---

## 13. Phase Gate Checklist

Use this checklist at every gate:

1. Unit tests green for phase modules.
2. No regressions in existing SCM core tests.
3. Metrics met for phase target.
4. Feature flag fallback validated.
5. Benchmark deltas documented.

If any item fails, phase does not close.

---

## 14. Final Acceptance Targets (Program-Level)

- Salient one-shot recall >= 0.85
- Post-sleep recall improvement >= 0.15 absolute
- Noise retention <= 0.20
- Contradiction resolution >= 0.90
- False-memory rate <= 0.05
- Retrieval p95 <= 150 ms on M1

Meeting these targets means SCM-HME achieves practical human-memory-like behavior.

---

## 15. Immediate Execution Sequence

1. Start Phase 0 this week (freeze and instrumentation).
2. Start Phase 1 immediately after baseline lock.
3. Keep weekly metric snapshots in `research/metrics/`.
4. Run full gate review at end of each phase.

---

**End of Phases Document**
