# SCM-HME Blueprint
## Human-Memory Behavior Engine Built on SCM

**Version**: 1.0  
**Date**: April 2026  
**Status**: Historical design blueprint (live status in `docs/PROJECT_STATUS.md`)  
**Base System**: SCM (Sleep-Consolidated Memory)

---

## Implementation Update (April 29, 2026)

This blueprint remains the design reference. Execution progress now includes:

- Phase 1-5 implementation complete in code.
- Phase 6 hardening scaffolding implemented:
  - `tests/human_memory_benchmark.py`
  - `tests/test_human_memory_behavior.py`
  - `tests/phase6_guardrails.py`
  - `research/demos/phase6_demo.py`

For live status and what remains, use `docs/PROJECT_STATUS.md`.

---

## 1. Mission

Build an AI memory system that behaves like human memory at the behavioral level:

- It remembers important things quickly.
- It does not remember everything.
- It links new information to existing knowledge.
- It benefits from sleep-like consolidation.
- It forgets weak or noisy information over time.

This document defines the next system, **HME (Human-Memory Engine)**, implemented on top of SCM.

---

## 2. Problem Statement

Current AI memory systems mostly behave like storage systems:

- store many chunks,
- retrieve by similarity,
- grow indefinitely,
- weakly handle conflict and forgetting.

Human memory behaves differently:

- selective encoding (attention and salience first),
- rapid one-shot grasp for meaningful items,
- associative recall,
- offline restructuring during sleep,
- adaptive forgetting and interference management.

**Goal**: Convert SCM from a memory store into a memory process.

---

## 3. Behavioral Contract (Non-Negotiable)

HME is considered successful only if all behaviors below are observable.

1. **Selective Memory**
   - Low-value input is filtered or weakly encoded.
   - High-value input receives stronger traces.

2. **Fast Grasp**
   - Salient facts can be recalled after one exposure.
   - Familiar-pattern input is encoded faster than unfamiliar noise.

3. **Associative Recall**
   - A cue activates related memories, not just exact matches.
   - Retrieval depends on context and current goal.

4. **Sleep Benefit**
   - Delayed recall quality improves after sleep cycles compared to no-sleep runs.
   - Memory graph becomes cleaner and more structured after sleep.

5. **Adaptive Forgetting**
   - Weak and unrehearsed traces decay.
   - Strong, repeated, or useful traces persist.

6. **Conflict Robustness**
   - New contradictory information does not blindly overwrite old memory.
   - System tracks versions and temporal validity.

---

## 4. Core Design Principles

1. **Attention before storage**
2. **Events before tokens**
3. **Associations before rankings**
4. **Sleep before scale**
5. **Forgetting as a feature, not a failure**
6. **Behavioral parity over biological literalism**

---

## 5. System Overview

HME adds six major layers on top of SCM.

1. `AttentionGate`
2. `EventCompiler`
3. `AssociationBinder`
4. `SpreadingActivationRetriever`
5. `SleepKernelV2` (micro + deep)
6. `ForgettingDynamics`

### 5.1 High-Level Flow

```text
Input Message
  -> AttentionGate (Should this be encoded strongly?)
  -> EventCompiler (Convert to event traces)
  -> AssociationBinder (Link to existing graph)
  -> WorkingMemory (priority buffer)
  -> LongTermMemory (trace graph)

Periodic Sleep
  -> MicroSleep (quick replay, cleanup)
  -> DeepSleep (reorganization, downscale, synthesis)

Query
  -> Cue extraction
  -> Spreading activation + semantic recall
  -> Hypothesis ranking + confidence
  -> Response + evidence-backed memory usage
```

---

## 6. Data Model Extensions

SCM concept graph remains the base. HME introduces additional trace metadata.

### 6.1 MemoryTrace Fields

- `trace_id`
- `concept_id` (or event_id)
- `salience_score` (attention output)
- `grasp_score` (one-shot encoding strength)
- `prediction_error`
- `schema_overlap`
- `rehearsal_count`
- `activation_count`
- `association_density`
- `decay_rate`
- `retention_score`
- `confidence`
- `state` (`buffered`, `active`, `consolidated`, `suppressed`, `archived`)
- `context_tags` (session, task, interlocutor, time window)
- `version_parent` (for contradiction/version tracking)

### 6.2 Event Structure

Each memory entry should be represented as an event frame when possible:

- `who`
- `what`
- `when`
- `where`
- `why` (if inferable)
- `source`
- `certainty`

---

## 7. Wake-Phase Algorithms

## 7.1 AttentionGate

Purpose: decide encoding intensity.

### Inputs
- novelty
- task relevance
- emotional weight
- prediction error
- repetition signal
- noise estimate

### Salience Formula

```text
S = w_n*Novelty
  + w_t*TaskRelevance
  + w_e*EmotionalWeight
  + w_p*PredictionError
  + w_r*RepetitionBoost
  - w_x*NoisePenalty
```

### Policy
- if `S >= high_threshold`: strong encode
- if `mid_threshold <= S < high_threshold`: normal encode
- if `S < mid_threshold`: weak encode or skip durable storage

## 7.2 Grasp Score (Fast Learning)

Purpose: mimic human fast understanding.

```text
G = a*S + b*SchemaOverlap + c*Clarity - d*CognitiveLoad
```

Higher `G` means stronger one-shot memory trace even after single exposure.

## 7.3 AssociationBinder

Immediately bind new traces to existing graph using:

- semantic similarity,
- entity overlap,
- causal/temporal relation hints,
- co-occurrence in recent episodes.

Edge strength update:

```text
W_ij(new) = W_ij(old) + eta * G_i * G_j
```

---

## 8. Retrieval Model (Human-Like Recall)

Current SCM retrieval is mostly ranked search. HME adds recall as inference.

## 8.1 Multi-Stage Recall

1. **Cue Decode**: extract key cues from query.
2. **Seed Selection**: identify top seed traces.
3. **Spreading Activation**: propagate energy across graph.
4. **Context Gate**: filter by session/task/person/time relevance.
5. **Hypothesis Rank**: score candidate memory sets.
6. **Confidence Check**: return answer + confidence, avoid hallucinated recall.

## 8.2 Activation Update

```text
A_j(t+1) = decay*A_j(t) + sum_i(A_i(t) * W_ij * gate_context)
```

Only high-activation candidates go to response generation.

---

## 9. SleepKernelV2

Sleep is split into two operational modes.

## 9.1 MicroSleep

Trigger: every N turns or entropy spike.

Operations:
- replay top unstable traces,
- strengthen repeated useful links,
- merge near-duplicates,
- apply light decay to weak traces.

Goal: keep online memory coherent with low latency cost.

## 9.2 DeepSleep

Trigger: explicit call, idle window, or long-session threshold.

Operations:
- broader replay across episodes,
- global proportional downscale,
- contradiction/version reconciliation,
- low-value pruning,
- optional dream-like synthesis of plausible new links.

### Downscale Rule

```text
W <- alpha * W
```

Preserves relative ordering, avoids saturation.

---

## 10. ForgettingDynamics

Forgetting is continuous and selective.

## 10.1 Retention Score

```text
R = p1*Grasp
  + p2*Salience
  + p3*Rehearsal
  + p4*AssociationDensity
  + p5*Recency
  - p6*Interference
```

## 10.2 Decay

```text
Strength(t+dt) = Strength(t) * exp(-lambda*dt)
```

`lambda` should be trace-specific:
- lower for important/rehearsed traces,
- higher for noisy/unreferenced traces.

## 10.3 State Transitions

- `active -> consolidated` (after successful sleep replay)
- `active -> suppressed` (low retention)
- `suppressed -> active` (reactivation by strong cue)
- `suppressed -> archived` (long-term inactivity)

---

## 11. Contradiction and Version Management

Never hard overwrite by default.

When contradiction appears:

1. create new trace version,
2. add `contradicts` relation,
3. assign temporal validity windows,
4. retrieval favors context-valid version,
5. keep old version for audit and recovery.

Example:
- old: "User lives in Mumbai" (valid until T1)
- new: "User moved to Pune" (valid from T2)

---

## 12. Module-by-Module Implementation Plan

This maps directly to current codebase.

### 12.1 New Files

- `src/core/attention_gate.py`
- `src/core/event_compiler.py`
- `src/core/association_binder.py`
- `src/core/prediction_error.py`
- `src/retrieval/spreading_activation.py`
- `src/retrieval/hypothesis_ranker.py`
- `src/sleep/micro_sleep.py`
- `src/sleep/deep_sleep.py`
- `src/sleep/forgetting_dynamics.py`

### 12.2 Existing Files to Extend

- `src/core/models.py` (trace metadata, states)
- `src/core/value_tagger.py` (salience + grasp integration)
- `src/core/encoder.py` (event compilation hooks)
- `src/core/long_term_memory.py` (versioning, activation fields)
- `src/chat/engine.py` (wake pipeline orchestration)
- `src/sleep/sleep_cycle.py` (route to micro/deep modes)

### 12.3 API Extensions

Add endpoints:

- `POST /memory/encode` with salience diagnostics
- `POST /memory/sleep/micro`
- `POST /memory/sleep/deep`
- `GET /memory/trace/{id}` (trace details and lineage)
- `GET /memory/health` (entropy, interference, decay pressure)

---

## 13. Delivery Phases

## Phase A: Selective Encoding Core (2 weeks)

Deliverables:
- AttentionGate
- Grasp score
- EventCompiler v1

Acceptance:
- Salient one-shot facts retained at least 80 percent in controlled tests.
- Noise storage reduced at least 40 percent vs baseline SCM.

## Phase B: Associative Recall (2 weeks)

Deliverables:
- AssociationBinder
- SpreadingActivation retriever

Acceptance:
- Cue-based related recall improves at least 25 percent over baseline retrieval.

## Phase C: SleepKernelV2 (2 to 3 weeks)

Deliverables:
- MicroSleep + DeepSleep
- Downscale and replay scheduler

Acceptance:
- Post-sleep delayed recall improves at least 15 percent vs no-sleep control.

## Phase D: Forgetting and Conflict Control (2 weeks)

Deliverables:
- ForgettingDynamics
- Versioned contradiction handling

Acceptance:
- False overwrite incidents near zero in contradiction benchmark.
- Memory bloat controlled while preserving key facts.

## Phase E: Benchmark and Hardening (2 weeks)

Deliverables:
- Human-memory benchmark suite
- dashboards and reports

Acceptance:
- Meets target metrics in Section 14.

---

## 14. Benchmark Suite (Own Standard)

Create `tests/test_human_memory_behavior.py` and `tests/human_memory_benchmark.py`.

Required tasks:

1. **One-Shot Salience Recall**
2. **Selective Retention under Noise Flood**
3. **Sleep vs No-Sleep Delayed Recall**
4. **Interference and Contradiction Handling**
5. **Associative Leap Recall**
6. **Context-Dependent Retrieval**
7. **False Memory Resistance**

### Target Metrics

- Salient one-shot recall: `>= 0.85`
- Post-sleep recall gain: `>= 0.15 absolute`
- Noise retention: `<= 0.20`
- Contradiction resolution accuracy: `>= 0.90`
- False memory rate: `<= 0.05`
- Retrieval latency p95: `<= 150 ms` (M1 local)

---

## 15. Configuration Parameters

Add to `.env`:

- `ATTENTION_HIGH_THRESHOLD`
- `ATTENTION_MID_THRESHOLD`
- `GRASP_SCHEMA_WEIGHT`
- `MICRO_SLEEP_INTERVAL_TURNS`
- `DEEP_SLEEP_MIN_IDLE_SECONDS`
- `DECAY_BASE_LAMBDA`
- `FORGETTING_RETENTION_THRESHOLD`
- `SPREADING_ACTIVATION_STEPS`
- `SPREADING_ACTIVATION_DECAY`

---

## 16. Risks and Mitigation

1. **Over-filtering important memory**
   - Mitigation: conservative thresholds + audit logs + replay fallback.

2. **Under-forgetting causing bloat**
   - Mitigation: adaptive decay and periodic deep-sleep pruning.

3. **Associative drift (wrong links)**
   - Mitigation: confidence gating + contradiction checks + relation aging.

4. **Latency increase from richer retrieval**
   - Mitigation: bounded activation steps + candidate caps + caching.

5. **Evaluation ambiguity**
   - Mitigation: deterministic benchmark harness and fixed scoring rules.

---

## 17. IP and Differentiation Notes

Potential unique claims for future filings:

- dual sleep scheduling (micro + deep) for inference-time memory maintenance,
- explicit grasp score driving one-shot encoding,
- retention score combining salience, association density, and interference,
- versioned contradiction memory with context-valid retrieval.

---

## 18. Immediate Next Actions

1. Implement Phase A (`AttentionGate`, `EventCompiler`, trace fields).
2. Add benchmark scaffolding for one-shot recall and sleep gain.
3. Integrate wake pipeline in `src/chat/engine.py` behind feature flags.
4. Run A/B tests against current SCM baseline.

---

## 19. Definition of Done

HME v1 is done when:

- behavioral contract (Section 3) is satisfied,
- benchmark targets (Section 14) are met,
- latency remains practical on M1 hardware,
- system remains stable across long multi-session runs.

At that point, SCM is no longer only a sleep-inspired memory architecture; it becomes a human-memory behavior engine suitable for advanced agents.

---

**End of Document**
