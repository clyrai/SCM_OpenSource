# SCM Baseline Comparison

**Date:** 2026-05-01

This report compares SCM against standard retrieval controls and summarizes the core behavioral lift from sleep-stage consolidation.

## Behavioral Comparison

Benchmark settings: `pair_count=96`, `seeds=7001..7010`.

| Method | Family | Disambiguation recall | Top-1 recall | Key retention | Noise retention | Mean latency (ms) | Pass rate | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lexical retrieval baseline | Standard retrieval | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 | 1.0000 | 0.54 | 0.00 | Token-overlap control. |
| Vector retrieval baseline | Standard retrieval | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 | 1.0000 | 15.44 | 0.00 | Deterministic text-embedding control. |
| SCM baseline (no sleep) | SCM control | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.93 | 0.00 | Graph retrieval before sleep consolidation. |
| SCM + MicroSleep | SCM sleep stage | 1.0000 ± 0.0000 | 0.1250 ± 0.0000 | 1.0000 | 0.0000 | 1.47 | 1.00 | Light replay, merging, and local reinforcement. |
| SCM + DeepSleep | SCM sleep stage | 0.9052 ± 0.0098 | 0.6292 ± 0.0615 | 1.0000 | 0.0000 | 31.14 | 1.00 | Full replay, synthesis, and pruning pass. |

## Human-Memory Suite

| Metric | Score | Target | Pass |
| --- | --- | --- | --- |
| One-shot recall accuracy | 1.0000 | >= 1.00 | yes |
| Selective forgetting key retention | 1.0000 | >= 0.80 | yes |
| Selective forgetting noise retention | 0.0000 | <= 0.35 | yes |
| Contradiction versioning accuracy | 1.0000 | >= 0.90 | yes |

## Reference System Features

| System | Meaning-based | Sleep consolidation | NREM + REM | Intentional forgetting | Multi-dim importance | Cross-session | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SleepGate | ✗ | △ | ✗ | △ | ✗ | ✗ | Sleep metaphor, but still cache-level eviction rather than semantic memory. |
| MemGPT | ✗ | ✗ | ✗ | ✗ | ✗ | △ | Strong memory-tiering idea, but no sleep or active forgetting. |
| Stateless retrieval libraries | △ | ✗ | ✗ | △ | △ | ● | Practical production memory, but awake-only and not sleep reorganized. |
| WSCL | ✗ | ● | ● | ✗ | ✗ | △ | Biologically inspired sleep stages, but not semantic conversational memory. |
| EWC | ✗ | ✗ | ✗ | ✗ | △ | △ | Important weighting mechanism, but not a memory architecture. |
| SCM | ● | ● | ● | ● | ● | ● | Semantic graph + sleep + forgetting + versioning in one system. |

## Interpretation

The control baselines stay weak on the adversarial duplicate-memory task, while SCM gains are driven by actual sleep consolidation rather than by raw retrieval tricks.
The external reference systems remain useful baselines, but the feature matrix shows that none of them combine semantic memory, multi-stage sleep, and intentional forgetting in one architecture.
