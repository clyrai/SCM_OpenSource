# SCM Paper Baseline (April 29, 2026)

This sheet captures the current citation-ready baseline for the next SCM manuscript draft.

## Artifact Snapshots

- `research/metrics/phase4_micro_deep_latest.json`  
  Timestamp: `2026-04-28T05:34:41.155929+00:00`
- `research/metrics/phase6_guardrails_latest.json`  
  Timestamp: `2026-04-29T04:47:27.059288+00:00`
- `research/metrics/phase6_human_memory_latest.json`  
  Timestamp: `2026-04-29T04:47:43.098260+00:00`

## Core Results (Current Baseline)

| Metric | Observed | Target | Pass |
|---|---:|---:|:---:|
| Phase 4 baseline disambiguation recall | 0.1389 | N/A | - |
| Phase 4 micro-sleep disambiguation gain (abs) | 0.8611 | >= 0.20 | ✅ |
| Phase 4 deep-sleep disambiguation gain (abs) | 0.3889 | >= 0.20 | ✅ |
| Phase 4 deep-sleep noise retention | 0.0000 | <= 0.40 | ✅ |
| Phase 6 one-shot recall accuracy | 1.0000 | >= 1.00 | ✅ |
| Phase 6 selective forgetting key retention | 1.0000 | >= 0.80 | ✅ |
| Phase 6 selective forgetting noise retention | 0.0000 | <= 0.35 | ✅ |
| Phase 6 contradiction versioning accuracy | 1.0000 | >= 0.90 | ✅ |
| Phase 6 guardrails overall status | true | true | ✅ |
| Phase 6 human-memory suite overall status | true | true | ✅ |

## Repro Commands

```bash
python tests/phase4_metrics.py
python tests/human_memory_benchmark.py
python tests/phase6_guardrails.py
python research/demos/phase6_demo.py
```

## Manuscript Naming Policy

- First mention: `SCM (Sleep-Consolidated Memory)`.
- Subsequent mentions: `SCM`.
- Use `SleepAI` only when referring to repository/product naming.
