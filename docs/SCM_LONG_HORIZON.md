# SCM Long-Horizon Memory

A reproducible multi-day retention benchmark for SCM.

## Summary

_Stable-fact hit rate and family-aware duplicate recall are the main release gates; strict duplicate-pair disambiguation remains a stress signal in the JSON artifact._

| Metric | Awake-only | Sleep-enabled | Lift |
| --- | --- | --- | --- |
| Final stable-fact hit rate | 1.0000 | 1.0000 | 0.0000 |
| Final duplicate-family recall | 0.0000 | 0.9812 | 0.9812 |
| Final noise retention | 1.0000 | 0.0000 | 1.0000 |
| Mean stable-fact hit rate | 1.0000 | 1.0000 | 0.0000 |
| Mean duplicate-family recall | 0.0000 | 0.9973 | 0.9973 |
| Anchor update accuracy | 1.0000 | 1.0000 | 0.0000 |
| Mean latency (ms) | 1.4056 | 1.7453 | 0.3397 |

## Day-by-Day Curve

| Day | Awake mode | Awake recall | Awake noise | Sleep mode | Sleep recall | Sleep noise |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | awake | 0.0000 | 1.0000 | micro | 1.0000 | 0.0000 |
| 2 | awake | 0.0000 | 1.0000 | micro | 1.0000 | 0.0000 |
| 3 | awake | 0.0000 | 1.0000 | deep | 1.0000 | 0.0000 |
| 4 | awake | 0.0000 | 1.0000 | micro | 1.0000 | 0.0000 |
| 5 | awake | 0.0000 | 1.0000 | micro | 1.0000 | 0.0000 |
| 6 | awake | 0.0000 | 1.0000 | deep | 1.0000 | 0.0000 |
| 7 | awake | 0.0000 | 1.0000 | deep | 0.9812 | 0.0000 |

## Duplicate Stress Signal

| Day | Awake strict recall | Sleep strict recall |
| --- | --- | --- |
| 1 | 0.0000 | 1.0000 |
| 2 | 0.0000 | 1.0000 |
| 3 | 0.0000 | 1.0000 |
| 4 | 0.0000 | 0.0000 |
| 5 | 0.0000 | 0.0000 |
| 6 | 0.0000 | 0.0000 |
| 7 | 0.0000 | 0.0000 |

## Interpretation

- This benchmark simulates repeated day-by-day interference, then checks whether the original memory still wins at the end of the horizon.
- The sleep-enabled path is expected to keep the target family stronger while pruning low-value noise more aggressively than the awake-only control.
- A late correction is also injected so we can verify versioning survives a longer memory history.
- Strict day-1 duplicate-pair recall is retained as a stress signal in the JSON appendix; the family-aware signal is the release gate.

## Acceptance Gate

- Overall pass: yes
- Sleep final recall pass: yes
- Sleep final noise pass: yes
- Sleep anchor pass: yes
- Sleep curve pass: yes
