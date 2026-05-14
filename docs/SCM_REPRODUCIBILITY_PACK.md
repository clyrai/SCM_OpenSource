# SCM Reproducibility Pack

This pack records the exact benchmark reruns, the environment, and the artifact fingerprints used for the current claim set.

## Verdict

- Overall pass: yes
- Git commit: `unknown`
- Branch: `unknown`

## Environment

- Python: `3.14.3`
- Platform: `macOS-26.2-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Working directory: `/Users/saish/Downloads/SleepAI`

## Benchmark Reruns

| Run | Status | Duration | Log |
| --- | --- | --- | --- |
| baseline_comparison | pass | 58443.53 ms | `research/reproducibility/logs/20260501T041700Z/baseline_comparison.log` |
| long_horizon | pass | 7296.59 ms | `research/reproducibility/logs/20260501T041700Z/long_horizon.log` |
| phase6_guardrails | pass | 21499.70 ms | `research/reproducibility/logs/20260501T041700Z/phase6_guardrails.log` |
| smoke_pytests | pass | 7291.73 ms | `research/reproducibility/logs/20260501T041700Z/smoke_pytests.log` |

## Artifact Fingerprints

| Artifact | Bytes | SHA256 |
| --- | --- | --- |
| `research/metrics/baseline_comparison_latest.json` | 6854 | `deedd32c24e03b307dc39e231e4d59e0c0c2905a56a636393c5a68f805a69003` |
| `research/metrics/long_horizon_latest.json` | 181393 | `f65d0af3650ce33b5f28a422c270e5d9e51f07b041f825b9aa8f0c0a9eae631f` |
| `research/metrics/phase6_guardrails_latest.json` | 12602 | `57a54d13b02d0406e919ea4245b57a67ad008d8ffa749fe163abbc965e1d8f57` |
| `research/reproducibility/reproducibility_smoke.json` | 483 | `878aee428d12aeea0946c722e5c02dd5bafdde2671eecebc5171460925c75072` |

## Key Summaries

### baseline_comparison
- overall_pass: True
- scm_deep_pass: True
- human_suite_pass: True

### long_horizon
- overall_pass: True
- final_family_recall: 0.9812
- final_strict_recall: 0.0
- final_noise_retention: 0.0
- anchor_accuracy: 1.0

### phase6_guardrails
- overall_pass: True
- phase2_pass: True
- phase4_pass: True
- human_pass: True
- pytest_pass: True
- warnings_pass: True
- deprecations_pass: True

### smoke_pytests
- overall_pass: True
- returncode: 0

## Reproduce

Run the same pack again with:

```bash
python tests/reproducibility_pack.py
```

