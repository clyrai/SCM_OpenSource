# SleepAI Tests

## Running Tests

```bash
# Install test deps
pip install pytest pytest-asyncio

# Run full suite
pytest tests/ -v

# Phase 6 guardrails (recommended hardening baseline)
python tests/phase6_guardrails.py

# Human-memory behavior benchmark
python tests/human_memory_benchmark.py

# Baseline comparison report
python tests/baseline_comparison.py

# Long-horizon memory evidence pack
python tests/long_horizon_benchmark.py

# Publication-grade reproducibility pack
python tests/reproducibility_pack.py
```

## Test Structure

- `test_encoder.py`, `test_value_tagger.py`, `test_working_memory.py` - Core memory primitives
- `test_phase2_pipeline.py`, `test_phase3_pipeline.py` - HME retrieval/association integration
- `test_micro_sleep.py`, `test_deep_sleep.py`, `test_sleep.py` - Sleep pipeline behavior
- `test_forgetting_dynamics.py`, `test_contradiction_versioning.py` - Phase 5 memory integrity
- `test_human_memory_behavior.py` - Phase 6 behavior regression checks
- `test_baseline_comparison.py` - Paper-ready baseline comparison smoke test
- `test_long_horizon_benchmark.py` - Long-horizon retention/update smoke test
- `reproducibility_pack.py` - One-command evidence rerun + artifact fingerprints
- `phase2_metrics.py`, `phase3_metrics.py`, `phase4_metrics.py`, `phase6_guardrails.py` - Machine-readable benchmark reports
