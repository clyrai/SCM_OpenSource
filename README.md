# SCM: Autonomous Lifelong Learning for Language Agents via Sleep-Stage Memory Consolidation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Abstract

Most production language agents do not learn after deployment. They retrieve, they answer, they generate — but the moment a conversation ends, no abstraction, no consolidation, no growth happens before the next one begins. The agent that wakes up tomorrow is exactly the agent that went to sleep last night.

SCM closes this gap with a complete biological memory lifecycle: bounded working memory, selective encoding with attention-gated intensity classification, event-structured semantic binding with Hebbian weight updates, spreading-activation retrieval with context-gated hypothesis ranking, dual-mode sleep consolidation (NREM + REM), adaptive value-based forgetting, contradiction-safe version lineage, and a Phase 7 autonomous-learning layer that activates while the user is away.

We additionally introduce a hierarchical encoding pathway inspired by recursive semantic chunking, in which incoming text is segmented into semantically coherent chunks before concept extraction. An A/B evaluation demonstrates that hierarchical encoding produces 2.3× more granular concepts for multi-topic inputs and 3.2× more for long narratives.

## Architecture

![SCM Architecture](docs/architecture.png)

| Phase | Component | Function |
|-------|-----------|----------|
| 1 | AttentionGate | Selective encoding with 4-tier intensity classification |
| 2 | EventCompiler | Structured event frames (who/what/when/where/why) |
| 3 | SpreadingActivation | Cue-driven graph propagation with context-gated hypothesis ranking |
| 4 | SleepKernel | Micro-sleep (NREM) + Deep-sleep (REM) consolidation |
| 5 | ForgettingDynamics | Adaptive value-based forgetting with contradiction-safe versioning |
| 6 | Guardrails | Paraphrase, evaluation harnesses, reproducibility packs |
| 7 | IdleLearner | Autonomous learning during user idle time (M1–M6) |

## Key Results

### Sleep Consolidation is the Mechanism

Under a 10-seed stress comparison with 96 memory pairs per seed, awake-only controls remain at 0.0 disambiguation recall. Only sleep-enabled SCM variants achieve non-zero performance.

| Condition | Recall (mean±std) | Noise Ret. | Pass |
|-----------|-------------------|------------|------|
| Lexical retrieval | 0.0000 ± 0.0000 | 1.0000 | 0/10 |
| Vector retrieval | 0.0000 ± 0.0000 | 1.0000 | 0/10 |
| SCM (no sleep) | 0.0000 ± 0.0000 | 1.0000 | 0/10 |
| SCM + MicroSleep | 1.0000 ± 0.0000 | 0.0000 | 10/10 |
| SCM + DeepSleep | 0.9052 ± 0.0098 | 0.0000 | 10/10 |

### Hierarchical Encoding Improves Granularity

| Input | Flat | Hierarchical | Factor | Δ Types |
|-------|------|-------------|--------|---------|
| Multi-topic (139w) | 6 | 14 | 2.3× | +preference |
| Long narrative (238w) | 6 | 19 | 3.2× | +location, +abstract |

### No Single System Wins All Workloads

LoCoMo++ evaluation across 3 conversations, 3 seeds, 36 total runs shows that memory architecture selection depends on workload:

| System | Clean recall | Contra-current | Disambig | Noise reject |
|--------|-------------|----------------|----------|--------------|
| Vector (Mem0-style) | 0.280 | **0.375** | 0.963 | 0.037 |
| SCM Phase-4-only | **0.430** | 0.104 | **1.000** | 0.167 |
| SCM HME-full | 0.022 | 0.000 | 0.704 | **1.000** |

## Paper

**SCM: Autonomous Lifelong Learning for Language Agents via Sleep-Stage Memory Consolidation**

SCM Research Team, June 2026

```bibtex
@article{scm2026,
  title={SCM: Autonomous Lifelong Learning for Language Agents via Sleep-Stage Memory Consolidation},
  author={SCM Research Team},
  year={2026}
}
```

## Repository Structure

```
src/                          # Core runtime
  chat/                       # ChatEngine, MemoryRetriever
  core/                       # Encoder, AttentionGate, ValueTagger, Models
  sleep/                      # MicroSleep, DeepSleep, NREM, REM, Forgetting
  lifecycle/                  # IdleLearner, Curiosity, WakeSummary
  retrieval/                  # SpreadingActivation, VectorIndex
  integrations/               # LangChain, MCP, REST API
tests/                        # 322 regression tests
examples/                     # Quickstart scripts
docs/                         # Deployment, integration guides
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- Author: Saish Shinde
- Email: blobopera@proton.me
- GitHub: [github.com/clyrai/SCM_OpenSource](https://github.com/clyrai/SCM_OpenSource)
