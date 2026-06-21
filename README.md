# SCM — Sleep-Consolidated Memory

> **Other memory layers store facts. SCM learns from them while you're awake.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Your AI agent forgets everything between sessions. SCM fixes that — not by storing more facts, but by **consolidating, abstracting, and growing memory while you're away**, the way the human brain does during sleep.

## The Core Claim

```
Awake-only memory:     Disambiguation recall = 0.0    (can't tell facts from noise)
Sleep-enabled memory:  Disambiguation recall = 0.9052 (sleep is the mechanism)
```

This is not a retrieval improvement. This is a **different category of system**.

## What It Does

When you go to sleep, your brain replays the day's experiences, strengthens important connections, prunes noise, and forms new associations. SCM does the same thing:

```
You:  "I work at GreenLeaf Cafe."
You:  "Actually, I switched to TechCorp last month."
You:  [goes idle for 30 minutes]

SCM:  [NREM] Replays both statements, strengthens TechCorp connection
SCM:  [REM]  Generates novel associations: TechCorp → platform team → new role
SCM:  [Forgetting] Archives GreenLeaf, preserves TechCorp as current
SCM:  [Schema] Detects: user changes jobs periodically

You:  "Where do I work?"
SCM:  "You work at TechCorp. You switched from GreenLeaf last month."
```

## The Numbers

| Metric | Value | What it means |
|--------|-------|---------------|
| Disambiguation recall (with sleep) | **0.9052** | Sleep-enabled SCM correctly identifies facts vs noise |
| Disambiguation recall (awake-only) | **0.0** | Without sleep, it can't distinguish them at all |
| Noise reduction | **90.9%** | Sleep removes 90.9% of irrelevant memories |
| One-shot recall | **1.0** | Hears a fact once, remembers it perfectly |
| Retrieval latency | **<0.3ms** | Sub-millisecond graph retrieval |

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    WAKE PHASE                           │
│  User Message → MeaningEncoder → ValueTagger →          │
│  AttentionGate → Working Memory (7 items) →             │
│  EventCompiler → AssociationBinder → Long-Term Memory   │
└─────────────────────────────────────────────────────────┘
                         ↓ (idle trigger)
┌─────────────────────────────────────────────────────────┐
│                    SLEEP PHASE                          │
│  MicroSleep: Replay + Hebbian strengthening             │
│  DeepSleep:  NREM consolidation + REM dreaming          │
│  Forgetting: Prune low-value memories (90.9% noise)     │
│  Schema:     Extract recurring patterns                 │
│  Curiosity:  Fill knowledge gaps autonomously           │
└─────────────────────────────────────────────────────────┘
                         ↓ (user returns)
┌─────────────────────────────────────────────────────────┐
│                    WAKE SUMMARY                         │
│  "While you were away I noticed three things:           │
│   1. You changed jobs — I've updated your profile.      │
│   2. Your Tuesday runs became a weekly pattern.         │
│   3. You mentioned OAuth five times — I read up on it." │
└─────────────────────────────────────────────────────────┘
```

## Why This Matters

Existing memory systems compete on **retrieval quality given a fixed snapshot of memory**. SCM competes on **how the snapshot itself improves between sessions**.

| System | Sleep? | Forgetting? | Schema? | Idle Learning? |
|--------|--------|-------------|---------|----------------|
| MemGPT | ✗ | ✗ | ✗ | ✗ |
| Mem0 | ✗ | ✗ | ✗ | ✗ |
| WSCL | ✓ | ✗ | ✗ | ✗ |
| **SCM** | **✓** | **✓** | **✓** | **✓** |

## Citation

```bibtex
@article{scm2026,
  title={SCM: Autonomous Lifelong Learning for Language Agents via Sleep-Stage Memory Consolidation},
  author={SCM Research Team},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE)

## Contact

Saish Shinde · [blobopera@proton.me](mailto:blobopera@proton.me) · [github.com/clyrai/SCM_OpenSource](https://github.com/clyrai/SCM_OpenSource)
