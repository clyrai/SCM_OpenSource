# SleepAI Research: Priority Paper Tiers

> Naming note (April 29, 2026): this historical prioritization doc uses "SleepAI". In current research writing, use **SCM (Sleep-Consolidated Memory)** as the canonical name.

## Purpose
Separate critical papers (must-read for building SleepAI) from supporting papers (good-to-know, not critical for architecture).

---

## TIER 1: CRITICAL (Must Analyze)
*Directly relevant to building SleepAI architecture. These form the core.*

| # | Paper | arXiv | Why Critical |
|---|-------|-------|---------------|
| 1 | **SleepGate** | 2603.14517 | Sleep micro-cycles, forgetting gate, entropy trigger |
| 2 | **Wake-Sleep Consolidated Learning (WSCL)** | 2401.08623 | NREM/REM differentiation, hippocampal-cortical split |
| 3 | **Nature 2022: Sleep Replay Consolidation** | - | Hebbian plasticity during sleep, memory rescue mechanism |
| 4 | **Forgetting Survey** | 2405.20620 | Forgetting taxonomy, "Goldilocks zone", intentional forgetting |

---

## TIER 2: HIGH (Should Analyze)
*Important for specific SleepAI components. Build on Tier 1.*

| # | Paper | arXiv | Why High |
|---|-------|-------|----------|
| 5 | **EWC** | 1612.00796 | Importance weighting, Fisher information for memory protection |
| 6 | **Mem0** | 2504.19413 | Production memory, LLM extraction, graph memory variant |
| 7 | **MemGPT** | 2310.08560 | Memory tiers, reflection mechanism, interrupt control |
| 8 | **SHY (Synaptic Homeostasis Hypothesis)** | - | Biological foundation: why sleep, what it does to synapses |

---

## TIER 3: SUPPORTING (Reference Only)
*Useful for understanding context, not critical for architecture.*

| # | Paper | Notes |
|---|-------|-------|
| 9 | Progressive Neural Networks | Architecture concept, not directly usable |
| 10 | Neural Turing Machine / DNC | Foundation for memory-augmented networks |
| 11 | Generative Model of Memory (Nature 2024) | Generative replay concept |
| 12 | H2O / StreamingLLM | KV cache eviction baselines |
| 13 | Intentional Forgetting in AI (2018) | Foundational but dated |
| 14 | Mem0 Graph Memory | Already covered in Mem0 paper |
| 15 | CLS Theory (Complementary Learning Systems) | Referenced in WSCL, not separate paper |

---

## TIER 4: IGNORE (Not Relevant)
*Interesting but not directly useful for SleepAI.*

- Language model specific papers
- RL-specific approaches without memory relevance
- Highly theoretical neuroscience without AI application

---

## Current Status

### Tier 1: COMPLETE ✓
- [x] SleepGate Analysis → `papers/01_SleepGate_Analysis.md`
- [x] WSCL Analysis → `papers/06_WSCL_Analysis.md`
- [x] Nature 2022 Sleep Replay → `papers/04_Sleep_Replay_Nature_2022_Analysis.md`
- [x] Forgetting Survey → `papers/07_Forgetting_Survey_Analysis.md`

### Tier 2: COMPLETE ✓
- [x] EWC Analysis → `papers/05_EWC_Analysis.md`
- [x] Mem0 Analysis → `papers/03_Mem0_Analysis.md`
- [x] MemGPT Analysis → `papers/02_MemGPT_Analysis.md`
- [x] SHY (Synaptic Homeostasis) → `papers/08_SHY_Analysis.md`

### Tier 3: NOT STARTED
- Will handle as needed for reference

---

## Priority Analysis Plan

**Next Steps (Continuing with Tier 2)**:

1. ✓ EWC — DONE
2. ✓ Mem0 — DONE
3. ✓ MemGPT — DONE
4. **→ SHY (Synaptic Homeostasis Hypothesis)** — NEXT
5. → Then: Consolidated Architecture Document
6. → Then: SleepAI Design Proposal

---

## Why This Tier System?

**Tier 1 (Critical)**: Papers that define WHAT SleepAI should be:
- SleepGate = how to trigger sleep
- WSCL = how to structure sleep (NREM/REM)
- Nature 2022 = why Hebbian during sleep works
- Forgetting Survey = what forgetting should do

**Tier 2 (High)**: Papers that define HOW SleepAI components work:
- EWC = importance protection
- Mem0 = production memory
- MemGPT = memory tiers
- SHY = biological grounding

**Tier 3 (Supporting)**: Implementation details when needed.

---

*Last Updated: April 2026*
*Project: SleepAI*
