# SleepAI Research: Paper Analysis Index

> Naming note (April 29, 2026): this historical paper index uses "SleepAI". In current research writing, use **SCM (Sleep-Consolidated Memory)** as the canonical name.

## Overview
This document indexes all papers analyzed for the SleepAI project. Each paper has a detailed analysis in the corresponding file.

---

## Papers Analyzed

### 1. SleepGate (arXiv:2603.14517)
**File**: `papers/01_SleepGate_Analysis.md`
**Date**: March 2026
**Authors**: Ying Xie (Kennesaw State University)
**Category**: Sleep-Inspired / KV Cache
**Problem**: Proactive interference in LLM context windows
**Key Innovation**: Sleep micro-cycles over KV cache with conflict-aware tagging, forgetting gate, consolidation
**Verdict**: Clever cache engineering, not true brain-inspired memory
**SleepAI Gap**: No meaning, no semantic encoding, no episodic memory, single-layer cache

---

### 2. MemGPT (arXiv:2310.08560)
**File**: `papers/02_MemGPT_Analysis.md`
**Date**: October 2023
**Authors**: Packer et al. (UC Berkeley)
**Category**: Memory Architecture / OS-Inspired
**Problem**: Limited LLM context windows
**Key Innovation**: Virtual context management with tiered memory, reflection, interrupts
**Verdict**: Good OS analogy, but no sleep, no semantic understanding
**SleepAI Gap**: No offline consolidation, no semantic meaning, no forgetting

---

### 3. Mem0 (arXiv:2504.19413)
**File**: `papers/03_Mem0_Analysis.md`
**Date**: April 2025
**Authors**: Chhikara et al.
**Category**: Production Memory System
**Problem**: Multi-session conversational memory
**Key Innovation**: Dynamic extraction + consolidation + retrieval, graph memory variant
**Verdict**: Best production system, but awake-only, no sleep
**SleepAI Gap**: No sleep consolidation, no intentional forgetting, reactive only

---

### 4. Sleep Replay Consolidation (Nature 2022)
**File**: `papers/04_Sleep_Replay_Nature_2022_Analysis.md`
**Date**: December 2022
**Authors**: González et al. (Nature Communications)
**Category**: Biological Foundation / Sleep-Inspired
**Problem**: Catastrophic forgetting in neural networks
**Key Innovation**: Hebbian plasticity + noise injection during simulated sleep enables memory rescue
**Verdict**: Scientific proof that sleep-like phases work in AI
**SleepAI Gap**: No semantic encoding, no value tagging, single sleep phase

---

### 6. Wake-Sleep Consolidated Learning (arXiv:2401.08623)
**File**: `papers/06_WSCL_Analysis.md`
**Date**: December 2023
**Authors**: Sorrenti et al.
**Category**: Sleep-Inspired / NREM+REM
**Problem**: Continual learning with sleep phases
**Key Innovation**: Explicit NREM (synaptic consolidation) + REM (dreaming/feature exploration), hippocampal short-term memory, forward transfer via dreams
**Verdict**: Most complete sleep architecture, forward transfer demonstrated
**SleepAI Gap**: Visual domain only, no semantic encoding, no value tagging

---

### 5. EWC (PNAS 2017)
**File**: `papers/05_EWC_Analysis.md`
**Date**: 2017
**Authors**: Kirkpatrick et al. (DeepMind)
**Category**: Continual Learning Foundation
**Problem**: Catastrophic forgetting
**Key Innovation**: Elastic weight consolidation using Fisher information
**Verdict**: Foundational, but weight-based not memory-based
**SleepAI Gap**: No structured memory, no forgetting, no sleep

---

### 7. Forgetting Survey (arXiv:2405.20620)
**File**: `papers/07_Forgetting_Survey_Analysis.md`
**Date**: May 2024
**Authors**: Sha, Nunes, Haller (ANU)
**Category**: Comprehensive Survey
**Problem**: All forgetting approaches across ML
**Key Innovation**: Shows forgetting is adaptive, not defective. Taxonomy of active vs passive forgetting. "Goldilocks zone" concept
**Verdict**: Essential reading for understanding forgetting in AI
**SleepAI Gap**: No sleep architecture, no meaning-based forgetting, no emotional tagging

---

## Papers to Analyze (Pending)

### Priority 1 (Key for SleepAI)
- [ ] **MemGPT / Mem0 detailed technical comparison**
- [ ] **Wake-Sleep Consolidated Learning** (González et al., 2020)
- [ ] **Selective Forgetting Survey** (ACM 2026)
- [ ] **Intentional Forgetting in AI** (2018)

### Priority 2 (Biological Foundation)
- [ ] **Synaptic Homeostasis Hypothesis (SHY)** - Tononi & Cirelli
- [ ] **Hippocampal Replay during Sleep** - Rasch & Born review
- [ ] **Memory Consolidation during Sleep** - Diekelmann & Born 2010

### Priority 3 (Supporting)
- [ ] **Differentiable Neural Computer (DNC)** - Graves et al.
- [ ] **Neural Turing Machine** - Graves et al.
- [ ] **Progressive Neural Networks** - Rusu et al.
- [ ] **Generative Model of Memory** (Nature 2024) - Spens et al.

---

## Gap Analysis: What All Papers Miss

### The Core Gaps for SleepAI

| Gap | Why It Matters | Papers That Address Partially |
|-----|----------------|------------------------------|
| **Meaning-Based Encoding** | AI processes tokens, not concepts | Mem0 (entity graph), MemGPT (extraction) |
| **True Sleep Consolidation** | Offline neural reorganization, not just cache eviction | Sleep Replay (Nature 2022), SleepGate |
| **Associative Linking** | "This causes that", not just "X related to Y" | Mem0-Graph (entity relations) |
| **Value/Emotional Tagging** | Importance signal beyond frequency | None fully |
| **Intentional Forgetting** | Active noise removal, not just decay/eviction | SleepGate (eviction gate), EWC (protection) |
| **Episodic Memory** | Events with temporal/contextual structure | None |
| **Hippocampal-Cortical Split** | Fast (working) vs slow (long-term) memory | MemGPT (tiered), but not biological |
| **NREM/REM Sleep Stages** | Different operations in different stages | None |
| **Dream-like Synthesis** | Novel combinations from old memories | None |
| **Cross-Session Memory** | Persistent across sessions with consolidation | Mem0, MemGPT |

---

## Architecture Comparison

### Memory System Comparison

| System | Memory Type | Consolidation | Forgetting | Sleep | Meaning |
|--------|-------------|---------------|------------|-------|---------|
| **SleepGate** | KV cache | Cache compression | Eviction gate | Micro-cycles | ✗ |
| **MemGPT** | Tiered paging | Summarization | Manual | ✗ | ✗ |
| **Mem0** | Graph DB | Update/merge | Implicit | ✗ | Entity-level |
| **EWC** | Weights | ✗ | Protection | ✗ | ✗ |
| **Nature 2022** | Weights | Hebbian replay | ✗ | Simulated | ✗ |
| **SleepAI** | ? | ? | ? | ? | Target |

---

## Key Insights Summary

### From SleepGate
- Sleep trigger based on entropy + conflict density
- Soft attention biasing for graceful degradation
- Dual-phase training (wake + sleep loss)

### From MemGPT
- Tiered memory architecture
- Reflection mechanism for extraction
- Interrupt-based control flow

### From Mem0
- LLM-powered dynamic extraction
- Graph memory for relations
- Importance scoring per fact

### From Nature 2022
- Hebbian plasticity works during offline "sleep"
- Noise injection enables spontaneous replay
- Memory rescue without stored data

### From EWC
- Not all weights/memory is equally important
- Fisher information = importance measure
- Protection with flexibility

---

## Next Steps for SleepAI Architecture

### Combine Best of Each:

1. **Sleep trigger** (SleepGate) → When to consolidate
2. **Memory tiers** (MemGPT) → Fast vs slow storage
3. **Extraction** (Mem0) → What to remember
4. **Hebbian replay** (Nature 2022) → How to consolidate offline
5. **Importance weighting** (EWC) → What to protect/forget
6. **Semantic graph** (Mem0-Graph) → How to link memories

### Add SleepAI's Unique Contributions:

7. **Meaning encoding** → Beyond tokens and entities
8. **Value tagging** → Emotional/task importance
9. **Intentional forgetting** → Active noise removal
10. **Sleep stages** → NREM + REM differentiation
11. **Dream synthesis** → Novel combination generation
12. **Episodic memory** → Temporal event structure

---

## Document Status

| Document | Status |
|----------|--------|
| Master Research Document | ✓ Complete |
| SleepGate Analysis | ✓ Complete |
| MemGPT Analysis | ✓ Complete |
| Mem0 Analysis | ✓ Complete |
| Sleep Replay (Nature 2022) Analysis | ✓ Complete |
| EWC Analysis | ✓ Complete |
| This Index | ✓ Complete |
| SHY Analysis | Pending |
| Wake-Sleep Learning Analysis | Pending |
| Intentional Forgetting Analysis | Pending |
| Other Papers | Pending |

---

*Last Updated: April 2026*
*Project: SleepAI*
