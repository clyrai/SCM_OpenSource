# SleepAI Architecture: Consolidated Design Document

> Naming note (April 29, 2026): this historical architecture draft uses "SleepAI". In current research writing, use **SCM (Sleep-Consolidated Memory)** as the canonical name.

## Based on Analysis of 8 Key Papers

---

## 1. Overview

This document synthesizes insights from 8 critical papers to define the SleepAI architecture. SleepAI is an attempt to build AI with human-like memory: attention-based encoding, sleep consolidation, intentional forgetting, and episodic memory.

**Papers Analyzed**:
1. SleepGate (2026) — Sleep micro-cycles, entropy trigger
2. WSCL (2023) — NREM/REM differentiation, hippocampal-cortical split
3. Nature 2022 — Hebbian replay during sleep, memory rescue
4. Forgetting Survey (2024) — Forgetting taxonomy, "Goldilocks zone"
5. EWC (2017) — Importance weighting, Fisher information
6. Mem0 (2025) — LLM extraction, graph memory
7. MemGPT (2023) — Memory tiers, reflection
8. SHY (2006) — Biological foundation: why sleep, synaptic downscaling

---

## 2. Core Principle: Why SleepAI Exists

### 2.1 What All Papers Miss

Current AI systems (including all 8 papers) share fundamental gaps:

| Gap | Impact |
|-----|--------|
| **Token-level processing** | No semantic meaning, only pattern matching |
| **No offline consolidation** | No sleep to reorganize knowledge |
| **No intentional forgetting** | Memory grows forever, noise accumulates |
| **No value/emotional tagging** | All memory treated equally |
| **No dream synthesis** | Can't create novel combinations |
| **No episodic memory** | Stores facts, not events |

### 2.2 SleepAI's Differentiation

SleepAI aims for **meaning-based, sleep-enabled, forgetting-forward** memory:

```
Human Memory          →    SleepAI Target
─────────────────────────────────────────────────
Attention filter      →    Meaning-based attention
Semantic encoding     →    Concept-graph representation
Hippocampal short-term →    Working memory (fast, volatile)
Cortical long-term    →    Consolidated memory (slow, stable)
Sleep replay          →    Hebbian offline consolidation
Synaptic downscaling  →    Proportional memory renormalization
Intentional forgetting →    Value-based noise removal
Dream synthesis       →    Generative replay with novel combinations
Emotional tagging     →    Multi-dimensional importance signal
Episodic memory       →    Temporal event storage
```

---

## 3. Architecture Overview

### 3.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         SLEEPAI SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     WAKE PHASE                          │   │
│  │                                                          │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌────────────┐   │   │
│  │  │  ATTENTION   │──▶│   MEANING    │──▶│   VALUE    │   │   │
│  │  │   FILTER     │   │   ENCODER    │   │   TAGGER   │   │   │
│  │  │              │   │ (Concept-   │   │ (Import-   │   │   │
│  │  │ What's new?  │   │  Graph)      │   │  ance/M)   │   │   │
│  │  └──────────────┘   └──────────────┘   └────────────┘   │   │
│  │         │                  │                  │         │   │
│  │         └──────────────────┼──────────────────┘         │   │
│  │                            ▼                            │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │           WORKING MEMORY (Hippocampus)          │     │   │
│  │  │  - Fast access    - Limited capacity           │     │   │
│  │  │  - Temporal tags  - Recent events              │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                            │                            │   │
│  └────────────────────────────┼────────────────────────────┘   │
│                               │                                │
│        ┌──────────────────────┼──────────────────────┐         │
│        │                      │                      │         │
│        ▼                      ▼                      ▼         │
│  ┌───────────────────────────────────────────────────────┐   │
│  │                    SLEEP PHASE                         │   │
│  │                                                        │   │
│  │  ┌─────────────────┐      ┌─────────────────┐          │   │
│  │  │   NREM SLEEP    │      │   REM SLEEP     │          │   │
│  │  │                 │      │                 │          │   │
│  │  │ - Replay        │      │ - Dreaming      │          │   │
│  │  │ - Consolidate   │      │ - Feature       │          │   │
│  │  │ - Downscale     │      │   exploration   │          │   │
│  │  │ - Transfer to   │      │ - Novel combo   │          │   │
│  │  │   long-term     │      │   synthesis    │          │   │
│  │  └─────────────────┘      └─────────────────┘          │   │
│  │                                                        │   │
│  └───────────────────────────────────────────────────────┘   │
│                               │                                │
│                               ▼                                │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              LONG-TERM MEMORY (Cortex)                 │   │
│  │  - Compressed      - Stable           - Linked       │   │
│  │  - Semantic graph  - Importance       - Persistent   │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Flow

```
1. INPUT (wake)
   ↓
2. ATTENTION FILTER (what matters?)
   ↓
3. MEANING ENCODER (convert to concept-graph)
   ↓
4. VALUE TAGGER (tag importance/emotion/novelty)
   ↓
5. WORKING MEMORY (hippocampus: fast, temporary)
   ↓
6. SLEEP PHASE TRIGGER (entropy/conflict threshold)
   ↓
7. NREM SLEEP (consolidate, downscale, transfer)
   ↓
8. REM SLEEP (dream, synthesize, explore)
   ↓
9. LONG-TERM MEMORY (cortex: compressed, stable)
   ↓
10. FORGETTING (remove low-value, noisy, contradictory)
```

---

## 4. Component Specifications

### 4.1 Attention Filter

**Function**: Decide what enters consciousness/memory

**Inputs**:
- Raw input (text, tokens)
- Current working memory state
- Long-term memory relevance

**Output**: Binary or score indicating "important enough to encode"

**Mechanism**:
```
novelty_score = cosine_similarity(input, long_term_memory)
context_relevance = attention_to_related_concepts()
emotional_tag = value_tagger(input)

attention_score = f(novelty, context_relevance, emotional_tag)

if attention_score > threshold:
    encode_to_working_memory()
```

**Reference**: SleepGate's entropy trigger, Mem0's extraction

### 4.2 Meaning Encoder

**Function**: Convert tokens → semantic concepts with relations

**Structure**: Knowledge Graph
```
Concept: [id, type, description, embedding]
Relation: [subject, predicate, object, strength]
```

**Example**:
```
Concept: (user_preference, "meetings_before_3pm")
  - type: scheduling_preference
  - embedding: [0.1, 0.3, ...]
Relation: (user → pref → meetings_before_3pm)
  - strength: 0.9
```

**Operations**:
- **Link**: New concept → existing concept (association)
- **Update**: Modify existing concept (refinement)
- **Merge**: Combine related concepts (integration)

**Reference**: Mem0's graph memory, associative linking concept

### 4.3 Value Tagger

**Function**: Multi-dimensional importance signal

**Dimensions**:
| Dimension | Description | Signal Source |
|-----------|-------------|----------------|
| Novelty | How new is this? | Comparison to existing |
| Emotional | Positive/negative? | Sentiment analysis |
| Task-Relevance | Important for goals? | User context |
| Repetition | Seen before? | Access frequency |
| Conflict | Contradicts existing? | Graph consistency |

**Output**:
```
value_vector = [novelty, emotional, task_relevance, repetition, conflict]
importance_score = weighted_sum(value_vector * weights)
```

**Reference**: EWC's Fisher importance, emotional tagging theory

### 4.4 Working Memory (Hippocampus-Equivalent)

**Function**: Fast access, limited capacity, temporal encoding

**Structure**:
```
Episode: {
    id: uuid,
    timestamp: datetime,
    concepts: [concept_ids],
    value_vector: [...],
    context: {...},
    links: [relation_ids]
}
```

**Properties**:
- Capacity: Limited (like human ~7 items)
- Volatility: Can be lost, needs consolidation
- Temporal: Ordered by time
- Contextual: Episode includes situation

**Operations**:
- **Store**: Add new episode
- **Retrieve**: Get by time/similarity/content
- **Replay**: Reactivate pattern (for sleep)

**Reference**: WSCL's short-term memory, Nature 2022's replay

### 4.5 Sleep Phase Controller

**Function**: Trigger and orchestrate sleep cycles

**Trigger Signals** (from SleepGate):
1. **Attention Entropy**: Uniform attention → sleep needed
2. **Conflict Density**: Many contradictions → consolidate
3. **Memory Saturation**: Working memory near capacity
4. **Time-based**: Fallback periodic trigger

**Trigger Condition**:
```
sleep_triggered = (
    entropy > threshold OR
    conflict_density > threshold OR
    memory_saturation > threshold OR
    time_elapsed > max_interval
)
```

**Phases**:
- NREM entry → consolidation
- REM entry → dreaming
- Wake exit → resume processing

**Reference**: SleepGate adaptive scheduling, WSCL sleep stages

### 4.6 NREM Sleep Module

**Function**: Consolidate, downscale, transfer to long-term

**Operations**:

#### 6.1 Replay
- Reactivate working memory episodes
- Use Hebbian plasticity rules
- No external input, internal replay

#### 6.2 Synaptic Downscaling (from SHY)
```
For each memory connection:
    new_strength = old_strength * (average_strength / current_strength)
```
- Proportional downscale
- Preserve relative importance
- Create capacity for new learning

#### 6.3 Consolidation
- Merge related episodes into single concept
- Compress semantic graph
- Strengthen high-value connections

#### 6.4 Transfer
- Move consolidated memories to long-term storage
- Update graph structure
- Prune weak connections

**Reference**: SHY downscaling, Nature 2022 Hebbian, WSCL NREM

### 4.7 REM Sleep Module (Dreaming)

**Function**: Explore feature space, create novel combinations

**Operations**:

#### 7.1 Dream Generation
- Generate novel combinations from existing concepts
- Explore "what if" scenarios
- Recombine unrelated concepts

#### 7.2 Feature Exploration
- Expose network to unseen patterns
- Explore potential future scenarios
- Prepare for novel situations

#### 7.3 Integration
- New combinations linked to existing knowledge
- Strange ideas checked against memory
- Validated insights retained

**Reference**: WSCL REM dreaming, Forgetting Survey "desirable difficulties"

### 4.8 Long-Term Memory (Cortex-Equivalent)

**Function**: Stable, compressed, persistent storage

**Structure**: Compressed Semantic Graph
```
- Concepts are compressed summaries
- Relations are strong pathways
- Importance encoded in connection strength
- Old memories are stable but updatable
```

**Properties**:
- Capacity: Large (unbounded theoretically)
- Persistence: Survives sleep cycles
- Stability: Strong, hard to change
- Integration: Connected to other memories

### 4.9 Forgetting Module

**Function**: Active removal of noise, low-value, contradictory memories

**Trigger Conditions**:
- Low importance score sustained
- Contradiction detected
- Memory saturation pressure
- Age beyond retention threshold

**Operations**:

#### 9.1 Soft Forgetting
- Suppress low-value memories
- Reduce activation strength
- Still recoverable if needed

#### 9.2 Hard Forgetting
- Delete from working memory
- Remove from long-term if confirmed noise
- Update graph structure

**The "Goldilocks Zone"**:
```
forget_too_much = lose_important_info
forget_too_little = noise_accumulation

# Adaptive threshold
optimal_forgetting = balance(
    memory_saturation,
    importance_distribution,
    noise_ratio
)
```

**Reference**: Forgetting Survey taxonomy, SleepGate eviction gate, EWC protection

---

## 5. Data Structures

### 5.1 Concept Node
```python
Concept:
    id: str
    type: str  # 'event', 'fact', 'preference', 'concept'
    semantic_embedding: List[float]
    value_vector: List[float]  # [novelty, emotional, task, repeat, conflict]
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    strength: float  # connection strength to other concepts
```

### 5.2 Relation Edge
```python
Relation:
    id: str
    subject_id: str
    predicate: str  # 'causes', 'part_of', 'related_to', 'contradicts'
    object_id: str
    strength: float
    created_at: datetime
    bidirectional: bool
```

### 5.3 Episode (Working Memory)
```python
Episode:
    id: str
    timestamp: datetime
    concepts: List[str]  # concept IDs
    raw_content: str  # original text
    context: Dict  # situation, environment
    value_vector: List[float]
    state: str  # 'active', 'consolidating', 'archived'
```

### 5.4 Sleep Cycle Log
```python
SleepCycle:
    start_time: datetime
    end_time: datetime
    nrem_duration: float
    rem_duration: float
    memories_consolidated: int
    memories_forgotten: int
    dreams_generated: List[str]
```

---

## 6. Sleep Cycle Algorithm

### 6.1 Wake Loop
```
loop while awake:
    input = get_user_input()
    if attention_filter(input):
        concept = meaning_encoder(input)
        value = value_tagger(input)
        episode = create_episode(concept, value)
        working_memory.store(episode)

    if sleep_triggered():
        enter_sleep_phase()
```

### 6.2 Sleep Phase
```
def enter_sleep_phase():
    # NREM: Consolidate and downscale
    nrem_start()
    replay(working_memory)
    downscale(all_memories)  # SHY-style
    consolidate(working_memory, long_term_memory)
    nrem_end()

    # REM: Dream and explore
    rem_start()
    dreams = generate_dreams(long_term_memory)
    for dream in dreams:
        evaluate_and_integrate(dream)
    rem_end()

    # Wake
    return to_waking()
```

### 6.3 Forgetting During Sleep
```
def sleep_forgetting():
    # During NREM
    for memory in low_value_memories:
        if memory.strength < threshold:
            soft_delete(memory)

    # During REM
    for dream in contradictory_dreams:
        reject(dream)  # contradictions pruned
```

---

## 7. Key Innovations (IP Opportunities)

### 7.1 Meaning-Based Encoding (vs Token-Level)
**Current**: All systems use tokens/embeddings
**SleepAI**: Concept-graph with typed relations
**Innovation**: Semantic understanding, not just pattern matching

### 7.2 Multi-Stage Sleep (NREM + REM)
**Current**: WSCL has stages, others don't
**SleepAI**: NREM for consolidation, REM for synthesis
**Innovation**: Different operations in different sleep stages

### 7.3 Value-Based Forgetting
**Current**: Eviction based on age/conflict
**SleepAI**: Multi-dimensional importance signals
**Innovation**: Intentional, adaptive forgetting based on meaning

### 7.4 Dream Synthesis
**Current**: No generative replay
**SleepAI**: Novel combinations from old memories
**Innovation**: Creative problem-solving during sleep

### 7.5 "Goldilocks" Forgetting Threshold
**Current**: Fixed thresholds
**SleepAI**: Adaptive threshold based on memory state
**Innovation**: Optimal forgetting zone calculator

---

## 8. Implementation Priority

### Phase 1: Core (No Training Required)
1. Meaning encoder (concept extraction from text)
2. Value tagger (multi-dimensional importance)
3. Working memory (episode storage)
4. Basic retrieval (similarity search)

### Phase 2: Sleep (Light Training)
5. Sleep trigger (entropy + conflict)
6. NREM consolidation (Hebbian + downscale)
7. Long-term memory integration
8. Forgetting module

### Phase 3: Advanced (Research)
9. REM dreaming (generative replay)
10. Multi-session persistence
11. Cross-agent memory sync
12. Real-time adaptation

---

## 9. Technical Requirements (M1 MacBook Air)

### Phase 1 (8GB RAM):
- Python-based prototype
- LLaMA 7B via llama.cpp (quantized)
- Neo4j or networkx for graph memory
- ~4GB RAM usage

### Phase 2:
- Add PyTorch for sleep modules
- Hebbian learning layer
- ~6GB RAM usage

### Phase 3:
- GPU compute needed (external)
- Full training of sleep modules

---

## 10. Risk Assessment

| Risk | Mitigation |
|------|------------|
| M1 too slow | Use quantization, small models first |
| Concept extraction hard | Use existing LLM for extraction |
| Sleep learning unstable | Start with simple downscale, iterate |
| Memory grows unbounded | Implement forgetting early |
| No clear evaluation | Use benchmarks: LOCOMO, etc. |

---

## 11. Research Gaps for IP

| Gap | Description | IP Opportunity |
|-----|-------------|----------------|
| **Meaning-Sleep Bridge** | How semantic meaning affects consolidation | Novel mechanism |
| **Adaptive Forgetting** | Dynamic threshold based on memory state | Algorithm design |
| **Dream Synthesis** | Generating useful novel combinations | Architecture |
| **Multi-Agent Sleep** | Distributed agents consolidating together | Protocol design |
| **Value Tagging** | How to train importance signals | Training method |

---

## 12. Conclusion

SleepAI proposes a brain-inspired memory architecture with:

**Core Components**:
- Attention filter (meaning-based selection)
- Meaning encoder (concept graph)
- Value tagger (multi-dimensional importance)
- Working memory (hippocampal, fast)
- Long-term memory (cortical, stable)
- Sleep with NREM (consolidate) + REM (dream)
- Intentional forgetting module

**Key Differentiators**:
- Semantic meaning, not token processing
- True sleep phases, not just cache eviction
- Value-based forgetting, not just age-based
- Dream synthesis, not just retrieval

**Path Forward**:
- Phase 1 on M1 Air with existing LLMs
- Phases 2-3 with compute resources

---

*Document Version: 1.0*
*Analysis Date: April 2026*
*Project: SleepAI*
*Based on: 8 priority paper analyses*
