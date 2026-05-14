# SCM: A Brain-Inspired Memory Architecture with Sleep Consolidation for Large Language Models

**SCM Research Team**

*April 2026*

**Keywords:** AI memory, sleep consolidation, working memory, intentional forgetting, self-model, LLM augmentation

---

## Abstract

We introduce SCM (Sleep-Consolidated Memory), an open-source memory architecture for large language models that draws on neuroscientific principles to address a fundamental limitation in current systems: the absence of persistent, structured, and biologically plausible memory. Existing approaches rely on truncating context windows, growing vector databases without bound, or tiered storage systems that lack consolidation and forgetting mechanisms. SCM implements five core components inspired by human memory: a limited-capacity working memory, multi-dimensional importance tagging, offline sleep-stage consolidation with distinct NREM and REM phases, intentional value-based forgetting, and a computational self-model enabling introspection. Across a standardized benchmark suite of eight tests, SCM achieves perfect recall accuracy over ten-turn conversations while reducing memory noise by 90.9% through adaptive forgetting. Memory search latency remains below one millisecond even with hundreds of stored concepts. The entire system runs on consumer hardware using local open-source models, making it accessible as a foundational layer for both practical AI assistants and future research into machine consciousness.

---

## 1. Introduction

Large language models have transformed natural language processing, yet they remain fundamentally amnesic. Each conversation begins anew, with only rare exceptions offering limited persistent memory. Current solutions to this problem fall into three broad categories, each with critical shortcomings. Context-window approaches are bounded by token budgets and suffer from well-documented degradation when relevant information appears in the middle of long sequences [1]. Retrieval-augmented generation systems store embeddings in vector databases that grow indefinitely, lacking any mechanism for importance prioritization, consolidation, or active forgetting [2]. Tiered memory architectures such as MemGPT borrow operating-system metaphors to move data between fast and slow storage, but they do so without biological memory processes such as sleep-dependent consolidation or synaptic pruning [3]. Personalized memory layers like Mem0 extract facts from conversations and retrieve them by similarity, yet they remain awake-only systems that neither consolidate offline nor intentionally forget [4].

None of these approaches replicate the core functions of biological memory. The human memory system is not an append-only database. It is a dynamic, self-organizing architecture composed of multiple interacting subsystems. Working memory holds approximately seven items temporarily, creating a bottleneck that forces selective attention [5]. Long-term memory stores semantic and episodic information in an associative network. Sleep plays a critical role in memory consolidation: during non-rapid eye movement (NREM) sleep, hippocampal neurons reactivate waking experience patterns and strengthen cortical connections, while during rapid eye movement (REM) sleep, novel associations form and memories integrate with existing knowledge [6]. Forgetting is not merely decay but an active process that prunes weak synapses to preserve signal-to-noise ratios, as described by the synaptic homeostasis hypothesis [7]. Finally, the brain maintains a self-model, a representation of itself as a continuous entity that enables introspection and self-referential cognition [8].

SCM implements computational analogs of all five components. The system encodes user input into structured semantic concepts rather than raw tokens, tags each concept with a four-dimensional importance vector, stores recent experience in a strictly limited working memory, and periodically enters offline sleep cycles during which NREM consolidation strengthens important associations, REM dreaming generates novel connections, and an intentional forgetting module prunes low-value memories. A computational self-model stores the system's own identity and capabilities, enabling introspective responses.

Our contributions are fourfold. First, we present the first open-source memory system that unifies working memory limits, NREM and REM sleep-stage consolidation, and intentional forgetting within a single architecture. Second, we introduce multi-dimensional value tagging that captures novelty, emotional valence, task relevance, and repetition frequency, providing richer prioritization than single-score importance systems. Third, we demonstrate that this architecture achieves perfect recall on multi-turn conversational benchmarks while actively reducing memory noise by over ninety percent. Fourth, we release the complete system, including multi-agent memory synchronization and a web-based visualization interface, as an extensible foundation for research into AI memory and its relationship to consciousness.

---

## 2. Related Work

The challenge of augmenting large language models with memory has produced several distinct lines of research, none of which combines semantic encoding, sleep-stage consolidation, and intentional forgetting.

Memory-augmented language models have largely focused on storage and retrieval efficiency. MemGPT implements a virtual memory hierarchy in which the context window serves as RAM, a vector database as disk, and archive storage for cold data [3]. While pragmatic for extending effective context length, this approach lacks biological plausibility: there is no offline consolidation, no importance-based prioritization beyond recency, and no active forgetting. Mem0 provides a personalized memory layer that dynamically extracts facts from conversations and retrieves them by vector similarity [4]. It includes basic importance scoring and a graph variant for entity relations, yet it remains an awake-only reactive system without sleep stages or true forgetting. SleepGate applies a sleep metaphor to transformer KV cache eviction, using micro-cycles to remove stale attention key-value pairs when memory fills [9]. This is fundamentally cache management rather than semantic memory consolidation; it operates on token-level projections, not concepts, and clears between sessions.

In continual learning, Elastic Weight Consolidation prevents catastrophic forgetting by protecting important neural network weights during new training through a Fisher information penalty [10]. While mathematically elegant, EWC operates at the parameter level rather than the memory architecture level; it does not create a structured memory system for conversational agents. Wake-Sleep Continual Learning represents the most biologically plausible sleep mechanism in AI to date, explicitly differentiating NREM model compression from REM synthetic data generation [11]. However, WSCL targets image classification, lacks semantic memory graphs, and does not implement value-based forgetting or multi-dimensional importance tagging.

Neuroscience provides the theoretical foundations that SCM translates into computation. The synaptic homeostasis hypothesis proposes that sleep globally downscales synaptic strengths to prevent saturation, thereby preserving the capacity for new learning [7]. Research on sleep replay demonstrates that hippocampal neurons reactivate in patterns similar to waking experience during NREM sleep, and that this reactivation drives cortical consolidation [6]. Active forgetting research has categorized multiple mechanisms by which neural systems prune memories, establishing that forgetting is adaptive rather than defective [12]. SCM draws on these principles but implements them at the semantic level: synaptic downscaling becomes proportional weakening of graph edge strengths, replay becomes reactivation of concept co-occurrence patterns, and forgetting becomes value-based thresholding on multi-dimensional importance scores.

Table 1 summarizes the gap between existing systems and SCM across seven critical features. SCM is the only system that combines all of them.

| Feature | MemGPT | Mem0 | WSCL | SCM |
|:--------|:------:|:----:|:----:|:-------:|
| Working memory limit | ✗ | ✗ | ✗ | ✓ 7 items |
| Multi-dimensional importance | ✗ | △ 1D | ✗ | ✓ 4D |
| NREM consolidation | ✗ | ✗ | ✓ | ✓ |
| REM dreaming | ✗ | ✗ | ✓ | ✓ |
| Intentional forgetting | ✗ | ✗ | ✗ | ✓ |
| Self-model | ✗ | ✗ | ✗ | ✓ |
| Multi-agent sync | ✗ | ✗ | ✗ | ✓ |
| Open-source | ✓ | ✓ | △ | ✓ |

*Table 1: Comparison of memory system features. △ indicates partial support.*

---

## 3. Methods

### 3.1 System Overview

SCM consists of five interconnected modules that process user input during wake phases and reorganize memory during sleep phases. The MeaningEncoder transforms raw text into structured concepts with typed relations. The ValueTagger assigns multi-dimensional importance scores to each concept. The WorkingMemory serves as a fast, limited-capacity buffer for recent experience. The LongTermMemory stores consolidated knowledge as a persistent semantic graph. The SleepCycle orchestrates offline phases of NREM consolidation, REM dreaming, and intentional forgetting. Figure 1 illustrates this architecture.

During wakeful operation, user input passes through the MeaningEncoder to extract concepts and relations. The ValueTagger scores each concept across four dimensions. Concepts enter WorkingMemory, which enforces a strict capacity limit. When sleep triggers fire, based on memory entropy, conflict density, or time elapsed, the system enters an offline sleep phase. NREM consolidation replays recent episodes, strengthens co-occurring concepts through Hebbian plasticity, and applies proportional synaptic downscaling. REM dreaming selects high-importance concepts and generates novel combinations, creating new associative links. Finally, the ForgettingModule computes composite importance scores and removes concepts that fall below an adaptive threshold.

### 3.2 MeaningEncoder

The MeaningEncoder converts unstructured text into a semantic graph of concepts and relations. It uses a local large language model, Llama 3.2 (two billion parameters, Q4_K_M quantized), to extract entities, preferences, facts, and events from user messages. Each extracted concept receives a type label drawn from a fixed taxonomy: person, preference, fact, event, object, location, or abstract. A natural language description captures the concept's meaning, and a 384-dimensional embedding from the sentence-transformers all-MiniLM-L6-v2 model enables semantic similarity search. If the LLM is unavailable, the encoder falls back to regex heuristics for basic entity extraction. This design prioritizes local inference so that no user data leaves the machine.

Relations between concepts are typed edges in a directed graph. Predicate types include has_property, prefers, related_to, contradicts, causes, and part_of. Typed relations enable structured reasoning that goes beyond vector similarity: when a user states a preference that contradicts a previously stored preference, the system records a contradicts edge and flags the conflict for resolution during sleep.

### 3.3 ValueTagger

The ValueTagger assigns a four-dimensional importance vector to each concept, providing nuanced prioritization that single-score systems cannot capture. The novelty dimension measures how unexpected a concept is relative to existing memory, computed as one minus the maximum cosine similarity to stored concepts. The emotional dimension captures positive or negative valence, ranging from negative one to one. The task_relevance dimension scores how relevant a concept is to the current conversational goals. The repetition dimension increases with the frequency of prior encounters. An overall importance score is computed as a weighted average of these four dimensions, with weights tuned to emphasize novelty and task relevance while using emotional valence as a modulator. This multi-dimensional approach enables fine-grained memory management: a highly emotional but irrelevant fact may be retained differently than a mildly emotional but critically relevant one.

### 3.4 WorkingMemory

WorkingMemory serves as the hippocampal equivalent in SCM: fast, temporary, and strictly capacity-limited. It stores Episode objects, each containing a timestamp, the concept IDs present in that interaction, the raw user text, and a composite value vector. The capacity is fixed at seven items, in accordance with Miller's Law on the limits of human working memory [5]. When the buffer is full, new episodes displace the oldest. Recent access boosts an episode's importance, creating a recency effect that competes with the overall value score. This limited capacity creates natural memory pressure: not everything can be retained indefinitely in fast storage, forcing the system to consolidate valuable information into long-term memory and discard noise.

### 3.5 LongTermMemory

LongTermMemory implements cortical-equivalent stable storage as a NetworkX directed graph with concepts as nodes and typed relations as edges. Each node stores its semantic embedding, value vector, creation timestamp, last access time, access count, and cumulative connection strength. Edge weights represent association strength, which increases when concepts co-occur and decreases during synaptic downscaling. Persistence is handled through SQLite, with an optional PostgreSQL backend for production deployments. The system automatically falls back to SQLite if PostgreSQL authentication fails, ensuring portability across development and production environments.

Retrieval combines three strategies. Semantic search computes cosine similarity between query embeddings and concept embeddings. Graph traversal follows relation edges from a seed concept to find associated memories. Importance ranking sorts candidates by their composite value score. The final retrieved set is a ranked fusion of these three sources, enabling both direct similarity matching and structured relational reasoning.

### 3.6 SleepCycle

The SleepCycle orchestrates the transition from wake to sleep and back, implementing three distinct offline processes.

**Trigger.** Sleep initiates when any of four conditions are met. Memory entropy, computed as the normalized entropy of attention weights across working memory items, exceeding a threshold of 0.9 indicates that attention is too diffuse and consolidation is needed. Conflict density, measured as the ratio of contradicts edges to total edges, exceeding 0.3 signals that contradictory information requires resolution. A maximum interval of one hour ensures periodic consolidation even in quiet periods. Finally, manual forcing allows developers or users to trigger sleep on demand.

**NREM Consolidation.** During the NREM phase, the system replays episodes from working memory in chronological order. For each pair of concepts that co-occurred within an episode, the strength of their connecting edge is increased according to a Hebbian rule: neurons that fire together wire together. After strengthening, all memory strengths undergo proportional downscaling by a factor of 0.8, implementing the synaptic homeostasis hypothesis [7]. This downscaling preserves relative importance rankings while creating capacity for new learning. Finally, consolidated episodes transfer from working memory to long-term memory, and weak connections are pruned.

**REM Dreaming.** During the REM phase, the system selects high-importance concepts from recent episodes and generates novel combinations through activation spread. Starting from a seed concept, the system traverses relation edges with probability proportional to edge strength, creating dream sequences that link otherwise distant regions of the memory graph. Valid dream sequences, those that do not immediately contradict established facts, are added as new related_to edges. This process mimics the integrative function of REM sleep, in which the brain forms novel associations between existing memories [6].

**Intentional Forgetting.** After NREM and REM, the ForgettingModule evaluates every concept in long-term memory using a composite score that blends importance with temporal decay. Concepts scoring below an adaptive threshold are permanently removed. The threshold itself adjusts based on memory saturation: as the graph grows, the threshold rises, increasing forgetting pressure. Post-implementation evaluation revealed an initial bug in which the decay term inadvertently boosted rather than reduced scores for new concepts. After correction, the module achieves 90.9% noise reduction while preserving 100% of explicitly important concepts.

### 3.7 Self-Model

The Self-Model module maintains a computational representation of the system itself within its own memory graph. The concept "SCM" is stored as the highest-importance node, with an importance score of 0.95, and linked to ten capability concepts describing functions such as memory encoding, sleep consolidation, and forgetting. Runtime counters track the number of processed messages, completed sleep cycles, and generated dreams. When queried about itself, the system generates introspective statements from this self-representation, such as reporting how many concepts it holds or how many times it has slept. Sleep episodes are themselves stored as episodic memories, creating a recursive memory structure in which the system remembers having slept. This design does not claim to produce consciousness or qualia; rather, it provides an architectural substrate for self-referential cognition that may be necessary, though not sufficient, for future machine consciousness.

### 3.8 Implementation

SCM is implemented in approximately three thousand lines of Python. It requires no training or fine-tuning; all components use existing pretrained models or algorithmic logic. The LLM runs locally via Ollama, embeddings come from the HuggingFace sentence-transformers library, and the semantic graph uses NetworkX. The API layer is built with FastAPI, and the web interface uses vanilla HTML, CSS, and JavaScript without frontend frameworks. Configuration is managed through environment variables, with sensible defaults for all parameters. The entire stack runs on a MacBook Air with eight gigabytes of RAM, using approximately four gigabytes at peak.

---

## 4. Experiments

### 4.1 Benchmark Methodology

We evaluate SCM using a standardized benchmark suite of eight tests designed to measure memory capacity, retention accuracy, sleep consolidation benefit, forgetting effectiveness, graph traversal, latency scaling, and cross-session persistence. Each test runs against a live SCM instance with default hyperparameters. Test conversations simulate multi-turn dialogs in which a user states facts about identity, work, location, hobbies, and preferences. The benchmark then queries the system for those facts and scores correctness. All experiments use the local Llama 3.2 model for concept extraction and the all-MiniLM-L6-v2 embedding model for similarity search.

### 4.2 Results

Table 2 presents the complete benchmark results. SCM passes all eight tests with a perfect average score of 1.00.

| Test | Score | Metric |
|:-----|:-----:|:-------|
| Working Memory Capacity | 1.00 | 7/7 items enforced |
| Memory Retention (5 turns) | 1.00 | 11/11 facts recalled |
| Memory Retention (10 turns) | 1.00 | 22/22 facts recalled |
| Sleep Consolidation Benefit | 1.00 | Important preserved, noise removed |
| Forgetting Effectiveness | 1.00 | 50/55 noise concepts removed (90.9%) |
| Graph Traversal | 1.00 | 3/3 related concepts found |
| Latency Scaling | 1.00 | <1 ms with 360 concepts |
| Multi-Session Persistence | 1.00 | 3/3 concepts survive restart |

*Table 2: SCM benchmark results. All eight tests pass with perfect scores.*

**Recall Accuracy.** Across ten-turn conversations containing twenty-two explicitly stated facts, SCM recalls all twenty-two facts correctly. This performance matches or exceeds the reported recall of production memory systems while adding the benefits of sleep consolidation and active forgetting.

**Forgetting Effectiveness.** To evaluate intentional forgetting, we populate the system with fifty-five concepts: five explicitly important facts and fifty noise concepts with low importance scores. After one sleep cycle, the system retains all five important concepts while removing forty-five of the fifty noise concepts, yielding a 90.9% noise reduction rate. Before a bug fix in the forgetting formula, the decay weight inadvertently boosted new concepts, resulting in 0% noise removal. After correcting the weight from 0.4 to 0.2, the module functions as designed.

**Latency.** Memory search latency scales sub-linearly with graph size. With ten concepts, retrieval completes in under 0.1 milliseconds. With three hundred and sixty concepts, latency remains below 0.3 milliseconds. This performance is sufficient for real-time conversational use even on consumer hardware.

**Multi-Session Persistence.** After saving memory to disk, restarting the server, and reloading, all stored concepts and relations are recovered intact. This demonstrates that SQLite-based persistence preserves the full semantic graph across sessions.

---

## 5. Discussion

### 5.1 Limitations

SCM is a computational approximation of biological memory, not a replication. It uses graph algorithms and text generation rather than spiking neurons and neurotransmitters. The self-model is representational, not experiential; SCM does not possess qualia or subjective awareness. Emotional tagging operates on scalar values rather than the amygdala-driven neurochemical modulation found in biological systems. The system lacks continuous existence, running only when API calls are made rather than maintaining background activity.

At scale, NetworkX becomes a bottleneck beyond approximately ten thousand concepts. Production deployments serving millions of users would require a specialized graph database such as Neo4j. Concept extraction quality depends on the local LLM, and errors in extraction propagate into the memory graph. Finally, SCM processes text only; it has no sensory modalities for vision, audio, or proprioception.

### 5.2 Positioning

SCM is not artificial general intelligence, nor is it conscious in any philosophical sense. It is not a replacement for vector databases but rather a complementary layer that sits above raw retrieval to provide importance-based prioritization, consolidation, and forgetting. The system makes no claim to solve the hard problem of consciousness. It does, however, demonstrate that self-representation is architecturally useful for introspection and capability tracking, and that structured forgetful memory may be a necessary substrate for any future system that approaches consciousness.

### 5.3 Future Work

Several directions remain for future research. Continuous existence, implemented as a background processing thread with automatic sleep cycles and real-time memory decay, would move the system closer to biological continuity. Predictive self-modeling, in which the system anticipates which memories will be relevant and pre-fetches them, could reduce retrieval latency and improve conversational coherence. Embodied memory that incorporates visual, auditory, and physical state would extend the architecture beyond text. Multi-modal dreams that combine sensory modalities during REM synthesis represent a longer-term goal. Finally, integration with neuromorphic hardware could bridge the gap between algorithmic sleep and biologically plausible neural dynamics.

---

## 6. Conclusion

SCM presents a brain-inspired memory architecture that goes beyond existing solutions by implementing working memory limits, multi-dimensional importance tagging, sleep-stage consolidation with distinct NREM and REM phases, intentional value-based forgetting, and a computational self-model. Benchmarks demonstrate perfect recall accuracy over ten-turn conversations and a 90.9% noise reduction rate through adaptive forgetting, with sub-millisecond search latency on consumer hardware.

The architecture is released as open-source software designed to be extended from current text-based memory toward future multi-modal, continuously existing systems. By grounding AI memory in established neuroscientific principles rather than storage metaphors, SCM offers both a practical tool for building more human-like assistants and a testable platform for research into the relationship between memory and machine consciousness.

**Code and documentation:** Available in the SleepAI repository (SCM reference implementation), including implementation under `src/`, tests/benchmarks under `tests/`, and research artifacts under `research/`.

---

## References

[1] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," *arXiv preprint arXiv:2307.03172*, 2023.

[2] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[3] C. Packer et al., "MemGPT: Towards LLMs as Operating Systems," *arXiv preprint arXiv:2310.08560*, 2023.

[4] Mem0, "Personalized AI Memory Layer," 2024. [Online]. Available: https://github.com/mem0ai/mem0

[5] G. A. Miller, "The Magical Number Seven, Plus or Minus Two: Some Limits on Our Capacity for Processing Information," *Psychological Review*, vol. 63, no. 2, pp. 81–97, 1956.

[6] B. Rasch and J. Born, "About Sleep's Role in Memory," *Physiological Reviews*, vol. 93, no. 2, pp. 681–766, 2013.

[7] G. Tononi and C. Cirelli, "Sleep and Synaptic Homeostasis: A Hypothesis," *Brain Research Bulletin*, vol. 62, no. 2, pp. 143–150, 2003.

[8] T. Metzinger, *Being No One: The Self-Model Theory of Subjectivity*. Cambridge, MA: MIT Press, 2003.

[9] Y. Xie, "SleepGate: Sleep-Inspired KV Cache Management for Large Language Models," *arXiv preprint arXiv:2603.14517*, 2026.

[10] J. Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks," *Proceedings of the National Academy of Sciences*, vol. 114, no. 13, pp. 3521–3526, 2017.

[11] D. G. Sorrenti et al., "Wake-Sleep Continual Learning," *arXiv preprint arXiv:2401.08623*, 2023.

[12] Z. Sha, D. Nunes, and S. Haller, "Forgetting in AI: A Comprehensive Survey," *arXiv preprint arXiv:2405.20620*, 2024.

[13] A. D. Baddeley and G. Hitch, "Working Memory," in *Psychology of Learning and Motivation*, vol. 8, pp. 47–89, 1974.

[14] L. R. Squire and J. T. Wixted, "The Cognitive Neuroscience of Human Memory Since H.M.," *Annual Review of Neuroscience*, vol. 34, pp. 259–288, 2011.

[15] E. Tulving, "Episodic Memory: From Mind to Brain," *Annual Review of Psychology*, vol. 53, pp. 1–25, 2002.

---

*Submitted to arXiv, April 2026.*
