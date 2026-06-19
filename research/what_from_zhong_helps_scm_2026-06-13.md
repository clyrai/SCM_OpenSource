# What From Zhong's Papers Can Practically Help SCM — 2026-06-13

**Purpose:** Identify concrete, implementable ideas from Zhong et al. that could improve SCM's architecture. Not citation advice — engineering advice.

---

## 1. Recursive Semantic Chunking for MeaningEncoder (HIGH IMPACT)

### Current SCM Behavior
`src/core/encoder.py:409` — `MeaningEncoder.extract()` takes raw text and produces a flat list of concepts. The LLM extracts entities, preferences, facts, and events in a single pass. No hierarchy.

### What Zhong Does
Zhong's chunking algorithm recursively segments text into a K-ary tree:
```
Document → K chunks → each chunk → K sub-chunks → ... → tokens
```
Each node is a semantically coherent span. K is the max branching factor (K* ∈ [2,6] empirically).

### What SCM Could Do
Instead of flat extraction, SCM could:
1. **First chunk** the input text into K semantically coherent segments
2. **Then extract** concepts from each segment independently
3. **Store hierarchy**: episode → segments → concepts (parent-child links)

### Concrete Implementation
```
Current:  text → LLM → [concept1, concept2, concept3, ...]
Proposed: text → chunk(K=4) → [segment1, segment2, ...] → LLM per segment → hierarchical concepts
```

### Benefits
- **Better concept quality**: LLM extracts from smaller, focused chunks rather than long text
- **Multi-resolution retrieval**: Search at segment level for broad queries, concept level for specific queries
- **Natural grouping**: Related concepts stay grouped under their source segment

### Effort
Medium. Requires modifying `MeaningEncoder._extract_with_llm()` to first chunk, then extract per chunk. The chunking itself could use the existing LLM with a simple prompt: "Split this text into at most 4 semantically coherent segments."

---

## 2. Entropy-Based Noise Metric for Forgetting (HIGH IMPACT)

### Current SCM Behavior
`src/sleep/forgetting_dynamics.py:29` — `ForgettingDynamics` computes retention score from 6 factors:
- grasp, salience, rehearsal, association density, recency, interference

### What Zhong Shows
Natural language carries ~80% redundancy. The entropy rate (~1 bit/char) means most tokens are predictable from context. Low-entropy content is redundant; high-entropy content carries information.

### What SCM Could Do
Add **source-text entropy** as a 7th factor in retention scoring:
- Compute entropy of the source text that produced a concept
- Low-entropy (predictable, redundant) content → lower retention → higher forgetting priority
- High-entropy (surprising, informative) content → higher retention

### Concrete Implementation
In `ForgettingDynamics._compute_retention_score()`:
```python
# New factor: source entropy
source_entropy = self._compute_text_entropy(concept.source_text)
entropy_factor = source_entropy / self.max_expected_entropy  # normalize to [0,1]
retention_score += self.entropy_weight * entropy_factor
```

### Benefits
- **Principled noise definition**: Instead of heuristic importance scoring, use information-theoretic redundancy
- **Aligns with Zhong's finding**: 80% of language is redundant → 80% of stored concepts should be forgettable
- **Validates SCM's 90.9% noise reduction**: The information-theoretic prediction matches SCM's empirical result

### Effort
Low. Add one computed field to concepts and one weight to the retention formula.

---

## 3. Adaptive K for Content-Aware Encoding (MEDIUM IMPACT)

### Current SCM Behavior
`src/core/attention_gate.py:36` — `AttentionGate` computes salience and encoding intensity (STRONG/NORMAL/WEAK/SKIP). But it doesn't adapt HOW MANY concepts to extract based on content complexity.

### What Zhong Shows
K* varies by corpus:
| Corpus | K* | Complexity |
|--------|-----|-----------|
| TinyStories | 2 | Simple |
| FairytaleQA | 3 | Low-moderate |
| RedditStories | 4 | Moderate |
| arXivAbstracts | 4 | Moderate-high |
| ModernPoetry | 6 | High |

### What SCM Could Do
Estimate content complexity and adapt extraction depth:
- Simple factual input ("My name is Saish") → K=2, extract 1-2 concepts
- Complex narrative ("Let me tell you about my week...") → K=5-6, extract more concepts with hierarchy

### Concrete Implementation
In `AttentionGate`, after computing salience:
```python
# Estimate content complexity
complexity = self._estimate_complexity(text)  # e.g., sentence count, entity density
K = max(2, min(6, int(complexity * 2)))  # map to [2,6]
# Pass K to MeaningEncoder for chunking depth
```

### Benefits
- **Efficient extraction**: Don't over-extract from simple inputs
- **Better coverage**: Extract more deeply from complex inputs
- **Matches human cognition**: Zhong shows K reflects working memory load during comprehension

### Effort
Low-medium. Add complexity estimation and pass K to encoder.

---

## 4. Entropy-Adaptive Sleep Triggers (MEDIUM IMPACT)

### Current SCM Behavior
Sleep triggers when:
- Memory entropy > 0.9 (attention is too diffuse)
- Conflict density > 0.3
- Max interval elapsed (1 hour)
- Manual trigger

### What Zhong Shows
Different content types have different "normal" entropy rates:
- Children's books: ~1.2 nats/token
- Reddit stories: ~2.5 nats/token
- Modern poetry: ~3.2 nats/token

### What SCM Could Do
Adapt the entropy threshold based on recent content complexity:
- If recent conversations are simple (low K*), lower the sleep threshold
- If recent conversations are complex (high K*), allow higher entropy before triggering

### Concrete Implementation
In `SleepTrigger`:
```python
# Compute recent content complexity
avg_K = self._estimate_recent_K(recent_episodes)
# Adjust threshold: simpler content → lower threshold
adaptive_threshold = 0.7 + (avg_K / 6) * 0.3  # range [0.7, 1.0]
```

### Benefits
- **Content-aware consolidation**: Simple conversations consolidate earlier
- **Avoids premature sleep**: Complex conversations get more processing time before consolidation
- **Matches biological reality**: Sleep need varies with cognitive load

### Effort
Low. Modify threshold computation in sleep trigger.

---

## 5. Hierarchical Wake Summary (MEDIUM IMPACT)

### Current SCM Behavior
`src/lifecycle/wake_summary.py` — Generates a flat narrative: "While you were away I noticed three things: ..."

### What Paper A (PRL) Shows
Human recall is hierarchical: people summarize at different levels. A 1000-word story gets summarized in 1-2 sentences, then details are provided on request.

### What SCM Could Do
Generate multi-level wake summaries:
- **Level 1 (overview)**: 1-2 sentence summary of what changed
- **Level 2 (topics)**: Bullet list of main themes/patterns
- **Level 3 (details)**: Specific facts, contradictions, schemas

User asks "what did you notice?" → Level 1
User says "tell me more" → Level 2
User says "specifically?" → Level 3

### Concrete Implementation
In `WakeSummaryBuilder`:
```python
def build_summary(self, ...):
    level1 = self._generate_overview(changes)  # 1-2 sentences
    level2 = self._generate_topic_list(changes)  # bullet list
    level3 = self._generate_details(changes)  # specific facts
    return {"overview": level1, "topics": level2, "details": level3}
```

### Benefits
- **Better UX**: Users get concise summaries, can drill down
- **Matches human recall**: Zhong shows recall is hierarchical
- **Reduces information overload**: Don't dump everything at once

### Effort
Medium. Restructure wake summary output format.

---

## 6. Lognormal Buffer Sizing (LOW IMPACT)

### What Zhong Shows
Chunk-size distributions at each tree level follow lognormal distributions. The mean and variance are predictable from K and level L.

### What SCM Could Do
Use lognormal estimates to predict how many concepts should be extracted from a given text length:
- Short message (10 tokens) → expect 2-3 concepts
- Long message (100 tokens) → expect 8-12 concepts
- This sets a natural bound on extraction depth

### Benefits
- **Prevents over-extraction**: Don't extract 50 concepts from a simple greeting
- **Prevents under-extraction**: Don't miss concepts from long narratives
- **Information-theoretic grounding**: Buffer sizes match language structure

### Effort
Low. Add a lookup table or formula based on text length.

---

## Priority Ranking

| Idea | Impact | Effort | Priority |
|------|--------|--------|----------|
| Recursive semantic chunking for MeaningEncoder | HIGH | Medium | **P1** |
| Entropy-based noise metric for forgetting | HIGH | Low | **P1** |
| Adaptive K for content-aware encoding | MEDIUM | Low-Medium | **P2** |
| Entropy-adaptive sleep triggers | MEDIUM | Low | **P2** |
| Hierarchical wake summary | MEDIUM | Medium | **P2** |
| Lognormal buffer sizing | LOW | Low | **P3** |

---

## What's NOT Useful

| Zhong's Focus | Why It Doesn't Help SCM |
|--------------|------------------------|
| Random tree ensemble mathematics | Too theoretical for implementation |
| Corpus-level KL divergence fitting | SCM is per-user, not corpus-level |
| Shannon entropy rate derivation | Already known, not actionable |
| LLM perplexity estimation | SCM doesn't need perplexity |
| Renormalization group analysis | Pure math, no implementation value |
| Beta distribution splitting | SCM doesn't use tree splitting |

---

## Bottom Line

The most valuable thing from Zhong's work is **the recursive chunking algorithm**. It's a concrete, implementable technique that could improve SCM's MeaningEncoder from flat extraction to hierarchical extraction. Everything else is secondary.

The second most valuable thing is **entropy as a noise metric**. It gives a principled, information-theoretic foundation for what SCM already does heuristically (90.9% noise reduction). Adding it as a factor in retention scoring would make the forgetting module more rigorous.

The rest (adaptive K, entropy-adaptive triggers, hierarchical summaries) are nice-to-haves that could be implemented incrementally.
