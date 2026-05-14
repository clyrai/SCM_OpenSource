# SCM / SleepAI — Product Vision

**Status:** Living document
**Last updated:** 2026-05-04
**Owner:** Project lead (saish)

---

## 1. The Vision in One Sentence

> **An AI that thinks and remembers like a human — selectively learning what matters,
> rapidly grasping new information, and continuing to learn during downtime so that
> when the user returns, the agent has grown.**

This is *not* a clone of any existing retrieval store or tiered storage layer.
This is an autonomous learning agent whose memory system has a biological
lifecycle — including learning-while-idle.

---

## 2. The Problem We're Actually Solving

Current AI memory systems are **stateless retrieval databases dressed up as memory**:

- They store facts when told to.
- They retrieve facts when asked.
- They sit idle when not called.
- They never learn anything the user didn't explicitly tell them.
- They don't consolidate, abstract, or grow on their own.

A human assistant who behaved this way would be useless. Imagine hiring an
assistant who:
- Forgot every pattern the moment you stopped talking.
- Never thought about anything between meetings.
- Had no internal life of their own.
- Could not connect Tuesday's conversation to Friday's.
- Could not say *"Hey, I noticed something while I was thinking about this..."*

That's what every memory system on the market is today.

**SCM's job is to be the first AI memory system that doesn't behave that way.**

---

## 3. The Five Human-Like Behaviors (in priority order)

### 3.1 Selective Learning — *"Not everything you say is worth remembering"*

Humans don't store every word of every conversation. They filter at multiple levels:
attention, salience, emotional relevance, recency, novelty, surprise. The result
is that a human assistant remembers your name, your preferences, the deadline you
mentioned — but not the small talk about the weather.

**What this means in the product:**
- The agent's encoding is *gated* by salience signals (novelty, task relevance,
  emotional weight, prediction error).
- Filler turns ("lol", "yeah", "did you see the game?") are buffered briefly and
  forgotten.
- Important turns ("I changed jobs", "my deadline is Friday", "I prefer X") are
  encoded with high intensity and made durable.
- The user feels the difference: the agent doesn't drown them in *"You also
  mentioned the weather on Tuesday."*

### 3.2 Quick Learning — *"One-shot grasp, like meeting a new person"*

When you tell a human your name, they don't need to hear it 5 times. One clear
exposure is usually enough. This is sometimes called *fast mapping* in cognitive
science — the brain can encode a strong, durable trace from a single salient
exposure.

**What this means in the product:**
- High-grasp concepts (clear, salient, schema-aligned) are encoded with high
  retention from one exposure.
- The agent can correctly recall your name, location, profession after you say
  them once — without needing repetition.
- This is measurable as *one-shot recall accuracy* — currently 1.0 on our
  Phase 6 human-memory suite.

### 3.3 Sleep-Time Learning — *"While the user sleeps, the agent learns"*

This is the **defining feature** of the vision. Human memory is profoundly
shaped by sleep:
- **NREM sleep** consolidates the day's experiences into long-term storage.
- **REM sleep** abstracts patterns, integrates new memories with old, and
  generates novel associations.
- **Slow-wave activity** prunes weak synapses to make room for tomorrow.

A human who skipped sleep for a week would be useless. An AI agent that doesn't
have an analog of this is also limited — it can never grow beyond the latest
session.

**What this means in the product:**
- When the user is away (laptop closed, agent idle, no API calls for N minutes),
  the agent enters its own "sleep."
- During this sleep, it runs consolidation, abstraction, and pruning — without
  the user having to ask.
- When the user comes back, the agent is **measurably better at retrieval and
  reasoning** than when they left.

### 3.4 A-Z Autonomous Learning — *"Don't just store what I told you"*

Humans don't only learn from explicit instruction. We read, observe, daydream,
explore. A great assistant studies the domain you work in even when you're not
looking, so they can be useful tomorrow on something you haven't discussed yet.

**What this means in the product:**
- The agent identifies **knowledge gaps** during sleep (concepts mentioned by
  the user but not understood; topics adjacent to user interests but missing
  context).
- During sleep, the agent optionally pulls from external knowledge sources
  (documents, web, Wikipedia, the user's notes folder) to fill those gaps.
- The agent grows "A-Z" — broader and broader coverage over time, like a human
  who keeps reading.

**This is opt-in and configurable.** Privacy-conscious users disable external
ingestion entirely. Power users grant access to their docs folder. Curious users
enable web search.

### 3.5 Continuous Existence — *"Most users don't run agents 24/7"*

A typical user runs an AI agent for ~1 hour a day. The remaining 23 hours, the
agent doesn't exist. This is wasteful: those 23 hours are when the agent should
be doing the deepest learning, because there's no interruption pressure.

**What this means in the product:**
- The agent's lifecycle is decoupled from API calls.
- A background daemon runs sleep cycles when idle, even while the laptop is
  closed (within reason — respects power state).
- Every "wake" can produce a summary: *"While you were away, here's what I
  consolidated, abstracted, and learned."*

---

## 4. The User Experience We're Designing For

### Day 1
User installs SCM, has a few conversations, closes laptop, goes to bed.

### Overnight
SCM's idle daemon runs:
1. Pulls today's working-memory and recent sessions.
2. Consolidates fact-level memories into the long-term graph.
3. Runs REM-style pattern extraction:
   - *"Three times today the user mentioned a deadline tied to a project named
     ProjectX. Schema: ProjectX has urgent deliverables this week."*
4. Identifies gap: user mentioned a tool called *Datadog* but the agent has no
   schema for it. (If external learning enabled:) fetches a brief from a
   configured source, integrates as a new concept node tagged
   `gap_filled_during_sleep`.
5. Forgets low-value items (small talk, redundant statements).
6. Writes a wake summary.

### Day 2 morning
User opens the agent. The agent's first message can be:
> *"Welcome back. While you were away I consolidated 23 memories from
> yesterday's conversations, noticed that you have three deadlines this week
> for ProjectX, and read up on Datadog so I can help when you bring it up
> again. Anything you want me to surface?"*

That experience does not exist anywhere in the AI assistant market today. That
is what SCM is for.

---

## 5. What "Success" Looks Like

We will know we have delivered the vision when these are all true:

| Criterion | Measurement |
|---|---|
| **Selective learning works** | Filler turns archived; durable facts retained. Measurable as noise-rejection rate ≥ 0.95 and clean-fact recall ≥ 0.80. |
| **Quick learning works** | One-shot recall accuracy ≥ 0.95 on stated facts (name, location, preference, schedule). |
| **Sleep-time learning works** | Closed-laptop test: idle for 8 hours; on wake, agent has consolidated >0 sessions, formed >0 abstract patterns, written a wake summary. |
| **A-Z autonomous learning works** | After 30 days of normal use: agent has formed schemas for at least 5 user-relevant topics it was never explicitly asked about. |
| **Continuous existence works** | Daemon runs reliably for 7+ days without manual intervention; respects laptop sleep/wake; never consumes >5% CPU when idle. |
| **The user feels the difference** | After 1 week of use, in a structured interview, ≥ 70% of users say the agent feels *"like it's actually thinking about my work between sessions"* (vs standard chat history / vector retrieval). |

---

## 6. What We Have Today (mapped to vision)

| Vision Capability | Status | Lives in |
|---|---|---|
| Selective learning (AttentionGate, salience, grasp) | ✅ **Built** | [src/core/attention_gate.py](../src/core/attention_gate.py), [src/core/value_tagger.py](../src/core/value_tagger.py) |
| Quick learning (one-shot grasp, schema overlap) | ✅ **Built** | `compute_grasp` in `value_tagger.py` |
| Forgetting dynamics (six-factor retention, decay, archival) | ✅ **Built** | [src/sleep/forgetting_dynamics.py](../src/sleep/forgetting_dynamics.py) |
| Contradiction-safe versioning | ✅ **Built** | [src/core/long_term_memory.py](../src/core/long_term_memory.py) |
| NREM consolidation (Hebbian + downscaling) | ✅ **Built** | [src/sleep/nrem.py](../src/sleep/nrem.py) |
| REM dreaming (concept replay, novel associations) | ✅ **Built** | [src/sleep/rem.py](../src/sleep/rem.py) |
| Dual-mode sleep (micro + deep) | ✅ **Built** | [src/sleep/sleep_cycle.py](../src/sleep/sleep_cycle.py) |
| Sleep-time paraphrase (clean fact form) | ✅ **Built** (Phase 6 fix) | [src/sleep/paraphrase.py](../src/sleep/paraphrase.py) |
| Self-model (the agent knows what it knows) | ✅ **Built** | [src/consciousness/self_model.py](../src/consciousness/self_model.py) |
| Reproducible benchmark (LOCOMO++) | ✅ **Built** | [tests/locomo_plus_benchmark.py](../tests/locomo_plus_benchmark.py) |
| IdleLearner daemon (autonomous sleep on inactivity) | ✅ **Built + tested** | [src/lifecycle/idle_learner.py](../src/lifecycle/idle_learner.py) |
| Cross-session memory pool | ✅ **Built + tested** | [src/core/cross_session_pool.py](../src/core/cross_session_pool.py) |
| Schema extraction during sleep | ✅ **Built + tested** | [src/sleep/schema_extractor.py](../src/sleep/schema_extractor.py) |
| Wake-up summary endpoint | ✅ **Built + tested** | [src/lifecycle/wake_summary.py](../src/lifecycle/wake_summary.py), `/chat/wake-summary/{session_id}` |
| Curiosity / gap-fill engine | ✅ **Built + tested** | [src/lifecycle/curiosity.py](../src/lifecycle/curiosity.py) |
| Lifecycle policy / resource gating | ✅ **Built + tested** | [src/lifecycle/lifecycle_policy.py](../src/lifecycle/lifecycle_policy.py) |
| PyPI package | ✅ **Published + tested** | `pip install scm-memory` |
| LangChain integration | ✅ **Built + tested** | [src/integrations/langchain_adapter.py](../src/integrations/langchain_adapter.py) |

**The core vision is now in code.** The remaining work is product distribution:
hosted demo, public launch assets, lighthouse users, and real-world validation.

---

## 7. What Landed in Phase 7

### M1. **IdleLearner daemon** ⭐ *the killer feature*
Built. SCM now tracks session activity and can trigger autonomous sleep cycles
when a user goes idle. This turns sleep from a manual demo action into an agent
lifecycle.

### M2. **Cross-session memory pool** ⭐ *the long-term self*
Built. Sleep can now consult recent prior sessions, so the agent can consolidate
"yesterday + today" instead of only the current working-memory window.

### M3. **Schema extraction in REM** ⭐ *the human-like pattern recognition*
Built. Deep sleep can emit abstract schema concepts from repeated patterns,
making wake-summary possible and making SCM more than fact storage.

### M4. **Wake-up summary** *the visible outcome*
Built. The user-visible product moment now exists: "while you were away, I
consolidated X, forgot Y, and noticed Z."

### M5. **Curiosity engine + external knowledge ingestion** *the A-Z learning*
Built as an opt-in path. SCM can identify knowledge gaps during sleep and fill
them from configured sources, while preserving the privacy-first default.

### M6. **Lifecycle scheduling polish** *the production behavior*
Built. The idle learner can be gated by policy, state can persist, and resource
conditions can block heavy background work.

**Phase 7 status:** complete enough for productization. The question is no
longer "can we build autonomous sleep memory?" It is now "can users try it,
understand it, and trust it quickly?"

---

## 8. Roadmap (Phases)

### Phase 7 — Continuous Existence
Complete. M1-M6 are implemented and tested, including idle learning,
cross-session consolidation, schema extraction, wake summary, curiosity, and
lifecycle policy.

### Phase 8 — Product Distribution
Current phase. Make SCM easy to try, easy to install, and easy to believe:
hosted demo, demo video, public repo, tutorials, and lighthouse users.

### Phase 9 — Real-World Validation
Run SCM with real users and real integrations. Collect failure cases, cost
data, wake-summary usefulness data, and head-to-head comparisons.

### Phase 10 — Multi-Modal Memory
Vision, audio, file ingestion. Out of scope for the immediate vision but the
natural extension.

---

## 9. Anti-Goals (what this is NOT)

To keep us focused, here is what SCM **explicitly does not aim to be**:

- ❌ Not a replacement for vector databases. We use them.
- ❌ Not optimized to win clean-fact retrieval benchmarks against pure vector
  RAG layers. SCM trades raw retrieval headroom for cross-session learning,
  and that trade is the point.
- ❌ Not artificial general intelligence. The agent does not reason about
  arbitrary topics; it reasons about its memory.
- ❌ Not consciousness. The self-model is representational, not experiential.
- ❌ Not always-on cloud surveillance. Idle learning runs on the user's
  device when configured locally; cloud mode is opt-in.
- ❌ Not a 24/7 background hog. Cooldowns, power-state awareness, and
  idle thresholds are first-class concerns.

---

## 10. How This Differentiates from Everything Else

| System | What it does | What it doesn't do |
|---|---|---|
| **Stateless retrieval libraries** | Fact extraction + vector retrieval | No sleep, no learning between sessions, no abstraction, no idle-time growth |
| **Tiered-storage memory layers** | Tiered storage with paging | No biological lifecycle, no consolidation, no autonomous learning |
| **Vector DBs** (Pinecone, etc.) | Raw embedding storage + search | Nothing about lifecycle |
| **Generative Agents** (Park 2023) | Reflection at end of episode | No continuous existence, no idle-time learning, no contradiction handling |
| **Voyager** (Wang 2023) | Skill library that grows | Domain-specific (Minecraft), not general memory |
| **MemoryBank** (Zhong 2024) | Memory that updates between sessions | No active learning during downtime |
| **WSCL** (Sorrenti 2024) | Wake-sleep training for image classifiers | Not a memory system for language agents |
| **SleepGate** (Xie 2024) | KV cache eviction with sleep metaphor | Token-level, not semantic; no learning |
| **SCM (this project)** | Selective + quick + sleep-time + A-Z autonomous learning agent for language models | Needs hosted demo, real-user validation, and public distribution |

**SCM is the only system in this list that even attempts the "agent learns
during user downtime" axis.** That is the moat. Defending it now means making
the behavior visible to users, not adding more hidden machinery.

---

## 11. The North-Star User Quote

If we get this right, the user testimonial we want to see is:

> *"I closed my laptop on Tuesday with a half-finished conversation about
> ProjectX. When I opened it Thursday, my agent greeted me with a one-line
> summary of what it had consolidated, named two patterns it had noticed about
> my workflow that I hadn't named myself, and surfaced a relevant doc from my
> notes folder it had read overnight. It was the first time using an AI
> assistant felt like having a real teammate."*

That quote is the entire product strategy in one paragraph. Everything else
is in service of producing that experience.

---

## 12. Decision Log

- **2026-05-01:** Vision document created. Confirmed by project lead that the
  *real* target is autonomous lifelong learning during idle, NOT competing
  on clean-fact retrieval benchmarks like LoCoMo. The workload-sensitivity
  paper content remains useful as supporting evidence, but the larger story
  is now autonomous idle-time learning.
- **Scope decision:** Multi-agent sync and self-model are existing features
  but not core to the vision; they will be maintained but not heavily
  invested in. Single-user, idle-learning agent is the primary product.
- **Privacy stance:** All external ingestion (M5) is opt-in. Default mode is
  "learn only from user conversations." Power users can grant additional
  sources.
- **2026-05-01 — Strategic sequencing committed:** Project lead confirmed the
  preferred order was to complete M1-M6 before public paper push. That gate is
  now passed; the next gate is product distribution: hosted demo, public repo,
  tutorials, and lighthouse users.
- **2026-05-04 — Phase 7 landed:** M1-M6 are built and tested. `scm-memory` is
  published/tested on PyPI, and LangChain integration has been tested. Strategy
  now shifts from "build autonomous memory" to "make the product publicly
  usable and validated."

---

## 13. Execution Sequence (committed 2026-05-01)

### Stage 1 — Build M1–M6

Complete.

### Stage 2 — Product readiness (current)

- Hosted demo with a two-click wake-summary experience.
- 60-90s product video showing ingest, idle sleep, and wake summary.
- Public repo readiness.
- PyPI install path documented as the default.
- npm package publish and JS quickstart.
- Lighthouse users and real integration feedback.

### Stage 3 — Paper rewrite (after product surface is real)

- New title direction: *"Autonomous Lifelong Learning for Language Agents
  via Sleep-Stage Memory Consolidation"*
- New benchmark: **idle-time learning effectiveness**, grounded in the M1-M6
  implementation and wake-summary usefulness.
- New baselines to add: Generative Agents (Park 2023), MemoryBank
  (Zhong 2024), Voyager (Wang 2023)
- The LoCoMo++ workload-sensitivity story stays as a supporting section,
  not the headline.

### Stage 4 — Publish (after product gate)

- arXiv preprint immediately after rewrite.
- Target venue: NeurIPS or ICLR workshop on AI agents / cognitive science
  for language models.

---

## 14. North-Star Metric for the Product

The single number that, if it improves, means the product is succeeding:

> **"Wake-up usefulness score"** — fraction of users who, after using SCM
> for 7+ days, agree that the agent's wake summary contained at least one
> insight or consolidation they found useful.

If this number is below 50%, we have not delivered the vision yet, no matter
what the benchmark numbers say.
