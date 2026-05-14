# SCM — memory that works like yours

> **The thesis.** Human memory has two phases: a **wake phase** that pays attention, encodes, retrieves, handles contradictions; and a **sleep phase** that consolidates the day, abstracts patterns, resolves conflicts, and fills knowledge gaps. **Existing AI memory only models the wake phase.** SCM models both.

> **One line.** SCM is working memory that behaves like yours — and learns about you while idle, the way your brain does during sleep.

---

## Why "like humans" is the whole pitch

Every other agent memory system — vector layers, graph stores, tiered RAG, off-the-shelf retrieval libraries — does some version of the same thing: **store facts, retrieve facts**. They differ in storage layout (vector / graph / tiered), but all of them are *awake-only reactive*. The agent never thinks about anything between sessions.

Your brain doesn't work that way. When you sleep, your hippocampus replays the day's experiences and your cortex consolidates them. Patterns emerge. Contradictions resolve. Knowledge gaps you noticed get noted for later. You wake up with a *better* version of yesterday's understanding — not the same one.

That's what SCM does for an agent. **Two phases, not one:**

| Phase | What it does | Biological analog | SCM modules |
|---|---|---|---|
| **Wake** | Selective attention. Encoding-by-importance. Cue-driven retrieval. Contradiction handling. Bounded working memory (~7 items). | Hippocampal encoding, prefrontal working memory, cue-driven recall | Phases 1-5 (AttentionGate, EventCompiler, SpreadingActivation, ContradictionVersioning) |
| **Sleep** | Pattern abstraction (schemas). Contradiction resolution. Adaptive forgetting. Knowledge-gap detection and curiosity-driven filling. Wake summary report. | NREM consolidation, REM dreaming, synaptic homeostasis | Phase 7 (M1-M6: IdleLearner, CrossSessionPool, SchemaExtractor, WakeSummary, CuriosityEngine, LifecyclePolicy) |

This is the entire pitch in one sentence:

> **SCM is the first agent memory layer that has both a wake phase and a sleep phase, like the only memory system in nature that actually works.**

---

## What it feels like to use

You build an agent. You drop SCM in front of it. Two things change:

**While the user is using the agent** (wake phase):
- The agent stops blurting back everything it heard. It pays attention selectively, like you do.
- It builds a structured event memory (who/what/when/where/why), not a flat vector blob.
- It retrieves by association, not just similarity. Asking "what should I avoid for lunch?" surfaces the peanut allergy mentioned three sessions ago, even though "lunch" wasn't in that conversation.
- When facts change, it versions them — knows the old value, knows the current value, knows the timeline.

**While the user is away** (sleep phase):
- The IdleLearner daemon notices the session has gone quiet.
- A deep-sleep cycle fires — only if battery, CPU, and cooldown allow.
- The cross-session memory pool pulls in episodes from prior days.
- The schema extractor finds patterns ("user runs Tuesdays", "user has Friday dinners with Mara").
- The curiosity engine notices terms the user has used but never explained ("OAuth flow", "Kubernetes ingress") and looks them up.
- Contradictions get resolved — when the user said they left Northstar and joined Atlas Labs, the old fact gets superseded.
- The WakeSummary is cached, ready for the next time the user comes back.

When the user returns, they ask: *"What did you notice while I was away?"*

```
While you were away I noticed three things:
  • You've changed jobs — I've moved you from Northstar Robotics to Atlas Labs in my records.
  • Your Tuesday-morning runs and Friday-night dinners with Mara have become weekly patterns.
  • You've mentioned 'OAuth flow' five times without explaining it; I read up on it.
    (Authorization protocol, redirect-based, token + scope.)
```

This moment is the product. **No other open-source memory system can produce it.**

---

## The wake summary — what the user actually sees

The interface to all of this is one endpoint:

```python
summary = engine.wake_summary(since_hours=24)
# → WakeSummary(
#     schemas_formed=[
#         "User runs every Tuesday morning",
#         "Friday dinners with Mara are weekly",
#         "User left Northstar Robotics; now at Atlas Labs"
#     ],
#     gaps_filled=[
#         {"term": "OAuth flow", "source": "static_dictionary"},
#         {"term": "Kubernetes ingress", "source": "llm"}
#     ],
#     contradictions_resolved=[
#         {"property": "employer", "old": "Northstar", "new": "Atlas"}
#     ],
#     narrative="While you were away I noticed you've changed jobs..."
# )
```

This is the feature people tweet about. It's the moment the agent feels **alive** — like it actually thought about you while you weren't there.

No other open-source memory system has this.

---

## How it works — three loosely-coupled layers

```
                ┌─────────────────────────────────────┐
                │  YOUR HARNESS                        │
                │  (LangChain, LlamaIndex, custom...)  │
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────────┐
                │  SCM CORE                            │
                │  Wake-phase: encode → bind → retrieve│
                │  Sleep-phase: consolidate → schema   │
                │  Phase 7:  idle daemon (M1)          │
                │            cross-session pool (M2)   │
                │            schema extraction (M3)    │
                │            wake summary (M4)         │
                │            curiosity engine (M5)     │
                │            lifecycle policy (M6)     │
                └────┬───────────────────────┬────────┘
                     │                       │
                     ▼                       ▼
          ┌──────────────────┐     ┌──────────────────┐
          │  LLM BACKEND     │     │  EMBEDDING       │
          │  (any provider)  │     │  (any provider)  │
          └──────────────────┘     └──────────────────┘
```

Three independent choices the deployer makes:
- **Harness** — what calls SCM
- **LLM** — what extracts concepts
- **Embedding model** — what powers vector retrieval

All three are swappable independently, at config time.

---

## Seamless with any LLM

SCM has zero LLM lock-in. Every LLM call goes through one abstraction (`LLMExtractor`) that ships with three backends already:

```python
from src.llm import LLMExtractor

# Local — Ollama (free, private)
llm = LLMExtractor(provider="ollama", model="llama3.2:latest")

# Cloud — DeepSeek (cheap)
llm = LLMExtractor(provider="deepseek")  # reads DEEPSEEK_API_KEY

# Cloud — OpenAI (best quality)
llm = LLMExtractor(provider="openai", model="gpt-4o-mini")

# Anthropic, Google, Anyscale, Together, Voyage — anything
# OpenAI-compatible — works via the openai_compat path
```

**Switch providers with a single env-var change.** Memory state isn't tied to the LLM — concepts are stored as text + structured metadata, not as embeddings of LLM internal representations.

The same interface accepts any LLM that:
- Returns text completions, OR
- Implements the OpenAI chat-completions API shape

That covers ~95% of all current and future LLM providers.

---

## Seamless with any embedding model

Same story for embeddings. `MeaningEncoder` accepts a backend selector:

```python
from src.core.encoder import MeaningEncoder

# Local sentence-transformers (default)
enc = MeaningEncoder(embedding_backend="sentence_transformers",
                     embedding_model_name="all-MiniLM-L6-v2")

# Local Ollama (free, higher quality, recommended)
enc = MeaningEncoder(embedding_backend="ollama",
                     embedding_model_name="nomic-embed-text")

# Cloud OpenAI / Voyage / Together
enc = MeaningEncoder(embedding_backend="openai_compat",
                     embedding_model_name="text-embedding-3-large")

# Hash fallback (offline, deterministic, for testing)
enc = MeaningEncoder(embedding_backend="hash")
```

Each backend is configured via env vars. Documented in `docs/DEPLOYMENT.md`.

---

## Seamless with any harness

The PyPI package exposes one Python class: `SCMEngine`. It's a 6-method interface:

```python
class SCMEngine:
    def message(text: str) -> tuple[str, dict]
    def sleep(mode: str = "deep") -> dict
    def memory_report() -> dict
    def wake_summary(since_hours: float | None = None) -> WakeSummary
    def export_memory() -> dict
    def import_memory(data: dict) -> None
```

This minimal surface fits anywhere. Concrete adapters:

### Plain Python (5 lines)

```python
from scm import SCMEngine

engine = SCMEngine(profile="chatbot")
response, meta = engine.message("Hello, my name is Saish")
print(response)  # SCM has stored the fact in LTM
```

### LangChain memory drop-in

```python
from src.integrations.langchain_adapter import SCMMemory

memory = SCMMemory(
    user_id="saish",
    base_url="http://localhost:8000/v1",
)
```

### LlamaIndex memory adapter

Same pattern — `BaseMemory` subclass that delegates to `ChatEngine`.

### FastAPI service mode (already exists)

```bash
venv/bin/uvicorn src.api.main:app
# POST /chat/message      ingest
# POST /chat/sleep        force consolidation
# GET  /chat/wake-summary read overnight learnings
# WS   /chat/ws/{session} streaming
```

Drop SCM in front of any agent loop, any framework, any UI. **Zero tight coupling.**

---

## What deployers see

Four pre-tuned profiles in `docs/DEPLOYMENT.md`. Pick one based on your constraints:

| Profile | LLM | Embedding | Cost / 30 turns | Privacy |
|---|---|---|---|---|
| **Offline-only** | heuristic regex | sentence-transformers | $0 | 100% local |
| **Ollama-only** | Ollama llama3 | Ollama nomic-embed-text | $0 | 100% local |
| **Hybrid (recommended)** | DeepSeek-chat | Ollama nomic-embed-text | ~$0.04 | text→cloud, vectors local |
| **All-cloud** | OpenAI gpt-4o-mini | OpenAI text-embedding-3-large | ~$0.06 | all→cloud |

Switch profiles with env var changes. No code changes.

---

## What SCM is NOT

Honest about scope so you know what you're getting:

- ❌ **Not a vector database.** It uses one (NetworkX in-memory + SQLite/Postgres backing), but the value is the lifecycle, not the index.
- ❌ **Not an LLM.** Bring your own. SCM doesn't generate responses — it manages what the LLM has access to.
- ❌ **Not a chat UI.** It's the backend memory layer; the UI is your problem (or use the demo at `localhost:8000/static`).
- ❌ **Not just a fact-extraction prompt over a vector DB.** It's an architecture — encoding, binding, retrieval, consolidation, forgetting, schema abstraction.
- ❌ **Not "set and forget" yet.** v0.1 is research-grade. The deployment guide closes most rough edges, but you'll need to read it.

---

## Why anyone should care

Three categories of user, three reasons:

### Researchers
- The first open-source platform for **lifelong agent memory** with explicit sleep stages.
- Brutal harness + LoCoMo++ benchmark are publishable contributions on their own.
- Honest reporting of failure modes (encoder dependence, workload sensitivity) makes it a credible reference architecture.

### Indie devs / hobbyists
- Privacy-first by default — Profile A/B run with zero cloud calls.
- Pluggable everything — match the LLM and embedding model to your hardware budget.
- Wake-Summary endpoint gives you a UX feature nobody else can ship.

### Product builders
- Differentiated positioning — autonomous learning is a category nobody else owns.
- Open-source MIT/Apache → no vendor lock.
- Real architecture: 35-page paper, 170+ unit tests, 4 brutal-harness scenarios that have already shipped fixes.

---

## How to start in 60 seconds

```bash
git clone <repo>
cd SleepAI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Profile B: Ollama-only (free, private)
ollama pull nomic-embed-text
ollama pull llama3.2

# Run the demo
python -c "
from scm import SCMEngine
e = SCMEngine(profile='chatbot')
e.message('Hi, I am Saish, and I run every Tuesday morning')
e.message('Tuesday again - out for a 5K')
e.sleep('deep')
print(e.wake_summary().narrative)
"
# → 'While you were away I noticed Tuesday-morning running has become a pattern...'
```

---

## The 18-month plan

| Quarter | Goal |
|---|---|
| **Q3 2026** | arXiv push, hosted demo, 60-sec video, first 100 GitHub stars, 5 lighthouse users |
| **Q4 2026** | LangChain + LlamaIndex official integrations, 1k stars, first paid customer |
| **Q1 2027** | SCM Cloud (managed hosted), 5k stars, public head-to-head benchmark suite |
| **Q2 2027** | Privacy-first vertical traction (legal / healthcare / on-prem enterprise), 10k stars, sustainable revenue |

This isn't a hope-and-pray. It's a known-distance run.

---

## Hero copy options (for the landing page)

Choose one. All express the same thesis:

**Option 1 — direct biological framing (recommended):**
> **Memory that works like yours.**
> Pays attention while awake. Consolidates while you're away.
> The first agent memory layer with a real sleep phase.

**Option 2 — outcome-focused:**
> **Your agent gets better between sessions, not just within them.**
> SCM consolidates the day's conversations into patterns, fills knowledge gaps,
> and produces a wake summary while you're not looking.

**Option 3 — competitor-anchored:**
> **Other memory layers store facts. SCM learns from them while you're idle.**
> Open-source agent memory with both wake and sleep phases.
> Works with any LLM. Privacy-first by default.

**Hero subline (any option):**
> Open-source · Works with any LLM · Privacy-first · 35-page paper · 170+ tests
> [Read the paper] [Try the demo] [Star on GitHub]

---

## Two-sentence elevator (for any introduction)

> SCM is the first agent memory layer that models both phases of human memory: a wake phase that pays attention and retrieves by association, and a sleep phase that consolidates the day's experiences into patterns, resolves contradictions, and fills knowledge gaps autonomously while the user is away. Drop it in front of any LLM, and your agent stops being amnesic between sessions — it gets better between them.
