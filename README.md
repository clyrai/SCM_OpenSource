# SCM — Memory that works like yours

> **Other memory layers store facts. SCM learns from them while you're idle.**

SCM (Sleep-Consolidated Memory) is the first agent memory layer with both a **wake phase** and a **sleep phase** — like the only memory system in nature that actually works.

[📄 Paper (35 pages)](research/SCM_Final_Paper.pdf) · [🛠 Deployment guide](docs/DEPLOYMENT.md) · [🔌 Integrations](docs/INTEGRATIONS.md) · [🗺 Roadmap](docs/ROADMAP.md) · [📊 Benchmarks](docs/BENCHMARKS.md)

---

## What's different

Every other agent memory product does the same thing in different shapes: **store facts, retrieve facts**. They never think between sessions. Your agent forgets to think the moment you stop talking to it.

That's not how memory works in any system that does it well — including yours. **Sleep is when memory consolidates.** When you sleep, your hippocampus replays the day's experiences and your cortex abstracts patterns from them. You wake up with a *better* version of yesterday's understanding, not the same one.

SCM does both phases:

| Phase | What happens | Bio analog | SCM modules |
|---|---|---|---|
| **Wake** | Selective attention. Encoding-by-importance. Cue-driven retrieval. Contradiction handling. Bounded working memory (~7 items). | Hippocampal encoding, working memory, cue-driven recall | Phases 1-5 |
| **Sleep** | Pattern abstraction. Contradiction resolution. Adaptive forgetting. Knowledge-gap detection and curiosity-driven filling. Wake summary report. | NREM consolidation, REM dreaming, synaptic homeostasis | Phase 7 (M1-M6) |

**The result the user sees:** when they come back from being away, the agent reports what it noticed.

```
> What did you notice while I was away?

While you were away I noticed three things:
  • You've changed jobs — I've moved you from Northstar Robotics to Atlas Labs.
  • Your Tuesday-morning runs and Friday-night dinners with Mara have become weekly patterns.
  • You've mentioned 'OAuth flow' five times without explaining it; I read up on it.
    (Authorization protocol, redirect-based, token + scope.)
```

This moment is the product. **No other open-source memory system has it.**

---

## Five-line quickstart

```python
from scm import SCMEngine

engine = SCMEngine(profile="chatbot")
engine.message("Hi, I'm Saish. I run every Tuesday morning.")
engine.message("Tuesday again — out for a 5K.")
engine.sleep("deep")
print(engine.wake_summary().narrative)
# → "While you were away I noticed Tuesday-morning running has become a pattern..."
```

Or, drop the SCM MCP server into Claude Desktop / Cursor / any MCP client and add five tools (`add_memory`, `search_memory`, `consolidate`, `wake_summary`, `forget`) automatically. See [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md).

---

## Install

```bash
pip install scm-memory
```

For local development from this repository:

```bash
git clone https://github.com/Saish15/sleepai.git
cd sleepai
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

For the recommended privacy-first profile (free, local, no cloud calls):

```bash
ollama pull nomic-embed-text     # 274 MB embedding model (recommended)
ollama pull llama3.2:latest      # ~4 GB chat / extraction model
```

That's it. SCM auto-detects Ollama and uses it. Four deployment profiles documented in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md):

| Profile | LLM | Embedding | Cost / 30 turns | Privacy |
|---|---|---|---|---|
| **A** Offline-only | heuristic regex | sentence-transformers MiniLM | $0 | 100% local |
| **B** Ollama-only (recommended) | Ollama llama3 | Ollama nomic-embed-text | $0 | 100% local |
| **C** Hybrid | DeepSeek-chat | Ollama nomic-embed-text | ~$0.04 | text→cloud, vectors local |
| **D** All-cloud | OpenAI gpt-4o-mini | OpenAI text-embedding-3-large | ~$0.06 | all→cloud |

---

## Works seamlessly with any LLM and any harness

SCM doesn't care which LLM you use. Concept extraction goes through `LLMExtractor`; switch providers with one env var:

```bash
LLM_PROVIDER=ollama       # local, free
LLM_PROVIDER=deepseek     # cheap cloud
LLM_PROVIDER=openai       # premium
# Anthropic / Voyage / Together / any OpenAI-compatible endpoint also works
```

Same for embeddings — sentence-transformers, Ollama, or any OpenAI-compatible provider.

Drop SCM behind any agent framework:
- **MCP server** (Claude Desktop, Cursor, ChatGPT-with-MCP) — `scm mcp` in your config
- **REST API** (`/v1/memories`, `/v1/wake-summary`, etc.) — OpenAPI 3.1 spec at `/v1/openapi.json`
- **Python SDK** — `from scm import SCMEngine`
- **JavaScript SDK** — `import { SCM } from "scm-memory"` (Node 18+, Bun, browsers, Edge runtime)
- **LangChain memory adapter** — drop-in `BaseChatMemory` subclass
- **Plain HTTP** — POST `/v1/memories` from anything

Tool definitions exported in OpenAI / Anthropic / Gemini / OpenAPI formats from one source. See [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md) for 7 integration recipes.

---

## What SCM is NOT

- ❌ Not a vector database. It uses one (NetworkX in-memory + SQLite/Postgres backing), but the value is the lifecycle, not the index.
- ❌ Not an LLM. Bring your own.
- ❌ Not a chat UI. It's the memory backend; the UI is your problem (or wire it to the included `/static` demo page).
- ❌ Not just a fact-extraction prompt over a vector DB. SCM is a complete memory pipeline — encoding, binding, retrieval, consolidation, forgetting, schema abstraction.
- ❌ Not 100% production-polished yet. v0.7.x is research-grade with strong tests; the deployment guide closes most rough edges. See [`docs/STATUS.md`](docs/STATUS.md) for an honest current-state read.

---

## Honest comparison

We don't claim to dominate every memory benchmark. We dominate a different axis.

| Capability | Stateless vector layer | **SCM** |
|---|---|---|
| Vector retrieval | ✅ | ✅ |
| Working-memory bound | ❌ | ✅ |
| Event-structured encoding | ❌ | ✅ |
| Spreading-activation retrieval | ❌ | ✅ |
| Contradiction-safe versioning | ❌ | ✅ |
| Sleep-stage consolidation | ❌ | ✅ |
| Schema extraction (REM) | ❌ | ✅ |
| Wake-summary endpoint | ❌ | ✅ |
| Curiosity-driven gap-filling | ❌ | ✅ |
| Idle-aware autonomous learning | ❌ | ✅ |

SCM does both jobs: vector retrieval (the table-stakes feature) plus continuous learning during idle time (the differentiator).

---

## Status

- **322 regression tests** passing (`pytest tests/ -q`)
- **143 focused regression tests** for Phase 7 + retrieval (`pytest tests/test_*spreading* tests/test_*idle* tests/test_*curiosity* -q`)
- **16/16 brutal LangChain harness scenarios** passing (multi-day persona, contradiction, idle wake-summary, multi-user isolation, failure mode)
- **5,561× p50 latency speedup** on `add_memory` since v0.7.2 (async ingest)
- **5-40× RAM saved at multi-user scale** since v0.7.3 (embedding-model singleton)

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for every measured number with reproduction instructions.

---

## Architecture

```
                ┌───────────────────────────────┐
                │  YOUR AGENT / HARNESS         │
                │  (LangChain, Claude, custom)  │
                └─────────────┬─────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │  SCM CORE                                  │
        │  Wake-phase: encode → bind → retrieve      │
        │  Sleep-phase: consolidate → schema → gap   │
        │  Phase 7:  M1 idle daemon                  │
        │            M2 cross-session pool           │
        │            M3 schema extraction (REM)      │
        │            M4 wake-summary endpoint        │
        │            M5 curiosity engine             │
        │            M6 lifecycle policy             │
        └────────┬─────────────────────┬─────────────┘
                 │                     │
                 ▼                     ▼
       ┌───────────────────┐   ┌───────────────────┐
       │  LLM BACKEND      │   │  EMBEDDING        │
       │  Ollama/DeepSeek/ │   │  Ollama/OpenAI/   │
       │  OpenAI/etc.      │   │  sentence-trans.  │
       └───────────────────┘   └───────────────────┘
```

35-page paper at [`research/SCM_Final_Paper.pdf`](research/SCM_Final_Paper.pdf). Documents architecture, formal definitions for all 11 equations, brutal-testing methodology, ALB pilot, LoCoMo + LoCoMo++ honest comparisons, encoder-dependence analysis.

---

## Project status

This project is in **active development**, working toward a public launch. See [`docs/STATUS.md`](docs/STATUS.md) for an unvarnished read of where it stands as a product, and [`docs/ROADMAP.md`](docs/ROADMAP.md) for what's coming next. The paper is ready and staged for arXiv submission but **held until the product-readiness checklist completes** — papers without products fade.

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Contact

`blobopera@proton.me`
