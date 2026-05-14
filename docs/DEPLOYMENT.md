# SCM Deployment Guide

SCM is **LLM-agnostic, embedding-agnostic, and database-agnostic by design**. There is no single "right" configuration — the tradeoffs between cost, privacy, latency, and recall quality differ enough that this guide describes four canonical profiles. Pick the one that matches your deployment constraints.

> **Two independent choices**: (1) which **LLM** to use for concept extraction at ingest time, and (2) which **embedding model** to use for vector retrieval. They do *not* have to be the same provider.

---

## At a glance

| Profile | LLM extractor | Embedding | Per-msg cost | Privacy | Quality |
|---|---|---|---|---|---|
| **A. Offline-only** | heuristic regex | sentence-transformers MiniLM | $0 | 100% local | floor |
| **B. Ollama-only** | local Ollama (llama3 / qwen) | local Ollama (nomic-embed-text) | $0 | 100% local | solid |
| **C. Hybrid cloud-LLM + local-embed** | DeepSeek or OpenAI | local Ollama (nomic-embed-text) | ~$0.04 / 30-turn day | text leaves device for extraction; embeddings stay local | high |
| **D. All-cloud** | OpenAI GPT-4o-mini or DeepSeek | OpenAI text-embedding-3 | ~$0.06 / 30-turn day | all text leaves device | best |

**ALB benchmark numbers** (v3/v4 pilot) were produced under **Profile B** (Ollama-only). Profile C is the recommended starting point for most consumer deployments.

---

## How to choose

```
                       Are you OK with text leaving the device?
                       │
                       ├─ No: Profile A (offline) or B (Ollama-only)
                       │   │
                       │   ├─ Have GPU or beefy CPU? → Profile B
                       │   └─ Bare metal, latency-sensitive → Profile A
                       │
                       └─ Yes:
                           │
                           ├─ Want lowest cost? → Profile C
                           └─ Want highest quality? → Profile D
```

---

## Profile A — Offline-only

**For:** air-gapped deployments, embedded devices, privacy-paranoid users, debug/development.
**Tradeoff:** the heuristic regex extractor produces flat salience signals. Deep-sleep forgetting may aggressively prune low-importance concepts because nothing differentiates them. This is the documented "encoder-dependence" failure mode (Section §11 of the paper, Bug A in `research/benchmarks/alb/README.md`).

### Configuration

```bash
# .env
SCM_EMBEDDING_BACKEND=sentence_transformers   # local 384-dim MiniLM
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# No LLM provider — heuristic extractor is the default fallback
LLM_PROVIDER=                                 # leave empty
```

### Code

```python
from src.chat.engine import ChatEngine

engine = ChatEngine(
    llm=None,                  # heuristic regex extractor
    sandbox_mode=False,
    profile="chatbot",
)
```

### Install

```bash
venv/bin/pip install sentence-transformers
# That's it.
```

### Verify

```bash
venv/bin/python -c "from src.core.encoder import MeaningEncoder; e = MeaningEncoder(); print(e._get_embedding('hello')[:5])"
# Should print 5 floats; sentence-transformer downloaded on first run.
```

---

## Profile B — Ollama-only

**For:** users with a local GPU or recent Apple Silicon, who want strong quality without ever sending data off the device.
**Tradeoff:** first-run model download (~300MB for `nomic-embed-text` + ~5GB for `llama3.2`). Latency is reasonable on Apple Silicon and modern x86; not great on low-end hardware.

### Configuration

```bash
# .env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:latest

SCM_EMBEDDING_BACKEND=ollama
SCM_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

### Code

```python
from src.chat.engine import ChatEngine
from src.core.encoder import MeaningEncoder
from src.llm import LLMExtractor

llm = LLMExtractor(provider="ollama")           # uses LLM_MODEL from env
encoder = MeaningEncoder(
    llm=llm,
    embedding_backend="ollama",
    embedding_model_name="nomic-embed-text",
)
engine = ChatEngine(llm=llm, encoder=encoder, profile="chatbot")
```

### Install

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull the models (one-time):
ollama pull llama3.2:latest
ollama pull nomic-embed-text

# 3. Verify Ollama is reachable:
curl -s http://localhost:11434/api/tags | head
```

### Verify

```bash
# End-to-end smoke:
venv/bin/python -c "
from src.llm import LLMExtractor
from src.core.encoder import MeaningEncoder
llm = LLMExtractor(provider='ollama')
enc = MeaningEncoder(llm=llm, embedding_backend='ollama', embedding_model_name='nomic-embed-text')
v = enc._get_embedding('Where do I work?', mode='query')
print('embed dim:', len(v))
print('extract:', llm.extract_concepts('My name is Saish'))
"
```

---

## Profile C — Hybrid (cloud-LLM + local-embed)

**For:** consumer-product deployments where extraction quality matters but every memory text being embedded shouldn't trigger an API call. This is the **recommended starting point**.
**Tradeoff:** text is sent to the cloud LLM at ingest time, but the vector index stays on the device. Per-message cost is bounded by the LLM extractor's cost, not the embedding cost (since the latter is free local).

### Configuration

```bash
# .env
LLM_PROVIDER=deepseek                          # or "openai"
DEEPSEEK_API_KEY=sk-...                        # see https://platform.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

SCM_EMBEDDING_BACKEND=ollama
SCM_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

### Code

```python
from src.chat.engine import ChatEngine
from src.core.encoder import MeaningEncoder
from src.llm import LLMExtractor

llm = LLMExtractor(provider="deepseek")
encoder = MeaningEncoder(
    llm=llm,
    embedding_backend="ollama",
    embedding_model_name="nomic-embed-text",
)
engine = ChatEngine(llm=llm, encoder=encoder, profile="chatbot")
```

### Install

```bash
venv/bin/pip install openai                    # OpenAI-compat client
ollama pull nomic-embed-text                   # local embedding model
```

### Cost ballpark

DeepSeek-chat is ~$0.27 / 1M input tokens, ~$1.10 / 1M output tokens (as of mid-2026). The HybridEncoder triages turns: only ~30-40% of turns escalate to the LLM. On a 30-turn conversation, expect ~$0.04 per session. Over a month of moderate use (~100 sessions), ~$4.

### Verify

```bash
# Same as Profile B's verify, swap provider="ollama" → provider="deepseek"
venv/bin/python -c "
from src.llm import LLMExtractor
llm = LLMExtractor(provider='deepseek')
print(llm._chat('Reply with the single word OK.', num_predict=10))
"
```

---

## Profile D — All-cloud

**For:** server-side deployments, evaluation runs where you want the absolute best retrieval, or any context where API latency and cost are not constraints.
**Tradeoff:** every memory text is embedded via API call. Privacy implications: every concept your agent stores gets sent to OpenAI. Cost is real.

### Configuration

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

SCM_EMBEDDING_BACKEND=openai_compat
SCM_EMBEDDING_MODEL=text-embedding-3-large    # 3072-dim
SCM_EMBEDDING_API_KEY=                        # leave empty to reuse OPENAI_API_KEY
SCM_EMBEDDING_BASE_URL=https://api.openai.com/v1
```

### Code

```python
from src.core.encoder import MeaningEncoder
from src.llm import LLMExtractor

llm = LLMExtractor(provider="openai")
encoder = MeaningEncoder(
    llm=llm,
    embedding_backend="openai_compat",
    embedding_model_name="text-embedding-3-large",
)
```

### Install

```bash
venv/bin/pip install openai
# Set OPENAI_API_KEY in your environment.
```

### Cost ballpark

- Extraction: gpt-4o-mini is ~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens.
- Embedding: text-embedding-3-large is ~$0.13 / 1M tokens.
- Combined: ~$0.06 per 30-turn session. ~$6 / 100 sessions.

You can also use this profile with **Voyage AI**, **Together**, or any OpenAI-compatible embedding endpoint by changing `SCM_EMBEDDING_BASE_URL` and `SCM_EMBEDDING_API_KEY`.

---

## Switching between profiles

You can swap embedding backends mid-deployment **only on a fresh database** — concept embeddings stored under one model are dimensionally incompatible with another. The system does not auto-rehydrate. To switch:

1. Export your concept graph: `engine.export_memory("backup.json")`
2. Change the env vars / restart with the new profile.
3. Re-ingest the conversation history if you want vector retrieval to work, OR rely on lexical (`search_by_text`) and graph-based retrieval until embeddings repopulate naturally.

The LLM extractor is hot-swappable — it doesn't affect stored data shape. You can switch `LLM_PROVIDER` between sessions freely.

---

## Configuration reference

All knobs live in `.env` (or environment). Defaults are documented in `.env.example`. A complete list of Phase 7 options:

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `LLM_PROVIDER` | LLMExtractor | `ollama` | `ollama` / `deepseek` / `openai` |
| `LLM_MODEL` | LLMExtractor | `llama3.2:latest` | Model name for the chosen provider |
| `LLM_TEMPERATURE` | LLMExtractor | `0.0` | Lower = more deterministic |
| `DEEPSEEK_API_KEY` | LLMExtractor | (none) | Required when `LLM_PROVIDER=deepseek` |
| `OPENAI_API_KEY` | LLMExtractor | (none) | Required when `LLM_PROVIDER=openai` |
| `SCM_EMBEDDING_BACKEND` | MeaningEncoder | `sentence_transformers` | `sentence_transformers` / `ollama` / `openai_compat` / `hash` |
| `SCM_EMBEDDING_MODEL` | MeaningEncoder | (provider-default) | Model name for the chosen backend |
| `OLLAMA_BASE_URL` | OllamaEmbeddingModel | `http://localhost:11434` | Local Ollama server |
| `SCM_EMBEDDING_BASE_URL` | OpenAICompatibleEmbeddingModel | OpenAI default | Override for non-OpenAI providers |
| `SCM_EMBEDDING_API_KEY` | OpenAICompatibleEmbeddingModel | falls back to `OPENAI_API_KEY` | Per-backend key |
| `EMBEDDING_DIM` | HashEmbeddingModel | `384` | Only used by hash fallback |
| `IDLE_LEARNER_ENABLED` | IdleLearner | `false` | Enable the M1 autonomous idle daemon |
| `IDLE_LEARNER_IDLE_THRESHOLD_SECONDS` | IdleLearner | `600` | M1 daemon idle threshold |
| `IDLE_LEARNER_MIN_SLEEP_INTERVAL_SECONDS` | IdleLearner | `1800` | M1 cooldown between autonomous cycles |
| `IDLE_LEARNER_TICK_INTERVAL_SECONDS` | IdleLearner | `60` | M1 daemon polling cadence |
| `CROSS_SESSION_POOL_ENABLED` | CrossSessionMemoryPool | `false` | Enable M2 continuity across sessions |
| `CROSS_SESSION_POOL_LOOKBACK_HOURS` | CrossSessionMemoryPool | `168` | M2 history window |
| `CROSS_SESSION_POOL_MAX_SESSIONS` | CrossSessionMemoryPool | `5` | M2 maximum sessions consulted |
| `SCHEMA_MIN_REPETITIONS` | SchemaExtractor | `3` | M3 sensitivity floor |
| `CURIOSITY_ENGINE_ENABLED` | CuriosityEngine | `false` | Enable M5 curiosity/gap filling |
| `CURIOSITY_LLM_SOURCE_ENABLED` | CuriosityEngine | `false` | M5 LLM source opt-in |
| `LIFECYCLE_POLICY_MIN_BATTERY_PERCENT` | LifecyclePolicy | `30` | M6 percent below which idle is vetoed |
| `LIFECYCLE_POLICY_MAX_CPU_PERCENT` | LifecyclePolicy | `80` | M6 percent above which idle is vetoed |

---

## Troubleshooting

**`OllamaEmbeddingModel` falls back to hash silently.**
Ollama isn't running, or the configured model isn't pulled. Run `ollama list` to verify, `ollama pull nomic-embed-text` if missing.

**`DEEPSEEK_API_KEY is not set`.**
Either the `.env` file isn't being loaded, or the key has been rotated. Re-export and re-load. Verify with `echo $DEEPSEEK_API_KEY`.

**Embedding-dimension mismatch on retrieval.**
You switched backends mid-deployment. See "Switching between profiles" above — concept embeddings are tied to the model that produced them. Easiest fix: clear LTM and re-ingest under the new profile.

**HME pipeline produces near-zero recall on the heuristic encoder.**
Encoder-dependence — see Section 11 of the paper. Either turn on `LLMExtractor` (Profile B/C/D), or set `FORGETTING_PROTECT_SALIENCE=0.5` (default) so high-salience concepts are protected from forgetting. The Phase 6 fixes already include this safety net.

**`LiteLLM` or `requests` not installed.**
`venv/bin/pip install requests openai` covers both the OpenAI-compat client and the basic HTTP fallback used by `OllamaEmbeddingModel`.

---

## What ALB v0.1 used

For full reproducibility, the ALB v0.1 pilot (`research/benchmarks/alb/`) ran under:

```bash
# v3 / v4 pilot:
SCM_EMBEDDING_BACKEND=ollama
SCM_EMBEDDING_MODEL=nomic-embed-text
LLM_PROVIDER=                                 # heuristic extractor (no LLM)
```

This is **Profile B without the LLM extractor** — i.e., heuristic concept extraction + Ollama embeddings. The encoder-dependence limitation observed in the v4 NIAL table (CSS = -1.000, CRAI_current = 0) is consistent with that profile. To rerun under Profile C (with DeepSeek LLM extractor), set:

```bash
ALB_USE_LLM_EXTRACTOR=1
ALB_LLM_PROVIDER=deepseek
```

before invoking `scripts/run_pilot.py`.
