# SCM + LangChain integration guide

Three integration shapes, in increasing order of agent autonomy:

1. **`SCMClient` + custom loop** — you drive the memory calls per turn (search → invoke → store). Three explicit lines, verified by 16/16 brutal scenarios. Use when the agent's loop is yours and you want full control.
2. **SCM as `@tool`s for a `create_agent`** — you hand SCM tools to the agent and let it decide when to recall, when to remember. Verified by `tests/agent_with_tools/test_tool_calling_agent.py`. Use when you want LangChain's tool-calling agent to manage memory autonomously.
3. **SCM in a LangGraph multi-agent system** — multiple specialist agents (researcher, profiler, writer, etc.) share or partition SCM memory in a `StateGraph`. Verified by `tests/agent_with_tools/test_multiagent_langgraph.py`. Use when you have a real agent team.

All three talk to the same `/v1/*` REST API on whatever SCM server you run.

---

## 0. Prerequisites — full setup from zero

### 0.1 Python 3.10+

```bash
python --version   # need 3.10 or newer
```

### 0.2 Install SCM and LangChain

```bash
# Create a clean venv (recommended)
python -m venv venv && source venv/bin/activate

# Core deps
pip install scm-memory langchain-core langchain-openai
```

Verify the SCM CLI:

```bash
scm version    # should print 0.7.7 (or newer)
```

### 0.3 Pick an embedding backend

SCM needs to embed text into vectors. **Three options**, pick one:

**Option A — Ollama (recommended, free, local).** Best quality, no API costs.

```bash
# Install Ollama first if you don't have it:
#   macOS: download from https://ollama.com/download
#   Linux: curl -fsSL https://ollama.com/install.sh | sh
# Then pull the embedding model (~270 MB):
ollama pull nomic-embed-text

# Start Ollama in a terminal you keep running:
ollama serve
```

**Option B — sentence-transformers (no extra installs, smaller model).** ~80 MB MiniLM, lower quality than Ollama but zero infra.

```bash
pip install scm-memory[embeddings]   # adds sentence-transformers
# No daemon to run. Use SCM_EMBEDDING_BACKEND=sentence_transformers below.
```

**Option C — OpenAI embeddings (cloud, paid).** Highest quality at scale, ~$0.02 per 1M tokens.

```bash
# No installs beyond `openai` (already pulled by scm-memory).
# You'll set OPENAI_API_KEY in 0.5 below.
```

### 0.4 Pick an LLM for chat replies

The LangChain agent needs an LLM. Anything OpenAI-compatible works. Cheap options:

| LLM | API endpoint | Where to get a key |
|---|---|---|
| **DeepSeek** (recommended for cost) | `https://api.deepseek.com` | https://platform.deepseek.com — $5 free trial credit |
| **OpenAI** | `https://api.openai.com/v1` | https://platform.openai.com/api-keys |
| **Together** | `https://api.together.xyz/v1` | https://api.together.ai |
| **Ollama (local LLM)** | `http://localhost:11434/v1` | already free if Ollama is running |

DeepSeek and Ollama-local are the cheapest paths. The examples below use DeepSeek; swap `base_url` and `model` for any other provider.

### 0.5 Create a `.env` file

In your project root:

```bash
# .env
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
# (optional) OPENAI_API_KEY=sk-your-openai-key-here

SCM_EMBEDDING_BACKEND=ollama
SCM_EMBEDDING_MODEL=nomic-embed-text
```

Load it in your script:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 0.6 Run the SCM server

The SCM server is a **separate process** that handles `/v1/*` HTTP requests. Your LangChain app talks to it over HTTP. Pick where to run it:

#### Option A — local dev (laptop, single user)

```bash
# In a terminal you keep open:
SCM_EMBEDDING_BACKEND=ollama \
SCM_EMBEDDING_MODEL=nomic-embed-text \
scm serve --port 8000
```

You should see:

```
LLM Status: {'available': True, 'provider': 'ollama', ...}
Database initialized at .../sleepai.db
SleepAI ready!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Your LangChain code uses `base_url="http://localhost:8000/v1"`. **Closing the terminal stops the server** — for development that's fine, for anything else use Option B or C.

#### Option B — local but persistent (background daemon)

Keep the server running across reboots / terminal closes. Uses the same Python install:

```bash
# Start in the background, log to a file, persist data dir
nohup env SCM_EMBEDDING_BACKEND=ollama \
          SCM_EMBEDDING_MODEL=nomic-embed-text \
          SCM_DATA_DIR=$HOME/.scm \
          scm serve --port 8000 > $HOME/.scm/server.log 2>&1 &
echo $! > $HOME/.scm/server.pid

# Stop later:
kill $(cat $HOME/.scm/server.pid)
```

For a more proper setup (auto-restart on crash, start on boot) wire SCM into systemd / launchd / a process manager you already use. Treat `scm serve --port 8000` as the command line.

#### Option C — Docker / production

Self-host with a Docker image. The repo ships a working [`Dockerfile`](../Dockerfile) and [`fly.toml`](../fly.toml) targeting Fly.io specifically; same image works on Railway, Render, EC2, your own k8s, anywhere a container runs.

```bash
docker build -t scm:0.7.7 .
docker run -d \
    -p 8000:8000 \
    -v scm_data:/data \
    -e SCM_DATA_DIR=/data \
    -e SCM_EMBEDDING_BACKEND=sentence_transformers \
    -e DEEPSEEK_API_KEY=sk-... \
    --name scm \
    scm:0.7.7
```

Full deployment walkthroughs (Fly.io, Railway, custom domain, persistent volumes, secrets) are in [`docs/HOSTED_DEMO.md`](HOSTED_DEMO.md). Production-specific concerns covered there:

- Persistent data volume (`SCM_DATA_DIR`) so a redeploy doesn't wipe memories
- Embedding-model selection inside the container (Ollama costs more RAM; sentence-transformers is the lighter default for hosted)
- Healthcheck endpoint (`/v1/health`)
- Custom domain + TLS

#### Option D — hosted SCM (coming)

A managed `https://scm.run/v1` endpoint where you skip running the server entirely is on the roadmap (not live yet). Until then, Options A–C above. The LangChain code is identical regardless — only `base_url` changes.

### 0.7 Verify the server is reachable

Whichever option you picked:

```bash
curl http://<your-host>:8000/v1/health
# {"ok":true,"active_users":0,"auto_sleep":true,...}
```

If that returns 200, the rest of the guide will work.

### 0.8 Quick sanity check (optional but recommended)

Before wiring LangChain, confirm raw SCM works:

```bash
curl -X POST http://localhost:8000/v1/memories \
     -H "Content-Type: application/json" \
     -d '{"text":"My favorite coffee is filter coffee.","user_id":"sanity"}'

curl -X POST http://localhost:8000/v1/memories/search \
     -H "Content-Type: application/json" \
     -d '{"query":"what do I drink","user_id":"sanity","wait_for_pending":true}'
```

The second call should return `memory_context` containing the coffee fact. If yes, you're ready.

### Troubleshooting prerequisites

| Symptom | Fix |
|---|---|
| `scm: command not found` | venv not activated, or `pip install scm-memory` failed silently — re-run |
| `ConnectionRefusedError: ... 11434` on server start | Ollama isn't running — `ollama serve` in a terminal, or use Option B / C above |
| `model 'nomic-embed-text' not found` | Run `ollama pull nomic-embed-text` once |
| `[LTM] PostgreSQL load failed` (warning, not error) | Harmless — SCM falls back to SQLite for local dev |
| `/v1/health` returns 404 | You started a different server, or wrong port |
| First request takes 20+ seconds | Cold-start: Ollama loading model + sentence-transformers warmup. Subsequent requests are fast |

---

## 1. The recommended pattern: `SCMClient` + custom loop (≈30 lines)

This is the pattern the brutal harness uses, verified against 16/16 scenarios. Three explicit calls per turn: **search, generate, store.** Works against modern LangChain (1.x+) which dropped `BaseChatMemory`.

```python
"""langchain_with_scm.py — minimal LangChain agent backed by SCM."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.integrations.langchain_adapter import SCMClient

# 1. SCM client — partition key is user_id. Different user_id = different memories.
scm = SCMClient(user_id="alex", base_url="http://localhost:8000/v1")

# 2. LLM — anything langchain-compatible. ChatOpenAI works against
#    DeepSeek, Together, Groq, etc. by setting base_url + api_key.
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-...",  # or load from env
    temperature=0.4,
)

history = []

def chat(user_input: str) -> str:
    # — Retrieve. wait_for_pending=True gives you read-your-writes (blocks
    #   until any prior add_memory finished its embedding step).
    search = scm.search_memory(user_input, limit=5, wait_for_pending=True)
    context = search.get("memory_context", "")
    wake = search.get("wake_summary_pending")

    # — Build the system prompt with retrieved context + any wake summary.
    sys_text = (
        "You are a helpful assistant with persistent memory provided by SCM. "
        "Use the retrieved memories to personalize your reply. Don't invent "
        "facts not in memory or the user's message.\n\n"
        f"Relevant memories:\n{context or '(none yet)'}"
    )
    if wake and wake.get("narrative"):
        sys_text += f"\n\n[While you were away: {wake['narrative']}]"

    # — Generate via LangChain.
    messages = [SystemMessage(content=sys_text)] + history + [HumanMessage(content=user_input)]
    reply = llm.invoke(messages).content
    history.extend([HumanMessage(content=user_input), AIMessage(content=reply)])

    # — Store ONLY the user's facts. Deliberately not the LLM's reply —
    #   it's a reformulation of stuff that's already in memory and would
    #   only dilute retrieval.
    scm.add_memory(text=user_input)
    return reply

print(chat("Hi, I'm Alex. I'm a backend engineer in Lisbon."))
print(chat("I run every Tuesday morning along the river."))
print(chat("Where do I work and where do I run?"))
# → "You're a backend engineer in Lisbon, and you run every Tuesday
#    morning along the river."
```

That's it. **Verified end-to-end** against this exact code path in [`tests/test_langchain_guide_example.py`](../tests/test_langchain_guide_example.py).

---

## 1b. SCM as tools for a tool-calling agent

When you want the agent to *decide* when to remember and when to recall — not just do it on every turn — expose SCM as `@tool`-decorated functions and let `create_agent` (LangChain 1.x) drive.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.integrations.langchain_adapter import SCMClient
from src.integrations.langchain_tools import make_scm_tools

# 1. SCM client — Bearer-authed against SCM Cloud (or unauthenticated against self-hosted)
scm = SCMClient(
    api_key="scm_live_...",
    base_url="https://scm.run/v1",
    user_id="customer_42",
)

# 2. Tools — search_memory, add_memory, consolidate, wake_summary
tools = make_scm_tools(scm)

# 3. LLM (BYOK)
llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com",
                 api_key="sk-...")

# 4. Agent
system_prompt = (
    "You are a helpful assistant with persistent memory. "
    "If the user asks about something they previously told you, call "
    "`search_memory`. If their question has multiple parts, call it "
    "SEPARATELY for each part. If they share a substantive fact about "
    "themselves, call `add_memory`. Don't invent facts not in your memory."
)
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# 5. Talk
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hi, I'm Alex, a backend engineer in Lisbon."}]
})
# → agent autonomously called add_memory(text="Alex is a backend engineer in Lisbon.")

result = agent.invoke({
    "messages": [{"role": "user", "content": "Where do I work and where do I run?"}]
})
# → agent decomposed into TWO search_memory calls (one for each clause)
# → reply: "You work as a backend engineer in Lisbon, and you run every Tuesday morning along the river."
```

**Verified end-to-end** by [`tests/agent_with_tools/test_tool_calling_agent.py`](../tests/agent_with_tools/test_tool_calling_agent.py):
- Agent calls `add_memory` when user shares a fact ✓
- Agent calls `search_memory` when user asks about a stored fact ✓
- Agent decomposes compound queries into multiple `search_memory` calls ✓
- Agent's reply is grounded in the retrieved memory verbatim ✓

### Tips for the system prompt

The exact wording matters because LangChain agents lean on docstrings. The default tool docstrings in `make_scm_tools` already nudge the agent correctly, but you can reinforce in the system prompt:

- *"If the question has multiple parts, call search_memory once per part."*
- *"Trust whatever search_memory returns — those are facts the user actually told you."*
- *"Don't invent facts not in memory or the current message."*

---

## 1c. Multi-agent systems with SCM (LangGraph)

When you have a team of specialist agents (researcher, profiler, writer, supervisor) that should share memory across the team, give each one its own `make_scm_tools(scm_client)` set, all bound to the same `user_id`.

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# All agents share ONE end-user namespace
end_user = "customer_42"

scm_researcher = SCMClient(api_key=..., base_url=..., user_id=end_user)
scm_profiler   = SCMClient(api_key=..., base_url=..., user_id=end_user)
scm_writer     = SCMClient(api_key=..., base_url=..., user_id=end_user)

researcher_llm = llm.bind_tools(make_scm_tools(scm_researcher))
profiler_llm   = llm.bind_tools(make_scm_tools(scm_profiler))
writer_llm     = llm.bind_tools(make_scm_tools(scm_writer))

class TeamState(TypedDict):
    user_msg: str
    notes: list
    profile: str
    reply: str

def researcher_node(state):  ...   # calls add_memory for any new facts
def profiler_node(state):    ...   # calls search_memory to build a user profile
def writer_node(state):      ...   # composes the final reply using the profile

graph = StateGraph(TeamState)
graph.add_node("researcher", researcher_node)
graph.add_node("profiler", profiler_node)
graph.add_node("writer", writer_node)
graph.set_entry_point("researcher")
graph.add_edge("researcher", "profiler")
graph.add_edge("profiler", "writer")
graph.add_edge("writer", END)
team = graph.compile()

result = team.invoke({"user_msg": "...", "notes": [], "profile": "", "reply": ""})
```

**Verified** by [`tests/agent_with_tools/test_multiagent_langgraph.py`](../tests/agent_with_tools/test_multiagent_langgraph.py) — three agents sharing memory across a `StateGraph`, with cross-account isolation enforced (Account B with the same `user_id` sees zero of Account A's memories).

### When to give each agent its own user_id (vs sharing one)

| Pattern | When to use |
|---|---|
| **One `user_id` shared by all agents** | The team is serving one human end-user. All agents need the same memory pool. |
| **One `user_id` per agent** | The agents are independent assistants for different tasks (a coding agent vs a research agent serving the same human). They each build their own context. |
| **`user_id` + per-agent metadata tag** | Hybrid — shared memory namespace, but each `add_memory` call passes `metadata={"agent": "researcher"}` so retrieval can scope by agent if needed. |

---

## 2. The legacy `SCMMemory` adapter (for ConversationChain users)

If you have an existing codebase using LangChain's classic `BaseChatMemory` API (the `langchain.chains.ConversationChain` pattern), `SCMMemory` is a drop-in replacement.

**Requires** the older `langchain` package, not just `langchain-core`:

```bash
pip install langchain  # in addition to scm-memory + langchain-openai
```

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from src.integrations.langchain_adapter import SCMMemory

memory = SCMMemory(user_id="alex", base_url="http://localhost:8000/v1")
llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", api_key="sk-...")
chain = ConversationChain(llm=llm, memory=memory)

print(chain.predict(input="Hi, I'm Alex. I'm a backend engineer in Lisbon."))
print(chain.predict(input="Where do I work?"))
```

For new code, prefer the `SCMClient` pattern in §1 — it's more flexible (you control which messages get stored, when sleep summaries surface, etc.) and avoids the LangChain-version coupling.

---

## 3. The wake-summary moment

This is the SCM-specific behavior most people miss on first read. Worth understanding because it's the differentiation.

When a user is idle past their configured sleep window, SCM autonomously:

1. Fires a deep-sleep cycle (NREM consolidation + REM schema extraction)
2. Builds a wake-summary narrative ("Here's what I have on you: …")
3. Caches it on the user's session

The next time that user calls `search_memory` or `add_memory`, the response includes a `wake_summary_pending` field — once. Your agent should surface it to the user before responding to whatever they typed.

Both `SCMMemory` and the manual pattern above already handle this. To hand-craft it:

```python
search = scm.search_memory(user_msg, wait_for_pending=True)
wake = search.get("wake_summary_pending")
if wake and wake.get("narrative"):
    print(f"[While you were away: {wake['narrative']}]")
    # then proceed with the regular reply
```

---

## 4. Configuring per-user sleep schedules (v0.7.7+)

By default a user's sleep window is 23:00–07:00 UTC, enabled. To match the user's actual timezone (so consolidation fires at *their* bedtime, not yours):

```python
import requests

requests.post(
    "http://localhost:8000/v1/users/alex/sleep-config",
    json={
        "timezone": "Europe/Lisbon",   # IANA name; use Intl.DateTimeFormat() in browsers
        "sleep_start": "23:00",        # local time
        "sleep_end": "07:00",
        "enabled": True,
    },
)
```

After this call, the MCP sweeper checks every 60 seconds whether Alex's local time has entered the window. If yes AND he has accumulated 3+ turns since his last cycle, it fires one deep-sleep — once per night. Like human circadian rhythm.

Read it back:

```python
r = requests.get("http://localhost:8000/v1/users/alex/sleep-config").json()
# {"user_id": "alex", "timezone": "Europe/Lisbon", "sleep_start": "23:00",
#  "sleep_end": "07:00", "enabled": true, "is_default": false, "last_sleep_at": null}
```

To disable nightly sleep for a user (e.g., they prefer manual `consolidate` only):

```python
requests.post(
    "http://localhost:8000/v1/users/alex/sleep-config",
    json={"enabled": False},
)
```

---

## 5. Multi-user agents (the right pattern)

A single SCM server handles many users. The key is to use a distinct `user_id` per logical user. Otherwise everyone shares one memory pool.

```python
def memory_for(user_id: str) -> SCMMemory:
    return SCMMemory(user_id=user_id, base_url="http://localhost:8000/v1")

# In your request handler:
mem = memory_for(request.user_id)
chain = ConversationChain(llm=llm, memory=mem)
reply = chain.predict(input=request.message)
```

SCM's per-user isolation is verified by tier 6 of the brutal LangChain harness ([tests/brutal_langchain/scenarios.py](tests/brutal_langchain/scenarios.py)) — Alice's memories never leak into Bob's retrieval results.

---

## 6. Forcing a sleep cycle (manual override)

Most callers don't need this — the nightly scheduler handles it. But for testing, demos, or end-of-session UX:

```python
scm.consolidate(mode="deep")     # full NREM + REM cycle
# or
scm.consolidate(mode="micro")    # quick intra-session pass
```

Returns:

```json
{
  "ok": true,
  "user_id": "alex",
  "mode": "deep",
  "schemas_formed": 2,
  "concepts_consolidated": 14,
  "concepts_forgotten": 0,
  "contradictions_resolved": 0
}
```

After a deep cycle, fetch the wake-summary:

```python
summary = scm.wake_summary(since_hours=24.0)
print(summary["narrative"])
# "Welcome back. I consolidated 14 memories...
#  Here's what I have on you:
#    • runs every Tuesday morning along the river
#    • Person: Alex
#    • Location: Lisbon
#    ..."
```

---

## 7. Common pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `search_memory` returns nothing fresh after `add_memory` | Async ingest hasn't drained yet | Pass `wait_for_pending=True` to `search_memory`, or call `add_memory(..., metadata={"sync": True})` for the writes that precede a search |
| Search results include SelfModel boilerplate ("I can remember conversations") | Old SCM version (<0.7.5) | Upgrade to ≥0.7.7 — `_internal=True` filter added |
| Sleep cycles fire every few minutes, not nightly | Legacy idle-timer mode | POST a per-user sleep-config (v0.7.7+); the user transitions to circadian on first config write |
| Embedding shape mismatch (`shapes (768,) and (384,)`) | Mixed-vintage data — switched embedding model with old concepts still around | Either wipe the data dir for a clean start, or accept the v0.7.6+ defensive fallback (treats as dissimilar, doesn't crash) |
| Multi-user data leaking | All calls using `user_id="default"` | Pass distinct `user_id` per logical user |
| `LangChain not installed` from `SCMMemory` import | LangChain optional in SCM core | `pip install langchain` (and `langchain-openai` for OpenAI/DeepSeek) |
| `403 / 401` from your LLM | API key wrong or rate-limited | Check `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, etc. |

---

## 8. Verifying the integration

The same scenarios that gate releases:

```bash
# Run the brutal harness — 16 scenarios across 7 tiers
python -m tests.brutal_langchain.runner
```

Expected output: `Pass rate: 16/16 (100%)`. Wall time ~14 minutes against a local Ollama. Verifies multi-day recall, contradiction handling, idle-fired wake summary, cross-session synthesis, adversarial storms, multi-user isolation, and graceful degradation when SCM is unreachable.

If you can run that against your stack and it passes, your SCM + LangChain integration is real.

---

## 9. Where to go from here

- **The five canonical SCM tools** in OpenAI/Anthropic/Gemini function-calling format: `GET /v1/tools?format=openai` (also `anthropic`, `gemini`, `openapi`)
- **OpenAPI 3.1 spec** for ChatGPT Custom GPT Actions: `GET /v1/openapi.json`
- **Brutal harness source** (real LangChain agent with 16-scenario test plan): [tests/brutal_langchain/](tests/brutal_langchain/)
- **MCP server** for Claude Desktop / Cursor instead of HTTP: `scm mcp` — same five tools exposed over stdio
