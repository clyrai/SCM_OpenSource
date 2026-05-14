# Integrating SCM into your agent

SCM is a **memory layer** — it doesn't generate, it remembers. You bolt it onto whatever LLM and harness you already have. Idle detection is automatic: SCM tracks per-user activity and fires sleep cycles in the background. The agent only needs to call two endpoints (`add_memory`, `search_memory`); everything else happens autonomously.

This doc shows the five canonical integrations. Pick the one that matches your stack.

---

## Quick mental model

For every user message your agent receives:

1. Call `search_memory(query=user_message, user_id=user_id)` to get relevant memories.
2. Inject the memories into the LLM's context.
3. Call `add_memory(text=user_message, user_id=user_id)` to store the message.
4. Generate a response with your LLM as usual.
5. **Optional:** if the response includes a fact worth remembering, call `add_memory(text=that_fact)` too.

Sleep cycles fire automatically when the user has been idle for `SCM_IDLE_THRESHOLD_SEC` seconds (default 300). On the next interaction, the response includes `wake_summary_pending` if there's anything new to surface to the user.

---

## 1. Claude Desktop / Cursor / VS Code Continue (via MCP)

**SCM speaks Anthropic's Model Context Protocol.** Add it to any MCP client with one config block.

### Install

```bash
pip install scm-memory
ollama pull nomic-embed-text  # recommended for retrieval quality
```

### Claude Desktop config

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "scm": {
      "command": "scm",
      "args": ["mcp"],
      "env": {
        "LLM_PROVIDER": "ollama",
        "SCM_EMBEDDING_BACKEND": "ollama",
        "SCM_EMBEDDING_MODEL": "nomic-embed-text",
        "SCM_DATA_DIR": "/Users/you/.scm",
        "SCM_IDLE_THRESHOLD_SEC": "300"
      }
    }
  }
}
```

Restart Claude Desktop. The five SCM tools (`add_memory`, `search_memory`, `consolidate`, `wake_summary`, `forget`) are now available to Claude. You can confirm by asking Claude *"What tools do you have?"*

### Cursor config

Cursor reads MCP config from `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "scm": {
      "command": "scm",
      "args": ["mcp"]
    }
  }
}
```

### VS Code Continue

Add to `.continue/config.json` under `mcpServers` — same shape as the Claude Desktop block.

### What you get

Claude/Cursor will autonomously call `add_memory` after meaningful exchanges, and `search_memory` when it suspects prior context. SCM's idle daemon fires sleep cycles in the background. Every few days, the user can ask *"What have you noticed about me lately?"* and Claude will call `wake_summary` to retrieve the cached report.

---

## 2. ChatGPT Custom GPT (via Actions)

ChatGPT Custom GPTs can call HTTP endpoints via "Actions" (OpenAPI-defined tools).

### Step 1: Run the SCM HTTP server

```bash
scm serve --host 0.0.0.0 --port 8000
# Or, exposed publicly via ngrok / Cloudflare Tunnel / Fly.io:
ngrok http 8000
```

### Step 2: Configure the GPT

Open a Custom GPT in `chat.openai.com/gpts/editor`, go to **Configure → Actions → Create new action**, and import:

```
https://YOUR-PUBLIC-HOST/v1/openapi.json
```

ChatGPT auto-imports the five SCM tools. Set authentication if your server requires it.

### Step 3: Tell the GPT how to use it

In the GPT's system prompt:

```
You have access to a long-term memory tool called SCM. Whenever the user
shares a fact about themselves, their preferences, or their context, call
add_memory(text=<the fact>). Before responding to any message that might
benefit from prior context, call search_memory(query=<the topic>) and
incorporate the returned memories into your response.

Do not call consolidate(); it fires automatically when the user is idle.
You may call wake_summary() if the user explicitly asks "what have you
noticed about me?" or returns from an absence.
```

That's it. The GPT now has multi-conversation memory.

---

## 3. OpenAI Agents SDK (or any function-calling client)

The five SCM tools export as standard OpenAI function-calling specs.

### Get the tool definitions

```python
import requests
tools = requests.get("http://localhost:8000/v1/tools?format=openai").json()["tools"]
```

Or via Python directly:

```python
from src.integrations.tools import export_all_openai
tools = export_all_openai()
```

### Pass them to the LLM

```python
from openai import OpenAI

client = OpenAI()
messages = [{"role": "user", "content": "Remember that I prefer vegan food."}]

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

# When the model returns a tool_call, route it to SCM:
for tool_call in resp.choices[0].message.tool_calls or []:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = requests.post(
        f"http://localhost:8000/v1/tools/{name}",
        json=args,
    ).json()
    # Append result to messages and re-invoke the LLM if needed.
```

The same pattern works with **DeepSeek**, **Together**, **Anyscale**, **Voyage**, and any other OpenAI-compatible provider.

---

## 4. Anthropic Claude API (function calling)

```python
import anthropic
import requests

client = anthropic.Anthropic()
tools = requests.get("http://localhost:8000/v1/tools?format=anthropic").json()["tools"]

resp = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "I'm allergic to peanuts."}],
)

for block in resp.content:
    if block.type == "tool_use":
        result = requests.post(
            f"http://localhost:8000/v1/tools/{block.name}",
            json=block.input,
        ).json()
        # Continue the conversation with the result.
```

---

## 5. Google Gemini (function calling)

```python
import google.generativeai as genai
import requests

tools = requests.get("http://localhost:8000/v1/tools?format=gemini").json()["tools"]
model = genai.GenerativeModel("gemini-2.5-flash", tools=tools)

resp = model.generate_content("My name is Saish.")
for part in resp.candidates[0].content.parts:
    if part.function_call:
        args = dict(part.function_call.args)
        result = requests.post(
            f"http://localhost:8000/v1/tools/{part.function_call.name}",
            json=args,
        ).json()
```

---

## 6. LangChain (BaseMemory adapter)

```python
from src.integrations.langchain_adapter import SCMMemory

memory = SCMMemory(user_id="alice", base_url="http://localhost:8000/v1")

# Drop into any LangChain chain that accepts BaseChatMemory
chain = ConversationChain(llm=llm, memory=memory)
```

This path has been tested with the brutal LangChain harness. The important
thing is that SCM is genuinely on the call path: the adapter exposes
`add_memory()` and `search_memory()` semantics through the `/v1` API rather
than relying on LangChain's in-context history.

---

## 7. Plain Python (no harness)

```python
from src.integrations.tools import get_tool
from src.integrations.mcp_server import UserEnginePool

pool = UserEnginePool(idle_threshold_sec=300)
pool.start()

engine = pool.get_or_create("alice")

# Add a memory
get_tool("add_memory").handler(
    {"text": "I run every Tuesday morning.", "user_id": "alice"},
    engine,
)

# Search later
result = get_tool("search_memory").handler(
    {"query": "what's my routine?", "user_id": "alice"},
    engine,
)
print(result["memories"])
```

---

## How idle detection works

SCM behaves like human memory at the API surface:

1. Every `add_memory` / `search_memory` call updates the user's `last_activity` timestamp.
2. A background sweeper checks every 30 seconds whether any user has been idle for more than `SCM_IDLE_THRESHOLD_SEC` (default 300).
3. Idle users get a deep-sleep cycle automatically — schema extraction, contradiction resolution, curiosity-driven gap filling all run.
4. The wake-summary is built and cached.
5. The next time the user calls any tool, the response includes `wake_summary_pending` with the report.

**The integrating agent doesn't need to manage any of this.** It just calls `add_memory` and `search_memory`. The lifecycle happens.

---

## Configuration matrix

Key env vars (full reference in [`docs/DEPLOYMENT.md`](DEPLOYMENT.md)):

| Variable | Default | Notes |
|---|---|---|
| `SCM_IDLE_THRESHOLD_SEC` | `300` | Seconds of inactivity before a sleep cycle fires |
| `SCM_AUTO_SLEEP_DISABLE` | `0` | Set to `1` to require manual `consolidate()` calls |
| `LLM_PROVIDER` | (empty) | `ollama` / `deepseek` / `openai` / empty (heuristic) |
| `SCM_EMBEDDING_BACKEND` | `sentence_transformers` | `ollama` recommended for retrieval quality |
| `SCM_EMBEDDING_MODEL` | depends on backend | `nomic-embed-text` for ollama |
| `SCM_DATA_DIR` | `~/.scm` | Where per-user memory persists |
| `SCM_PUBLIC_URL` | `http://localhost:8000` | Used in OpenAPI servers field for ChatGPT GPTs |

---

## What this looks like in production

Imagine a Claude Desktop user named Alice:

- **Day 1, 9am**: Alice tells Claude *"I'm starting a new job at Atlas Labs tomorrow."* Claude calls `add_memory`. SCM stores the fact.
- **Day 1, 9:05am**: Alice closes Claude.
- **Day 1, 9:10am**: SCM detects Alice's session has been idle for 5 minutes. It fires a deep-sleep cycle. The schema extractor doesn't find a recurring pattern (only one mention), but the curiosity engine notes *"Atlas Labs"* as an unfamiliar entity. The cycle completes; a wake-summary is cached.
- **Day 2, 9am**: Alice opens Claude again, says *"Morning."* Claude calls `add_memory`. The response includes `wake_summary_pending` with: *"While you were away I noticed you mentioned starting at Atlas Labs. I looked them up — they do reinforcement learning for industrial robots."*
- Claude surfaces this to Alice naturally: *"Good morning! Hope your first day at Atlas Labs went well. I read up on them — RL for industrial robots, right?"*

Alice didn't ask Claude to remember anything. Claude didn't manage any memory state. Both worked because SCM was sitting in the middle, doing what human memory does.

---

## What's next

- Hosted SCM at `api.scm.run` so you don't have to self-host (in development)
- LangChain + LlamaIndex official integrations
- Native JavaScript SDK (`npm install scm-memory`)
- Browser extension that intercepts ChatGPT.com / Claude.ai conversations and records to SCM (privacy-safe, opt-in)
