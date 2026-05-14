# Build a chatbot with persistent memory in 10 minutes

You'll build a Python chatbot that:
1. Remembers what users tell it across sessions
2. Notices patterns autonomously while users are away
3. Surfaces a "wake summary" the next time the user comes back

We'll use **OpenAI** for the chat LLM, **Ollama** for embeddings (free, local), and **SCM** as the memory layer. Total elapsed time: ~10 minutes.

---

## What you'll build

```
You> Hi, I'm Alex. I'm a backend engineer in Lisbon.
Bot> Nice to meet you, Alex!

You> I run every Tuesday morning along the river.
Bot> Got it.

You> /sleep
[SCM consolidating...]

You> Tuesday again, did 6km this time.
Bot> Nice — sounds like Tuesday morning is your running day.

You> /quit

# ...come back the next day...

You> Morning!
Bot> Good morning, Alex! While you were away I noticed Tuesday-morning running has become a weekly pattern. Hope today's run was good.
```

That last message — the wake summary — is what makes SCM different from every other memory layer.

---

## Prerequisites

- Python 3.10+
- Ollama installed (https://ollama.com/download) — for free local embeddings
- An OpenAI API key (or DeepSeek, or Anthropic — anything OpenAI-compatible)

---

## Step 1: Install (1 min)

```bash
mkdir scm-chatbot && cd scm-chatbot
python -m venv venv && source venv/bin/activate
pip install scm-memory openai
ollama pull nomic-embed-text   # ~270 MB embedding model
```

`scm-memory` is the default install path. For local development inside this repo, use `pip install -e .`.

---

## Step 2: Configure (1 min)

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
SCM_EMBEDDING_BACKEND=ollama
SCM_EMBEDDING_MODEL=nomic-embed-text
SCM_DATA_DIR=./scm-data
```

---

## Step 3: Write the chatbot (5 min)

Save as `chatbot.py`:

```python
"""Persistent-memory chatbot using SCM + OpenAI."""
import os

from dotenv import load_dotenv
from openai import OpenAI

# SCM imports
from scm import SCMEngine

load_dotenv()


# 1. Build the SCM memory engine.
# SCM_EMBEDDING_BACKEND=ollama is auto-picked up from env.
engine = SCMEngine(profile="chatbot", auto_sleep=False)

# 2. Wire OpenAI for the chat generation.
openai_client = OpenAI()


def respond(user_message: str) -> str:
    """One conversation turn — search memory, generate reply, store the turn."""
    # Search SCM for memories relevant to what the user just said
    from src.integrations.tools import _search_memory_handler
    search = _search_memory_handler({"query": user_message, "user_id": "alex"}, engine.engine)
    memories = search.get("memory_context", "") or ""

    # Pass memories into the LLM as system context
    sys_prompt = (
        "You are a helpful assistant with persistent memory provided by SCM. "
        "Use the memories below to personalize your reply. Don't invent facts. "
        "Keep responses to 1-2 short sentences.\n\n"
        f"Relevant memories:\n{memories}"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=120,
    )
    reply = response.choices[0].message.content

    # Store the user's input in SCM (the assistant's reply is optional;
    # what matters for memory is what the USER says).
    engine.message(user_message)

    return reply


def show_wake_summary():
    """Print what the agent learned during the most recent idle period."""
    summary = engine.wake_summary(since_hours=24)
    if summary and summary.narrative:
        print(f"\n💤 While you were away:\n{summary.narrative}\n")


def main():
    print("Chatbot ready. Type /sleep to consolidate, /wake to see what was learned, /quit to exit.\n")

    # On startup, show any pending wake summary (if this is a return session)
    show_wake_summary()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/sleep":
            print("[SCM consolidating...]")
            engine.sleep("deep")
            print("[Done.]")
            continue
        if user_input == "/wake":
            show_wake_summary()
            continue

        reply = respond(user_input)
        print(f"Bot> {reply}\n")


if __name__ == "__main__":
    main()
```

---

## Step 4: Run it (3 min)

```bash
python chatbot.py
```

```
Chatbot ready. Type /sleep to consolidate, /wake to see what was learned, /quit to exit.

You> Hi, I'm Alex. I'm a backend engineer in Lisbon.
Bot> Nice to meet you, Alex! How can I help you today?

You> I run every Tuesday morning along the river.
Bot> Got it. Sounds like a great way to start the week.

You> /sleep
[SCM consolidating...]
[Done.]

You> Tuesday again, did 6km this time.
Bot> Nice — Tuesday running really is your thing.

You> /quit
```

Quit. Restart the script.

```bash
python chatbot.py
```

```
Chatbot ready. ...

💤 While you were away:
Welcome back, Alex! While you were away I noticed Tuesday-morning runs along
the river have become a weekly pattern. I also have you logged as a backend
engineer in Lisbon.

You> Morning!
Bot> Good morning, Alex! Hope today's run along the river was good.
```

That `💤 While you were away` block is the wake-summary surfacing. SCM noticed the pattern, abstracted it, and surfaced it on the next session.

**That's the chatbot. ~50 lines of code. ~10 minutes of your time.**

---

## What just happened

| Step | What ran |
|---|---|
| `engine.message("...")` | Selective encoding → event compilation → spreading-activation retrieval → working memory updated |
| `engine.sleep("deep")` | NREM consolidation + REM schema extraction + adaptive forgetting + curiosity gap-filling |
| `engine.wake_summary()` | Generated the human-readable narrative from the most recent sleep cycle's outputs |
| `respond()` | Combined the retrieved memories with the OpenAI completion |

You didn't write any of the memory pipeline. SCM did it.

---

## Try these next

### Add contradiction handling

```
You> I work at Stripe.
You> /sleep
You> Actually I just left Stripe — I'm at PostgreSQL Inc now.
You> /sleep
You> Where do I work?
Bot> You currently work at PostgreSQL Inc; you were previously at Stripe.
```

The contradiction-versioning machinery (Phase 5) handles this automatically.

### Add cross-session synthesis

```
You> My office in Lisbon has a snack basket.
You> /sleep
You> I'm allergic to peanuts.
You> /sleep
You> What should I avoid in the office?
Bot> Avoid peanuts — your office's snack basket may contain them, and you're allergic.
```

The agent combined the office-context fact (one session) with the allergy fact (another session) — that's spreading activation across days.

### Switch to autonomous sleep

Set `auto_sleep=True` in the `SCMEngine(...)` constructor. Now you don't need `/sleep` — the M1 IdleLearner daemon fires sleep cycles automatically when the user has been quiet for `IDLE_LEARNER_IDLE_THRESHOLD_SECONDS` (default 600s).

### Switch to MCP for Claude Desktop

Skip the Python chatbot entirely and add the MCP server to Claude Desktop. See [`docs/INTEGRATIONS.md`](INTEGRATIONS.md). Claude gets the same memory layer with zero Python code on your end.

### Switch to the REST API

```bash
scm serve  # starts the FastAPI server on :8000
```

Then `curl` it from anywhere:

```bash
curl -X POST http://localhost:8000/v1/memories \
     -d '{"text":"I love filter coffee","user_id":"alex"}' \
     -H "Content-Type: application/json"
```

---

## Where to go from here

- [`docs/INTEGRATIONS.md`](INTEGRATIONS.md) — wire SCM into Claude Desktop, Cursor, ChatGPT Custom GPT, OpenAI Agents SDK, Anthropic, Gemini, LangChain, LlamaIndex
- [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) — four configuration profiles (offline / Ollama-only / hybrid / all-cloud)
- [`docs/BENCHMARKS.md`](BENCHMARKS.md) — performance and quality numbers with reproduction instructions
- [The paper](../research/SCM_Final_Paper.pdf) — 35 pages on the architecture

---

## Common errors

**"`SCM_EMBEDDING_BACKEND` is sentence_transformers but model unavailable"** — install with `pip install scm-memory[embeddings]` to pull sentence-transformers, or `ollama pull nomic-embed-text` and set `SCM_EMBEDDING_BACKEND=ollama`.

**"Connection refused on port 11434"** — Ollama isn't running. Start it: open the Ollama app, or `ollama serve &`.

**"OpenAI API key not set"** — your `.env` isn't being loaded. `pip install python-dotenv` and ensure `load_dotenv()` runs before any OpenAI import.

**Wake summary is empty** — you only had one or two turns, not enough for a pattern. Run more turns about the same topic and try again.

---

## Cost

| Configuration | Cost per 30-turn session |
|---|---|
| OpenAI gpt-4o-mini for chat + Ollama for embeddings | ~$0.02 (chat tokens only) |
| OpenAI gpt-4o + Ollama embeddings | ~$0.10 |
| OpenAI gpt-4o-mini + OpenAI text-embedding-3-large | ~$0.04 |
| All-Ollama (free local) | $0 |
