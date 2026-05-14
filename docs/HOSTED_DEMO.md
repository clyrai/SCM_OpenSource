# Hosted demo runbook — `/demo` on the public web

The SCM demo at `https://<your-host>/demo` lets anyone — no login, no install — type messages, watch the agent learn about them, and on a return visit see the wake-summary moment surface in a banner. This is the artifact the launch tweet links to.

The demo is a self-contained module:

- **Frontend:** [`src/api/static/demo.html`](../src/api/static/demo.html) — single-file vanilla HTML/JS, no build step
- **Backend:** [`src/api/demo_router.py`](../src/api/demo_router.py) — FastAPI router mounted at `/demo`
- **Engine:** per-slug `ChatEngine(sandbox_mode=True)` — in-memory only, no SQLite writes, no cross-session pollution
- **LLM for replies:** DeepSeek (`DEEPSEEK_API_KEY` env var). Falls back to a memory-echo reply if unset.

---

## Run locally (smoke test before deploying)

```bash
cd /Users/saish/Downloads/SleepAI
source venv/bin/activate

# Confirm Ollama running with nomic-embed-text pulled
ollama pull nomic-embed-text

# Confirm DeepSeek key in .env
grep DEEPSEEK_API_KEY .env

# Start the server
SCM_EMBEDDING_BACKEND=ollama \
SCM_EMBEDDING_MODEL=nomic-embed-text \
scm serve --port 8000
```

Open http://localhost:8000/demo — you'll be redirected to a fresh URL like `http://localhost:8000/demo/s/k7oiwlbqoaim`. **Bookmark that URL.** Type 3-4 messages, click `/sleep`, then refresh. The wake banner appears at the top of the conversation.

---

## URL flow

```
GET  /demo                  → 302 to /demo/s/<random-12-char-slug>
GET  /demo/s/{slug}         → serves static/demo.html
POST /demo/api/chat/{slug}  → ingest + retrieve + LLM reply (returns wake_summary if pending)
POST /demo/api/sleep/{slug} → force a deep sleep cycle now
GET  /demo/api/wake/{slug}  → fetch latest wake summary
GET  /demo/api/history/{slug} → restore conversation on page reload
```

The slug is the entire session identity. No cookies, no auth. To "share your conversation" with someone, send them the URL. To "log out", close the tab — server-side memory persists for the lifetime of the process.

**Important:** because sessions are in-memory, restarting the server wipes all sessions. That is intentional for a demo (clean slate every deployment); production would swap `sandbox_mode=True` → per-slug SQLite.

---

## Deploy to Fly.io (recommended — free tier)

```bash
# One-time setup
curl -L https://fly.io/install.sh | sh
fly auth login

cd /Users/saish/Downloads/SleepAI
fly launch --no-deploy
# When prompted: pick a name (e.g. "scm-demo"); default region; no Postgres/Redis

# Set the runtime secrets
fly secrets set \
  DEEPSEEK_API_KEY=sk-... \
  SCM_EMBEDDING_BACKEND=ollama \
  SCM_EMBEDDING_MODEL=nomic-embed-text \
  LLM_PROVIDER=deepseek

# Deploy
fly deploy

# Open it
fly open /demo
```

The existing [`Dockerfile`](../Dockerfile) and [`fly.toml`](../fly.toml) already work; the only addition was the `/demo` route, which is included automatically because it's part of the FastAPI app.

Cost: ~$0/mo for low traffic (Fly's free tier covers a 1-cpu, 256MB machine that auto-stops when idle). If you keep Ollama in the container too, bump to a 1GB-RAM machine (~$2-5/mo).

---

## Deploy to Railway (alternative)

```bash
npm install -g @railway/cli
railway login
cd /Users/saish/Downloads/SleepAI
railway init
railway variables set \
  DEEPSEEK_API_KEY=sk-... \
  SCM_EMBEDDING_BACKEND=ollama \
  SCM_EMBEDDING_MODEL=nomic-embed-text
railway up
```

Railway gives you a `*.up.railway.app` URL. Append `/demo` to it and you're live.

---

## Custom domain (`scm.run` or whatever)

Once Fly/Railway is up:

```bash
# Fly
fly certs create scm.run
# add the CNAME they show you in your DNS provider

# Railway
railway domain  # then add the CNAME
```

Wait ~5 minutes for the cert to issue.

---

## Tunable env vars

| Var | Default | Purpose |
|---|---|---|
| `SCM_DEMO_IDLE_SEC` | `30` | seconds of idle before the demo sweeper auto-fires sleep |
| `SCM_DEMO_SWEEP_SEC` | `5` | how often the sweeper checks for idle sessions |
| `DEEPSEEK_API_KEY` | unset | required for fluent LLM replies; falls back to memory-echo if missing |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | override for self-hosted |
| `DEMO_LLM_MODEL` | `deepseek-chat` | swap for other DeepSeek/OpenAI-compat models |
| `SCM_EMBEDDING_BACKEND` | `auto-detect` | `ollama` or `sentence_transformers` |
| `SCM_EMBEDDING_MODEL` | `nomic-embed-text` | embedding model name |

---

## What the user actually experiences

1. **First visit:** Browser opens `your-host/demo`, redirects to `your-host/demo/s/k7oiwlbqoaim`. Empty conversation panel, prompt: *"Tell me about yourself. I'll remember what you say, and while you're idle I'll think about it. Come back later — same URL — to see what I figured out."*

2. **They type 3-5 messages** — name, profession, hobby, a fact. Each turn: their message → SCM stores it → SCM retrieves anything related → DeepSeek composes a 1-2 sentence reply that uses the retrieved context.

3. **They click `/sleep`** OR wait 30 seconds (the sweeper fires automatically). A deep sleep cycle runs: NREM consolidation, REM schema extraction, freshness-floor protection.

4. **They send the next message** (or close the tab and come back later — same URL). The reply payload includes a `wake_summary` field, and the UI surfaces it as a banner at the top of the conversation:

   ```
   💤 While you were away
   Welcome back.
   I consolidated 15 memories, generated 5 associative dreams.
   Here's what I have on you:
     • Coffee (implied by visiting a coffee shop)
     • Person: Alex
     • I run every Tuesday morning along the river.
   I'm ready when you are.
   ```

   That banner is the **demo moment** — the thing the entire pitch is built around. It only appears once per idle period (`wake_consumed_at` flag prevents re-surfacing).

---

## Reset & abuse mitigations (for production)

- The `/reset` button on the UI just creates a new slug. Old sessions stay in memory until the server restarts.
- Slug entropy: 12 chars from `[a-z0-9]` → 36¹² ≈ 4.7×10¹⁸ space. URL guessing is impractical.
- No persistent storage means abuse is bounded: a bad actor can fill RAM, but a server restart wipes everything.
- For real production, swap `sandbox_mode=True` for per-slug SQLite + add a TTL eviction loop on `_DemoPool`.

---

## Verification checklist before announcing the URL

- [ ] `https://<host>/demo` redirects to a slug URL (302, not 500)
- [ ] Page renders with the dark theme
- [ ] Type a message → bot replies within ~3 seconds
- [ ] Click `/sleep` → narrative banner with multiple bullet points appears
- [ ] Refresh the page → conversation history is preserved (proves session persists)
- [ ] Open the same URL in incognito → see the same conversation (proves URL is the session)
- [ ] Open `/demo` in a new tab → fresh empty conversation (proves redirect creates new sessions)
- [ ] Click `/reset` → confirms, new slug, empty state

That's the demo.
