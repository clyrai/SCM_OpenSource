# Human-Like User Trial (SCM)

This guide is for live product demos where users should feel human-like memory behavior directly.

## 1. Start the App

```bash
python -m src.api.main
```

Open: `http://localhost:8000`

## 2. Run the Guided Demo

In the right sidebar:

1. Find **Human-Like Demo**.
2. Click **Run Full Demo**.
3. Watch the auto-flow:
- one-shot learning (`name`, `location`)
- micro sleep consolidation
- contradiction update (`morning` -> `evening`)
- deep sleep consolidation
- final preference query and summary check

Backend equivalent (for scripted demos / integrations):

```bash
curl -X POST http://localhost:8000/chat/product-demo/demo_session_001
```

Single-call backend smoke check (no UI needed):

```bash
curl -X POST http://localhost:8000/chat/backend-smoke/demo_session_001
```

## 3. Manual Demo (Optional)

Use these prompts manually if you want to narrate each step:

1. `My name is Alice.`
2. `I live in Seattle.`
3. `I prefer morning meetings.`
4. `What is my name?`
5. `Where do I live?`
6. `Noise token zxq-91`
7. Click **Run Micro Sleep**
8. `I prefer evening meetings.`
9. Click **Run Deep Sleep**
10. `What do I prefer?`

## 4. What To Highlight To Users

- SCM learns important facts quickly (one-shot behavior).
- Sleep cycles are functional operations, not visual gimmicks.
- Memory quality is maintained by selective forgetting pressure.
- Contradictions are handled as updates, not blind overwrites.
- Even fallback responses now sound more human for name, location, profession, and preference questions.

## 5. Product Diagnostics Endpoint

Use this endpoint to fetch product-level memory-health signals:

```bash
curl http://localhost:8000/chat/product-report/demo_session_001
```

Signals include:
- one-shot readiness
- selective forgetting presence
- contradiction-versioning presence
- average retention score
- benchmark pack status (Phase 4 / Phase 6 / guardrails)
- overall product readiness score
