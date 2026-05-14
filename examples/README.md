# SCM Examples

Three ready-to-run scripts. Pick one based on what you want to verify.

| Script | What it shows | Requires |
|---|---|---|
| [01_quickstart.py](01_quickstart.py) | The smallest possible SCM example. Ingest, sleep, retrieve. | Nothing — runs offline. |
| [02_wake_summary.py](02_wake_summary.py) | The killer feature. A simulated five-day persona, then the wake-summary the user would read on day 6. | Nothing — runs offline. |
| [03_with_ollama.py](03_with_ollama.py) | The privacy-first profile (Profile B in `docs/DEPLOYMENT.md`). Real LLM extraction + real semantic embeddings, fully local. | `ollama` running with `llama3.2` and `nomic-embed-text` pulled. |

## How to run

From the project root, with the venv activated:

```bash
source venv/bin/activate
python examples/01_quickstart.py
python examples/02_wake_summary.py
python examples/03_with_ollama.py     # needs Ollama, see below
```

## Setting up Ollama for example 03

```bash
# Install: https://ollama.com/download
ollama pull llama3.2:latest
ollama pull nomic-embed-text

# Verify
curl -s http://localhost:11434/api/tags | python3 -m json.tool | head
```

If Ollama isn't installed, example 03 still runs but falls back to heuristic
extraction + hash embeddings, and you'll see the quality drop noticeably.
That's the point of the example.

## After running

```bash
scm wake-summary --hours 24    # if you've installed the CLI
scm status                     # see how many concepts / schemas formed
```
