"""
SCM with full Ollama stack (LLM extractor + embedding model).

  $ ollama pull llama3.2
  $ ollama pull nomic-embed-text
  $ python examples/03_with_ollama.py

This is the recommended privacy-first profile (Profile B in
docs/DEPLOYMENT.md). Everything runs locally; nothing leaves the device.

The script:
  1. Configures SCM to use Ollama for both concept extraction and
     embeddings (via env vars set inside the script for clarity).
  2. Walks through a 4-turn conversation about a job change — the kind
     of multi-fact statement that the heuristic extractor handles
     poorly but a real LLM extractor handles well.
  3. Asks a question whose phrasing differs from the stored facts to
     show that the higher-quality embeddings bridge the abstraction.

If Ollama isn't running or the models aren't pulled, the script still
runs — SCM falls back to heuristic extraction + hash embeddings — but
retrieval quality drops noticeably. That's the point of Profile B.
"""
import os

# Configure environment BEFORE importing SCM so the encoder picks up
# the right backend at construction time.
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "llama3.2:latest")
os.environ.setdefault("SCM_EMBEDDING_BACKEND", "ollama")
os.environ.setdefault("SCM_EMBEDDING_MODEL", "nomic-embed-text")

from src.chat.engine import ChatEngine
from src.core.encoder import MeaningEncoder
from src.llm import LLMExtractor


def main() -> int:
    print(f"LLM provider:        {os.environ['LLM_PROVIDER']} ({os.environ['LLM_MODEL']})")
    print(f"Embedding backend:   {os.environ['SCM_EMBEDDING_BACKEND']} ({os.environ['SCM_EMBEDDING_MODEL']})")
    print()

    try:
        llm = LLMExtractor(provider="ollama")
    except Exception as e:
        print(f"WARNING: LLM extractor failed to initialise ({e}); falling back to heuristic.")
        llm = None

    encoder = MeaningEncoder(
        llm=llm,
        embedding_backend="ollama",
        embedding_model_name="nomic-embed-text",
    )
    engine = ChatEngine(llm=llm, encoder=encoder, profile="chatbot",
                        enable_auto_sleep=False)

    print("→ telling the agent about a job change…")
    engine.chat("I used to work at Northstar Robotics doing backend stuff.")
    engine.chat("As of today I left Northstar — I'm now at Atlas Labs.")
    engine.chat("Atlas does reinforcement learning for industrial robots, very different stack.")
    engine.chat("Their office is in Kendall Square; the commute is shorter than Northstar's.")

    print("→ consolidating with deep sleep…")
    engine.force_sleep("deep")

    print("→ asking with question phrasing that doesn't share tokens with the storage form…")
    response, meta = engine.chat("Where do I work?")
    print(f"\n  agent: {response}")
    print(f"  retrieved {meta.get('memories_retrieved', 0)} memories\n")

    print("→ asking about the previous employer (versioning probe)…")
    response, meta = engine.chat("Where did I used to work?")
    print(f"\n  agent: {response}")
    print(f"  retrieved {meta.get('memories_retrieved', 0)} memories\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
