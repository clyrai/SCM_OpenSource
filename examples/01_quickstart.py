"""
SCM Quickstart — the smallest possible example.

  $ python examples/01_quickstart.py

This script:
  1. Builds an SCM ChatEngine with default heuristic-only configuration
     (no LLM required — runs fully offline).
  2. Ingests four user turns establishing a fact and a routine.
  3. Forces a deep-sleep cycle to consolidate.
  4. Asks the agent to retrieve the stored fact.

Expected output: the agent finds the fact even though the question phrasing
doesn't share tokens with the original sentence — the magic of associative
retrieval.
"""
from src.chat.engine import ChatEngine


def main() -> int:
    engine = ChatEngine(profile="chatbot", enable_auto_sleep=False)

    # Ingest a few turns
    print("→ ingesting turns…")
    engine.chat("Hi, my name is Saish.")
    engine.chat("I live in Bangalore and I love filter coffee.")
    engine.chat("I'm a backend engineer working on distributed systems.")
    engine.chat("Every Sunday I go for a long walk along the lake.")

    # Force a sleep cycle so the schema extractor has something to chew on
    print("→ running deep-sleep cycle…")
    stats = engine.force_sleep("deep")
    if stats:
        print(f"  schemas formed: {stats.get('schemas_formed', 0)}")

    # Retrieve
    print("→ asking a question whose phrasing differs from the storage form…")
    response, meta = engine.chat("What do I do for work?")
    print(f"  agent: {response}")
    print(f"  retrieved {meta.get('memories_retrieved', 0)} memories")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
