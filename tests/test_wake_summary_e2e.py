"""End-to-end test of the demo flow: multi-turn chat → sleep → restart → wake-summary.

The single most important test we have right now: does the wake-summary actually
surface a meaningful narrative for a cold user? If yes, the pitch is real. If no,
the entire demo video / launch story is fiction.
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Force a clean data dir so we test cold-start behavior
_DATA_DIR = tempfile.mkdtemp(prefix="scm_e2e_")
os.environ["SCM_DATA_DIR"] = _DATA_DIR
os.environ["SCM_EMBEDDING_BACKEND"] = "ollama"
os.environ["SCM_EMBEDDING_MODEL"] = "nomic-embed-text"
os.environ["LLM_PROVIDER"] = "deepseek"

# Make sure DeepSeek key is loaded
from dotenv import load_dotenv
load_dotenv()

assert os.environ.get("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY missing from .env"

print(f"=== Cold-start E2E test ===")
print(f"Data dir: {_DATA_DIR}")
print(f"LLM: deepseek | Embedding: ollama nomic-embed-text\n")

# Step 1: Session 1 — ingest a multi-turn conversation
print("--- Session 1: ingesting 5 turns ---")
from src.chat.engine import ChatEngine
from datetime import datetime, timedelta, timezone

engine = ChatEngine(profile="e2e_test", enable_auto_sleep=False)

turns = [
    "Hi, I'm Alex. I'm a backend engineer in Lisbon.",
    "I run every Tuesday morning along the river.",
    "I work at a startup called Filtrum building data pipelines.",
    "Tuesday again — did 6km this time.",
    "My favorite coffee shop is Hello Kristof in Anjos.",
]

for i, t in enumerate(turns, 1):
    print(f"  [{i}] {t}")
    engine.chat(t)

# Step 2: Force a deep sleep cycle
print("\n--- Forcing deep sleep (NREM consolidation + REM schema extraction) ---")
import time
t0 = time.time()
sleep_result = engine.force_sleep("deep")
sleep_dt = time.time() - t0
print(f"Sleep cycle took {sleep_dt:.1f}s")
if sleep_result:
    print(f"Sleep result keys: {list(sleep_result.keys()) if isinstance(sleep_result, dict) else type(sleep_result).__name__}")

# Step 3: Tear down to simulate "user closed the app"
del engine
print("\n--- Engine torn down — simulating restart ---")

# Step 4: Cold restart and pull wake-summary
from src.chat.engine import ChatEngine as ChatEngineNew
from src.lifecycle.wake_summary import WakeSummaryBuilder

engine2 = ChatEngineNew(profile="e2e_test", enable_auto_sleep=False)
builder = WakeSummaryBuilder(engine=engine2)

since = datetime.now(timezone.utc) - timedelta(hours=24)
summary = builder.build(since=since)

print("\n--- WAKE SUMMARY ---")
if summary is None:
    print("[FAIL] No summary generated.")
elif not summary.narrative:
    print("[FAIL] Summary built but narrative is empty.")
    print(f"Object: {summary}")
else:
    print(summary.narrative)
    print(f"\n(Narrative: {len(summary.narrative)} chars, {len(summary.narrative.split())} words)")

# Step 5: Quick retrieval test — does it remember Alex / Tuesday / Filtrum?
print("\n--- Retrieval probe: 'where do I work?' ---")
from src.integrations.tools import _search_memory_handler
res = _search_memory_handler({"query": "where do I work", "user_id": "alex"}, engine2)
print(f"  retrieved_count: {res.get('retrieved_count')}")
print(f"  memory_context: {res.get('memory_context','')[:300] or '(empty)'}")
for m in res.get("memories", [])[:5]:
    print(f"  - {m.get('description','')[:120]}")

print("\n--- Retrieval probe: 'when do I run?' ---")
res = _search_memory_handler({"query": "when do I go running", "user_id": "alex"}, engine2)
print(f"  retrieved_count: {res.get('retrieved_count')}")
print(f"  memory_context: {res.get('memory_context','')[:300] or '(empty)'}")
for m in res.get("memories", [])[:5]:
    print(f"  - {m.get('description','')[:120]}")

# Diagnostic: dump ALL concepts in the graph
all_concepts = engine2.long_term_memory.get_all_concepts()
print(f"\n--- All {len(all_concepts)} concepts in graph ---")
for c in all_concepts:
    tags = c.context_tags if isinstance(c.context_tags, dict) else {}
    print(f"  [{c.type}] {c.description[:120]}  tags={list(tags.keys())[:3]}")

# Direct search_by_text test
print("\n--- LTM.search_by_text('Filtrum') ---")
hits = engine2.long_term_memory.search_by_text("Filtrum", limit=5)
for h in hits:
    print(f"  - {h.description[:120]}")

print("\n--- LTM.search_by_text('Tuesday') ---")
hits = engine2.long_term_memory.search_by_text("Tuesday", limit=5)
for h in hits:
    print(f"  - {h.description[:120]}")

# Check spreading activation directly
sa = getattr(engine2, "_spreading_activation", None)
print(f"\n--- spreading_activation present: {sa is not None} ---")
if sa is not None:
    activated, stats = sa.retrieve("where do I work", context_tags={"session_id": "default", "person": None})
    print(f"  activated count: {len(activated)}")
    print(f"  stats: {stats}")
    for c in activated[:5]:
        print(f"  - {c.description[:120]}")

# Cleanup
shutil.rmtree(_DATA_DIR, ignore_errors=True)
print(f"\n=== Done. Cleaned up {_DATA_DIR} ===")
