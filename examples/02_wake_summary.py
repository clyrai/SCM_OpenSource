"""
SCM Wake-Summary demo — the killer moment.

  $ python examples/02_wake_summary.py

This script simulates a five-day chat history compressed into seconds,
then asks for the wake-summary that the agent would deliver to the user
on day 6 morning.

The transcript is hand-crafted so that the schema extractor *should*
find:
  • a recurring Tuesday-morning run
  • a Friday dinner co-occurrence (Mara + dinner)
  • an employer change (Northstar → Atlas Labs) that the contradiction
    versioning should resolve

If your environment has Ollama with `nomic-embed-text` installed, the
results will be richer because semantic retrieval is far stronger. The
script otherwise runs fully offline.
"""
import os
from datetime import datetime, timedelta

from src.chat.engine import ChatEngine
from src.lifecycle.wake_summary import WakeSummaryBuilder


# A compressed five-day persona transcript.
TURNS = [
    # day 1 — Monday
    "Tuesday morning is when I run, before standup. Today I biked to work instead.",
    "Standup at Northstar Robotics. Rough day, OAuth flow is blocking three teams.",
    # day 2 — Tuesday
    "Tuesday — went for the usual morning run along the river. 5K, brutal headwind.",
    "OAuth flow is fixed; the redirect-handler scopes were wrong.",
    # day 3 — Friday
    "Friday already. Dinner with Mara at the noodle place tonight, our standing thing.",
    "Mentioned the Atlas Labs offer to Mara at dinner. She thinks I should take it.",
    # day 4 — Tuesday
    "Tuesday run, 6K this time. Felt strong.",
    "Big news — left Northstar Robotics today. I'm at Atlas Labs starting Monday.",
    # day 5 — Friday
    "Friday again. Dinner with Mara. We've kept this tradition through three job changes.",
    "First week at Atlas Labs done. Onboarding had me set up a Kubernetes ingress.",
]


def main() -> int:
    engine = ChatEngine(profile="research", enable_auto_sleep=False)
    wake_builder = WakeSummaryBuilder(engine=engine)

    print(f"→ ingesting {len(TURNS)} turns spanning a simulated work week…")
    for t in TURNS:
        engine.chat(t)

    print("→ running deep-sleep consolidation…")
    stats = engine.force_sleep("deep")
    if stats:
        print(f"  schemas formed: {stats.get('schemas_formed', 0)}")
        print(f"  contradictions: {stats.get('contradictions_resolved', 0)}")

    # Build the wake summary as if the user just returned next morning
    since = datetime.utcnow() - timedelta(hours=24)
    summary = wake_builder.build(since=since)

    print("\n" + "=" * 60)
    print("WAKE SUMMARY  (this is what the user reads next session)")
    print("=" * 60)
    print(summary.narrative or "(no narrative produced)")
    print("=" * 60)

    insights = getattr(summary, "insights", []) or []
    if insights:
        print(f"\n{len(insights)} insight(s):")
        for i in insights[:8]:
            print(f"  • {i}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
