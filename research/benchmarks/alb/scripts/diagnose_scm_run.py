"""
Diagnostic harness — runs SCM through persona_001, then dumps:
  - what schemas were formed (and their signatures)
  - what gaps the curiosity engine sees
  - what each probe query returned
  - which expected items are missing

Use this to debug adapter wiring, NOT to score systems.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from adapters.scm_adapter import SCMAdapter
from runner import load_persona, run_persona, score_run


def main() -> int:
    persona = load_persona(HERE / "personas" / "persona_001.json")
    adapter = SCMAdapter()

    print("=" * 70)
    print("Running SCM adapter on persona_001 (idle_on=True)…")
    print("=" * 70)
    run = run_persona(persona, adapter, seed=0, idle_on=True)
    scored = score_run(run, persona, adapter=adapter)
    print(scored.summary())
    print()

    # Dump schemas formed
    print("=" * 70)
    print(f"Schemas formed by SCM ({len(run.final_schemas)}):")
    print("=" * 70)
    for s in run.final_schemas[:25]:
        print(f"  [{s.type}] sig={s.signature} desc={s.raw_text[:80]!r}")
    if len(run.final_schemas) > 25:
        print(f"  ...and {len(run.final_schemas) - 25} more")
    print()

    # What patterns did we expect, and which match?
    print("=" * 70)
    print("Ground-truth patterns vs system schemas:")
    print("=" * 70)
    from metrics import match_pattern
    for p in persona["ground_truth"]["patterns"]:
        sid, score = match_pattern(p, run.final_schemas)
        status = "MATCH" if sid else "MISS"
        print(f"  {status}  {p['pattern_id']:30s} type={p['type']:18s} → {sid or '(no match)'}")
    print()

    # Gaps
    print("=" * 70)
    print(f"Open gaps reported by SCM ({len(run.final_gaps)}):")
    print("=" * 70)
    for g in run.final_gaps[:10]:
        print(f"  [{g.gap_id}] term={g.term!r} occ={g.occurrences} filled={g.has_been_filled}")
    print()
    print("Expected gaps from persona:")
    for gt in persona["ground_truth"]["gaps"]:
        print(f"  {gt['gap_id']:25s} term={gt['term']!r}")
    print()

    # Probe responses
    print("=" * 70)
    print("Probe outcomes:")
    print("=" * 70)
    for p in run.probe_outcomes:
        print(f"  [{p.target_metric:12s}] {p.query_id:25s}")
        print(f"      query: {p.query_text!r}")
        print(f"      response: {(p.response_text or '(empty)')[:200]!r}")
        print()

    # Concept inventory by type
    from collections import Counter
    print("=" * 70)
    print("Concept inventory:")
    print("=" * 70)
    type_counts = Counter()
    schema_type_counts = Counter()
    has_employer_atlas = False
    has_employer_northstar = False
    for c in adapter._all_concepts():
        type_counts[str(c.type)] += 1
        tags = c.context_tags or {}
        if tags.get("schema_type"):
            schema_type_counts[tags["schema_type"]] += 1
        desc_low = (c.description or "").lower()
        if "atlas" in desc_low:
            has_employer_atlas = True
        if "northstar" in desc_low:
            has_employer_northstar = True
    print(f"  total concepts: {sum(type_counts.values())}")
    print(f"  by type: {dict(type_counts)}")
    print(f"  schema types: {dict(schema_type_counts)}")
    print(f"  has Atlas concept: {has_employer_atlas}")
    print(f"  has Northstar concept: {has_employer_northstar}")
    print()

    # Sample concepts mentioning employers
    print("=" * 70)
    print("Sample of employer-related concepts:")
    print("=" * 70)
    seen = 0
    for c in adapter._all_concepts():
        d = (c.description or "").lower()
        if "atlas" in d or "northstar" in d:
            print(f"  state={c.state} cur_ver={getattr(c, 'is_current_version', '?')} type={c.type}")
            print(f"    desc: {(c.description or '')[:160]!r}")
            seen += 1
            if seen >= 6:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
