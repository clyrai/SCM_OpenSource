# SCM Bug Discovery Log

Every real bug found by the brutal harness, including its symptom, root cause, fix, verification, and the date discovered. This is the "honest engineering" record the paper §9.10-9.11 documents at the architectural level — this file documents at the line-of-code level.

---

## Bug #1 — Schema-ID instability ("85,310 consolidated concepts")

**Discovered:** 2026-04-30 (brutal harness Caroline persona, single-agent)
**Affected:** v0.6 series
**Severity:** Critical (memory blow-up, would have OOMed in production within hours)

### Symptom

Running the 31-turn Caroline persona produced 85,310 "consolidated" concepts. Each replay added more.

### Root cause

[`src/sleep/schema_extractor.py`](../src/sleep/schema_extractor.py) was minting a fresh UUID for each schema concept on every cycle. Two replays of the same persona produced two distinct concept IDs for the same `COOCCUR(running, tuesday)` pattern, polluting the graph with near-duplicate schemas.

### Fix

Derive schema IDs deterministically: `SHA1(schema_type || sort(entities))`. Same logical schema collapses to the same concept across cycles. Cumulative metadata (occurrence count, last-seen, supporting episode IDs) is updated additively on re-derivation.

```python
sig = "|".join([self.schema_type, ":".join(sorted(e.lower() for e in self.entities))])
digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()
stable_id = f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
```

### Verification

Tier 5 brutal scenario re-runs the persona twice and asserts schema concept count stays bounded (<50). Currently passing.

---

## Bug #2 — Spreading-activation seed purging

**Discovered:** 2026-04-30 (brutal harness, 0/4 multi-day recall scenarios)
**Affected:** v0.6 series
**Severity:** Critical (silent retrieval failure)

### Symptom

0/4 multi-day recall scenarios returned the seeded fact even when the cue concept was a direct match.

### Root cause

The spreading-activation propagator applied a default decay of 0.45 over 3 steps, dropping seed activation below the 0.05 inclusion threshold by the final round even though the seed itself was the cue match. The activated-concept output filter then dropped the seed silently.

### Fix

Always preserve cue-matched seeds in the output set, gated only by the `is_current_version` flag (see Bug #4). Decay still applies to non-seed neighbors.

```python
# Always preserve cue-matched seeds in the result
included_ids = {c.id for c, _ in activated_concepts}
for seed_concept, seed_init in seeds:
    if seed_concept.id in included_ids:
        continue
    if not getattr(seed_concept, "is_current_version", True):
        continue
    seed_act = activation.get(seed_concept.id, seed_init)
    activated_concepts.append((seed_concept, seed_act))
```

### Verification

Tier 5 scenario asserts that for each `expected_recall` entry in the persona, the seeded concept appears in the retriever's output for the matching cue.

---

## Bug #3 — Dead cross-session pool in single-user case

**Discovered:** 2026-04-30 (brutal harness, single-user multi-day persona)
**Affected:** v0.6 series M2 default config
**Severity:** High (silent feature failure)

### Symptom

A user with a single logical session ID who chatted across multiple days saw zero schema formation, despite each individual day's content containing strong patterns.

### Root cause

The cross-session pool's default policy was "include all session IDs except the current one" — which made sense for multi-tenant deployments but silently disabled cross-day pattern detection in the single-user case (the only candidate was the current session, which got excluded).

### Fix

Added an `include_current_session` flag (default `False` for backward compat) that, when set, includes the current session in the candidate pool so older persisted-but-cleared episodes can be replayed.

```python
if self.config.include_current_session:
    candidate_ids = list(all_recent_ids)
    if self.current_session_id and self.current_session_id not in candidate_ids:
        candidate_ids.insert(0, self.current_session_id)
else:
    candidate_ids = [s for s in all_recent_ids if s != self.current_session_id]
```

### Verification

Tier 5 scenario runs a two-day single-session persona and asserts that at least one schema is formed by day two. Currently passing.

---

## Bug #4 — Superseded-concept retrieval leakage

**Discovered:** 2026-04-30 (brutal harness, contradiction handling)
**Affected:** v0.6 series spreading-activation retriever
**Severity:** High (correctness violation — versioning machinery existed but was bypassed)

### Symptom

After a contradiction (e.g., *"I work at GreenLeaf"* → *"I work at TechCorp"*), queries about the user's employer returned both old and new concepts.

### Root cause

The Phase 5 contradiction-versioning machinery correctly marked the old concept's `is_current_version = False`, but the spreading-activation retriever did not consult this flag when selecting seeds, propagating activation, or filtering the final output.

### Fix

Added the `is_current_version` filter at all three sites (seed selection, propagation, final output filter).

### Verification

Tier 5 scenario asserts that for each `contradiction-current` expected outcome, the retriever returns only the current concept. Currently passing.

---

## Bug #5 — `SCMClient` API method name mismatch (the silent killer)

**Discovered:** 2026-05-02 (brutal LangChain harness diagnostic)
**Affected:** v0.7.0
**Severity:** Critical (SCM was completely uncalled — but tests "passed" because LangChain history did the work)

### Symptom

5 brutal LangChain harness runs (v1-v5) reported 13/16 - 16/16 pass rates. Subsequent diagnostic revealed `AttributeError: 'SCMClient' object has no attribute 'search_memory'` on every chat call.

### Root cause

[`src/integrations/langchain_adapter.py`](../src/integrations/langchain_adapter.py) defined methods named `add()` and `search()`. The agent in [`tests/brutal_langchain/agent.py`](../tests/brutal_langchain/agent.py) called `add_memory()` and `search_memory()`. Every call raised `AttributeError`, which the agent's broad `except` swallowed silently. **SCM was completely uncalled.** The tests "passed" because LangChain's in-context message history kept the user's prior turns visible to the LLM — exactly the behavior we were claiming SCM provided.

### Fix

Renamed `SCMClient.add` → `add_memory`, `SCMClient.search` → `search_memory`. Added aliases for backward compat. Updated all callers.

### Lesson

This is the canonical example of why end-to-end tests must verify mechanism, not just outcome. The output looked right (LLM said "Bangalore" when asked about city) but for the wrong reason. We added explicit per-turn `error` reporting to the agent so future silent failures can't hide.

### Verification

After the fix, the brutal LangChain harness produced **5,561× different latency numbers** — proving SCM was now genuinely on the call path. All 16/16 tiers pass with SCM verifiably handling the storage.

---

## Bug #6 — `_context_gate` AttributeError on empty working memory

**Discovered:** 2026-05-02 (during diagnosis of the v3 ALB pilot — alice's seafood retrieval failure)
**Affected:** v0.7.0, v0.7.1
**Severity:** Critical (entire spreading-activation path broken after deep-sleep)

### Symptom

After a `force_consolidate("deep")` cycle that cleared working memory, the next `_retrieve_hme` call raised `AttributeError: 'NoneType' object has no attribute 'lower'` — silently breaking all retrieval.

### Root cause

In [`src/retrieval/spreading_activation.py`](../src/retrieval/spreading_activation.py), `_context_gate` did:

```python
if context_tags['person'].lower() == concept_tags['person'].lower():
```

When `working_memory.retrieve()` returned an empty list (just after consolidation), `context_tags['person']` was `None`. `.lower()` crashed.

### Fix

Guarded with `isinstance(..., str)` checks. Also fixed the same issue for `session_id`.

```python
ctx_person = context_tags.get('person')
con_person = concept_tags.get('person')
if isinstance(ctx_person, str) and isinstance(con_person, str):
    if ctx_person.lower() == con_person.lower():
        score *= 1.3
```

### Verification

Three new regression tests in [`tests/test_spreading_activation.py`](../tests/test_spreading_activation.py):
- `test_retrieve_with_none_person_tag_does_not_crash`
- `test_retrieve_with_no_context_tags_does_not_crash`
- `test_retrieve_with_missing_person_key_does_not_crash`

All passing.

---

## Bug #7 — Spreading-activation sort weights backward

**Discovered:** 2026-05-02 (during the diag_alice7 seafood test, after Bug #6 fix)
**Affected:** v0.6 / v0.7.0 / v0.7.1
**Severity:** Medium (search returned wrong-priority results)

### Symptom

Searching alice's memory for "What food am I allergic to?" returned `["I completed sleep cycle 1...", "can remember conversations", ...]` — SelfModel concepts at the top, the actual seafood concept buried.

### Root cause

`SpreadingActivationRetriever` ranked activated concepts via:

```python
key = 0.85 * compute_consolidation_score(c) + 0.15 * cue_match_score(c, cues)
```

Well-rehearsed background concepts (SelfModel facts) had high consolidation scores, beating the actual cue-matching answer.

### Fix

Inverted the weighting:

```python
key = 0.70 * cue_match_score(c, cues) + 0.30 * compute_consolidation_score(c)
```

Cue-match dominates; consolidation is the tiebreaker.

### Verification

Same diag scenario after the fix returns `["seafood allergy concept"]` first. ALB v3 pilot CGC_id metric improved from 0.250 → 0.500.

---

## Bug #8 — `FORGETTING_PROTECT_SALIENCE=0.0` by default

**Discovered:** 2026-05-02 (during diagnosis of why "I'm allergic to seafood" was being archived)
**Affected:** v0.6 / v0.7.0 (Phase 6 fix existed but was opt-in)
**Severity:** Critical (user-stated facts archived seconds after ingestion)

### Symptom

In the multi-agent harness, alice ingested "I'm allergic to seafood" then triggered `force_consolidate("deep")`. Subsequent search returned 0 memories. Tracing showed the concept was moved to SUPPRESSED state during the sleep cycle.

### Root cause

The Phase 6 fix `FORGETTING_PROTECT_SALIENCE` existed but defaulted to `0.0`, which disabled it ("0.0 preserves legacy behavior" per the paper). The forgetting machinery archived any low-salience concept, including freshly-ingested user facts.

### Fix

[`src/core/config.py`](../src/core/config.py): default raised from `0.0` → `0.5`. Default `FORGETTING_MIN_REHEARSAL_BEFORE_ARCHIVE` raised from `0` → `1`.

### Side effects (test updates)

5 forgetting tests had to opt back into legacy aggressive behavior via `protect_salience=0.0` in their setup, because they exercised the forgetting mechanics directly:
- `tests/test_sleep.py::TestForgettingModule.setUp`
- `tests/test_deep_sleep.py::test_deep_sleep_downscale_and_forgetting`
- `tests/test_deep_sleep.py::test_deep_sleep_suppresses_low_importance_hubs`
- `tests/phase4_metrics.py` distractor concepts now have explicit `salience_score=0.1`

### Verification

`diag_alice9` test (heuristic encoder + Ollama embedding) now correctly returns the seafood concept after consolidation. 322/322 regression passing with new defaults. Brutal LangChain harness 16/16.

---

## Bug #9 — Multi-user isolation leak via shared SQLite singleton

**Discovered:** 2026-05-02 (brutal LangChain harness Tier 6)
**Affected:** v0.7.0, v0.7.1
**Severity:** Critical (privacy violation — users can see each other's facts)

### Symptom

Alice told the agent her secret pet name "Mr. Whiskers" with `user_id="t6_alice"`. Bob ingested cars at `user_id="t6_bob"`. Alice's later query for "What car do I drive?" retrieved Bob's Honda Civic concept.

### Root cause

`UserEnginePool` created one `ChatEngine` per user_id, but the underlying `sqlite_db` module is a process-global singleton. All users wrote to the same SQLite. `LongTermMemory.search_by_text` didn't filter by session_id at the LTM level.

### Fix (v0.7.1)

`UserEnginePool._build_engine` now constructs each `ChatEngine` with `sandbox_mode=True` and `enable_persistence=False`. Each engine has its own in-memory NetworkX graph. No cross-user leak via SQLite.

### Cost

Sandbox mode loses memory across server restarts. For v0.7.x this is the right tradeoff (correctness > durability). v0.8 work: per-user persistent SQLite that doesn't depend on the global singleton.

### Verification

Tier 6 of the brutal LangChain harness: bob retrieves 0 memories for "pet name", alice retrieves 0 for "car". Each user retains own memories. Currently 3/3 passing.

---

## Bug #10 — Async-ingest race / consolidation ordering

**Discovered:** 2026-05-04 (during v0.7.2 design)
**Affected:** Would have appeared in v0.7.2 if not handled
**Severity:** Latent (would have caused intermittent test failures)

### Symptom (anticipated, prevented)

`add_memory` returns immediately. If a test/agent immediately calls `consolidate` on the same user, the consolidation cycle would see a stale graph (the just-added concept hasn't been extracted yet).

### Fix

`UserEnginePool.fire_sleep_now` always calls `wait_for_pending(user_id, timeout=10.0)` before invoking `engine.force_sleep()`. Same for the `consolidate` path in `memories_api._invoke`.

### Verification

Brutal LangChain harness with async ingest enabled: 16/16 passing (was 16/16 with sync). Race avoided.

---

## Format note

This file is append-only. Bugs are not retroactively edited; they're updated with a `**Update:**` section if new information emerges. The file is the bug-discovery record for SCM as a whole, including bugs that no longer exist in the current version.
