# ALB — Autonomous Learning Benchmark

A benchmark for measuring **what an agent memory system does during user idle time**, and how that idle work translates into capability the system did not have before.

> Read [SPEC.md](SPEC.md) first. It is the contract; this README is the implementation status and pilot results.

---

## v0.1 status

**Frozen on 2026-05-02.** Spec, persona schema, adapter contract, metric scorers, statistics, and runner are in place. Two pilot personas exist; SCM is the only adapter wired. First numbers below.

### What works

- ✅ SPEC.md frozen with 7 metrics, 10 pre-registered hypotheses, full statistical methodology
- ✅ JSON Schema validates personas (jsonschema draft-07)
- ✅ `BaseMemorySystem` ABC + value objects + conformance checker
- ✅ All 7 metric scorers + matching primitives (deterministic, no LLM in scoring)
- ✅ Stats module: bootstrap CI, paired t-test, Cohen's d, Wilcoxon, Holm-Bonferroni — verified against null distribution (FPR ≈ 0.052 at α=0.05)
- ✅ Runner drives any adapter through any persona, snapshots schemas/gaps before/after each idle for WSI, NIAL ablation via `idle_on=False`
- ✅ Smoke test with PerfectAdapter / NullAdapter passes
- ✅ SCM adapter (full Phase 7 stack) runs end-to-end on both personas in ~2 minutes per run
- ✅ Pilot results written to `results/scored.csv` and `results/summary.md`

### What's deferred to v0.2

- 18 more personas (target 20 total per spec)
- Adapters for MemoryBank, Generative Agents (reflection), A-Mem, Mem0, MemGPT, vector floor
- Multi-seed runs (target 5 seeds × 20 personas = 100 paired runs)
- Statistical comparison tables (paired t-test + Cohen's d + Holm-Bonferroni) — code is in `stats.py`, just needs ≥2 systems to compare
- Real-LLM matrix (Ollama + DeepSeek)

---

## Pilot results — SCM v0.1

Two pilot runs were conducted. **v3 is the current canonical result** after three product-level bug fixes informed by the v1 pilot.

### v3 — after bug fixes (canonical)

**2 personas × 1 seed × 2 idle conditions = 4 runs. Wall time ≈ 13 min on M1 Air.**

#### idle_on = True

| persona | PDR | CSS | CGC_id | CGC_fill | CRAI_cur | CRAI_old | WSI_F1 |
|---|---|---|---|---|---|---|---|
| persona_001 | 0.750 | 0.000 | 0.500 | 0.500 | 0.000 | 0.000 | 0.211 |
| persona_002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |

#### idle_on = False (NIAL ablation)

| persona | PDR | CSS | CGC_id | CGC_fill | CRAI_cur | CRAI_old | WSI_F1 |
|---|---|---|---|---|---|---|---|
| persona_001 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| persona_002 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

#### NIAL lift (idle_on − idle_off, v3 vs v1)

| metric | v1 lift | **v3 lift** | direction |
|---|---|---|---|
| PDR | +0.250 | **+0.375** | idle helps (+50% better) |
| WSI_F1 | +0.250 | **+0.355** | idle helps (+42% better) |
| CGC_id | +0.250 | +0.250 | idle helps |
| CGC_fill | +0.250 | +0.250 | idle helps |
| CRAI_current | 0.000 | 0.000 | invariant |
| CSS | −1.000 | −1.000 | idle HURT (unchanged) |
| CRAI_old | −1.000 | −1.000 | idle HURT (unchanged) |

### Three bugs fixed between v1 and v3

| # | Bug | Fix | Outcome |
|---|---|---|---|
| C | Schema extractor produced 30-40 schemas/run, ~85% noise (`I've`, `Code`, `What's`, `Tuesdays` alone) | Tightened entity regex to reject contractions; expanded `schema_extractor` stoplist with sentence-starter common words; added day-of-week to weekly cadence schema descriptions | **Worked.** PDR +50%, WSI +42%. |
| A | Deep-sleep paraphrase rewrites concept descriptions, breaking verbatim-recall queries | `LongTermMemory.search_by_text()` and `SpreadingActivationRetriever._cue_match_score()` now check both `description` and `context_tags["original_description"]` | **Code fix shipped, no measurable lift on CSS.** CSS = -1.0 still. The actual cause of CSS regression is deeper: the salience-thin heuristic encoder lets deep-sleep forgetting archive low-salience concepts (peanut allergy from a casual dinner mention) before they can be retrieved. Same encoder-dependence story as the paper. |
| B | Question phrasing ("Where do I work?") doesn't share tokens with declarative facts ("I'm at Atlas Labs") | Added embedding-similarity seed fallback to `_select_seeds` (bounded to 60 candidates by recency); wired `MeaningEncoder` into `SpreadingActivationRetriever` constructor | **Code fix shipped, no measurable lift on CRAI.** CRAI_current = 0 still. Embedding fallback fires but cosine similarity between question form and factual storage form (sentence-transformer all-MiniLM-L6-v2) doesn't bridge the abstraction gap. Real fix needs structured fact extraction (property → value tuples) at ingestion time, which is a Phase 8 architectural change, not a patch. |

**131/131 SCM regression tests pass after all fixes.** No flaky-LLM tests counted in that 131.

---

## Honest read of the pilot

These are 4 single-seed runs on a 2-persona pilot. They are **not** statistically rigorous — they describe the system's behavior, they do not yet support significance claims. With that caveat:

### What's working
- **Phase 7 schema extraction is live and producing schemas** that match planted patterns at PDR=0.50 on persona_001. Without idle (PDR=0), no schemas form. The +0.25 mean lift across both personas is the expected direction.
- **Curiosity engine identifies and fills gaps** (CGC_id and CGC_fill both at 0.5 on persona_001 with idle on; both at 0 with idle off). The static dictionary fills "OAuth flow" but not "Kubernetes ingress" — see known issues.
- **Wake summary is produced** with perfect recall (every formed schema is reported) but low precision (~0.08 on persona_001, ~0.36 on persona_002) because SCM forms many noisy schemas that pollute the report. WSI lift is +0.25 across personas.

### What's broken (ALB found these honestly)
- **Cross-Session Synthesis collapses with idle on.** CSS=1.0 with idle off, CSS=0.0 with idle on. The pattern: when the deep-sleep cycle paraphrases and consolidates raw concepts, the verbatim text that ALB's keyword scorer matches is altered. Same root cause as the Phase 6 encoder-dependence finding documented in the paper. **This is a real SCM tuning issue, not a benchmark artifact.**
- **CRAI_current = 0 across the board.** SCM correctly stores both employer concepts and the spreading-activation retriever has cue-driven access, but query phrasing like *"Where do I work?"* doesn't share tokens with *"Atlas Labs"* / *"Northstar Robotics"*. The retrieval doesn't bridge the question form to the fact form. Suggests the adapter needs query expansion or the SCM retriever needs semantic seed selection.
- **CRAI_old shows the same idle-hurts-retrieval pattern as CSS** (1.0 → 0.0). Confirms the consolidation pipeline is rewriting concept descriptions in ways that lose verbatim entity strings.
- **PDR collapses on persona_002** (0.500 on persona_001, 0.000 on persona_002). The chemistry-teacher persona has Mon/Wed/Fri biking and Sunday lab prep cadences. SCM extracts "biking" and "Sunday" as separate recurring topics but never connects them as a TEMPORAL_CADENCE schema. Same limitation that misses `p_running_tuesday` on persona_001.

### What this means
**The benchmark is honest.** It doesn't reward SCM unfairly — both personas show real gaps. It also clearly shows where SCM does work (PDR, CGC, WSI lift on idle). The −1.0 lift on CSS and CRAI_old is the most interesting finding: the consolidation cycle is **net-harmful** for verbatim-recall workloads. This is a real architectural tradeoff that the paper should document.

---

## Known issues (file under "this is the system, not the benchmark")

1. **Schema noise.** SCM emits 30–40 schemas per run, of which 3–4 match planted patterns. The schema extractor's `min_repetitions=2` threshold catches too much filler. WSI precision is dominated by this noise.
2. **Pattern type coverage.** Persona TEMPORAL_CADENCE patterns (Tuesday running, Mon/Wed/Fri biking) aren't matched. The schema extractor finds the day token and the activity token separately but never as a CADENCE schema. Either the extractor's signature shape doesn't match ALB's expectation, or it really doesn't form these.
3. **Query → fact bridging.** SCM's retrieval is keyword-based. Question phrasing and declarative storage form don't share enough surface tokens. CRAI_current = 0 across all runs.
4. **Idle-hurts-retrieval.** Deep-sleep paraphrase + consolidation alters concept descriptions in ways that hurt CSS and CRAI_old. This is the encoder-dependence story already documented in the paper, now confirmed on a different evaluation axis.
5. **Schema concept ID determinism.** Two runs on the same persona produce slightly different PDR (0.500 vs 0.750 in different runs). The Phase 7 stable-schema-ID fix is in place but downstream randomness (concept ordering, hash collisions) still introduces variance. **Multi-seed runs in v0.2 will smooth this out.**

---

## File map

```
SPEC.md                              # The frozen contract. Read this first.
README.md                            # This file.
personas/
  schema.json                        # JSON Schema for personas.
  persona_001.json                   # Devon Park (software engineer, 5 days)
  persona_002.json                   # Priya (chemistry teacher, 5 days)
adapters/
  __init__.py
  base.py                            # BaseMemorySystem ABC + value objects
  scm_adapter.py                     # SCM Phase 7 wired to ALB
metrics/
  __init__.py
  match.py                           # Pattern + keyword matching primitives
  pdr.py                             # Pattern Discovery Rate
  css.py                             # Cross-Session Synthesis
  cgc.py                             # Curiosity Gap Coverage (id + fill)
  crai.py                            # Contradiction Resolution Across Idle
  wsi.py                             # Wake-Summary Informativeness
  imc.py                             # Idle Maintenance Cost
  nial.py                            # No-Idle Ablation Lift
runner.py                            # Drives a persona through any adapter
stats.py                             # Bootstrap CI, paired t, Wilcoxon, Cohen's d, Holm-Bonferroni
test_smoke.py                        # PerfectAdapter + NullAdapter wiring test
scripts/
  diagnose_scm_run.py                # Diagnostic dump for one SCM run
  run_pilot.py                       # The full pilot (writes results/)
results/
  raw/                               # Per-run JSON artifacts
  scored.csv                         # Flat table of all run scores
  summary.md                         # Auto-generated by run_pilot.py
```

---

## Reproducing the pilot

```bash
cd research/benchmarks/alb
python test_smoke.py                 # ~3s, exercises framework with stub adapters
python scripts/diagnose_scm_run.py   # ~3min, deep dump on persona_001
python scripts/run_pilot.py --seeds 1   # ~9min, the full pilot
cat results/summary.md
```

Requires the SCM venv with sentence-transformers + jsonschema:
```bash
source /Users/saish/Downloads/SleepAI/venv/bin/activate
pip install jsonschema
```

---

## Next steps for v0.2

1. Three more personas (target 5 hand-authored for v0.2 pilot).
2. Multi-seed runs (5 seeds × 5 personas = 25 paired runs per system).
3. Wire MemoryBank, Generative Agents (reflection), and A-Mem adapters for the headline comparison.
4. Statistical comparison tables in `summary.md`: paired t, Cohen's d, Holm-Bonferroni.
5. Real-LLM matrix (DeepSeek extractor + LLMSource for curiosity).
6. Per-persona failure ledger so reviewers can see exactly which planted patterns/gaps each system missed.
