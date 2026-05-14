# ALB: Autonomous Learning Benchmark

**Specification v0.1 — pre-registration draft**

A benchmark for measuring **what an agent memory system does during user idle time**, and how that idle work translates into capability the system did not have before.

---

## 0. Why this benchmark exists

Existing memory benchmarks (LoCoMo, LongMemEval, MemoryBank's eval set) score a single axis: *given a fixed snapshot of memory, how well does the system answer questions?* This rewards retrieval-family architectures and is silent on a different axis: *how does the snapshot itself improve between sessions, without being asked?*

ALB is designed to measure the second axis. It is not designed to make any specific system win. The metrics are chosen so that a system can score zero on every metric without being broken — they measure presence of capability, not absence of failure. A trivially correct retrieval system (e.g., raw vector cosine over conversation turns) should score near zero on ALB's headline metrics. A system that does *something* during idle time should score above zero. The interesting question is **how much above zero, and at what cost.**

---

## 1. Scope

### 1.1 What ALB measures

1. **Pattern Discovery Rate (PDR)** — does the system autonomously detect repeating structure in the conversation stream?
2. **Cross-Session Synthesis (CSS)** — can the system answer queries that require combining facts from multiple sessions?
3. **Curiosity Gap Coverage (CGC)** — does the system identify and (autonomously, where possible) fill knowledge gaps?
4. **Contradiction Resolution Across Idle (CRAI)** — does the system reconcile conflicts that span sessions?
5. **Wake-Summary Informativeness (WSI)** — when the user returns from idle, what does the system report it learned, and is the report accurate?
6. **Idle Maintenance Cost (IMC)** — what does idle processing cost in CPU time, memory growth, and wall-clock?
7. **No-Idle Ablation Lift (NIAL)** — for each system, the difference between "idle processing enabled" and "idle processing disabled" on metrics 1–5.

### 1.2 What ALB does NOT measure

- **Raw retrieval latency.** Use existing latency benchmarks; ALB is a quality benchmark.
- **Generation quality.** ALB is memory-only; the LLM is held fixed across all systems.
- **Long-horizon stability beyond 5 days.** Multi-week deployments are out of scope for v0.1.
- **Multi-user / privacy guarantees.** Single-user; multi-tenant tests are out of scope.
- **Multi-modal memory.** Text only.
- **Tool use.** Memory-only; no external API calls except the configured LLM.

### 1.3 What ALB is NOT
- It is **not** a replacement for LoCoMo. Systems should be evaluated on both: ALB for the autonomous-learning axis, LoCoMo (or LongMemEval) for the clean-fact-retrieval axis. **Each benchmark answers a different question.**
- It is **not** a leaderboard. There is no single composite score. Each metric has its own table.

---

## 2. Pre-registration

To preempt the "you tuned this benchmark to make your system win" critique:

- **The spec is frozen before any system is run on it.** Changes after the first run are tracked in a CHANGELOG with rationale.
- **Hypotheses are stated before runs.** See §10.
- **All systems run with frozen versions.** Version pin recorded with results.
- **All raw outputs are released**, not just summary statistics. Anyone can re-score.
- **Failure modes are reported with the same emphasis as wins.** A separate per-persona failure ledger is published alongside means.

---

## 3. Workload: persona-driven multi-day streams

### 3.1 Persona structure

A persona is a JSON document conforming to `personas/schema.json` (companion file). Each persona contains:

- **Demographics + role.** Hand-authored summary (occupation, location, social context). Used by the generator only; not seen by the system under test.
- **Days.** A list of 3–7 days. Each day has a wake time, a sleep time, and an ordered list of turns.
- **Turns.** A list of `{timestamp_utc, speaker: "user"|"agent", text}`. Agent turns are scripted (the system under test does NOT generate them — see §3.4).
- **Idle periods.** Implicit from timestamps. The gap between the last turn of day N and the first turn of day N+1 is one idle period. The runner advances simulated time across these gaps.
- **Probe queries.** A list of `{timestamp_utc, query, target_metric, scoring_rule}` injected at controlled time points.
- **Ground truth.** Hidden from the system under test; used by the scorer.

### 3.2 Ground truth structure

Each persona ships a `ground_truth` block declaring what is *plantable* in the persona:

```jsonc
{
  "patterns": [
    {
      "pattern_id": "p_running_tuesday",
      "type": "TEMPORAL_CADENCE",
      "signature": {"event": "running", "cadence": "weekly", "day_of_week": "tuesday"},
      "planted_in_sessions": [1, 2, 3],          // which days the pattern is observable
      "expected_discovery_after_session": 3      // by end of day 3 a working system should have it
    }
  ],
  "gaps": [
    {
      "gap_id": "g_oauth",
      "term": "OAuth flow",
      "expected_definition_keywords": ["authorization", "token", "redirect", "client"],
      "planted_in_sessions": [1],
      "min_keyword_match": 2                     // 2 of 4 keywords = pass
    }
  ],
  "contradictions": [
    {
      "contradiction_id": "c_employer",
      "old_value": "GreenLeaf Cafe",
      "new_value": "TechCorp",
      "old_planted_session": 1,
      "new_planted_session": 4,
      "probe_query_id": "q_employer_current"
    }
  ],
  "cross_session_questions": [
    {
      "question_id": "q_lunch_safe",
      "query": "what should you avoid bringing to my office for lunch?",
      "evidence_sessions": [1, 3],               // requires combining day 1 (workplace) + day 3 (allergy)
      "correct_answer_keywords": ["peanut", "nut", "allergy"],
      "incorrect_leak_keywords": []              // optional negative-test
    }
  ]
}
```

### 3.3 Persona generation

ALB v0.1 ships **20 hand-authored personas** spanning diverse occupations, social contexts, and routine densities. Personas are not LLM-generated for v0.1 because LLM-generated personas have hidden statistical regularities that bias evaluation.

Persona authoring rules (enforced by `lint_personas.py`):
1. Every plantable pattern must appear $\geq 2$ times in the persona.
2. Every gap term must be used $\geq 2$ times without being defined.
3. Every contradiction's old + new values must each appear $\geq 1$ time, separated by $\geq 2$ days.
4. Every cross-session question requires evidence from $\geq 2$ distinct sessions.
5. No persona may contain explicit hints (e.g., "by the way, I run every Tuesday — that's a pattern!").

For v0.2, a templated generator may be added if reproducibility audits show the hand-authored set is small enough to be memorized by tested LLMs.

### 3.4 Agent turns

Agent turns in the persona are **scripted neutral acknowledgements** ("Got it.", "Thanks for letting me know."). They exist to make the conversation feel like a dialog without giving the system under test any signal. The system under test does NOT generate turns; ALB measures memory, not generation.

This is a deliberate methodology choice: it isolates the memory architecture as the only source of variance.

### 3.5 Probe query placement

Probe queries are injected at the **start of the next day after the relevant content appears**. This is the "wake up and ask" pattern — the user has been away (idle period), comes back, and asks. This is precisely the moment where a system that did useful idle work should outperform a system that didn't.

Some probe queries are placed mid-day to test within-session retrieval as a control.

---

## 4. Metrics

Each metric below specifies: **what it measures**, **how it's scored**, and **what the floor / ceiling values mean**.

### 4.1 Pattern Discovery Rate (PDR)

**What.** Of the patterns planted in the persona, how many does the system have in its memory at the relevant probe time?

**How.** For each `pattern` in ground truth, the runner issues a probe query of the form *"What patterns have you noticed about [domain]?"* at the `expected_discovery_after_session` boundary. The scorer:
1. Asks the system for its current schema/abstraction set (via the `list_schemas()` adapter method, see §6).
2. For each ground-truth pattern, checks whether the system reports a schema whose `signature` matches by **deterministic rule**:
   - `REPETITION`: the system reports a schema mentioning the planted event.
   - `COOCCUR`: the system reports a schema mentioning both planted entities.
   - `TEMPORAL_CADENCE`: the system reports a schema mentioning the event AND the cadence.
   - `TRAJECTORY`: the system reports a schema mentioning the planted sequence in order.

**Score.** $\text{PDR} = \frac{\#\text{patterns discovered}}{\#\text{patterns planted}} \in [0, 1]$.

**Floor.** A system with no abstraction mechanism (e.g., raw vector store) scores 0. **This is correct** — they don't do this kind of work.

**Ceiling.** 1.0. A perfect system finds every planted pattern by the expected boundary.

### 4.2 Cross-Session Synthesis (CSS)

**What.** Can the system answer questions whose evidence is split across sessions?

**How.** For each `cross_session_question`, run the question through the system at the placement timestamp. Score by **token-level keyword match** against `correct_answer_keywords` and `incorrect_leak_keywords`:
- $+1$ if at least one correct keyword appears in the response.
- $-1$ if any incorrect leak keyword appears.
- $0$ otherwise.

Final score per question: clip to $\{0, 1\}$ — partial credit is not granted because cross-session reasoning is binary (you either combined the evidence or you didn't).

**Score.** $\text{CSS} = \frac{\#\text{questions scored 1}}{\#\text{cross-session questions}} \in [0, 1]$.

**Floor.** Systems that don't do cross-session reasoning score $\leq$ the rate at which one session contains all evidence by chance.

### 4.3 Curiosity Gap Coverage (CGC)

**What.** Does the system identify gaps and (autonomously) attempt to fill them?

**How.** Two sub-metrics:
- **CGC-id (identification rate)**: at the post-idle probe, ask the system *"Which terms used recently are you unsure about?"* via the `list_open_questions()` adapter method. Compare against ground-truth gaps.
- **CGC-fill (fill rate)**: ask the system *"Define [term]"* for each gap. Score by `min_keyword_match` against `expected_definition_keywords`.

Systems that don't expose `list_open_questions()` get **CGC-id = 0** (they don't identify gaps as a separate operation; this is a real architectural gap, not a bug).

**Score.**
- $\text{CGC-id} = \frac{\#\text{gaps system identified}}{\#\text{gaps planted}} \in [0, 1]$
- $\text{CGC-fill} = \frac{\#\text{gaps system filled correctly}}{\#\text{gaps planted}} \in [0, 1]$
- $\text{CGC-fill}$ is computed against gaps the system filled *autonomously during idle*, not gaps it answers reactively when asked. This is detected via the adapter's `was_autonomous_fill(term)` method.

**Honesty note.** Systems without a curiosity mechanism will score 0 on CGC-fill. This is correct. They aren't doing this work. Reporting CGC-fill = 0 for those systems is informative, not unfair.

### 4.4 Contradiction Resolution Across Idle (CRAI)

**What.** When a user asserts X, then several days later asserts ¬X or X', does the system know X' supersedes X after idle processing?

**How.** For each contradiction, the runner injects two probe queries on the day **after** the new value is planted (so an idle period has occurred):
- **Current probe**: *"What's the current value of [property]?"* — should return new_value.
- **Old probe**: *"What was the previous value of [property]?"* — should return old_value (only if the system supports versioning).

Scoring per contradiction:
- $+1$ on current probe if response contains new_value AND not old_value.
- $+1$ on old probe if response contains old_value (and is not penalized if it lacks old_value — many systems don't version).

**Score.**
- $\text{CRAI-current} = \frac{\#\text{current probes correct}}{\#\text{contradictions}} \in [0, 1]$
- $\text{CRAI-old} = \frac{\#\text{old probes correct}}{\#\text{contradictions}} \in [0, 1]$ (versioning systems only)

### 4.5 Wake-Summary Informativeness (WSI)

**What.** When the user "returns from idle," does the system report what it learned, and is the report accurate?

**How.** At the start of each day after the first, the runner calls `wake_summary()` on the adapter. The summary is a structured report:
```python
WakeSummary(
    schemas_formed: List[SchemaRef],
    contradictions_resolved: List[ContradictionRef],
    gaps_filled: List[GapRef],
    narrative: str,                    # human-readable
)
```

Scoring against ground truth (events that *actually happened* in the previous idle period, computed from event tracing):
- **WSI-precision**: of items the summary reports, how many are real?
- **WSI-recall**: of real items, how many are reported?
- **WSI-f1**: harmonic mean.

Systems that don't have a wake-summary endpoint score **0** on all three. (Floor = no capability, score = 0 is honest.)

### 4.6 Idle Maintenance Cost (IMC)

**What.** What does idle processing cost?

**How.** During each idle period, the runner records:
- **Wall-clock time** spent in idle processing (from adapter's `idle()` start to return).
- **Peak resident memory** (RSS) during idle, via `psutil`.
- **CPU-seconds** during idle, via `psutil.Process.cpu_times()`.

Reported as means + 95% bootstrap CIs across persona × seed.

**Score.** Three numbers per system, reported as a **frontier** alongside quality metrics. There is no "best" — a system that does more during idle should cost more. The question is whether the cost is justified by the lift on quality metrics.

### 4.7 No-Idle Ablation Lift (NIAL)

**What.** Is the autonomous-learning machinery actually doing useful work, or are quality gains attributable to retrieval alone?

**How.** Each system is run twice per persona × seed:
- **idle-on**: full configuration; idle processing enabled.
- **idle-off**: same configuration; idle processing disabled (the adapter's `idle()` is a no-op).

For each metric in §4.1–§4.5, report:
- $\text{lift} = \text{score}_{\text{idle-on}} - \text{score}_{\text{idle-off}}$
- $\text{relative lift} = \text{lift} / \max(\text{score}_{\text{idle-off}}, \epsilon)$

**Floor.** A system with no idle work to disable has $\text{lift} = 0$ across the board. **This is the headline finding for retrieval-family systems**: they do not benefit from idle time because they don't use it.

---

## 5. Statistical methodology

### 5.1 Sampling structure

Per system: 20 personas × 5 seeds = **100 runs** per condition (idle-on, idle-off).
Total: 5 systems × 2 conditions × 100 runs = **1000 runs** for the full benchmark.

For real-LLM extraction (Ollama, DeepSeek), reduced to **10 personas × 3 seeds = 30 runs** per condition due to time/cost. Real-LLM matrix is reported separately.

Seeds control:
- Persona shuffling order across the run pool (irrelevant to scoring but recorded).
- Stochastic LLM sampling (temperature, top_p frozen at adapter-specified defaults).
- Adapter-internal randomness (memory tie-breaking, schema-ID sort order on collisions).

### 5.2 Bootstrap confidence intervals

For every metric reported, compute a **95% bootstrap CI** with 10,000 resamples.

```python
def bootstrap_ci(scores: List[float], n_boot: int = 10000, alpha: float = 0.05):
    boots = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
             for _ in range(n_boot)]
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
```

CIs reported alongside means in every table. Tables without CIs are draft-only.

### 5.3 Paired significance tests

For each (SCM, baseline) pair on each metric, compute a **paired t-test** over the 100 paired runs (same persona × same seed, different system). Report:
- Mean difference.
- 95% CI on the difference.
- $t$, df, $p$.
- **Cohen's $d$** (paired) for effect size.

If the metric is bounded $[0, 1]$ and the distribution is heavy-tailed, additionally report a **Wilcoxon signed-rank test** as a non-parametric companion.

### 5.4 Multi-comparison correction

The full table has $5 \text{ baselines} \times 7 \text{ metrics} = 35$ pairwise comparisons. Apply **Holm-Bonferroni** correction (less conservative than full Bonferroni but still controls family-wise error rate).

A finding is reported as significant only if it survives Holm-Bonferroni at $\alpha = 0.05$.

### 5.5 Minimum reportable effect

Effects with Cohen's $d < 0.2$ are reported as "not meaningful" even if statistically significant. (A 100-run sample can detect tiny effects that don't matter in practice.)

### 5.6 Power analysis

Pre-run power analysis: with $n = 100$ paired observations and $\alpha = 0.05$ (Holm-Bonferroni-corrected), we have 80% power to detect Cohen's $d = 0.4$ (a "small-to-medium" effect). Effects smaller than this require larger samples.

---

## 6. Adapter contract

Every system must implement `BaseMemorySystem`:

```python
class BaseMemorySystem(ABC):
    """ALB adapter contract. All baselines must implement this."""

    @abstractmethod
    def reset(self, persona_id: str, seed: int) -> None:
        """Fresh state. Called once per run."""

    @abstractmethod
    def ingest(self, message: Message, sim_time: datetime) -> None:
        """Add a turn. sim_time is the persona's timestamp, not wall clock."""

    @abstractmethod
    def idle(self, duration_seconds: float, sim_time: datetime,
             allow_compute: bool = True) -> IdleReport:
        """Simulated idle period. Systems do their autonomous-learning work here.
        Returns: what the system did during the idle (for IMC + WSI scoring)."""

    @abstractmethod
    def query(self, text: str, sim_time: datetime) -> QueryResult:
        """Answer a probe query."""

    @abstractmethod
    def list_schemas(self) -> List[Schema]:
        """Return current abstraction/schema set. Empty list if unsupported."""

    @abstractmethod
    def list_open_questions(self) -> List[Gap]:
        """Return identified knowledge gaps. Empty list if unsupported."""

    @abstractmethod
    def wake_summary(self, since_sim_time: datetime) -> Optional[WakeSummary]:
        """Report on idle work since the given time. None if unsupported."""

    @abstractmethod
    def stats(self) -> SystemStats:
        """Memory size, CPU/mem usage since last reset, etc."""

    @abstractmethod
    def supports(self) -> Set[Capability]:
        """What this system claims to do.
        Used for sanity checks, NOT for scoring.
        Capabilities: {SCHEMA_EXTRACTION, GAP_TRACKING, WAKE_SUMMARY,
                       VERSIONING, CROSS_SESSION_POOL, IDLE_PROCESSING}"""
```

A system that returns `[]` from `list_schemas()` honestly scores 0 on PDR. We do not penalize systems for not implementing capabilities they don't claim. We **do** penalize systems that claim a capability and fail to deliver it.

---

## 7. Baselines

ALB v0.1 ships adapters for:

| System | Adapter location | Capability claims |
|---|---|---|
| **SCM (full Phase 7)** | `adapters/scm_adapter.py` | All six |
| **SCM (no Phase 7)** | `adapters/scm_adapter.py` (idle disabled) | Internal ablation |
| **MemoryBank** | `adapters/memorybank_adapter.py` | WAKE_SUMMARY, IDLE_PROCESSING (daily summary) |
| **Generative Agents (reflection)** | `adapters/genagents_adapter.py` | SCHEMA_EXTRACTION (reflection), IDLE_PROCESSING |
| **A-Mem** | `adapters/amem_adapter.py` | IDLE_PROCESSING (note construction) |
| **Mem0 v2** | `adapters/mem0_adapter.py` | (none; floor baseline) |
| **MemGPT** | `adapters/memgpt_adapter.py` | (none; floor baseline) |
| **Vector + LLM rewrite** | `adapters/vector_baseline.py` | (none; sanity baseline) |

Mem0 + MemGPT + vector are included **only as floor baselines**. They are not the headline comparison; they are reported in a separate "no-idle baselines" table to establish that the lift on idle-axis metrics is non-trivial.

The headline comparison is **SCM full vs MemoryBank vs Generative Agents (reflection) vs A-Mem**.

---

## 8. Reproducibility

### 8.1 Frozen versions

Every result is annotated with a **manifest** recording:
- Git SHA of each adapter's underlying system.
- LLM backend + version (e.g., `deepseek-chat-v2.5`, `llama3:8b`).
- Embedding model + version.
- Python version, platform, key library versions.
- Total run wall-clock time, total LLM cost.

Manifest is committed alongside results JSON. A re-run on the same manifest must produce results within 5% on every metric (modulo LLM stochasticity).

### 8.2 Cost

| Configuration | Approximate cost |
|---|---|
| Stub-LLM (offline, all metrics) | $0, ~30 min on M1 Air |
| Ollama (real-LLM extraction, 30-run subset) | $0, ~12-15 hours on M1 Air |
| DeepSeek (real-LLM extraction, 30-run subset) | ~$5–8, ~2-3 hours wall-clock |

Stub-LLM is the headline matrix. Real-LLM is reported as a robustness check.

### 8.3 Open data

- All 20 personas: released as `personas/persona_*.json` in the repo.
- All raw run outputs: released as `results/raw/*.jsonl`.
- All scored outputs: released as `results/scored/*.csv`.
- Per-persona failure ledger: `results/failures.md`.
- Plotting scripts: `scripts/plot_alb.py`.

---

## 9. Threats to validity

This benchmark has limitations. We list them so reviewers don't have to.

1. **Hand-authored personas are a small sample.** 20 personas may not capture the distribution of real conversational patterns. v0.2 should expand to 50–100 personas with templated diversity.

2. **Synthetic ground truth is binary.** Real autonomous learning is graded; ALB scores are mostly $\in \{0, 1\}$. This privileges systems that produce schemas in the exact form ALB expects.

3. **The scoring rules favor systems that expose explicit `list_schemas()` / `list_open_questions()`.** A system whose schemas are implicit (e.g., embedded in vector clusters) may score low even if behaviorally equivalent. We mitigate this by also scoring via probe queries — but the bias is real.

4. **No real users.** All workloads are synthetic. The benchmark cannot tell you whether real users would benefit from the autonomous-learning capability. A user study is future work.

5. **No multi-week runs.** ALB v0.1 is bounded at 7 days per persona. Long-horizon stability (weeks-to-months) is not tested.

6. **English only.** Personas are in English. Re-localization is straightforward via the persona JSON but is not tested.

7. **The scoring of WSI narrative is structural-only.** A correctly-content but poorly-phrased narrative scores the same as a well-phrased one. Human-rated narrative quality is future work.

8. **System-claimed capabilities are self-reported.** We cannot verify a system actually does what its `supports()` method claims. We mitigate by cross-checking against the score on the corresponding metric: if a system claims SCHEMA_EXTRACTION but scores 0 on PDR, that's reported.

---

## 10. Pre-registered hypotheses

These are the predictions we commit to before any system is run. After the first run, deviations from these hypotheses are reported with the same emphasis as confirmations.

| H# | Hypothesis | Rationale |
|---|---|---|
| H1 | SCM (full) > MemoryBank on PDR with $d \geq 0.5$. | SCM has typed multi-pass schema extractor (REPETITION, COOCCUR, TRAJECTORY, CADENCE); MemoryBank has only daily summary. |
| H2 | SCM (full) > A-Mem on CSS with $d \geq 0.3$. | SCM has cross-session memory pool; A-Mem builds links within session primarily. |
| H3 | SCM (full) > all baselines on CGC-fill (no $d$ commitment). | SCM is the only system in the matrix with an autonomous LLM-source curiosity engine. |
| H4 | SCM (full) ~ Generative Agents (reflection) on PDR (CI overlap). | Reflection is the closest analog to schema extraction. We do not predict dominance. |
| H5 | SCM (no Phase 7) ~ Mem0 on idle-axis metrics (both near floor). | Without Phase 7, SCM does not do autonomous learning. This is the ablation control. |
| H6 | NIAL > 0.1 for SCM full on PDR, CSS, CGC. | If Phase 7 is doing useful work, idle-on should beat idle-off by a meaningful margin. |
| H7 | NIAL ≈ 0 for Mem0 / MemGPT / vector. | These systems do nothing during idle; idle-on and idle-off should be indistinguishable. |
| H8 | IMC for SCM is between 1.5× and 5× the floor baselines. | Phase 7 daemon + cross-session pool + schema extractor + curiosity must cost more than retrieval-only. |
| H9 | CRAI-current: vector + LLM rewrite > SCM (no Phase 7), but SCM (full) > vector. | Without versioning, vector retrieves both; with Phase 7's superseded-concept filter, SCM should resolve correctly. |
| H10 | SCM (full) does not dominate on every metric. | A real benchmark has tradeoffs. We commit to reporting where SCM loses. |

If $\geq 4$ of these hypotheses fail in the actual run, the paper text is rewritten to reflect findings, not to defend the predictions.

---

## 11. Versioning

| Version | Date | Changes |
|---|---|---|
| v0.1 | 2026-05-01 | Initial spec. Frozen before first run. |

Subsequent versions track changes with rationale. **No silent edits.**

---

## 12. Implementation status

| Component | Status | Owner |
|---|---|---|
| SPEC.md (this file) | ✅ v0.1 frozen | — |
| `personas/schema.json` | ⏳ next | — |
| 20 personas | ⏳ pending | — |
| `adapters/base.py` | ⏳ next | — |
| Metric implementations (§4.1–§4.7) | ⏳ pending | — |
| Adapters: SCM, MemoryBank, GenAgents, A-Mem, Mem0 | ⏳ pending | — |
| `runner.py` | ⏳ pending | — |
| `stats.py` (bootstrap, paired t, Cohen's d) | ⏳ pending | — |

Nothing in this benchmark has been run yet. **No system has any results on ALB at the time this spec is frozen.**
