# arXiv Submission Metadata — SCM Paper

> ⚠️ **DO NOT SUBMIT YET.** As of 2026-05-04, saish has explicitly held the paper push until the product is publicly ready.
>
> The bundle is **staged** here for fast turnaround once the product-ready checklist in [`docs/ROADMAP.md`](../../docs/ROADMAP.md) is fully green (hosted demo + PyPI + npm + 60-90s video + tutorial + 2-3 lighthouse users + repo public).
>
> When that checklist is complete, this file becomes the submission cheat-sheet. Until then, it's just a parked artifact.

---

This file is the submission cheat-sheet — copy-paste fields into the arXiv submission form.

---

## Title

**SCM: Autonomous Lifelong Learning for Language Agents via Sleep-Stage Memory Consolidation**

## Authors

```
SCM Research Team
```

(Update this line once we decide whether to use real names. arXiv requires at least one author with a verified email.)

Contact: `blobopera@proton.me`

## Abstract

Use the paper's `\begin{abstract} ... \end{abstract}` body verbatim. The current abstract reads:

> We present SCM (Sleep-Consolidated Memory), the first open-source memory architecture for language agents that learns autonomously during user idle time. Existing memory systems are stateless when not in use: they store facts when told and retrieve them when asked, but never consolidate, abstract, or grow on their own. SCM closes this gap with a complete biological memory lifecycle: bounded working memory, selective encoding with attention-gated intensity classification, event-structured semantic binding with Hebbian weight updates, spreading-activation retrieval with context-gated hypothesis ranking, dual-mode sleep consolidation, and contradiction-safe version lineage --- and on top of these, a Phase 7 autonomous-learning layer that activates while the user is away. Phase 7 contributes six new subsystems: an idle-aware background daemon that fires sleep cycles when sessions go quiet (M1); a cross-session memory pool that lets consolidation span days rather than single conversations (M2); a schema extractor that emits typed pattern concepts during REM (M3); a wake-up summary endpoint that surfaces what the agent learned overnight (M4); a curiosity engine with pluggable knowledge sources, including a static dictionary, a local-docs scanner, and an opt-in LLM source (M5); and a power-state-aware lifecycle policy with crash-safe state persistence (M6). All locale-specific knowledge --- regex patterns, stoplists, narrative templates, paraphrase rules --- lives in an externalized JSON file rather than in code. We validate the system through a brutal testing methodology that augments unit tests with persona-driven multi-day simulations, adversarial inputs, failure-mode injection, and scale runs. The harness uncovered four real bugs that 170+ unit tests missed (schema-ID instability, spreading-activation seed purging, dead cross-session pool in single-user case, and superseded-concept retrieval leakage); each was root-caused, fixed, and verified by an additional brutal scenario. Real-LLM validation on Ollama (free, local) and DeepSeek (cloud, paid) confirms the system functions end-to-end with non-stub extractors. We release the complete system, the brutal harness, and all benchmark artifacts as reproducible open-source infrastructure.

## arXiv Categories

**Primary:** `cs.AI` — Artificial Intelligence
(SCM is an AI agent memory architecture; this is the natural home.)

**Secondary (cross-list):**
- `cs.CL` — Computation and Language (LLM agents are the deployment target; relevant to the NLP community)
- `cs.LG` — Machine Learning (lifelong learning, consolidation, forgetting are ML topics)
- `cs.NE` — Neural and Evolutionary Computing (biologically-inspired architecture)

Suggested order on the submission form: `cs.AI; cs.CL; cs.LG; cs.NE`

## Comments (free-form text shown above the abstract on arXiv)

```
35 pages, 5 figures, 8 tables. Open-source: https://github.com/Saish15/sleepai
```

If we want to acknowledge it's a tech report flavor:
```
35 pages, 5 figures, 8 tables. Technical report. Open-source release of the
SCM runtime, brutal testing harness, and benchmark artifacts at
https://github.com/Saish15/sleepai
```

## License

Code: MIT (see `LICENSE` in repo).
Paper text: arXiv default (perpetual non-exclusive) — fine for a tech report.

## Funding / Disclosure

None to declare. Self-funded research. No conflicts of interest.

## Suggested Reviewers

Not applicable for arXiv (it's a preprint server, not peer review).

## Submission process — step by step

1. Create / sign in at https://arxiv.org/user/
2. Use endorsement code (need an endorser if first submission to cs.AI; ask anyone with prior cs.AI submissions)
3. Click "Start a new submission"
4. License: choose **arXiv perpetual, non-exclusive license** (default; allows others to read)
5. Upload the source bundle: `scm.tex` (single file). arXiv will compile.
6. Verify the auto-built PDF matches `scm.pdf` in this folder
7. Add metadata (title, authors, abstract, categories, comments) from the fields above
8. Submit. Moderation typically takes 1-2 business days.

## What we're NOT including

- No accompanying datasets file (the brutal personas + ALB benchmark are in the open-source repo, linked from the paper)
- No supplementary materials PDF (everything is in the main paper)
- No video / demo (would help adoption, but separate from arXiv submission)

## Versioning plan

- **v1**: this submission (SCM v0.7.1 architecture + ALB v0.1 results)
- **v2** (later, if needed): updated benchmark numbers including v0.7.2 latency improvements + multi-agent harness + head-to-head against MemoryBank/A-Mem on ALB

## Files in this directory

| File | Purpose |
|---|---|
| `scm.tex` | Single-file LaTeX source for arXiv to compile |
| `scm.pdf` | Local-built reference PDF (verify against arXiv's build) |
| `SUBMISSION.md` | This file (metadata cheat-sheet) |

## Author note (for the cover letter, if asked)

> SCM is a memory architecture for language agents that adds a sleep phase to the conventional "wake-only" memory pattern. Six interlocking Phase-7 subsystems (M1–M6) let the agent consolidate, abstract, and fill knowledge gaps during user idle time. We validate via a "brutal" testing methodology that uncovered four real architectural bugs unit tests missed. The full system, the brutal harness, and all benchmarks are released as open-source infrastructure.
