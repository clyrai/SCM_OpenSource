# Research Document Audit (April 29, 2026)

This audit pass reviewed the primary project and research documents for consistency before next-paper drafting.

## Audited Document Set

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/PHASE1_COMPLETE_DOCUMENTATION.md`
- `tests/README.md`
- `research/10_SCM_Human_Memory_Blueprint.md`
- `research/11_SCM_HME_Phases_Document.md`
- `research/SleepAI_arXiv_Paper.md`
- `research/latex/sleepai.tex`
- `research/latex/sleepai.pdf` (rebuilt from updated `.tex`)
- `research/papers/*.md` (reference analyses spot-checked for naming consistency)

## Fixes Applied In This Audit

1. Updated test documentation:
- `tests/README.md` now reflects Phase 2-6 suites and benchmark/guardrail commands.

2. Clarified historical vs live status docs:
- Added historical-context notes and live-status pointers in:
  - `docs/PHASE1_COMPLETE_DOCUMENTATION.md`
  - `research/10_SCM_Human_Memory_Blueprint.md`
  - `research/11_SCM_HME_Phases_Document.md`

3. Aligned code-availability statements with repository reality:
- Updated `research/latex/sleepai.tex` Code Availability section.
- Updated `research/SleepAI_arXiv_Paper.md` code/documentation path statement.

4. Kept README roadmap consistent with current execution:
- `README.md` now marks Phase 6 as "in progress".

5. Rebuilt paper PDF:
- Recompiled `research/latex/sleepai.pdf` after `.tex` update.

6. SCM naming normalization follow-up:
- Updated `research/SleepAI_arXiv_Paper.md` to SCM-first terminology.
- Added naming notes to historical strategy/reference docs that still use "SleepAI":
  - `research/03_SleepAI_Architecture.md`
  - `research/04_Comparative_Gap_Analysis.md`
  - `research/priority_papers.md`
  - `research/papers/00_MASTER_INDEX.md`
- Added citation-ready baseline summary:
  - `docs/SCM_PAPER_BASELINE_2026-04-29.md`

## Paper-Writing Readiness Notes

- Live implementation and what remains: use `docs/PROJECT_STATUS.md`.
- Hardening/benchmark evidence for paper tables:
  - `research/metrics/phase6_guardrails_latest.json`
  - `research/metrics/phase6_human_memory_latest.json`
  - `research/metrics/phase6_demo_latest.json`
- If writing a single canonical manuscript, explicitly decide naming convention:
  - `SCM` (used in LaTeX paper)
  - `SleepAI` (used in README and arXiv-style markdown draft)

## Recommended Canonical Sources For Next Paper

1. Methods/architecture baseline:
- `research/latex/sleepai.tex`

2. Current implementation status and claims boundary:
- `docs/PROJECT_STATUS.md`

3. Reproducible experimental artifacts:
- `research/metrics/phase4_micro_deep_latest.json`
- `research/metrics/phase6_guardrails_latest.json`
- `research/metrics/phase6_human_memory_latest.json`
