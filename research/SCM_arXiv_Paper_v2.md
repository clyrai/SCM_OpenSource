# SCM: Sleep-Consolidated Memory for Language Agents  
## A Reproducible Evaluation of One-Shot Encoding, Sleep-Stage Consolidation, and Selective Forgetting

**Authors:** SCM Research Team  
**Date:** May 1, 2026  
**Intended venue:** arXiv preprint (paper-first release)

---

## Abstract

Large language models remain limited by memory designs that either depend on short-lived context windows or accumulate unbounded retrieval stores without biologically motivated consolidation and forgetting. We present SCM (Sleep-Consolidated Memory), a memory architecture that combines bounded working memory, event-aware encoding, micro-sleep and deep-sleep consolidation, selective forgetting dynamics, and contradiction-safe version lineage. We evaluate SCM using artifact-backed benchmark suites already integrated in the repository. In the Phase 6 human-memory benchmark, SCM achieves one-shot recall accuracy of 1.0, micro-sleep disambiguation gain of 1.0, deep-sleep disambiguation gain of 1.0, deep-sleep noise retention of 0.0, and contradiction-versioning accuracy of 1.0 (`research/metrics/phase6_human_memory_latest.json`). In a 10-seed baseline comparison stress test with 96 memory pairs per seed, lexical retrieval, vector retrieval, and SCM without sleep all remain at 0.0 disambiguation recall, while SCM with deep sleep reaches mean disambiguation recall 0.9052 and deep noise retention 0.0 (`research/metrics/baseline_comparison_latest.json`). In long-horizon evaluation, sleep-enabled SCM attains final family-aware disambiguation recall 0.9812 with final noise retention 0.0, compared with awake-only recall 0.0 and noise retention 1.0 (`research/metrics/long_horizon_latest.json`, `research/reproducibility/reproducibility_pack_latest.json`). Reproducibility-pack reruns pass end to end with `overall_pass: true`. These results support SCM as a strong, testable memory-lifecycle architecture and establish a rigorous baseline for future user studies and broader external comparisons.

---

## 1. Introduction

Recent progress in large language models has not removed a persistent systems bottleneck: stable, high-quality memory over long interactions. Most production memory layers can be grouped into three families. The first family extends context windows and buffering, but retention quality degrades as dialogue length grows and as conflicting updates accumulate [1]. The second family uses vector retrieval and semantic stores, but typically treats memory as append-only and does not include explicit offline consolidation or intentional forgetting [2, 4]. The third family introduces hierarchical memory management metaphors inspired by operating systems, improving scalability but still lacking a biologically grounded memory lifecycle [3].

Human memory, by contrast, is not a static store. It uses bounded working capacity, replay and consolidation, targeted forgetting, and conflict resolution across time [5, 6, 7]. Motivated by these principles, SCM models memory as a lifecycle rather than a retrieval cache. The architecture is organized around five operational requirements: rapid episodic encoding, constrained short-term buffering, staged sleep consolidation, selective forgetting under interference, and contradiction-safe version lineage.

The central question of this paper is straightforward: can a sleep-centered memory lifecycle produce measurable gains over awake-only and standard retrieval controls under controlled stress testing? We answer this question using the current SCM artifact stack. The evidence indicates large gains in disambiguation and noise control, while preserving one-shot and contradiction-update behaviors.

This work is explicitly paper-first. We do not frame SCM as a sales product in this manuscript. We frame it as a reproducible research system with clear strengths, clear limitations, and clear next experiments.

---

## 2. Contributions

This paper makes four concrete contributions.

First, it defines SCM as an end-to-end memory-lifecycle architecture with explicit micro-sleep and deep-sleep stages, rather than an always-awake retrieval wrapper.

Second, it introduces an integrated contradiction-versioning pathway in which preference updates preserve lineage across consolidation, instead of naive overwrite semantics.

Third, it provides a stress-evaluated behavioral comparison showing that sleep stages, rather than baseline retrieval mechanics, are responsible for the measured disambiguation gains.

Fourth, it presents a reproducibility pathway that replays baseline comparison, long-horizon evaluation, guardrails, and smoke suites with recorded artifacts.

---

## 3. Method

### 3.1 Architecture Overview

SCM combines a bounded working-memory substrate with a persistent graph-based long-term memory. Incoming dialogue is encoded into typed concepts and event metadata, then written to working memory and long-term memory with value signals and context tags. Retrieval operates through semantic cues and graph propagation. Sleep orchestration periodically executes consolidation and forgetting passes, then syncs the updated memory state back into long-term storage.

The architecture spans five development phases that remain active in the current codebase: selective encoding and grasp estimation (Phase 1), event and association binding (Phase 2), spreading-activation retrieval with hypothesis ranking (Phase 3), micro-sleep and deep-sleep consolidation (Phase 4), and forgetting dynamics with contradiction-safe versioning (Phase 5). Phase 6 contributes evaluation harnesses and guardrails.

### 3.2 Sleep Stages

Micro-sleep performs lightweight replay, local reinforcement, and duplicate-pressure cleanup. It is designed to improve near-term recall under moderate memory pressure with low disruption.

Deep sleep performs broader replay and consolidation, relation synthesis, and forgetting dynamics. In current SCM, deep-sleep outputs now preserve retired concept state in sync payloads, which is important for maintaining contradiction-version lineage through consolidation.

### 3.3 Selective Forgetting and Version Lineage

Forgetting dynamics estimate retention using grasp, salience, rehearsal, association density, recency, and interference. Concepts transition through active, suppressed, and archived states.

Contradictions are represented with version lineage rather than overwrite. A new conflicting memory can supersede a prior version while preserving auditability through parent/root fields and contradiction relations. This mechanism is now regression-tested through deep-sleep transitions.

---

## 4. Experimental Protocol

All numbers reported in this paper come from repository artifacts generated by the benchmark harnesses and regression suites.

The human-memory suite is sourced from `research/metrics/phase6_human_memory_latest.json`.  
The baseline stress comparison is sourced from `research/metrics/baseline_comparison_latest.json`.  
The long-horizon behavior report is sourced from `research/metrics/long_horizon_latest.json` and corroborated in `research/reproducibility/reproducibility_pack_latest.json`.  
Guardrail status is sourced from `research/metrics/phase6_guardrails_latest.json`.  
Backend smoke stability is sourced from `research/metrics/phase6_backend_smoke_latest.json`.

We report means and pass flags exactly as stored by the harnesses, and we avoid retrospective metric reinterpretation.

---

## 5. Results

### 5.1 Human-Memory Suite

In the latest Phase 6 human-memory artifact, all top-level gates pass. One-shot recall is 1.0. Micro-sleep disambiguation gain is 1.0. Deep-sleep disambiguation gain is 1.0. Deep-sleep noise retention is 0.0. Contradiction-versioning accuracy is 1.0. The aggregate status is `overall_pass: true`.

These results indicate that SCM preserves core target behaviors simultaneously: fast encoding, post-sleep strengthening, and contradiction-safe update handling.

### 5.2 Baseline Stress Comparison

In the 10-seed, 96-pair stress comparison, three controls remain at disambiguation recall 0.0: lexical retrieval baseline, vector retrieval baseline, and SCM baseline without sleep. By contrast, SCM with micro-sleep reaches disambiguation recall mean 1.0 and SCM with deep sleep reaches disambiguation recall mean 0.9052. Deep-noise retention remains 0.0, and deep-stage pass rate is 1.0.

This pattern is important for attribution. The gains do not appear in awake-only variants; they appear when sleep stages are enabled. That supports the claim that consolidation dynamics, not merely retrieval plumbing, are responsible for the measured improvements.

### 5.3 Long-Horizon Behavior

In long-horizon evaluation, sleep-enabled SCM reaches final family-aware disambiguation recall 0.9812, final noise retention 0.0, and anchor accuracy 1.0. The awake-only mode remains at final disambiguation recall 0.0 with final noise retention 1.0. The comparative disambiguation lift is 0.9812 and noise reduction is 1.0.

Strict duplicate-pair metrics remain a transparent stress appendix in the project and are not hidden; the headline release gate is family-aware duplicate behavior aligned with shipped retrieval semantics.

### 5.4 Reliability and Regression Safety

The Phase 6 guardrail artifact reports `overall_pass: true`, with `warnings_count: 0` and deprecation hits 0 for tracked patterns (`datetime.utcnow(` and `.dict(`).

The reproducibility pack reports `overall_pass: true` and records passing summaries for baseline comparison, long horizon, guardrails, and smoke pytests.

The backend smoke brutal artifact reports 20/20 HTTP success and 20/20 overall pass, with average runtime latency 45.2 ms, p50 22.7 ms, and p90 26.3 ms.

---

## 6. Discussion

The empirical story is consistent across suites. SCM appears strongest where memory quality depends on lifecycle transformations, especially disambiguation under pressure and selective noise removal over time. The data also indicates that contradiction handling is not merely symbolic at ingestion time; it can remain operational through consolidation when lineage-preserving sync paths are correct.

A second observation is methodological. Because awake-only controls and no-sleep SCM controls remain near zero on disambiguation in the stress setting, the large gains in sleep-enabled variants are difficult to explain as incidental retrieval effects.

A third observation concerns scope. The present evidence is internal but rigorous: deterministic harnesses, multi-seed stress, regression packs, and reproducibility artifacts. This is strong engineering-science evidence, but it is not yet external user evidence.

---

## 7. Limitations

This paper has several limitations that should be explicit.

First, evaluations are primarily synthetic and harness-driven. They capture targeted memory phenomena but do not yet represent broad naturalistic user populations.

Second, comparisons are currently strongest against internal control baselines and standard retrieval controls. Wider external baselines should be added in future revisions.

Third, strict duplicate-pair behavior under long-horizon pressure remains less stable than family-aware behavior, which is why the project now reports both layers separately.

Fourth, this manuscript does not claim consciousness or full human cognition. It evaluates human-like memory dynamics in a bounded systems sense.

---

## 8. Ethics and Claim Boundaries

SCM is presented as a memory architecture research system, not a claim of sentience. We avoid anthropomorphic claims beyond measured memory behaviors. The relevant ethical standard here is truthful capability communication: one-shot recall, sleep-stage consolidation, selective forgetting, and contradiction-safe updating are operational claims supported by artifacts; broader claims about consciousness, autonomy, or societal replacement are not made.

---

## 9. Reproducibility Statement

The project includes explicit reproducibility infrastructure.

Core scripts include `tests/baseline_comparison.py`, `tests/long_horizon_benchmark.py`, `tests/phase6_guardrails.py`, and `tests/reproducibility_pack.py`. The latest aggregate reproducibility artifact is `research/reproducibility/reproducibility_pack_latest.json`, with corresponding narrative in `docs/SCM_REPRODUCIBILITY_PACK.md`.

As of May 1, 2026, the reproducibility pack reports `overall_pass: true`.

---

## 10. Conclusion

SCM demonstrates that a sleep-consolidated memory lifecycle can produce large, reproducible gains over awake-only and standard retrieval controls on targeted memory tasks. Across current artifacts, SCM maintains one-shot behavior, contradiction-safe updates, and long-horizon selective forgetting while sustaining strong disambiguation performance under stress. The work supports a clear research direction: memory quality in language agents improves when memory is treated as a lifecycle process, not only as retrieval storage.

The immediate next step is external validation through a structured user study, followed by broader cross-system benchmarking. With those additions, SCM can be evaluated not only as an internal engineering success but as a generalizable memory architecture contribution.

---

## References

[1] Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172, 2023.

[2] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., and Rocktaschel, T. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.

[3] Packer, C., Fang, V., Patil, S. G., Lin, K., Wooders, J., and Gonzalez, J. E. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560, 2023.

[4] Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413, 2025.

[5] Miller, G. A. The Magical Number Seven, Plus or Minus Two: Some Limits on Our Capacity for Processing Information. Psychological Review, 63(2):81-97, 1956.

[6] Rasch, B., and Born, J. About Sleep's Role in Memory. Physiological Reviews, 93(2):681-766, 2013.

[7] Tononi, G., and Cirelli, C. Sleep and Synaptic Homeostasis: A Hypothesis. Brain Research Bulletin, 62(2):143-150, 2003.

[8] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., and Grabska-Barwinska, A. Overcoming Catastrophic Forgetting in Neural Networks. PNAS, 114(13):3521-3526, 2017.

[9] Sorrenti, D. G., Serafini, A., Calderara, S., and Cucchiara, R. Wake-Sleep Consolidated Learning. arXiv:2401.08623, 2024.

[10] Xie, Y. Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models. arXiv:2603.14517, 2026.
