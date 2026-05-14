# SCM Roadmap — 2026-05-04 (revised)

Honest, executable plan. Update on each release.

**Strategic call (2026-05-04):** the paper is **HELD until the product is publicly ready.** Reasoning: papers without products fade; products with papers compound. Pushing the paper now without a hosted demo / PyPI install / lighthouse users would convert "interested readers" into "bouncing readers." Better to delay the paper, ship the product, then publish with traction signal.

The arXiv submission bundle stays staged at `research/arxiv_submission/` — **DO NOT upload until the product-ready checklist below is complete.**

Categorization:
- **Distribution** = stuff between SCM and the market (not code)
- **Product** = features that ship in the runtime
- **Research** = paper + benchmark work
- **Operations** = CI, packaging, infra

---

## Product-ready checklist (gating arXiv push)

The paper publishes only when a curious reader who finds it can:

- [ ] Click a hosted demo URL and try SCM in 2 clicks (no install required)
- [ ] Watch a 60-90s video showing the wake-summary moment in action
- [x] Run `pip install scm-memory` (PyPI publish complete) and have it work
- [ ] Publish the next PyPI patch release containing the public `SCMEngine.wake_summary()` SDK surface
- [ ] Run `npm install scm-memory` (npm publish complete) and have it work
- [ ] Open the README and see a 10-line "first chatbot" example that just works
- [ ] Find a "build a persistent-memory chatbot in 10 min" tutorial
- [ ] See evidence of real usage — at least 2-3 lighthouse users public, or visible GitHub activity, or HN/Twitter mentions

When all 7 boxes are checked, the paper push happens within 24 hours.

---

## This week (7 days) — PRODUCT-READINESS, NOT PAPER

**Goal: get from "good code" to "anyone can try it in 5 min."**

| Item | Category | Effort | Why | Gates arXiv? |
|---|---|---|---|---|
| Make GitHub repo public | Distribution | 5 min | Required for PyPI / npm / HN / hosted demo | yes |
| Replace `README.md` with the pitch from `docs/PITCH.md` | Distribution | 30 min | First impression for repo visitors | yes |
| Verify PyPI install in a fresh venv before public launch | Operations | 30 min | Avoid broken first installs after any final release bump | yes |
| Publish JS SDK to npm (`scm-memory`) | Operations | 1 hour | Discoverability via JS ecosystem | yes |
| Add GitHub Actions CI (lint + test on PR) | Operations | half day | Future contributors blocked otherwise; signals quality | no but good |
| Record 60-90s demo video (wake-summary moment) | Distribution | 4-6 hours | Single most-shared asset | yes |
| Stand up hosted demo (Fly.io / Railway / Vercel) | Distribution | 2-3 days | Lets people try without install | yes |
| Walk-through tutorial: "10-min persistent-memory chatbot" | Distribution | 1 day | Concrete first-success path | yes |
| Multi-agent brutal harness — overnight re-run on 16GB box | Operations | 0 active effort | Verify multi-agent 21/21 before claiming it works | no but useful |

**Done state:** the product-ready checklist above is fully checked.

**NOT this week:**
- arXiv push (held)
- HN launch (held — fires after demo + lighthouse users)
- Conference / workshop submission (held)

---

## Next 30 days — LIGHTHOUSE USERS + ADOPTION SIGNAL

**Goal: real users + first community signal.**

| Item | Category | Effort | Why |
|---|---|---|---|
| Recruit 5 lighthouse users (free, hand-held, 2-week trial) | Distribution | 1 week elapsed | Real-user signal is irreplaceable |
| Per-user persistent SQLite (replace sandbox-only multi-tenant) | Product | 1 week | Sandbox loses data on restart; lighthouse users won't tolerate that |
| Aggressive hybrid encoder (cut LLM extraction frequency to <15%) | Product | 2-3 days | Cost story for production deployments |
| Cookbook: 5 production patterns (chatbot / agent / RAG-with-memory / multi-tenant SaaS / on-device) | Distribution | 1 week | Convert curious-evaluators to integrators |
| Discord / GitHub Discussions community channel | Distribution | half day setup; ongoing | Place for users to ask + complain |
| Wire MemoryBank adapter to ALB benchmark | Research | 2-3 days | Real head-to-head competitor — DATA for the paper |
| Wire Generative Agents (reflection) adapter to ALB | Research | 2-3 days | Closest architectural prior — DATA for the paper |
| Cost ablation in benchmarks: SCM vs cloud-LLM extraction baseline on per-1k-msgs LLM cost | Research | 2 days | Differentiator we can quantify — DATA for the paper |
| HackerNews launch (Show HN) — once hosted demo + 1+ lighthouse user | Distribution | 1 hour | Triggers when adoption signal exists |

**Done state:** at least 2-3 lighthouse users running SCM in something real. HN launch fired with hosted-demo link. Real numbers from MemoryBank / Generative Agents adapters in `research/benchmarks/alb/results/`.

---

## Next 60-90 days — PAPER WITH TRACTION

**Goal: arXiv push with the product-readiness story already in motion.**

| Item | Category | Effort | Why |
|---|---|---|---|
| Update paper §9 with MemoryBank / GenAgents head-to-head numbers | Research | 1 week | Reviewers will demand this |
| Update paper §11 Limitations with v0.7.2 / v0.7.3 numbers (latency, RAM) | Research | half day | Honest about what's improved since v0.6 |
| Add "Adoption" or "Production Use" section to paper if lighthouse users visible | Research | half day | Strongest signal a memory-layer paper can have |
| **arXiv push** | Distribution | 30 min | Triggers AFTER all above + product-ready checklist green |
| Twitter / HN / Reddit launch coordinated with arXiv | Distribution | 1 day | Maximum visibility within 48h of upload |
| Conference talk submission (NeurIPS workshops, MemAI, ReAlign) | Distribution | 1 week elapsed | Researcher community visibility |

**Done state:** paper on arXiv with traction signal in the README/abstract footnote ("X production deployments confirmed; live demo at scm.run"). 1k+ GitHub stars within 30 days of paper push.

---

## Next 1 year (mid-2027)

**Goal: Sustainable position in agent-memory category.**

| Item | Category | Effort | Why |
|---|---|---|---|
| SCM Cloud (managed hosted) — paid tier | Distribution / Product | 1-2 months | Revenue capture for users who don't self-host |
| Conference paper (ACL Findings or NeurIPS Datasets) | Research | 2-3 months iterating | Academic legitimacy |
| Multi-week real-user study with 20+ users | Research | 1-2 months elapsed | Real lifelong-learning data |
| GDPR / SOC2 compliance audit | Operations | 1-2 months | Required for any enterprise sale |
| Vertical templates: legal-AI memory, healthcare-AI memory, on-device assistant memory | Distribution | 1 month each | Specific market wedge |
| Ecosystem integrations: official LangChain partner, Llamaindex partner, AutoGen integration | Distribution | 1 week each | Distribution via existing platforms |

**Done state:** SCM is the default suggestion when someone asks "what should I use for agent memory if I care about lifelong learning?"

---

## Beyond 1 year (speculative)

| Item | When | Why |
|---|---|---|
| Multi-modal memory (image, audio embeddings) | 2027 H2 | Tracks the broader agent ecosystem |
| Federated SCM (peer agents share schemas without sharing raw memory) | 2028 | Privacy-preserving cross-agent learning |
| On-device foundation model integration (Apple Intelligence, Llama-on-iPhone) | 2028 | Scale to consumer devices |
| Mobile SDK (iOS / Android native) | 2027 H2 | If on-device wedge takes off |

---

## Next 30 days

**Goal: First lighthouse users + community foundation.**

| Item | Category | Effort | Why |
|---|---|---|---|
| Hosted demo at a free domain (Fly.io / Railway / Vercel) | Distribution | 2-3 days | Lets people try in browser without install |
| Walk-through tutorial: "Add SCM to your Claude Desktop in 5 min" | Distribution | 1 day | Concrete first-success path |
| Walk-through tutorial: "Persistent memory chatbot in 30 lines of Python" | Distribution | 1 day | Concrete dev appeal |
| Recruit 5 lighthouse users (free, hand-held, 2-week trial) | Distribution | 1 week elapsed | Real-user signal is irreplaceable |
| Add GitHub Actions CI (lint + test on PR) | Operations | half day | Future contributors blocked otherwise |
| Cost ablation in benchmarks: SCM vs cloud-LLM extraction baseline on per-1k-msgs LLM cost | Research | 2 days | Differentiator we can quantify |
| Wire MemoryBank adapter to ALB benchmark | Research | 2-3 days | Real head-to-head competitor |
| Wire Generative Agents (reflection) adapter to ALB | Research | 2-3 days | Closest architectural prior |
| Multi-agent brutal harness — overnight re-run on a 16GB box | Operations | 0 active effort | Verify we hit 21/21 multi-agent |

**Done state:** real users with real data informing real iteration. Public benchmark vs the 2-3 closest competitors.

---

## Next 90 days (Q3 2026)

**Goal: Real adoption signal — 1k+ stars, real production usage.**

| Item | Category | Effort | Why |
|---|---|---|---|
| Per-user persistent SQLite (replace sandbox-only multi-tenant) | Product | 1 week | Sandbox loses data on restart; real users need persistence |
| Aggressive hybrid encoder (cut LLM extraction frequency to <15%) | Product | 2-3 days | Cost story for production deployments |
| Multi-namespace single-call search | Product | 1 day | One fewer HTTP roundtrip on multi-agent |
| Real-LLM ALB run with DeepSeek + 5 personas + 5 seeds | Research | 2 days compute, $5-10 cost | Statistical significance on the autonomous-learning axis |
| ALB v0.2 with 20 personas | Research | 1-2 weeks (hand-author) | Power for Holm-Bonferroni significance tests |
| Conference talk submission (NeurIPS workshops, MemAI, ReAlign) | Distribution | 1 week elapsed | Researcher community visibility |
| Cookbook: 5 production patterns (chatbot / agent / RAG-with-memory / multi-tenant SaaS / on-device) | Distribution | 1 week | Convert curious-evaluators to integrators |
| Discord / GitHub Discussions community channel | Distribution | half day setup; ongoing | Place for users to ask + complain |

**Done state:** SCM is a recognized name in agent-memory conversations; 1-3 production deployments confirmed; benchmark numbers ready for arXiv v2.

---

## Next 1 year (mid-2027)

**Goal: Sustainable position in agent-memory category.**

| Item | Category | Effort | Why |
|---|---|---|---|
| SCM Cloud (managed hosted) — paid tier | Distribution / Product | 1-2 months | Revenue capture for users who don't self-host |
| Conference paper (ACL Findings or NeurIPS Datasets) | Research | 2-3 months iterating | Academic legitimacy |
| Multi-week real-user study with 20+ users | Research | 1-2 months elapsed | Real lifelong-learning data |
| GDPR / SOC2 compliance audit | Operations | 1-2 months | Required for any enterprise sale |
| Vertical templates: legal-AI memory, healthcare-AI memory, on-device assistant memory | Distribution | 1 month each | Specific market wedge |
| Ecosystem integrations: official LangChain partner, Llamaindex partner, AutoGen integration | Distribution | 1 week each | Distribution via existing platforms |

**Done state:** SCM is the default suggestion when someone asks "what should I use for agent memory if I care about lifelong learning?"

---

## Beyond 1 year (speculative)

| Item | When | Why |
|---|---|---|
| Multi-modal memory (image, audio embeddings) | 2027 H2 | Tracks the broader agent ecosystem |
| Federated SCM (peer agents share schemas without sharing raw memory) | 2028 | Privacy-preserving cross-agent learning |
| On-device foundation model integration (Apple Intelligence, Llama-on-iPhone) | 2028 | Scale to consumer devices |
| Mobile SDK (iOS / Android native) | 2027 H2 | If on-device wedge takes off |

---

## Decisions deferred (not committed)

- **Funding:** open-source-only vs YC vs angel rounds. Decision deferred until lighthouse-user traction signal.
- **Open-core vs fully open-source:** would the cloud be paid + the OSS free, or pure OSS with consulting/support model? Deferred.
- **Paper venue:** arXiv-only forever vs aim for ACL/NeurIPS revisions. Deferred until v1 is on arXiv.
- **Hiring:** solo-developer vs collaborator vs team. Deferred until distribution traction informs need.

---

## What we're explicitly NOT doing

- **Not** chasing benchmark perfection on every metric. SCM has a workload-sensitivity story; we own the autonomous-learning axis, not the clean-fact-retrieval axis.
- **Not** marketing as a "killer" of any specific incumbent. The category-different framing ("memory that works like yours") is more durable.
- **Not** building a chat UI as the primary product. SCM is a memory layer behind agents; UI users would dilute the developer-focused positioning.
- **Not** locking in on a specific LLM provider. The agnostic stance is part of the value prop.

---

## How we'll know we're succeeding

| Signal | Threshold | Timeframe |
|---|---|---|
| GitHub stars | 1k | 3 months from public launch |
| arXiv reads (per arXiv stats) | 500 | 1 month from upload |
| Twitter mentions in agent-memory threads | "SCM" referenced unprompted by 3+ accounts | 3 months |
| External integration | At least 1 third-party tutorial in the agent-memory ecosystem mentioning SCM | 6 months |
| Production deployment | 5+ confirmed (via discord / issues / direct contact) | 6 months |
| Conference presence | At least 1 paper accepted to a workshop | 9-12 months |

If we miss most of these by 50% or more, the strategy is wrong; pause and reconsider. If we hit them, the strategy is working; double down.
