"""
Seven tiers of brutal multi-agent scenarios.

All scoring is on what SCM returned (retrieved memories), not on what the
LLM hallucinated in its response. Same methodological discipline as the
single-agent brutal harness.
"""
from __future__ import annotations

import time
import uuid
from typing import List

from .agents import SpecialistAgent, AgentTurn
from .runner import Scenario


def _retrieved_blob(turn: AgentTurn, scope: str = "both") -> str:
    """Concatenate retrieved memories for scoring.

    scope: 'private', 'shared', or 'both'.
    """
    parts: List[str] = []
    if scope in ("private", "both"):
        parts.extend(turn.private_retrieved)
        if turn.private_context:
            parts.append(turn.private_context)
    if scope in ("shared", "both"):
        parts.extend(turn.shared_retrieved)
        if turn.shared_context:
            parts.append(turn.shared_context)
    return " | ".join(parts).lower()


def _passes(blob: str, must_contain: List[str], must_not_contain: List[str] = None) -> bool:
    must_not_contain = must_not_contain or []
    pos = any(kw.lower() in blob for kw in must_contain)
    neg = any(kw.lower() in blob for kw in must_not_contain)
    return pos and not neg


def _build_team(server_url: str, suffix: str = "") -> tuple:
    """Construct a fresh trio with a unique shared user_id."""
    shared = f"user_team_{suffix}_{uuid.uuid4().hex[:8]}"
    return (
        SpecialistAgent("researcher", scm_url=server_url, shared_user_id=shared),
        SpecialistAgent("coder",      scm_url=server_url, shared_user_id=shared),
        SpecialistAgent("reviewer",   scm_url=server_url, shared_user_id=shared),
        shared,
    )


# ─── Tier 1 — per-agent specialty memory ──────────────────────────────────


def tier1_per_agent_specialty(server_url: str) -> List[Scenario]:
    """Each agent develops its own niche knowledge in private memory."""
    researcher, coder, reviewer, _shared = _build_team(server_url, "t1")
    out = []

    # Each agent records a domain-specific decision.
    researcher.remember_decision("Decided to evaluate FastAPI vs Litestar for the rewrite.")
    coder.remember_decision("Picked Pydantic v2 for the data models.")
    reviewer.remember_decision("Enforced ruff + mypy in CI as the lint baseline.")

    # Force consolidation
    researcher.force_consolidate("deep")
    coder.force_consolidate("deep")
    reviewer.force_consolidate("deep")

    # Probe each agent for ITS OWN specialty
    t = researcher.chat("What framework comparison did we discuss?")
    out.append(Scenario(
        name="researcher recalls FastAPI/Litestar evaluation",
        tier=1,
        passed=_passes(_retrieved_blob(t, "private"), ["fastapi", "litestar"]),
        detail=f"private retrieved: {_retrieved_blob(t, 'private')[:200]!r}",
    ))

    t = coder.chat("Which data-model library did we pick?")
    out.append(Scenario(
        name="coder recalls Pydantic v2 choice",
        tier=1,
        passed=_passes(_retrieved_blob(t, "private"), ["pydantic"]),
        detail=f"private retrieved: {_retrieved_blob(t, 'private')[:200]!r}",
    ))

    t = reviewer.chat("What lint tools did we standardize on?")
    out.append(Scenario(
        name="reviewer recalls ruff + mypy CI standard",
        tier=1,
        passed=_passes(_retrieved_blob(t, "private"), ["ruff", "mypy"]),
        detail=f"private retrieved: {_retrieved_blob(t, 'private')[:200]!r}",
    ))
    return out


# ─── Tier 2 — shared user-memory handoff ──────────────────────────────────


def tier2_shared_user_handoff(server_url: str) -> List[Scenario]:
    """User tells one agent something. Another agent should see it."""
    researcher, coder, reviewer, _shared = _build_team(server_url, "t2")
    out = []

    # User talks to Researcher
    researcher.chat("My name is Saish and I prefer Python over JavaScript for everything.")
    researcher.chat("I work on a payment system that processes about 5M transactions per day.")

    researcher.force_consolidate("deep")
    coder.force_consolidate("deep")  # coder hasn't talked yet but consolidates anyway

    # Now Coder asks something — should see the user's facts via shared memory
    t = coder.chat("What language should I use for this implementation?")
    out.append(Scenario(
        name="coder sees user's Python preference (shared memory)",
        tier=2,
        passed=_passes(_retrieved_blob(t, "shared"), ["python"]),
        detail=f"shared retrieved: {_retrieved_blob(t, 'shared')[:240]!r}",
    ))

    # Reviewer should also see it
    t = reviewer.chat("What's the scale of the payment system we're building for?")
    out.append(Scenario(
        name="reviewer sees user's transaction-volume context (shared memory)",
        tier=2,
        passed=_passes(
            _retrieved_blob(t, "shared"),
            ["payment", "5m", "5 m", "transaction", "million"],
        ),
        detail=f"shared retrieved: {_retrieved_blob(t, 'shared')[:240]!r}",
    ))
    return out


# ─── Tier 3 — agents disagree; each holds its own private view ────────────


def tier3_cross_agent_contradiction(server_url: str) -> List[Scenario]:
    """Researcher concludes X; Coder concludes !X. Each agent keeps own view."""
    researcher, coder, reviewer, _shared = _build_team(server_url, "t3")
    out = []

    researcher.remember_decision("Recommended PostgreSQL for the primary datastore (ACID guarantees needed).")
    coder.remember_decision("Implemented with SQLite for the prototype (simpler deploy).")

    researcher.force_consolidate("deep")
    coder.force_consolidate("deep")

    # Researcher's view: PostgreSQL
    t = researcher.chat("What database did we pick for primary datastore?")
    out.append(Scenario(
        name="researcher holds PostgreSQL recommendation",
        tier=3,
        passed=_passes(_retrieved_blob(t, "private"), ["postgres"]),
        detail=f"private retrieved: {_retrieved_blob(t, 'private')[:200]!r}",
    ))

    # Coder's view: SQLite (different from researcher's)
    t = coder.chat("What database did we go with for the prototype?")
    out.append(Scenario(
        name="coder holds SQLite implementation choice",
        tier=3,
        passed=_passes(_retrieved_blob(t, "private"), ["sqlite"]),
        detail=f"private retrieved: {_retrieved_blob(t, 'private')[:200]!r}",
    ))

    # Critical: researcher should NOT see SQLite in private memory; coder
    # should NOT see PostgreSQL in private. (Both might appear in shared
    # if they were spoken aloud — but these are remember_decision calls
    # which write only to private, so this is a clean test.)
    t = researcher.chat("Did anyone consider SQLite?")
    blob = _retrieved_blob(t, "private")
    out.append(Scenario(
        name="researcher's private memory doesn't contain SQLite",
        tier=3,
        passed="sqlite" not in blob,
        detail=f"researcher private retrieved: {blob[:200]!r}",
    ))
    return out


# ─── Tier 4 — per-agent autonomous wake summary ───────────────────────────


def tier4_per_agent_wake_summary(server_url: str) -> List[Scenario]:
    """Each agent's idle period → its own wake summary; doesn't cross-contaminate."""
    researcher, coder, reviewer, _shared = _build_team(server_url, "t4")
    out = []

    # Researcher does some activity
    researcher.chat("Investigated three options: Pyright, Mypy, Pytype.")
    researcher.chat("Pyright was fastest; Mypy had the most plugins; Pytype was the most accurate.")
    researcher.chat("Recommendation: Pyright for speed, Mypy if plugin ecosystem matters more.")

    # Coder does completely different activity
    coder.chat("Wrote the auth middleware in 80 lines.")
    coder.chat("Added rate limiting via fastapi-limiter.")

    # Wait for the auto-sleep sweeper to fire on both private namespaces
    print("    [t4] waiting 25s for per-agent auto-sleep sweepers…")
    time.sleep(25)

    # Each agent's NEXT call should surface its OWN wake summary
    t = researcher.chat("Anything you noticed while I was away?")
    out.append(Scenario(
        name="researcher's wake summary surfaces on resume",
        tier=4,
        passed=t.wake_summary_pending is not None
               and bool(t.wake_summary_pending.get("narrative", "").strip()),
        detail=(
            f"wake_summary_pending: {bool(t.wake_summary_pending)}\n"
            f"narrative: "
            f"{(t.wake_summary_pending or {}).get('narrative', '')[:160]!r}"
        ),
    ))

    t = coder.chat("Catch me up on what's new.")
    out.append(Scenario(
        name="coder's wake summary surfaces on resume",
        tier=4,
        passed=t.wake_summary_pending is not None
               and bool(t.wake_summary_pending.get("narrative", "").strip()),
        detail=(
            f"wake_summary_pending: {bool(t.wake_summary_pending)}\n"
            f"narrative: "
            f"{(t.wake_summary_pending or {}).get('narrative', '')[:160]!r}"
        ),
    ))
    return out


# ─── Tier 5 — collaborative task with per-agent retrieval ─────────────────


def tier5_collaborative_task(server_url: str) -> List[Scenario]:
    """User establishes context; all 3 agents collaborate on a task using
    both their private and the shared memory."""
    researcher, coder, reviewer, _shared = _build_team(server_url, "t5")
    out = []

    # User shares a constraint via Researcher
    researcher.chat("I run a SaaS with strict GDPR compliance — no user data leaves EU.")
    researcher.chat("Latency target: <100ms p95 for the API.")

    # Each agent records its own decision
    researcher.remember_decision("Surveyed AWS Frankfurt, GCP Belgium, Hetzner — all GDPR-compliant.")
    coder.remember_decision("Decided on async asyncpg + uvloop for the lowest latency.")
    reviewer.remember_decision("Required GDPR data-residency tags on all DB columns containing PII.")

    researcher.force_consolidate("deep")
    coder.force_consolidate("deep")
    reviewer.force_consolidate("deep")

    # Cross-cutting probe: user asks Coder about deployment
    # Coder should see user's GDPR constraint via SHARED, and its own
    # asyncpg decision via PRIVATE.
    t = coder.chat("Can you summarize what we've decided for the implementation?")
    blob_shared = _retrieved_blob(t, "shared")
    blob_private = _retrieved_blob(t, "private")

    out.append(Scenario(
        name="coder sees user's GDPR constraint via shared memory",
        tier=5,
        passed=_passes(blob_shared, ["gdpr", "eu"]),
        detail=f"shared blob: {blob_shared[:200]!r}",
    ))
    out.append(Scenario(
        name="coder recalls own asyncpg decision via private memory",
        tier=5,
        passed=_passes(blob_private, ["asyncpg", "uvloop"]),
        detail=f"private blob: {blob_private[:200]!r}",
    ))
    return out


# ─── Tier 6 — strict isolation: agent A never sees B's private facts ──────


def tier6_strict_isolation(server_url: str) -> List[Scenario]:
    """Each agent's private decisions stay private."""
    researcher, coder, reviewer, _shared = _build_team(server_url, "t6")
    out = []

    researcher.remember_decision("Researcher's secret password is 'sphinx-9482'.")
    coder.remember_decision("Coder's hidden code is 'falcon-2026'.")
    reviewer.remember_decision("Reviewer's review token is 'ratchet-7711'.")

    researcher.force_consolidate("deep")
    coder.force_consolidate("deep")
    reviewer.force_consolidate("deep")

    # Researcher should not see coder's or reviewer's secrets
    t = researcher.chat("What was the coder's hidden code?")
    blob = _retrieved_blob(t, "private")
    out.append(Scenario(
        name="researcher's private memory excludes coder's secret",
        tier=6,
        passed="falcon" not in blob,
        detail=f"researcher's blob: {blob[:200]!r}",
    ))

    t = researcher.chat("What was the reviewer's token?")
    blob = _retrieved_blob(t, "private")
    out.append(Scenario(
        name="researcher's private memory excludes reviewer's secret",
        tier=6,
        passed="ratchet" not in blob,
        detail=f"researcher's blob: {blob[:200]!r}",
    ))

    # Coder should not see researcher's password
    t = coder.chat("What's the researcher's password?")
    blob = _retrieved_blob(t, "private")
    out.append(Scenario(
        name="coder's private memory excludes researcher's secret",
        tier=6,
        passed="sphinx" not in blob,
        detail=f"coder's blob: {blob[:200]!r}",
    ))

    # Each agent CAN see its own secret
    t = researcher.chat("What was my own password?")
    blob = _retrieved_blob(t, "private")
    out.append(Scenario(
        name="researcher CAN see its own secret",
        tier=6,
        passed="sphinx" in blob,
        detail=f"researcher's blob: {blob[:200]!r}",
    ))
    return out


# ─── Tier 7 — DeepSeek extraction depth ──────────────────────────────────


def tier7_deepseek_extraction_depth(server_url: str) -> List[Scenario]:
    """An ambiguous, multi-fact statement should yield richer concept
    extraction with DeepSeek v4 than with the heuristic fallback."""
    researcher, _coder, _reviewer, _shared = _build_team(server_url, "t7")
    out = []

    researcher.chat(
        "I'm Saish, lead engineer at Northstar Robotics. I run every "
        "Tuesday morning, am allergic to peanuts, and just got engaged "
        "to my partner Mara who's a chemistry teacher in Bangalore."
    )
    researcher.force_consolidate("deep")

    # Probe the multi-fact statement on multiple axes
    t = researcher.chat("Where do I work?")
    out.append(Scenario(
        name="DeepSeek extracted employer (Northstar Robotics)",
        tier=7,
        passed=_passes(_retrieved_blob(t, "shared"), ["northstar"]),
        detail=f"retrieved: {_retrieved_blob(t, 'shared')[:200]!r}",
    ))

    t = researcher.chat("What food am I allergic to?")
    out.append(Scenario(
        name="DeepSeek extracted allergy (peanuts)",
        tier=7,
        passed=_passes(_retrieved_blob(t, "shared"), ["peanut"]),
        detail=f"retrieved: {_retrieved_blob(t, 'shared')[:200]!r}",
    ))

    t = researcher.chat("Who is my partner?")
    out.append(Scenario(
        name="DeepSeek extracted partner name (Mara)",
        tier=7,
        passed=_passes(_retrieved_blob(t, "shared"), ["mara"]),
        detail=f"retrieved: {_retrieved_blob(t, 'shared')[:200]!r}",
    ))

    t = researcher.chat("What city do I live in?")
    out.append(Scenario(
        name="DeepSeek extracted city (Bangalore)",
        tier=7,
        passed=_passes(_retrieved_blob(t, "shared"), ["bangalore", "bengaluru"]),
        detail=f"retrieved: {_retrieved_blob(t, 'shared')[:200]!r}",
    ))

    return out
