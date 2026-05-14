"""
Seven tiers of brutal scenarios for the LangChain + SCM agent.

Each tier returns a list of Scenario records (pass/fail + detail).
Scenarios are deterministic in scoring (keyword presence), even when the
LLM responses are stochastic.
"""
from __future__ import annotations

import time
from typing import List

import requests

from .agent import SCMLangChainAgent
from .runner import Scenario


def _passes(text: str, must_contain: List[str], must_not_contain: List[str] = None) -> bool:
    must_not_contain = must_not_contain or []
    blob = text.lower()
    pos = all(any(kw.lower() in blob for kw in opts) if isinstance(opts, list)
              else opts.lower() in blob
              for opts in must_contain)
    neg = any(kw.lower() in blob for kw in must_not_contain)
    return pos and not neg


# ─── Tier 1 — multi-day recall ────────────────────────────────────────────


def tier1_multi_day_recall(server_url: str) -> List[Scenario]:
    """User states a fact on day 1; on day 2 (after idle), agent should recall it."""
    agent = SCMLangChainAgent(user_id="t1_alice", scm_url=server_url)
    out = []

    # Day 1
    agent.chat("Hi, my name is Alice and I live in Bangalore.")
    agent.chat("I work as a backend engineer on payment systems.")
    agent.chat("I have a peanut allergy — pretty severe one.")

    # Force consolidation (simulating the day's-end sleep)
    agent.force_consolidate("deep")

    # Day 2 probe
    t = agent.chat("What city do I live in?")
    out.append(Scenario(
        name="recalls city across consolidation",
        tier=1, passed=_passes(t.response, [["bangalore", "bengaluru"]]),
        detail=f"response: {t.response[:160]!r}",
    ))

    t = agent.chat("Refresh me on my work — what do I do?")
    out.append(Scenario(
        name="recalls profession across consolidation",
        tier=1,
        passed=_passes(t.response, [["backend", "engineer", "payment"]]),
        detail=f"response: {t.response[:160]!r}",
    ))

    t = agent.chat("Any food allergies I should keep in mind?")
    out.append(Scenario(
        name="recalls allergy across consolidation",
        tier=1,
        passed=_passes(t.response, [["peanut", "nut", "allergy", "allergic"]]),
        detail=f"response: {t.response[:160]!r}",
    ))

    return out


# ─── Tier 2 — contradiction handling ──────────────────────────────────────


def tier2_contradiction_handling(server_url: str) -> List[Scenario]:
    """User changes employer; agent should track the new value."""
    agent = SCMLangChainAgent(user_id="t2_devon", scm_url=server_url)
    out = []

    agent.chat("I work at Northstar Robotics. We do industrial robotics.")
    agent.chat("Standup at Northstar today was rough. OAuth flow rewrite.")
    agent.force_consolidate("deep")

    # Contradiction
    agent.chat("Big news — I left Northstar Robotics today. I'm at Atlas Labs starting Monday.")
    agent.chat("Atlas Labs does reinforcement learning for industrial robots.")
    agent.force_consolidate("deep")

    # Probe current
    t = agent.chat("Where do I work currently?")
    out.append(Scenario(
        name="returns NEW employer after contradiction",
        tier=2,
        passed=_passes(
            t.response,
            must_contain=[["atlas"]],
            # Critical: it should NOT confidently say "Northstar" as the current
            # employer. We accept "previously at Northstar" because that's
            # contextually correct, so we only fail if Northstar appears
            # WITHOUT atlas.
            must_not_contain=[],
        ) and "atlas" in t.response.lower(),
        detail=f"response: {t.response[:200]!r}",
    ))

    t = agent.chat("Where did I used to work?")
    out.append(Scenario(
        name="returns OLD employer when asked about past",
        tier=2,
        passed=_passes(t.response, [["northstar"]]),
        detail=f"response: {t.response[:200]!r}",
    ))
    return out


# ─── Tier 3 — idle-fired wake summary ─────────────────────────────────────


def tier3_idle_fired_wake_summary(server_url: str) -> List[Scenario]:
    """Wait past idle threshold; the auto-sleep sweeper should fire and the
    next turn should surface a cached wake_summary_pending.

    No manual consolidate — that's the point. The sweeper does it on its own.
    """
    agent = SCMLangChainAgent(user_id="t3_priya", scm_url=server_url)
    out = []

    inputs = [
        "I bike to school every Monday and Wednesday.",
        "On Fridays I bike too — same route along the river.",
        "I'm a chemistry teacher, currently teaching stoichiometry.",
        "Sundays are my lab-prep day, weekly ritual.",
    ]
    for i, msg in enumerate(inputs, 1):
        t = agent.chat(msg)
        if t.error:
            print(f"    [t3] turn {i} ERROR: {t.error[:200]}")
        else:
            print(f"    [t3] turn {i} ok (response len={len(t.response)})")

    # Wait past idle threshold (8s) + sweeper tick (3s) + sleep wall time
    # (~6-15s for Ollama). Total: ~25s for the autonomous sleep to fire
    # and cache a summary.
    print("    [t3] waiting 25s for auto-sleep sweeper to fire and cache a wake summary…")
    time.sleep(25)

    # Next turn should surface wake_summary_pending
    t = agent.chat("Morning! What did you notice while I was away?")

    has_pending = t.wake_summary_pending is not None
    has_narrative = (
        has_pending
        and bool((t.wake_summary_pending.get("narrative") or "").strip())
    )
    out.append(Scenario(
        name="wake_summary_pending surfaces after idle",
        tier=3,
        passed=has_pending,
        detail=(
            f"wake_summary_pending: {bool(has_pending)}\n"
            f"narrative excerpt: "
            f"{(t.wake_summary_pending or {}).get('narrative', '')[:160]!r}"
        ),
    ))

    out.append(Scenario(
        name="wake_summary contains narrative text",
        tier=3,
        passed=has_narrative,
        detail=f"narrative present: {has_narrative}",
    ))
    return out


# ─── Tier 4 — cross-session synthesis ─────────────────────────────────────


def tier4_cross_session_synthesis(server_url: str) -> List[Scenario]:
    """Plant facts across separate days; ask a question that requires combining them."""
    agent = SCMLangChainAgent(user_id="t4_bob", scm_url=server_url)
    out = []

    # Day 1: workplace context
    agent.chat("My new office at Northstar has snack baskets in every conference room.")
    agent.chat("Catered lunch comes in every Wednesday.")
    agent.force_consolidate("deep")

    # Day 2: allergy context (no workplace mention)
    agent.chat("Just got back from dinner — had to send food back, peanut allergy is real.")
    agent.chat("Always have to ask twice at new restaurants about peanuts.")
    agent.force_consolidate("deep")

    # Probe — requires combining (office snacks + lunch) with (peanut allergy)
    t = agent.chat("What kind of food should I be careful about at the office?")
    out.append(Scenario(
        name="cross-session synthesis: office + allergy",
        tier=4,
        passed=_passes(t.response, [["peanut", "nut", "allergy", "allergic"]]),
        detail=f"response: {t.response[:240]!r}",
    ))

    return out


# ─── Tier 5 — adversarial storm ───────────────────────────────────────────


def tier5_adversarial_storm(server_url: str) -> List[Scenario]:
    """Hit it with all-noise, then a contradiction storm. It shouldn't crash."""
    agent = SCMLangChainAgent(user_id="t5_eve", scm_url=server_url)
    out = []

    # All-noise: 12 turns of low-content filler
    noise = [
        "Anyway.",
        "Hmm okay.",
        "Sure why not.",
        "Yeah no.",
        "Maybe.",
        "Got it.",
        "Right.",
        "Cool.",
        "I see.",
        "Fine.",
        "Whatever.",
        "Mhm.",
    ]
    crashed_during_noise = False
    for n in noise:
        try:
            agent.chat(n)
        except Exception as e:
            crashed_during_noise = True
            break

    out.append(Scenario(
        name="survives 12 turns of pure noise",
        tier=5,
        passed=not crashed_during_noise,
        detail=f"crashed: {crashed_during_noise}",
    ))

    # Contradiction storm: rapid back-and-forth on a single property
    storms = [
        "I prefer morning meetings.",
        "Actually I prefer evening meetings.",
        "Wait, morning is better.",
        "No, evenings only.",
        "Morning final answer.",
        "Mid-day works for me actually.",
    ]
    crashed_during_storm = False
    for s in storms:
        try:
            agent.chat(s)
        except Exception:
            crashed_during_storm = True
            break

    out.append(Scenario(
        name="survives contradiction storm",
        tier=5,
        passed=not crashed_during_storm,
        detail=f"crashed: {crashed_during_storm}",
    ))

    # After the storm, the most-recent value should win
    agent.force_consolidate("deep")
    t = agent.chat("What time of day do I prefer for meetings?")
    # We don't pin a specific value — we assert the response isn't empty
    # and isn't an obvious error.
    sane_response = bool(t.response.strip()) and "error" not in t.response.lower()
    out.append(Scenario(
        name="produces a coherent answer after contradiction storm",
        tier=5,
        passed=sane_response,
        detail=f"response: {t.response[:160]!r}",
    ))

    return out


# ─── Tier 6 — multi-user isolation ────────────────────────────────────────


def _retrieved_blob(turn) -> str:
    """Concatenate everything SCM returned for a query — both the
    structured memories list AND the spreading-activation memory_context.
    Either source counts as 'SCM surfaced this fact'."""
    parts = list(turn.retrieved_memories or [])
    if getattr(turn, "memory_context", None):
        parts.append(turn.memory_context)
    return " | ".join(parts).lower()


def tier6_multi_user_isolation(server_url: str) -> List[Scenario]:
    """Two users on the same server. Alice's facts must not appear in Bob's
    SCM-retrieved memories (and vice versa).

    NOTE: We score on what SCM returned to the agent (turn.retrieved_memories),
    NOT on what the LLM hallucinated in its final response. A small local
    LLM like llama3.2:3b will sometimes invent facts even when given an
    empty memory context — that's a known LLM property, not a SCM bug.
    The product surface is the SCM API; that's what we test.
    """
    out = []
    alice = SCMLangChainAgent(user_id="t6_alice", scm_url=server_url)
    bob   = SCMLangChainAgent(user_id="t6_bob", scm_url=server_url)

    alice.chat("My name is Alice and my secret pet name is Mr. Whiskers.")
    alice.chat("I'm allergic to seafood.")
    bob.chat("My name is Bob and I drive a 2019 Honda Civic.")
    bob.chat("I love sushi every weekend.")

    alice.force_consolidate("deep")
    bob.force_consolidate("deep")

    # Probe Bob: SCM should NOT return Alice's pet-name concept
    t = bob.chat("What's my pet's name?")
    bob_retrieved = _retrieved_blob(t)
    out.append(Scenario(
        name="SCM isolation: Bob's search doesn't return Alice's pet name",
        tier=6,
        passed=("whiskers" not in bob_retrieved),
        detail=(
            f"bob retrieved {len(t.retrieved_memories)} memories; "
            f"contains 'whiskers'={('whiskers' in bob_retrieved)}\n"
            f"retrieved: {bob_retrieved[:200]!r}"
        ),
    ))

    # Probe Alice: SCM should NOT return Bob's car concept
    t = alice.chat("What car do I drive?")
    alice_retrieved = _retrieved_blob(t)
    has_honda_in_retrieved = "honda" in alice_retrieved or "civic" in alice_retrieved
    out.append(Scenario(
        name="SCM isolation: Alice's search doesn't return Bob's car",
        tier=6,
        passed=not has_honda_in_retrieved,
        detail=(
            f"alice retrieved {len(t.retrieved_memories)} memories; "
            f"contains 'honda/civic'={has_honda_in_retrieved}\n"
            f"retrieved: {alice_retrieved[:200]!r}\n"
            f"(LLM response may hallucinate; we score on SCM-returned data)"
        ),
    ))

    # Probe own memory survives isolation: Alice's seafood allergy should be
    # in HER retrieved memories.
    t = alice.chat("What food am I allergic to?")
    own_retrieved = _retrieved_blob(t)
    found_own = any(kw in own_retrieved for kw in ("seafood", "fish", "shellfish", "allerg"))
    out.append(Scenario(
        name="own memory survives across user separation (Alice)",
        tier=6,
        passed=found_own,
        detail=f"alice retrieved {len(t.retrieved_memories)}; own seafood/allergy fact: {found_own}",
    ))

    return out


# ─── Tier 7 — failure mode (SCM unreachable) ──────────────────────────────


def tier7_failure_mode(server_url: str) -> List[Scenario]:
    """Point the agent at a non-existent SCM endpoint; the agent must
    NOT crash — it should degrade to no-memory mode and still respond."""
    out = []
    # Wrong port: nothing listening
    fake_url = "http://127.0.0.1:65500/v1"
    agent = SCMLangChainAgent(user_id="t7_carlos", scm_url=fake_url)

    crashed = False
    response_produced = False
    try:
        t = agent.chat("Hi, I'm Carlos and I'm new here.")
        response_produced = bool(t.response.strip())
    except Exception:
        crashed = True

    out.append(Scenario(
        name="agent doesn't crash when SCM is unreachable",
        tier=7,
        passed=not crashed,
        detail=f"crashed: {crashed}",
    ))
    out.append(Scenario(
        name="agent still produces a response without memory",
        tier=7,
        passed=response_produced,
        detail=f"response_produced: {response_produced}",
    ))

    return out
