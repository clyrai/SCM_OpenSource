"""
LOCOMO++ perturbation generator.

Takes a clean LOCOMO conversation and augments it with realistic perturbations
that production memory systems face but the original LOCOMO benchmark does not
exercise:

  1. Noise turns       — irrelevant chatter (filler, small talk)
  2. Contradictions    — fact updates that should supersede prior versions
  3. Entity confusion  — turns about similar-named distractor entities
  4. Long-horizon      — paraphrased restatements of real facts

The output is a perturbed conversation plus an augmented QA set that includes
the original LOCOMO questions plus four new question categories matched to the
perturbation types.

Everything is deterministic via seed. No LLM calls. No API spend.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─── Noise template bank (50+ realistic small-talk turns) ───────────────────

NOISE_TURNS: List[str] = [
    "Did you see the weather today? It's so unpredictable.",
    "I had pizza for lunch, it was really good.",
    "Lol that's hilarious.",
    "Hmm interesting.",
    "I'm so tired today, didn't sleep well.",
    "Did you watch the game last night?",
    "What time is it there?",
    "Got to grab some coffee, brb.",
    "How's everything going on your end?",
    "I keep forgetting to charge my phone.",
    "The traffic was insane this morning.",
    "Anyway, what were we saying?",
    "Yeah totally.",
    "Mhm makes sense.",
    "That reminds me of something but I forget what.",
    "Oh by the way did you see that meme?",
    "I'm thinking about getting a haircut.",
    "Random but I really want sushi tonight.",
    "Ugh my back hurts from sitting all day.",
    "I should probably exercise more.",
    "Did you hear about that new show?",
    "I forgot what I was going to say.",
    "Coffee is the only thing keeping me going today.",
    "It's so hot outside.",
    "I love rainy days.",
    "Just got back from a walk.",
    "I'm so behind on emails it's not even funny.",
    "What did you have for breakfast?",
    "I keep meaning to clean my desk.",
    "Hold on, my phone's buzzing.",
    "Sorry, got distracted there for a sec.",
    "lol",
    "haha yeah",
    "right?",
    "for real",
    "Anyway.",
    "k",
    "👍",
    "Sorry, what was the question again?",
    "I missed the start of what you said.",
    "Oh that reminds me, I need to call my mom.",
    "I'm starving.",
    "Where do you want to eat?",
    "I should switch to decaf.",
    "My cat keeps walking on my keyboard.",
    "It's already that late?",
    "Time really flies.",
    "I need a vacation.",
    "Have you been sleeping ok lately?",
    "I keep getting these random headaches.",
    "Whatever, it's fine.",
]


# ─── Contradiction generator (template-driven, no LLM) ──────────────────────

# ─── Synthetic anchor facts (injected at conversation start, contradicted at end) ──
#
# Each entry is a tuple of (fact_kind, anchor_template, contradiction_template,
# old_value, new_value). We inject the anchor near the start so the system has
# many turns to consolidate it, then issue the contradiction near the end. QA
# pairs ask for the CURRENT value (should be `new`) and explicitly NOT the OLD.

ANCHOR_FACTS: List[Dict[str, str]] = [
    {
        "kind": "employer",
        "anchor": "By the way, just for context — I currently work at GreenLeaf Cafe.",
        "contradiction": "Quick update on me — I left GreenLeaf Cafe last month. I'm at TechCorp now.",
        "old": "GreenLeaf Cafe",
        "new": "TechCorp",
        "current_q": "Where does the speaker work now (current employer)?",
        "old_q": "Where did the speaker used to work (former employer)?",
    },
    {
        "kind": "city",
        "anchor": "Just a heads up — I'm based in Seattle these days.",
        "contradiction": "Oh by the way, I moved out of Seattle a few weeks ago. I'm in Boston now.",
        "old": "Seattle",
        "new": "Boston",
        "current_q": "Which city does the speaker live in now (current city)?",
        "old_q": "Which city did the speaker move away from?",
    },
    {
        "kind": "hobby",
        "anchor": "Random fact about me — my main hobby right now is rock climbing.",
        "contradiction": "Hobby update: I stopped rock climbing. I picked up sculpting instead.",
        "old": "rock climbing",
        "new": "sculpting",
        "current_q": "What is the speaker's current main hobby?",
        "old_q": "What hobby did the speaker stop doing?",
    },
    {
        "kind": "diet",
        "anchor": "I should mention — I've been vegetarian for years.",
        "contradiction": "Diet change: I'm not vegetarian anymore. I went pescatarian last month.",
        "old": "vegetarian",
        "new": "pescatarian",
        "current_q": "What is the speaker's current diet?",
        "old_q": "What diet did the speaker used to follow?",
    },
    {
        "kind": "transport",
        "anchor": "FYI — I drive a blue Toyota to work every day.",
        "contradiction": "I sold the Toyota — I'm biking to work now.",
        "old": "Toyota",
        "new": "biking",
        "current_q": "How does the speaker currently commute to work?",
        "old_q": "How did the speaker used to commute?",
    },
    {
        "kind": "phone",
        "anchor": "I just got a new iPhone last week, by the way.",
        "contradiction": "Returned the iPhone — I'm using a Pixel now.",
        "old": "iPhone",
        "new": "Pixel",
        "current_q": "What phone does the speaker currently use?",
        "old_q": "What phone did the speaker use before switching?",
    },
    {
        "kind": "pet",
        "anchor": "Side note — I have a cat named Mochi.",
        "contradiction": "Mochi went to live with my sister. I have a dog named Bagel now.",
        "old": "Mochi",
        "new": "Bagel",
        "current_q": "What is the name of the speaker's current pet?",
        "old_q": "What was the name of the speaker's previous pet?",
    },
    {
        "kind": "schedule",
        "anchor": "My weekly support group meets on Sundays.",
        "contradiction": "The support group switched days — it's on Wednesdays now, not Sundays.",
        "old": "Sundays",
        "new": "Wednesdays",
        "current_q": "What day does the speaker's support group meet now?",
        "old_q": "What day did the support group used to meet?",
    },
]


# ─── Entity confusion: distractor names ─────────────────────────────────────

DISTRACTOR_NAMES = [
    "Caryn", "Catherine", "Christine", "Coral", "Connie",
    "Marcus", "Marcel", "Martin", "Marvin", "Michael",
    "Diana", "Daphne", "Darla", "Donna", "Dorothy",
]


# ─── Paraphrase rules (for long-horizon replay) ────────────────────────────

PARAPHRASE_PREFIXES = [
    "By the way,",
    "Just to recap,",
    "Reminder:",
    "Like I said before,",
    "Going back to what I mentioned —",
    "Quick recap:",
]


@dataclass
class PerturbationStats:
    noise_turns_added: int = 0
    contradiction_turns_added: int = 0
    distractor_turns_added: int = 0
    paraphrase_turns_added: int = 0
    contradictions_tracked: List[Dict[str, Any]] = field(default_factory=list)
    distractor_entities: List[str] = field(default_factory=list)
    noise_phrases: List[str] = field(default_factory=list)


@dataclass
class PerturbationConfig:
    """Knobs controlling LOCOMO++ severity."""
    noise_density: float = 0.30        # fraction of new turns that are noise
    contradiction_count: int = 8       # # of facts to contradict per conversation
    distractor_entity_count: int = 3   # # of fake similar-named entities
    distractor_facts_per_entity: int = 6
    paraphrase_replays: int = 0        # extra paraphrased copies of real turns (0 = off)
    seed: int = 42


class LocomoPlusPerturber:
    """Augment a LOCOMO conversation with realistic perturbations."""

    def __init__(self, config: Optional[PerturbationConfig] = None):
        self.cfg = config or PerturbationConfig()
        self.rng = random.Random(self.cfg.seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def perturb(self, conv: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], PerturbationStats]:
        """
        Returns (perturbed_conversation, augmented_qa, stats).

        The perturbed conversation has the same shape as the LOCOMO original
        so existing benchmark code can ingest it without changes. The augmented
        QA set is an EXTRA set of questions designed to stress the perturbation
        types (use this in addition to or instead of the original conv['qa']).
        """
        stats = PerturbationStats()
        new_conv = json.loads(json.dumps(conv))  # deep copy

        speaker_a = new_conv["conversation"].get("speaker_a", "Speaker A")
        speaker_b = new_conv["conversation"].get("speaker_b", "Speaker B")
        speakers = [speaker_a, speaker_b]

        session_keys = sorted(
            [k for k in new_conv["conversation"].keys()
             if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )

        # 0. Inject synthetic ANCHOR facts at the start of session_1 so the
        #    contradictions later have something concrete to supersede.
        if session_keys:
            first_session = session_keys[0]
            anchor_turns = self._inject_anchor_facts(speakers, stats)
            new_conv["conversation"][first_session] = (
                anchor_turns + new_conv["conversation"][first_session]
            )

        # 1. Inject noise turns (interleaved within sessions)
        for sk in session_keys:
            self._inject_noise_into_session(new_conv["conversation"][sk], speakers, stats)

        # 2. Inject contradictions for the anchor facts at end of last session
        if session_keys:
            last_session = session_keys[-1]
            contradictions = self._build_anchor_contradictions(speakers, stats)
            new_conv["conversation"][last_session].extend(contradictions)

        # 3. Inject distractor entities (append to last session)
        if session_keys:
            last_session = session_keys[-1]
            distractor_turns = self._build_distractor_entities(speakers, stats)
            new_conv["conversation"][last_session].extend(distractor_turns)

        # 4. Optional long-horizon paraphrases
        if self.cfg.paraphrase_replays > 0 and session_keys:
            last_session = session_keys[-1]
            paraphrases = self._build_paraphrases(new_conv["conversation"], speakers, stats)
            new_conv["conversation"][last_session].extend(paraphrases)

        # Build augmented QA set
        augmented_qa = self._build_augmented_qa(stats)

        return new_conv, augmented_qa, stats

    # ── Noise injection ─────────────────────────────────────────────────────

    def _inject_noise_into_session(
        self,
        session_turns: List[Dict[str, Any]],
        speakers: List[str],
        stats: PerturbationStats,
    ) -> None:
        if not session_turns:
            return
        target_density = self.cfg.noise_density
        n_real = len(session_turns)
        # We want noise/(real+noise) = density → noise = real * density / (1 - density)
        if target_density >= 1.0:
            target_density = 0.95
        n_noise = int(round(n_real * target_density / (1.0 - target_density)))

        # Insert at random positions
        insertion_positions = sorted(
            self.rng.sample(range(n_real + 1), min(n_noise, n_real + 1))
        ) if n_noise else []

        # Walk through original turns, inserting noise at chosen positions
        new_turns: List[Dict[str, Any]] = []
        ins_idx = 0
        for i, turn in enumerate(session_turns):
            while ins_idx < len(insertion_positions) and insertion_positions[ins_idx] == i:
                phrase = self.rng.choice(NOISE_TURNS)
                speaker = self.rng.choice(speakers)
                new_turns.append({"speaker": speaker, "text": phrase, "_perturbation": "noise"})
                stats.noise_turns_added += 1
                stats.noise_phrases.append(phrase)
                ins_idx += 1
            new_turns.append(turn)
        # Trailing inserts
        while ins_idx < len(insertion_positions):
            phrase = self.rng.choice(NOISE_TURNS)
            speaker = self.rng.choice(speakers)
            new_turns.append({"speaker": speaker, "text": phrase, "_perturbation": "noise"})
            stats.noise_turns_added += 1
            stats.noise_phrases.append(phrase)
            ins_idx += 1

        session_turns.clear()
        session_turns.extend(new_turns)

    # ── Synthetic anchor + contradiction injection ──────────────────────────

    def _inject_anchor_facts(
        self,
        speakers: List[str],
        stats: PerturbationStats,
    ) -> List[Dict[str, Any]]:
        """Add anchor-fact turns near the start of the conversation."""
        n = min(self.cfg.contradiction_count, len(ANCHOR_FACTS))
        chosen = self._sample_unique(ANCHOR_FACTS, n)
        # Stash chosen anchors so the contradiction step can refer back.
        self._chosen_anchors = chosen

        # Use the first speaker so anchors read as one consistent profile
        # (production assistants typically track ONE user).
        primary_speaker = speakers[0]
        turns: List[Dict[str, Any]] = []
        for anchor in chosen:
            turns.append({
                "speaker": primary_speaker,
                "text": anchor["anchor"],
                "_perturbation": "anchor",
                "_kind": anchor["kind"],
                "_old": anchor["old"],
                "_new": anchor["new"],
            })
        return turns

    def _build_anchor_contradictions(
        self,
        speakers: List[str],
        stats: PerturbationStats,
    ) -> List[Dict[str, Any]]:
        """Generate contradiction turns for each previously injected anchor."""
        anchors = getattr(self, "_chosen_anchors", [])
        if not anchors:
            return []
        primary_speaker = speakers[0]
        out: List[Dict[str, Any]] = []
        for anchor in anchors:
            out.append({
                "speaker": primary_speaker,
                "text": anchor["contradiction"],
                "_perturbation": "contradiction",
                "_kind": anchor["kind"],
                "_old": anchor["old"],
                "_new": anchor["new"],
            })
            stats.contradiction_turns_added += 1
            stats.contradictions_tracked.append({
                "speaker": primary_speaker,
                "kind": anchor["kind"],
                "old": anchor["old"],
                "new": anchor["new"],
                "current_q": anchor["current_q"],
                "old_q": anchor["old_q"],
                "anchor_text": anchor["anchor"],
                "contradiction_text": anchor["contradiction"],
            })
        return out

    def _sample_unique(self, items: List[Any], k: int) -> List[Any]:
        if k >= len(items):
            return list(items)
        return self.rng.sample(items, k)

    # ── Distractor entities ─────────────────────────────────────────────────

    def _build_distractor_entities(
        self,
        speakers: List[str],
        stats: PerturbationStats,
    ) -> List[Dict[str, Any]]:
        """
        Inject turns about fake similar-named entities to test disambiguation.
        Each distractor entity gets ONE unique signature fact (used by QA) plus
        several padding facts that don't appear in QA.
        """
        n = self.cfg.distractor_entity_count
        if n <= 0:
            return []
        chosen_distractors = self.rng.sample(DISTRACTOR_NAMES, min(n, len(DISTRACTOR_NAMES)))
        stats.distractor_entities.extend(chosen_distractors)

        # Signature facts: one per distractor, with a question phrase the QA
        # set can target unambiguously. Cycled deterministically by index.
        signature_pool = [
            ("runs a small bakery downtown", "Who runs a small bakery downtown?"),
            ("drives a vintage red Mustang", "Who drives a vintage red Mustang?"),
            ("plays the cello in a chamber group", "Who plays the cello in a chamber group?"),
            ("collects vintage typewriters", "Who collects vintage typewriters?"),
            ("works night shifts at the hospital", "Who works night shifts at the hospital?"),
            ("is learning Mandarin on weekends", "Who is learning Mandarin on weekends?"),
            ("grew up in rural Vermont", "Who grew up in rural Vermont?"),
            ("has three dogs named after planets", "Who has three dogs named after planets?"),
        ]

        # Padding facts (not used in QA — just clutter to make distractors feel real)
        padding_pool = [
            "is allergic to shellfish",
            "just got back from a hiking trip in the Alps",
            "hates pineapple on pizza",
            "once met a celebrity at an airport",
            "loves stargazing on weekends",
        ]

        turns: List[Dict[str, Any]] = []
        # Track signature mapping for QA
        if not hasattr(stats, "_disambig_signatures"):
            stats_signatures = []
            stats._disambig_signatures = stats_signatures  # type: ignore[attr-defined]
        else:
            stats_signatures = stats._disambig_signatures  # type: ignore[attr-defined]

        for i, distractor in enumerate(chosen_distractors):
            # 1) Signature fact (cycle through pool for determinism)
            sig_fact, sig_question = signature_pool[i % len(signature_pool)]
            speaker = self.rng.choice(speakers)
            sig_text = f"My friend {distractor} {sig_fact}."
            turns.append({
                "speaker": speaker,
                "text": sig_text,
                "_perturbation": "distractor",
                "_distractor_entity": distractor,
                "_distractor_fact": sig_fact,
            })
            stats.distractor_turns_added += 1
            stats_signatures.append({
                "entity": distractor,
                "question": sig_question,
                "fact": sig_fact,
            })

            # 2) Padding facts (no QA target)
            pad_count = max(0, self.cfg.distractor_facts_per_entity - 1)
            for fact in self.rng.sample(padding_pool, min(pad_count, len(padding_pool))):
                speaker = self.rng.choice(speakers)
                text = f"My friend {distractor} {fact}."
                turns.append({
                    "speaker": speaker,
                    "text": text,
                    "_perturbation": "distractor",
                    "_distractor_entity": distractor,
                    "_distractor_fact": fact,
                })
                stats.distractor_turns_added += 1

        return turns

    # ── Long-horizon paraphrases ────────────────────────────────────────────

    def _build_paraphrases(
        self,
        conv_data: Dict[str, Any],
        speakers: List[str],
        stats: PerturbationStats,
    ) -> List[Dict[str, Any]]:
        if self.cfg.paraphrase_replays <= 0:
            return []
        # Pick a few real turns and create paraphrased restatements
        real_turns: List[Tuple[str, str]] = []
        for sk, turns in conv_data.items():
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if isinstance(turn, dict) and not turn.get("_perturbation"):
                    text = turn.get("text", "")
                    if len(text) > 20:
                        real_turns.append((turn.get("speaker", ""), text))

        if not real_turns:
            return []
        chosen = self._sample_unique(real_turns, min(self.cfg.paraphrase_replays, len(real_turns)))
        out: List[Dict[str, Any]] = []
        for speaker, text in chosen:
            prefix = self.rng.choice(PARAPHRASE_PREFIXES)
            paraphrase = f"{prefix} {text}"
            out.append({"speaker": speaker, "text": paraphrase, "_perturbation": "paraphrase"})
            stats.paraphrase_turns_added += 1
        return out

    # ── Augmented QA generation ─────────────────────────────────────────────

    def _build_augmented_qa(self, stats: PerturbationStats) -> List[Dict[str, Any]]:
        qa: List[Dict[str, Any]] = []

        # Contradiction questions: ask for the CURRENT value (correct = new)
        # and the OLD value (correct = old). A retrieval-only system without
        # versioning will conflate the two.
        for c in stats.contradictions_tracked:
            qa.append({
                "category": 100,
                "_qtype": "contradiction_current",
                "question": c["current_q"],
                "answer": str(c["new"]),
                "evidence": {
                    "kind": c["kind"],
                    "expected": c["new"],
                    "incorrect_value": c["old"],
                },
            })
            qa.append({
                "category": 103,
                "_qtype": "contradiction_old",
                "question": c["old_q"],
                "answer": str(c["old"]),
                "evidence": {
                    "kind": c["kind"],
                    "expected": c["old"],
                    "incorrect_value": c["new"],
                },
            })

        # Entity disambiguation: each distractor has ONE unique signature fact
        # (e.g. "X runs a small bakery downtown"). The question targets that
        # specific fact; the correct answer is the distractor name. A retrieval
        # system without entity grounding will surface multiple distractors and
        # fail to attribute the fact correctly.
        signatures = getattr(stats, "_disambig_signatures", []) or []
        for sig in signatures:
            qa.append({
                "category": 101,
                "_qtype": "entity_disambig_distractor",
                "question": sig["question"],
                "answer": sig["entity"],
                "evidence": {
                    "expected_entity": sig["entity"],
                    "fact": sig["fact"],
                    "other_entities": [s["entity"] for s in signatures if s["entity"] != sig["entity"]],
                },
            })

        # Noise rejection questions: phrased so the correct retrieval should
        # NOT surface this noise content as a memory about the speaker.
        for phrase in stats.noise_phrases[:6]:
            qa.append({
                "category": 102,
                "_qtype": "noise_reject",
                "question": f"What significant fact did the speaker share that mentions: '{phrase[:50]}'?",
                "answer": "no significant fact",  # any system that returns the noise verbatim fails
                "evidence": {"noise_phrase": phrase},
                "_negative_substring": phrase[:50].lower(),  # used by scorer to detect noise pollution
            })

        return qa


# ─── CLI for quick inspection ───────────────────────────────────────────────


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo/locomo10.json")
    parser.add_argument("--conv-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-density", type=float, default=0.30)
    parser.add_argument("--contradictions", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=3)
    args = parser.parse_args()

    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, args.data)
    with open(data_path) as f:
        all_conv = json.load(f)
    conv = all_conv[args.conv_idx]

    cfg = PerturbationConfig(
        seed=args.seed,
        noise_density=args.noise_density,
        contradiction_count=args.contradictions,
        distractor_entity_count=args.distractors,
    )
    perturber = LocomoPlusPerturber(cfg)
    new_conv, aug_qa, stats = perturber.perturb(conv)

    print(f"Conversation {args.conv_idx} ({conv['conversation'].get('speaker_a','?')}/{conv['conversation'].get('speaker_b','?')})")
    print(f"  noise_turns_added         = {stats.noise_turns_added}")
    print(f"  contradiction_turns_added = {stats.contradiction_turns_added}")
    print(f"  distractor_turns_added    = {stats.distractor_turns_added}")
    print(f"  paraphrase_turns_added    = {stats.paraphrase_turns_added}")
    print(f"  augmented_qa_pairs        = {len(aug_qa)}")
    print()
    print("Sample contradictions:")
    for c in stats.contradictions_tracked[:3]:
        print(f"  {c['kind']}: {c['old']!r} -> {c['new']!r}")
        print(f"    anchor: {c.get('anchor_text', '?')[:60]}")
        print(f"    contra: {c.get('contradiction_text', '?')[:60]}")
    print()
    print("Distractor entities:")
    for d in stats.distractor_entities:
        print(f"  - {d}")
    print()
    print("Sample augmented QA:")
    for q in aug_qa[:5]:
        print(f"  [{q['_qtype']}] {q['question']} -> {q['answer']!r}")


if __name__ == "__main__":
    _main()
