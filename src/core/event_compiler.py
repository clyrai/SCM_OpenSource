"""
EventCompiler: Convert episodes into structured event frames.
"""
from __future__ import annotations

import re
from typing import List, Optional

from .models import EventSchema, Episode


TIME_PATTERNS = [
    r"\b(today|tonight|tomorrow|yesterday|now|later)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]

WHERE_PATTERNS = [
    r"\bto\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
    r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
    r"\bat\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
    r"\bfrom\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
]

WHY_PATTERNS = [
    r"\bbecause\s+([^.!?]+)",
    r"\bso\s+that\s+([^.!?]+)",
    r"\bdue\s+to\s+([^.!?]+)",
]

CONTRADICTION_PATTERNS = [
    r"\bactually\b.*\bno\b",
    r"\bi was wrong\b",
    r"\bchanged my mind\b",
    r"\bnot anymore\b",
    r"\bcorrection\b",
    r"\bon second thought\b",
]


class EventCompiler:
    """
    Converts raw episodes into structured event frames.

    Phase 2 behavior:
    - Create who/what/when/where/why fields using deterministic heuristics.
    - Carry salience/grasp/certainty metadata from encode step.
    - Detect potential contradiction events for later versioning support.
    """

    def compile_episode(
        self,
        episode: Episode,
        interlocutor: Optional[str] = None,
        task_context: Optional[str] = None,
    ) -> EventSchema:
        text = (episode.raw_content or "").strip()

        who = self._extract_who(text, episode.source, interlocutor)
        what = self._extract_what(text)
        when_value = self._extract_when(text)
        where_value = self._extract_where(text)
        why = self._extract_why(text)
        certainty = self._estimate_certainty(text, episode)
        contradiction = self._is_contradiction(text)

        event = EventSchema(
            who=who,
            what=what,
            when=when_value,
            where=where_value,
            why=why,
            source=episode.source,
            certainty=certainty,
            salience=episode.salience_score,
            grasp=episode.grasp_score,
            is_contradiction=contradiction,
            raw_episode_id=episode.id,
        )
        event.event_key = self._make_event_key(event)
        return event

    def compile_batch(
        self,
        episodes: List[Episode],
        interlocutor: Optional[str] = None,
        task_context: Optional[str] = None,
    ) -> List[EventSchema]:
        return [
            self.compile_episode(
                ep,
                interlocutor=interlocutor,
                task_context=task_context,
            )
            for ep in episodes
        ]

    def is_duplicate(self, candidate: EventSchema, existing: List[EventSchema], threshold: float = 0.82) -> bool:
        """
        Duplicate check to prevent event inflation.

        Uses lightweight token-overlap on (who + what + where + when).
        """
        candidate_tokens = self._event_tokens(candidate)
        if not candidate_tokens:
            return False

        for event in existing:
            tokens = self._event_tokens(event)
            if not tokens:
                continue
            overlap = self._jaccard(candidate_tokens, tokens)
            if overlap >= threshold:
                return True
        return False

    def _extract_who(self, text: str, source: str, interlocutor: Optional[str]) -> str:
        if source == "assistant":
            return "assistant"

        lowered = text.lower()
        if re.search(r"\b(i|i'm|my|me)\b", lowered):
            return interlocutor or "user"

        proper_names = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        if proper_names:
            return proper_names[0]

        return interlocutor or "user"

    def _extract_what(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return "(empty)"
        if len(cleaned) <= 160:
            return cleaned
        return cleaned[:157].rstrip() + "..."

    def _extract_when(self, text: str) -> Optional[str]:
        for pattern in TIME_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_where(self, text: str) -> Optional[str]:
        for pattern in WHERE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_why(self, text: str) -> Optional[str]:
        for pattern in WHY_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _estimate_certainty(self, text: str, episode: Episode) -> float:
        lowered = text.lower()
        uncertainty_terms = [
            "maybe",
            "perhaps",
            "i think",
            "not sure",
            "possibly",
            "probably",
            "might",
        ]
        confidence = max(0.0, min(1.0, episode.salience_score * 0.6 + episode.grasp_score * 0.4))
        penalty = 0.0
        for term in uncertainty_terms:
            if term in lowered:
                penalty += 0.08
        return max(0.1, min(1.0, round(confidence - penalty, 4)))

    def _is_contradiction(self, text: str) -> bool:
        lowered = text.lower()
        return any(re.search(pattern, lowered) for pattern in CONTRADICTION_PATTERNS)

    def _make_event_key(self, event: EventSchema) -> str:
        parts = [
            event.who.lower().strip(),
            event.what.lower().strip(),
            (event.where or "").lower().strip(),
            (event.when or "").lower().strip(),
        ]
        return "|".join(parts)

    def _event_tokens(self, event: EventSchema) -> set[str]:
        blob = " ".join([
            event.who or "",
            event.what or "",
            event.when or "",
            event.where or "",
        ]).lower()
        return set(re.findall(r"[a-z0-9]+", blob))

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0
