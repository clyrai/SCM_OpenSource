"""
Unresolved tension detection for sleep-time replay.

Tension is a functional signal, not an emotion claim. It marks memories that
look unfinished, contradictory, stressful, uncertain, or goal-related so REM
replay can prioritize the kinds of traces humans tend to keep processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from ..core.models import Concept, Episode, PredicateType, Relation
from ..core.time_utils import ensure_utc, utc_now


@dataclass(frozen=True)
class TensionSignal:
    concept_id: str
    description: str
    score: float
    signals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "description": self.description,
            "score": round(self.score, 4),
            "signals": list(self.signals),
        }


class TensionDetector:
    """Rank concepts that are likely still unresolved."""

    _SIGNAL_KEYWORDS = {
        "unfinished_task": (
            "todo",
            "to do",
            "need to",
            "have to",
            "pending",
            "unfinished",
            "not done",
            "follow up",
        ),
        "stress": (
            "stress",
            "stressed",
            "anxious",
            "worried",
            "pressure",
            "overwhelmed",
            "urgent",
        ),
        "uncertainty": (
            "maybe",
            "uncertain",
            "not sure",
            "unclear",
            "confused",
            "might",
            "could",
        ),
        "recurring_concern": (
            "again",
            "keeps",
            "recurring",
            "repeat",
            "still",
        ),
        "goal_open": (
            "goal",
            "deadline",
            "milestone",
            "launch",
            "plan",
            "next step",
            "project",
        ),
        "relationship_thread": (
            "team",
            "manager",
            "client",
            "friend",
            "family",
            "partner",
            "conflict",
        ),
    }

    _RESOLUTION_MARKERS = (
        "resolved",
        "closed",
        "completed",
        "fixed",
        "finished",
        "shipped",
        "settled",
        "handled",
        "no longer an issue",
    )

    _STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "is",
        "are",
        "was",
        "were",
        "it",
        "this",
        "that",
        "we",
        "i",
        "you",
        "they",
        "have",
        "has",
        "had",
        "still",
        "not",
    }

    def close_resolved(
        self,
        concepts: List[Concept],
        episodes: List[Episode],
        min_overlap: float = 0.28,
    ) -> List[Dict[str, Any]]:
        """Mark tension concepts closed when recent episode text resolves them."""
        closures: List[Dict[str, Any]] = []
        closure_episodes = [
            episode for episode in episodes
            if self._has_resolution_marker(episode.raw_content or "")
        ]
        if not closure_episodes:
            return closures

        for concept in concepts:
            tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
            if tags.get("_internal") or tags.get("tension_resolved"):
                continue
            concept_tokens = self._tokens(concept.description or "")
            if not concept_tokens:
                continue

            for episode in closure_episodes:
                episode_tokens = self._tokens(episode.raw_content or "")
                if not episode_tokens:
                    continue
                overlap = len(concept_tokens & episode_tokens) / max(1, len(concept_tokens))
                explicit_link = concept.id in set(episode.concept_ids or [])
                if not explicit_link and overlap < min_overlap:
                    continue

                tags["tension_resolved"] = True
                tags["tension_resolved_at"] = utc_now().isoformat()
                tags["tension_resolution_evidence"] = episode.raw_content
                concept.context_tags = tags
                closures.append({
                    "concept_id": concept.id,
                    "description": concept.description,
                    "resolved_at": tags["tension_resolved_at"],
                    "evidence": episode.raw_content,
                    "overlap": round(overlap, 4),
                })
                break

        return closures

    def detect(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        episodes: List[Episode],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        contradiction_ids = self._contradiction_ids(relations)
        episode_hits = self._episode_signal_hits(episodes)

        ranked: List[TensionSignal] = []
        for concept in concepts:
            tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
            if tags.get("_internal") or tags.get("tension_resolved"):
                continue

            text = (concept.description or "").strip()
            if not text:
                continue

            signals = self._signals_for_text(text)
            signals.extend(str(s) for s in tags.get("tension_signals", []) if s)
            if concept.id in contradiction_ids:
                signals.append("contradiction")
            signals.extend(episode_hits.get(concept.id, []))

            signals = sorted(set(signals))
            if not signals:
                continue

            score = self._score(concept, signals)
            if score <= 0:
                continue
            ranked.append(TensionSignal(concept.id, text, score, signals))

        ranked.sort(key=lambda item: (-item.score, item.description.lower()))
        return [item.to_dict() for item in ranked[:limit]]

    def annotate(self, concepts: List[Concept], tensions: List[Dict[str, Any]]) -> None:
        by_id = {t.get("concept_id"): t for t in tensions}
        for concept in concepts:
            tension = by_id.get(concept.id)
            if not tension:
                continue
            tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
            tags["unresolved_tension_score"] = float(tension.get("score", 0.0) or 0.0)
            tags["unresolved_tension_signals"] = list(tension.get("signals", []) or [])
            concept.context_tags = tags

    def _score(self, concept: Concept, signals: List[str]) -> float:
        importance = concept.importance
        emotional = abs(float(importance.emotional if importance else 0.0))
        task = float(importance.task_relevance if importance else 0.0)
        repetition = float(importance.repetition if importance else 0.0)
        recency = self._recency_score(getattr(concept, "last_accessed", None))
        signal_strength = min(1.0, len(signals) / 4.0)

        return min(
            1.0,
            emotional * 0.28
            + task * 0.25
            + repetition * 0.16
            + recency * 0.16
            + signal_strength * 0.15,
        )

    def _signals_for_text(self, text: str) -> List[str]:
        lower = text.lower()
        if "not done" in lower or "not finished" in lower or "not resolved" in lower:
            pass
        elif self._has_resolution_marker(lower):
            return []
        signals: List[str] = []
        for signal, keywords in self._SIGNAL_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                signals.append(signal)
        return signals

    def _has_resolution_marker(self, text: str) -> bool:
        lower = text.lower()
        unresolved_negations = ("not done", "not finished", "not resolved", "not fixed", "not closed")
        if any(negation in lower for negation in unresolved_negations):
            return False
        return any(marker in lower for marker in self._RESOLUTION_MARKERS)

    @classmethod
    def _tokens(cls, text: str) -> Set[str]:
        cleaned = []
        for char in text.lower():
            cleaned.append(char if char.isalnum() else " ")
        return {
            token for token in "".join(cleaned).split()
            if len(token) >= 3 and token not in cls._STOPWORDS
        }

    @staticmethod
    def _contradiction_ids(relations: List[Relation]) -> Set[str]:
        out: Set[str] = set()
        for relation in relations:
            predicate = relation.predicate.value if hasattr(relation.predicate, "value") else str(relation.predicate)
            if predicate == PredicateType.CONTRADICTS.value or "contradict" in predicate.lower():
                out.add(relation.subject_id)
                out.add(relation.object_id)
        return out

    def _episode_signal_hits(self, episodes: List[Episode]) -> Dict[str, List[str]]:
        hits: Dict[str, List[str]] = {}
        for episode in episodes:
            signals = self._signals_for_text(episode.raw_content or "")
            if not signals:
                continue
            for concept_id in episode.concept_ids:
                hits.setdefault(concept_id, []).extend(signals)
        return hits

    @staticmethod
    def _recency_score(value: Any) -> float:
        ts = ensure_utc(value)
        if ts is None:
            return 0.5
        now = utc_now()
        age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        if age_hours <= 1:
            return 1.0
        if age_hours >= 168:
            return 0.1
        return max(0.1, 1.0 - (age_hours / 168.0))
