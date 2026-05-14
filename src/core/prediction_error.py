"""
PredictionError: Surprise detection for HME
Detects when incoming input violates expected patterns,
indicating high prediction error (surprise) which boosts memory encoding.
"""
from typing import List, Optional, Dict
from collections import deque
from dataclasses import dataclass
import math


@dataclass
class PredictionContext:
    recent_concepts: List[str]
    recent_episodes: List[str]
    topic_history: List[str]
    entity_history: List[str]


class PredictionErrorEngine:
    """
    Computes prediction error (surprise) for incoming content.

    High prediction error = input violates expectations = strong memory trace.
    Low prediction error = expected content = weaker memory trace.

    Tracks a rolling window of recent concepts and topics to build
    an expectation model. Incoming content that diverges from recent
    patterns scores high on surprise.
    """

    def __init__(
        self,
        window_size: int = 5,
        decay: float = 0.85,
        entity_bonus: float = 0.2,
        topic_bonus: float = 0.15,
        contradiction_bonus: float = 0.3,
    ):
        self.window_size = window_size
        self.decay = decay
        self.entity_bonus = entity_bonus
        self.topic_bonus = topic_bonus
        self.contradiction_bonus = contradiction_bonus

        self._recent_concepts: deque = deque(maxlen=window_size)
        self._recent_topics: deque = deque(maxlen=window_size)
        self._recent_entities: deque = deque(maxlen=window_size)
        self._episode_summaries: deque = deque(maxlen=window_size)

    def compute(
        self,
        text: str,
        extracted_entities: Optional[List[str]] = None,
        topic: Optional[str] = None,
        episode_summary: Optional[str] = None,
        prior_concepts: Optional[List[str]] = None,
    ) -> float:
        """
        Compute prediction error score [0, 1] for the given content.

        Higher = more surprising = stronger memory encoding.
        Lower = expected = weaker encoding.

        Args:
            text: The raw input text
            extracted_entities: Named entities detected in the text
            topic: Topic label if known
            episode_summary: Brief summary of the episode content
            prior_concepts: Prior concept descriptions to compare against

        Returns:
            float: prediction error score in [0, 1]
        """
        surprise_score = 0.0
        reasons = []

        entities = extracted_entities or []
        topic_str = topic or self._infer_topic(text)

        entity_overlap = self._compute_entity_overlap(entities)
        topic_familiarity = self._compute_topic_familiarity(topic_str)
        summary_familiarity = self._compute_summary_familiarity(episode_summary)
        content_uniqueness = self._compute_content_uniqueness(text, prior_concepts)

        novelty_from_novel_entities = len(entities) * self.entity_bonus * (1.0 - entity_overlap)
        topic_surprise = (1.0 - topic_familiarity) * self.topic_bonus
        summary_surprise = (1.0 - summary_familiarity) * 0.2

        base_surprise = max(
            novelty_from_novel_entities,
            topic_surprise,
            summary_surprise,
            content_uniqueness * 0.2,
        )

        surprise_score = min(1.0, base_surprise)

        if self._is_contradiction(text):
            surprise_score = min(1.0, surprise_score + self.contradiction_bonus)
            reasons.append("contradiction_detected")

        self._update_history(entities, topic_str, episode_summary)

        return round(surprise_score, 4)

    def compute_batch(self, texts: List[str]) -> List[float]:
        """Compute prediction error for a batch of texts."""
        return [self.compute(t) for t in texts]

    def _compute_entity_overlap(self, entities: List[str]) -> float:
        """How familiar are these entities based on recent history?"""
        if not entities or not self._recent_entities:
            return 0.0

        entity_set = set(e.lower() for e in entities)
        recent_set = set(e.lower() for e in self._recent_entities)

        if not entity_set:
            return 0.0

        overlap = len(entity_set & recent_set)
        return overlap / len(entity_set)

    def _compute_topic_familiarity(self, topic: str) -> float:
        """How familiar is this topic from recent history?"""
        if not topic or not self._recent_topics:
            return 0.0

        topic_lower = topic.lower()
        recent = [t.lower() for t in self._recent_topics]

        if topic_lower in recent:
            freq = recent.count(topic_lower)
            return min(1.0, freq * 0.25)
        return 0.0

    def _compute_summary_familiarity(self, summary: Optional[str]) -> float:
        """How familiar is this episode content from recent episodes?"""
        if not summary or not self._episode_summaries:
            return 0.0

        summary_words = set(summary.lower().split())
        best_match = 0.0

        for ep_summary in self._episode_summaries:
            ep_words = set(ep_summary.lower().split())
            if summary_words and ep_words:
                intersection = len(summary_words & ep_words)
                union = len(summary_words | ep_words)
                match = intersection / union if union > 0 else 0.0
                best_match = max(best_match, match)

        return best_match

    def _compute_content_uniqueness(
        self,
        text: str,
        prior_concepts: Optional[List[str]],
    ) -> float:
        """
        How unique is this content compared to what was recently seen?
        Uses simple word-level Jaccard against prior concept descriptions.
        """
        if not prior_concepts:
            return 0.5

        text_words = set(text.lower().split())
        if not text_words:
            return 0.0

        max_uniqueness = 0.0
        for concept_text in prior_concepts[-self.window_size:]:
            concept_words = set(concept_text.lower().split())
            intersection = len(text_words & concept_words)
            union = len(text_words | concept_words)
            if union > 0:
                jaccard = intersection / union
                uniqueness = 1.0 - jaccard
                max_uniqueness = max(max_uniqueness, uniqueness)

        return max_uniqueness

    def _is_contradiction(self, text: str) -> bool:
        """Detect explicit contradiction signals in text."""
        text_lower = text.lower()
        contradiction_signals = [
            ("actually", "no"), ("wait", "not"),
            ("i was wrong", "take it back"),
            ("on second thought", "changed my mind"),
            ("changed my mind", "instead"),
            ("not ", "anymore"), ("never mind", "i take it back"),
            ("i mean", "sorry"), ("i'm wrong", "apologies"),
            ("was wrong", "instead"), ("was wrong", "correction"),
        ]
        count = sum(1 for a, b in contradiction_signals if a in text_lower and b in text_lower)
        return count > 0

    def _infer_topic(self, text: str) -> Optional[str]:
        """Simple heuristic topic inference."""
        text_lower = text.lower()
        topic_keywords = {
            "work": ["meeting", "project", "deadline", "office", "boss", "colleague", "job", "career"],
            "personal": ["family", "home", "kids", "spouse", "parent", "friend"],
            "health": ["doctor", "exercise", "gym", "diet", "sick", "medicine", "hospital"],
            "finance": ["money", "bank", "invest", "salary", "budget", "expense", "debt"],
            "travel": ["trip", "flight", "hotel", "vacation", "destination", "travel"],
            "food": ["restaurant", "cook", "recipe", "dinner", "lunch", "meal"],
            "hobby": ["paint", "music", "read", "game", "sport", "hobby"],
            "news": ["news", "politics", "world", "event", "happened"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        return None

    def _update_history(
        self,
        entities: List[str],
        topic: Optional[str],
        episode_summary: Optional[str],
    ):
        """Update rolling history after processing."""
        for entity in (entities or []):
            self._recent_entities.append(entity)
        if topic:
            self._recent_topics.append(topic)
        if episode_summary:
            self._episode_summaries.append(episode_summary)

    def get_context(self) -> PredictionContext:
        """Expose current history for debugging/inspection."""
        return PredictionContext(
            recent_concepts=list(self._recent_concepts),
            recent_episodes=list(self._episode_summaries),
            topic_history=list(self._recent_topics),
            entity_history=list(self._recent_entities),
        )

    def reset(self):
        """Clear all history."""
        self._recent_concepts.clear()
        self._recent_topics.clear()
        self._recent_entities.clear()
        self._episode_summaries.clear()
