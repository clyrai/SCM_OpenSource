"""
ValueTagger: Multi-dimensional importance tagging
"""
from typing import Dict, List
from datetime import datetime, timedelta

from .models import Concept, ImportanceVector
from .time_utils import ensure_utc, utc_now


def get_concept_type(concept):
    """Helper to get concept type as string"""
    if isinstance(concept.type, str):
        return concept.type
    return concept.type.value


class ValueTagger:
    """
    Assigns multi-dimensional importance signals to concepts.
    Dimensions: novelty, emotional, task_relevance, repetition
    """

    def __init__(self):
        self.history = {}  # Track seen concepts for repetition scoring

    def tag(self, concept: Concept, context: Dict = None) -> ImportanceVector:
        """
        Tag a concept with importance vector.

        Args:
            concept: The concept to tag
            context: Optional context (task info, user preferences, etc.)

        Returns:
            ImportanceVector with scores for each dimension
        """
        context = context or {}

        novelty = self._compute_novelty(concept)
        emotional = self._compute_emotional(concept, context)
        task_relevance = self._compute_task_relevance(concept, context)
        repetition = self._compute_repetition(concept)

        return ImportanceVector(
            novelty=novelty,
            emotional=emotional,
            task_relevance=task_relevance,
            repetition=repetition
        )

    def _compute_novelty(self, concept: Concept) -> float:
        """
        How new is this concept compared to existing memory?
        Higher = more novel (new person, new preference, etc.)
        """
        concept_text = concept.description.lower()

        # Check if we've seen similar concepts
        for existing_desc, last_seen in self.history.items():
            # Simple word overlap check
            existing_words = set(existing_desc.split())
            current_words = set(concept_text.split())
            overlap = len(existing_words & current_words) / max(len(existing_words), len(current_words))

            if overlap > 0.7:
                # Similar concept exists, lower novelty
                # Decay novelty based on recency
                hours_since = (utc_now() - ensure_utc(last_seen)).total_seconds() / 3600
                novelty = max(0.1, 1.0 - (hours_since / 24) * 0.5)
                return novelty

        return 0.9  # Very novel if no similar found

    def _compute_emotional(self, concept: Concept, context: Dict) -> float:
        """
        Positive/negative emotional valence.
        Higher = positive, Lower = negative, 0 = neutral
        """
        text = concept.description.lower()
        context_text = str(context).lower()

        # Positive indicators
        positive_words = ['like', 'love', 'prefer', 'enjoy', 'great', 'awesome', 'happy', 'excited', 'good', 'nice']
        # Negative indicators
        negative_words = ['hate', 'dislike', 'avoid', 'never', 'bad', 'sad', 'angry', 'upset', 'worried', 'fear']

        score = 0.0

        for word in positive_words:
            if word in text or word in context_text:
                score += 0.3

        for word in negative_words:
            if word in text or word in context_text:
                score -= 0.3

        # Preference type tends to be positive (personal ownership)
        concept_type = get_concept_type(concept)
        if concept_type == 'preference':
            score += 0.2

        return max(-1.0, min(1.0, score))

    def _compute_task_relevance(self, concept: Concept, context: Dict) -> float:
        """
        How relevant is this for current task/goals?
        """
        concept_type = get_concept_type(concept)

        # If context contains current task, check relevance
        if 'task' in context:
            task = context['task'].lower()
            concept_desc = concept.description.lower()

            # Check for task-related keywords
            task_keywords = ['meeting', 'project', 'deadline', 'work', 'task', 'goal', 'schedule']
            for keyword in task_keywords:
                if keyword in concept_desc and keyword in task:
                    return 0.8

        # Personal info is often task-relevant for assistants
        if concept_type in ['person', 'preference', 'fact']:
            return 0.6

        return 0.3  # Default moderate relevance

    def _compute_repetition(self, concept: Concept) -> float:
        """
        How many times has this concept been reinforced?
        Higher = more repetition
        """
        concept_text = concept.description.lower()

        if concept_text in self.history:
            # Already seen - compute repetition score
            count = 1  # Would need to track counts properly
            return min(1.0, 0.3 + count * 0.1)

        return 0.0  # Never seen - no repetition yet

    def update_history(self, concept: Concept):
        """Record that we've seen this concept"""
        self.history[concept.description.lower()] = utc_now()

    def tag_batch(self, concepts: List[Concept], context: Dict = None) -> List[ImportanceVector]:
        """Tag multiple concepts at once"""
        return [self.tag(c, context) for c in concepts]

    def compute_grasp(
        self,
        concept: Concept,
        importance: ImportanceVector,
        salience: float,
        schema_overlap: float,
    ) -> float:
        """
        Compute one-shot grasp score for a concept.

        Higher grasp = concept is understood and encoded strongly
        even after single exposure.

        Args:
            concept: The concept to score
            importance: Pre-computed ImportanceVector
            salience: Pre-computed salience score from AttentionGate
            schema_overlap: How much this concept matches existing schemas

        Returns:
            float: grasp score in [0, 1]
        """
        clarity = self._compute_concept_clarity(concept)
        cognitive_load = self._compute_concept_load(concept)

        from .config import (
            GRASP_WEIGHT_SALIENCE,
            GRASP_WEIGHT_SCHEMA,
            GRASP_WEIGHT_CLARITY,
            GRASP_WEIGHT_COGNITIVE_LOAD,
        )

        grasp = (
            GRASP_WEIGHT_SALIENCE * salience
            + GRASP_WEIGHT_SCHEMA * schema_overlap
            + GRASP_WEIGHT_CLARITY * clarity
            - GRASP_WEIGHT_COGNITIVE_LOAD * cognitive_load
        )

        return max(0.0, min(1.0, grasp))

    def _compute_concept_clarity(self, concept: Concept) -> float:
        """Clarity for a concept: how well-defined is it?"""
        desc = concept.description
        if len(desc) < 5:
            return 0.1
        if len(desc) > 200:
            return 0.4

        has_type = concept.type is not None
        has_embedding = concept.embedding is not None

        clarity = 0.5
        if has_type:
            clarity += 0.2
        if has_embedding:
            clarity += 0.15
        if len(desc.split()) <= 10:
            clarity += 0.1

        return max(0.0, min(1.0, clarity))

    def _compute_concept_load(self, concept: Concept) -> float:
        """Cognitive load: how complex is this concept to encode?"""
        desc = concept.description
        words = desc.split()

        load = 0.3
        load += min(0.4, len(words) / 50.0)

        complex_indicators = ["maybe", "perhaps", "might", "could be", "uncertain"]
        if any(ind in desc.lower() for ind in complex_indicators):
            load += 0.15

        return max(0.0, min(1.0, load))
