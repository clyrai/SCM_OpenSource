"""
AttentionGate: Salience-based encoding intensity gate for HME
Decides HOW strongly to encode each piece of input based on
attention signals: novelty, relevance, emotional weight, repetition, and prediction error.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .models import (
    Concept,
    Episode,
    ImportanceVector,
    EncodeIntensity,
    EncodeResult,
    AttentionGateResult,
    GraspResult,
)
from .config import (
    SALIENT_ENCODE_THRESHOLD,
    NORMAL_ENCODE_THRESHOLD,
    SKIP_ENCODE_THRESHOLD,
    SALIENCE_WEIGHT_NOVELTY,
    SALIENCE_WEIGHT_TASK,
    SALIENCE_WEIGHT_EMOTIONAL,
    SALIENCE_WEIGHT_REPETITION,
    SALIENCE_WEIGHT_PREDICTION,
    GRASP_WEIGHT_SALIENCE,
    GRASP_WEIGHT_SCHEMA,
    GRASP_WEIGHT_CLARITY,
    GRASP_WEIGHT_COGNITIVE_LOAD,
    ONE_SHOT_GRASP_THRESHOLD,
)
from .prediction_error import PredictionErrorEngine


class AttentionGate:
    """
    Human-like selective attention for memory encoding.

    Decides encoding intensity based on salience signals:
    - Novelty: how new is this vs existing memory?
    - Task relevance: does this matter for current goals?
    - Emotional weight: positive or negative signal?
    - Repetition: has this been reinforced before?
    - Prediction error: is this surprising?

    Output is one of four encode intensities:
    - STRONG: high-salience → durable one-shot memory trace
    - NORMAL: medium-salience → standard encoding
    - WEAK: low-salience → minimal trace, prone to fast decay
    - SKIP: near-zero value → buffered only, not durable
    """

    def __init__(
        self,
        salient_threshold: float = SALIENT_ENCODE_THRESHOLD,
        normal_threshold: float = NORMAL_ENCODE_THRESHOLD,
        skip_threshold: float = SKIP_ENCODE_THRESHOLD,
        w_novelty: float = SALIENCE_WEIGHT_NOVELTY,
        w_task: float = SALIENCE_WEIGHT_TASK,
        w_emotional: float = SALIENCE_WEIGHT_EMOTIONAL,
        w_repetition: float = SALIENCE_WEIGHT_REPETITION,
        w_prediction: float = SALIENCE_WEIGHT_PREDICTION,
        one_shot_threshold: float = ONE_SHOT_GRASP_THRESHOLD,
        enable_prediction_error: bool = True,
    ):
        self.salient_threshold = salient_threshold
        self.normal_threshold = normal_threshold
        self.skip_threshold = skip_threshold
        self.w_novelty = w_novelty
        self.w_task = w_task
        self.w_emotional = w_emotional
        self.w_repetition = w_repetition
        self.w_prediction = w_prediction
        self.one_shot_threshold = one_shot_threshold
        self.enable_prediction_error = enable_prediction_error

        self.prediction_engine = PredictionErrorEngine()

        self._prior_concepts: List[str] = []

    def evaluate(
        self,
        text: str,
        importance: ImportanceVector,
        concept: Optional[Concept] = None,
        prior_concepts: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EncodeResult:
        """
        Evaluate a piece of input and return encoding decision.

        Args:
            text: Raw input text
            importance: Pre-computed ImportanceVector
            concept: Optional Concept object (for schema overlap)
            prior_concepts: List of prior concept descriptions (for novelty/schema)
            context: Optional context dict (task, interlocutor, session)

        Returns:
            EncodeResult with decision and all component scores
        """
        context = context or {}

        if prior_concepts:
            self._prior_concepts = prior_concepts

        salience = self._compute_salience(importance, text, context)
        prediction_error = 0.0
        schema_overlap = 0.0
        noise_penalty = 0.0

        if self.enable_prediction_error:
            prediction_error = self.prediction_engine.compute(
                text=text,
                topic=context.get("task"),
                prior_concepts=self._prior_concepts,
            )

        if concept and self._prior_concepts:
            schema_overlap = self._compute_schema_overlap(
                concept, self._prior_concepts
            )

        noise_penalty = self._estimate_noise_penalty(text)

        adjusted_salience = max(
            0.0,
            salience
            + self.w_prediction * prediction_error
            - noise_penalty
        )

        intensity, reason = self._classify(adjusted_salience, noise_penalty)

        return EncodeResult(
            should_encode=intensity != EncodeIntensity.SKIP,
            intensity=intensity,
            salience=round(adjusted_salience, 4),
            grasp=round(
                self._compute_grasp_score(
                    salience=adjusted_salience,
                    schema_overlap=schema_overlap,
                    text=text,
                ),
                4,
            ),
            prediction_error=round(prediction_error, 4),
            reason=reason,
            noise_penalty=round(noise_penalty, 4),
            schema_overlap=round(schema_overlap, 4),
        )

    def evaluate_episode(
        self,
        episode: Episode,
        prior_concepts: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EncodeResult:
        """
        Convenience method for evaluating a full Episode.

        Uses episode's raw_content and importance.
        """
        return self.evaluate(
            text=episode.raw_content,
            importance=episode.importance,
            prior_concepts=prior_concepts,
            context=context,
        )

    def evaluate_batch(
        self,
        texts: List[str],
        importances: List[ImportanceVector],
        prior_concepts: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[EncodeResult]:
        """
        Evaluate a batch of inputs efficiently.
        """
        results = []
        for text, importance in zip(texts, importances):
            result = self.evaluate(
                text=text,
                importance=importance,
                prior_concepts=prior_concepts,
                context=context,
            )
            results.append(result)
        return results

    def _compute_salience(
        self,
        importance: ImportanceVector,
        text: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Compute composite salience score from importance dimensions.

        Human brains weight task relevance and emotional content heavily.
        We follow the same pattern: task relevance > novelty > emotional > repetition.
        """
        novelty_comp = importance.novelty * self.w_novelty
        task_comp = importance.task_relevance * self.w_task
        emotional_comp = abs(importance.emotional) * self.w_emotional
        repetition_comp = importance.repetition * self.w_repetition

        raw_salience = novelty_comp + task_comp + emotional_comp + repetition_comp

        interlocutor_boost = 0.0
        if context.get("interlocutor"):
            interlocutor_boost = 0.05

        return min(1.0, raw_salience + interlocutor_boost)

    def _compute_grasp_score(
        self,
        salience: float,
        schema_overlap: float,
        text: str,
    ) -> float:
        """
        Compute grasp score: how well this input is understood in one shot.

        High grasp = clear, familiar-pattern input that creates strong
        one-shot memory even after single exposure.

        G = w_s*S + w_schema*SchemaOverlap + w_clarity*Clarity - w_load*CognitiveLoad
        """
        clarity = self._compute_clarity(text)
        cognitive_load = self._compute_cognitive_load(text)

        grasp = (
            GRASP_WEIGHT_SALIENCE * salience
            + GRASP_WEIGHT_SCHEMA * schema_overlap
            + GRASP_WEIGHT_CLARITY * clarity
            - GRASP_WEIGHT_COGNITIVE_LOAD * cognitive_load
        )

        return max(0.0, min(1.0, grasp))

    def _compute_clarity(self, text: str) -> float:
        """
        How linguistically clear is the input?
        Short, declarative sentences with named entities = high clarity.
        Complex multi-clause or filler-heavy sentences = low clarity.
        """
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
        avg_words_per_sentence = word_count / sentence_count

        has_entity = any(c.isupper() for c in text if c.isalpha())
        has_numbers = any(c.isdigit() for c in text)

        complex_connectors = [
            "however", "although", "nevertheless", "whereas",
            "therefore", "consequently", "furthermore", "moreover",
            "whereby", "hereinafter", "paradoxically",
        ]
        simple_connectors = [
            "and", "but", "or", "so", "then", "because", "when", "if",
        ]

        word_lower = text.lower()
        complex_count = sum(1 for w in complex_connectors if w in word_lower)
        simple_count = sum(1 for w in simple_connectors if w in word_lower)

        clarity = 0.5
        if 5 <= avg_words_per_sentence <= 25:
            clarity += 0.15
        if has_entity:
            clarity += 0.15
        if has_numbers:
            clarity += 0.1
        if complex_count >= 2:
            clarity -= 0.2
        if simple_count >= 2 and complex_count == 0:
            clarity += 0.1
        if word_count > 80:
            clarity -= 0.15
        if word_count > 150:
            clarity -= 0.15
        if avg_words_per_sentence > 40:
            clarity -= 0.2

        return max(0.0, min(1.0, clarity))

    def _compute_cognitive_load(self, text: str) -> float:
        """
        Estimate cognitive processing difficulty.
        High load = complex, long, or ambiguous input.
        """
        words = text.split()
        word_count = len(words)

        indicators_hard = [
            "however", "although", "nevertheless", "whereas",
            "therefore", "consequently", "furthermore", "moreover",
            "despite", "whereby", "hereinafter",
        ]
        indicators_easy = [
            "because", "so", "and", "but", "or",
            "then", "when", "if", "that",
        ]

        word_lower = text.lower()
        hard_count = sum(1 for w in indicators_hard if w in word_lower)
        easy_count = sum(1 for w in indicators_easy if w in word_lower)

        load = 0.3
        load += min(0.3, word_count / 200.0)
        load += min(0.2, hard_count * 0.05)
        load -= min(0.2, easy_count * 0.05)

        return max(0.0, min(1.0, load))

    def _compute_schema_overlap(
        self,
        concept: Concept,
        prior_concepts: List[str],
    ) -> float:
        """
        How much does this concept overlap with existing memory schemas?
        High overlap = familiar pattern = faster encoding.
        """
        if not prior_concepts or not concept:
            return 0.0

        concept_words = set(concept.description.lower().split())
        if not concept_words:
            return 0.0

        max_overlap = 0.0
        for prior in prior_concepts[-10:]:
            prior_words = set(prior.lower().split())
            if prior_words:
                intersection = len(concept_words & prior_words)
                union = len(concept_words | prior_words)
                jaccard = intersection / union if union > 0 else 0.0
                max_overlap = max(max_overlap, jaccard)

        return max_overlap

    def _estimate_noise_penalty(self, text: str) -> float:
        """
        Estimate how "noisy" this input is.
        High noise penalty reduces encoding strength.
        """
        text_lower = text.lower()
        word_count = len(text.split())

        filler_words = [
            "um", "uh", "like", "you know", "basically",
            "actually", "literally", "honestly", "so basically",
            "i mean", "you see", "right?",
        ]
        filler_count = sum(1 for fw in filler_words if fw in text_lower)

        punctuation_ratio = (text.count(".") + text.count("!") + text.count("?")) / max(1, word_count)

        filler_penalty = min(0.3, filler_count * 0.08)
        complexity_penalty = 0.0
        if word_count > 80:
            complexity_penalty = min(0.15, (word_count - 80) / 200.0)
        low_punct_penalty = 0.0
        if punctuation_ratio < 0.1 and word_count > 20:
            low_punct_penalty = 0.1

        return min(0.5, filler_penalty + complexity_penalty + low_punct_penalty)

    def _classify(
        self,
        salience: float,
        noise_penalty: float,
    ) -> tuple[EncodeIntensity, str]:
        """Map salience score to encode intensity with reasoning."""
        if salience >= self.salient_threshold and noise_penalty < 0.2:
            return EncodeIntensity.STRONG, (
                f"High salience ({salience:.3f}) above threshold "
                f"({self.salient_threshold}) with low noise ({noise_penalty:.3f})"
            )
        elif salience >= self.normal_threshold:
            return EncodeIntensity.NORMAL, (
                f"Medium salience ({salience:.3f}) in normal range "
                f"[{self.normal_threshold}, {self.salient_threshold})"
            )
        elif salience >= self.skip_threshold:
            return EncodeIntensity.WEAK, (
                f"Low salience ({salience:.3f}) below normal threshold "
                f"({self.normal_threshold}), weak encoding"
            )
        else:
            return EncodeIntensity.SKIP, (
                f"Very low salience ({salience:.3f}) below skip threshold "
                f"({self.skip_threshold}), buffering only"
            )

    def compute_grasp_result(
        self,
        text: str,
        salience: float,
        schema_overlap: float,
    ) -> GraspResult:
        """Compute and return detailed grasp score breakdown."""
        clarity = self._compute_clarity(text)
        cognitive_load = self._compute_cognitive_load(text)

        grasp = (
            GRASP_WEIGHT_SALIENCE * salience
            + GRASP_WEIGHT_SCHEMA * schema_overlap
            + GRASP_WEIGHT_CLARITY * clarity
            - GRASP_WEIGHT_COGNITIVE_LOAD * cognitive_load
        )
        grasp = max(0.0, min(1.0, grasp))

        return GraspResult(
            grasp_score=round(grasp, 4),
            schema_overlap=round(schema_overlap, 4),
            clarity=round(clarity, 4),
            cognitive_load=round(cognitive_load, 4),
            one_shot_capable=grasp >= self.one_shot_threshold,
        )

    def update_prior_concepts(self, concepts: List[str]):
        """Update the concept history used for novelty and schema overlap."""
        self._prior_concepts = concepts[-50:]
