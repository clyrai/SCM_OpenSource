from typing import List, Dict, Tuple, Optional
import re
import time
from datetime import datetime

from ..core.models import Concept, Episode, MemoryState
from ..core.memory_scoring import compute_consolidation_score
from ..core.working_memory import WorkingMemory
from ..core.long_term_memory import LongTermMemory
from ..core.time_utils import utc_now
from ..core import config as _config_module


def _state_value(concept: Concept) -> str:
    state = getattr(concept, "state", MemoryState.ACTIVE.value)
    return state.value if hasattr(state, "value") else str(state)


class SpreadingActivationRetriever:
    """
    Implements cue-driven associative recall via bounded spreading activation.

    Unlike ranked search, this propagates activation energy from seed concepts
    outward through the association graph. Activation decays per step and is
    gated by context relevance (session, task, person, recency).

    Formula per step:
        A_j(t+1) = decay * A_j(t) + sum_i(A_i(t) * W_ij * gate_context)

    No LLM calls in the critical path — all heuristics.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        long_term_memory: LongTermMemory,
        spreading_steps: int = 3,
        spreading_decay: float = 0.45,
        activation_threshold: float = 0.05,
        max_candidates: int = 50,
        context_gate_weight: float = 0.30,
        encoder: Optional[object] = None,
        vector_index: Optional[object] = None,
    ):
        self.working_memory = working_memory
        self.long_term_memory = long_term_memory
        self.spreading_steps = spreading_steps
        self.spreading_decay = spreading_decay
        self.activation_threshold = activation_threshold
        self.max_candidates = max_candidates
        self.context_gate_weight = context_gate_weight
        # Optional encoder for embedding-based seed fallback. When token
        # overlap finds no seeds (e.g., the query phrases a property like
        # "where do I work?" but the stored fact is "I'm at Atlas Labs"),
        # we compute query+concept embeddings and pick top-K by cosine
        # similarity. Without an encoder, we fall back to recent high-
        # importance concepts (legacy behavior).
        self._encoder = encoder
        # Optional VectorIndex for sub-linear ANN seed selection. When
        # present, the embedding-similarity fallback skips the O(n) scan
        # and queries the index directly. The index is owned by
        # LongTermMemory and kept in sync as concepts are added/updated.
        self._vector_index = vector_index

    def retrieve(
        self,
        query: str,
        context_tags: Optional[Dict] = None,
        max_seed_concepts: int = 8,
        max_activation_steps: Optional[int] = None,
    ) -> Tuple[List[Concept], Dict]:
        """
        Main entry point: cue-driven associative recall.

        Args:
            query: The user query string (used for cue extraction).
            context_tags: Optional dict with keys like session_id, task_id,
                         person, time_window for context gating.
            max_seed_concepts: Cap on initial seed set size.
            max_activation_steps: Override for spreading_steps.

        Returns:
            Tuple of (activated_concepts sorted by activation, retrieval_stats).
        """
        context_tags = context_tags or {}
        steps = max_activation_steps if max_activation_steps is not None else self.spreading_steps

        cues = self._extract_cues(query)
        seeds = self._select_seeds(cues, max_seed_concepts, query=query)
        activated, stats = self._propagate_activation(seeds, context_tags, steps, cues)

        stats['cues'] = cues
        stats['seeds'] = len(seeds)
        stats['steps'] = steps
        stats['total_concepts_activated'] = len(activated)

        return activated, stats

    def _extract_cues(self, query: str) -> List[str]:
        """Extract semantic cue tokens from the query."""
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) >= 3]
        stop = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'her', 'was', 'one', 'our', 'out', 'has', 'have',
                'had', 'what', 'when', 'where', 'which', 'their', 'there',
                'really', 'love'}
        return [t for t in tokens if t not in stop]

    def _select_seeds(
        self,
        cues: List[str],
        limit: int,
        query: str = "",
    ) -> List[Tuple[Concept, float]]:
        """
        Select seed concepts by token overlap with cues, with an embedding-
        similarity fallback when token overlap finds nothing.

        Returns list of (concept, initial_activation) sorted by activation desc.
        """
        now = utc_now()
        if not cues:
            return self._fallback_seeds(limit, now=now)

        all_concepts = self.long_term_memory.get_all_concepts(include_suppressed=False)
        seed_scores = []

        for concept in all_concepts:
            # Phase 7 brutal-fix: never seed on a superseded concept. The
            # versioning system has already declared this fact stale.
            if not getattr(concept, "is_current_version", True):
                continue
            score = self._cue_match_score(concept, cues)
            if score > 0:
                consolidation = compute_consolidation_score(concept, now=now)
                combined = score * 0.6 + consolidation * 0.4
                seed_scores.append((concept, combined, score, consolidation))

        seed_scores.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        if seed_scores and seed_scores[0][1] > 0:
            return [(concept, score) for concept, score, _, _ in seed_scores[:limit]]

        # Embedding-similarity fallback: cue-token overlap missed (the query's
        # tokens don't appear in any stored concept's description). If we
        # have an encoder, use semantic similarity; otherwise fall back to
        # recency+importance.
        if query and self._encoder is not None:
            embed_seeds = self._embedding_seeds(query, all_concepts, limit, now)
            if embed_seeds:
                return embed_seeds

        return self._fallback_seeds(limit, now=now)

    def _cue_match_score(self, concept: Concept, cues: List[str]) -> float:
        """Compute how well a concept matches the query cues.

        Phase 6 paraphrase moves the original raw text into
        context_tags['original_description']. We match against both the
        (possibly rewritten) description and the preserved original so
        verbatim entity strings remain seedable after consolidation.
        """
        text_blob = (concept.description or "").lower()
        tags = concept.context_tags or {}
        orig = tags.get("original_description")
        if isinstance(orig, str) and orig:
            text_blob = text_blob + " " + orig.lower()
        desc_tokens = set(
            re.sub(r'[^\w\s]', ' ', text_blob).split()
        )
        matched_cues = sum(1 for c in cues if c in desc_tokens)
        if not cues:
            return 0.0
        return matched_cues / len(cues)

    # Maximum candidates to score in the embedding fallback. Beyond this,
    # we use recency to pre-filter — full cosine over a 1000-concept LTM
    # would otherwise dominate per-query latency.
    _EMBEDDING_FALLBACK_MAX_CANDIDATES = 60

    def _embedding_seeds(
        self,
        query: str,
        all_concepts: List[Concept],
        limit: int,
        now: datetime,
    ) -> List[Tuple[Concept, float]]:
        """Pick top-K concepts by cosine similarity of stored embedding to query.

        Bounded by a per-call candidate cap so concept-graph growth doesn't
        translate into linear query-latency regression. Candidates are
        pre-filtered by recency + importance so we score the most relevant
        slice first.
        """
        try:
            # mode="query" so instruction-tuned embedding models (nomic-embed-text,
            # bge) get the right task prefix — concept embeddings were created
            # with mode="document" at ingest time.
            q_embed = self._encoder._get_embedding(query, mode="query")
        except Exception:
            try:
                q_embed = self._encoder._get_embedding(query)
            except Exception:
                return []
        if not q_embed:
            return []

        # Fast path: if a VectorIndex is wired up, ask it directly. ANN
        # search is O(log n) for FAISS/HNSW backends and O(n) but cache-
        # friendly for the InMemory backend. Either way, we skip the
        # eligible-candidate sort + cosine loop below.
        if self._vector_index is not None:
            try:
                hits = self._vector_index.search(
                    embedding=q_embed,
                    top_k=limit,
                    min_score=0.0,
                )
                if hits:
                    out: List[Tuple[Concept, float]] = []
                    for h in hits:
                        c = self.long_term_memory.get_concept(h.concept_id)
                        if c is None:
                            continue
                        if not getattr(c, "is_current_version", True):
                            continue
                        out.append((c, max(0.05, float(h.score))))
                    if out:
                        return out
            except Exception:
                # Fall back to the legacy O(n) path on any index error.
                pass

        # Pre-filter: keep only concepts that are current-version, have an
        # embedding, and rank in the top _EMBEDDING_FALLBACK_MAX_CANDIDATES
        # by (recency, importance). This bounds the cosine work to O(60).
        eligible = []
        for c in all_concepts:
            if not getattr(c, "is_current_version", True):
                continue
            if not getattr(c, "embedding", None):
                continue
            eligible.append(c)
        if not eligible:
            return []

        def _recency_key(c):
            ts = getattr(c, "last_accessed", None) or getattr(c, "created_at", None)
            return ts.timestamp() if ts else 0.0

        eligible.sort(
            key=lambda c: (
                _recency_key(c),
                getattr(c.importance, "overall", 0.0)
                if getattr(c, "importance", None) is not None else 0.0,
            ),
            reverse=True,
        )
        candidates = eligible[: self._EMBEDDING_FALLBACK_MAX_CANDIDATES]

        # Cosine similarity (manual to avoid numpy dependency in the hot path).
        def cos(a, b):
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = na = nb = 0.0
            for x, y in zip(a, b):
                dot += x * y
                na += x * x
                nb += y * y
            if na == 0 or nb == 0:
                return 0.0
            return dot / ((na * nb) ** 0.5)

        scored = []
        for c in candidates:
            sim = cos(q_embed, c.embedding)
            if sim <= 0:
                continue
            scored.append((c, sim))
        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(c, max(0.05, sim)) for c, sim in scored[:limit]]

    def _fallback_seeds(self, limit: int, now: Optional[datetime] = None) -> List[Tuple[Concept, float]]:
        """When no cues match, use most recently accessed high-importance concepts."""
        now = now or utc_now()
        all_concepts = self.long_term_memory.get_all_concepts(include_suppressed=False)
        scored = []
        for c in all_concepts:
            consolidation = compute_consolidation_score(c, now=now)
            recency = self._recency_factor(c.last_accessed)
            scored.append((c, 0.75 * consolidation + 0.25 * recency))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _propagate_activation(
        self,
        seeds: List[Tuple[Concept, float]],
        context_tags: Dict,
        steps: int,
        cues: Optional[List[str]] = None,
    ) -> Tuple[List[Concept], Dict]:
        """
        Bounded spreading activation from seed concepts.

        Returns (activated_concepts, stats) where concepts are sorted by
        final activation value descending.
        """
        if not seeds:
            return [], {'propagation_steps': 0, 'activated_beyond_seeds': 0}

        activation: Dict[str, float] = {c.id: init for c, init in seeds}

        for step in range(steps):
            new_activation: Dict[str, float] = {}
            for concept_id, current_a in activation.items():
                decayed = current_a * self.spreading_decay
                if decayed < self.activation_threshold:
                    new_activation[concept_id] = decayed
                    continue

                neighbors = self.long_term_memory.get_related_concepts(concept_id, depth=1)
                gate = self._context_gate(concept_id, context_tags)

                for neighbor in neighbors:
                    if _state_value(neighbor) in {
                        MemoryState.SUPPRESSED.value,
                        MemoryState.ARCHIVED.value,
                    }:
                        continue
                    # Phase 7 brutal-fix: also drop superseded concepts
                    # (versioning has marked is_current_version=False but the
                    # state may still be ACTIVE if forgetting hasn't run yet).
                    # Without this, a contradiction-versioned old concept
                    # leaks into propagation and ranks as if it were current.
                    if not getattr(neighbor, "is_current_version", True):
                        continue
                    edge_strength = self._get_edge_strength(concept_id, neighbor.id)
                    contribution = current_a * edge_strength * gate
                    new_activation[neighbor.id] = (
                        new_activation.get(neighbor.id, 0.0) + contribution
                    )

                new_activation[concept_id] = max(new_activation.get(concept_id, 0.0), decayed)

            activation = new_activation

        activated_concepts = []
        seen_ids = {c.id for c, _ in seeds}
        beyond_seeds = 0

        for concept_id, act in activation.items():
            if act >= self.activation_threshold:
                concept = self.long_term_memory.get_concept(concept_id)
                if concept:
                    # Drop superseded concepts at the final filter too
                    if not getattr(concept, "is_current_version", True):
                        continue
                    if concept_id not in seen_ids:
                        beyond_seeds += 1
                    activated_concepts.append((concept, act))

        # Always preserve cue-matched seeds in the result, even if their
        # activation decayed below threshold over `steps` iterations. They
        # were the original cue matches and are by definition relevant —
        # the threshold is meant to filter PROPAGATION noise, not seeds.
        # Skip superseded seeds (versioned-out facts).
        included_ids = {c.id for c, _ in activated_concepts}
        for seed_concept, seed_init in seeds:
            if seed_concept.id in included_ids:
                continue
            if not getattr(seed_concept, "is_current_version", True):
                continue
            seed_act = activation.get(seed_concept.id, seed_init)
            activated_concepts.append((seed_concept, seed_act))

        cue_terms = cues or []
        # Relevance ranking: cue-match dominates (the user asked about THIS
        # topic, so concepts mentioning the topic should rank first).
        # Consolidation acts as a tiebreaker for equally-relevant concepts.
        # Bug fix: previously 0.85 * consolidation + 0.15 * cue_match made
        # well-rehearsed background concepts (e.g., self-model facts)
        # outrank the actual cue-matching answer to a query.
        activated_concepts.sort(
            key=lambda x: (
                0.70 * self._cue_match_score(x[0], cue_terms)
                + 0.30 * compute_consolidation_score(x[0])
            ),
            reverse=True,
        )
        concepts_only = [c for c, _ in activated_concepts[: self.max_candidates]]

        stats = {
            'propagation_steps': steps,
            'activated_beyond_seeds': beyond_seeds,
            'final_activation_count': len(concepts_only),
        }
        return concepts_only, stats

    def _context_gate(self, concept_id: str, context_tags: Dict) -> float:
        """
        Compute a [0, 1] context relevance multiplier for a concept.

        Factors: session match, task match, person match, recency.
        """
        concept = self.long_term_memory.get_concept(concept_id)
        if not concept:
            return 1.0

        score = 1.0

        concept_tags = concept.context_tags or {}

        if 'session_id' in context_tags and 'session_id' in concept_tags:
            ctx_sess = context_tags.get('session_id')
            con_sess = concept_tags.get('session_id')
            if ctx_sess and con_sess and ctx_sess == con_sess:
                score *= 1.2
            elif ctx_sess and con_sess:
                score *= 0.7

        # Person match: skip silently if either side is missing or non-string.
        # Bug fix: when working_memory is empty after consolidation,
        # context_tags['person'] is None and .lower() crashes the gate.
        ctx_person = context_tags.get('person')
        con_person = concept_tags.get('person')
        if isinstance(ctx_person, str) and isinstance(con_person, str):
            if ctx_person.lower() == con_person.lower():
                score *= 1.3

        recency = self._recency_factor(concept.last_accessed)
        score *= (0.6 + 0.4 * recency)

        return min(1.5, max(0.3, score))

    def _recency_factor(self, last_accessed) -> float:
        """Recency factor in [0, 1]; newer = higher."""
        if last_accessed is None:
            return 0.5
        current_time = self._resolve_current_time()
        if isinstance(last_accessed, datetime):
            last_accessed = last_accessed.timestamp()
        age = max(0.0, current_time - float(last_accessed))
        hour = 3600.0
        if age <= hour:
            return 1.0
        if age >= 24 * hour:
            return 0.2
        return 1.0 - (age - hour) / (23 * hour)

    @staticmethod
    def _resolve_current_time() -> float:
        """
        Resolve current time for recency calculations.

        Benchmarks/tests may inject `_config_module.current_time`; production
        falls back to wall clock when that override is unset.
        """
        configured = getattr(_config_module, "current_time", 0.0)
        try:
            configured_value = float(configured or 0.0)
        except Exception:
            configured_value = 0.0
        return configured_value if configured_value > 0.0 else time.time()

    def _get_edge_strength(self, from_id: str, to_id: str) -> float:
        """Get edge strength between two concepts from LTM graph."""
        graph = self.long_term_memory.graph
        if graph.has_edge(from_id, to_id):
            return graph[from_id][to_id].get('strength', 0.5)
        return 0.5

    def retrieve_with_recency_boost(
        self, query: str, context_tags: Optional[Dict] = None, recency_boost: float = 0.2
    ) -> Tuple[List[Concept], Dict]:
        """
        Variant: retrieve with additional recency bias on top of spreading activation.
        """
        concepts, stats = self.retrieve(query, context_tags)
        if not concepts:
            return concepts, stats

        denom = max(1, len(concepts))
        # Approximate baseline activation from the original retrieval order.
        base_activation = {
            concept.id: 1.0 - (idx / denom)
            for idx, concept in enumerate(concepts)
        }
        boosted_activation = {
            concept.id: base_activation[concept.id] + recency_boost * self._recency_factor(concept.last_accessed)
            for concept in concepts
        }

        concepts_sorted = sorted(
            concepts,
            key=lambda c: (
                self._get_activation(c.id, boosted_activation),
                (c.importance.overall if c.importance else 0.0)
            ),
            reverse=True
        )
        stats_with_boost = dict(stats)
        stats_with_boost["recency_boost_applied"] = True
        stats_with_boost["recency_boost_weight"] = recency_boost
        stats_with_boost["max_boosted_activation"] = max(boosted_activation.values()) if boosted_activation else 0.0
        return concepts_sorted, stats_with_boost

    def _get_activation(
        self, concept_id: str, activation_map: Dict[str, float]
    ) -> float:
        return float(activation_map.get(concept_id, 0.0))
