"""
Memory Retriever: Intelligent context retrieval for conversations
Builds relevant memory context from Working Memory + Long-Term Memory
"""
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from ..core.models import Concept, Episode, MemoryState
from ..core.memory_scoring import compute_consolidation_score
from ..core.working_memory import WorkingMemory
from ..core.long_term_memory import LongTermMemory


def _state_value(concept: Concept) -> str:
    state = getattr(concept, "state", MemoryState.ACTIVE.value)
    return state.value if hasattr(state, "value") else str(state)


class MemoryRetriever:
    """
    Retrieves the most relevant memories for a given conversation context.

    Strategy:
    1. Working Memory: Always include recent episodes (short-term context)
    2. LTM Semantic Search: Find concepts similar to current query
    3. LTM Graph Traversal: Follow relations from retrieved concepts
    4. Importance Boost: High-importance memories get priority
    5. Recency Decay: Older memories fade unless highly important
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        long_term_memory: LongTermMemory,
        max_context_tokens: int = 1500,  # Approximate token budget
        working_memory_weight: float = 0.4,
        semantic_weight: float = 0.35,
        graph_weight: float = 0.25
    ):
        self.working_memory = working_memory
        self.long_term_memory = long_term_memory
        self.max_context_tokens = max_context_tokens
        self.working_memory_weight = working_memory_weight
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight

    def retrieve_context(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        max_working_memories: int = 5,
        max_semantic_concepts: int = 5,
        max_graph_depth: int = 1
    ) -> Tuple[str, Dict]:
        """
        Retrieve relevant memory context for a conversation query.

        Returns:
            Tuple of (context_string, retrieval_stats)
        """
        stats = {
            'wm_retrieved': 0,
            'ltm_semantic': 0,
            'ltm_graph': 0,
            'total_tokens_estimate': 0
        }

        context_parts = []

        # 1. Working Memory - Recent context (always included)
        wm_context = self._get_working_memory_context(max_working_memories)
        if wm_context:
            context_parts.append(f"## Recent Conversation\n{wm_context}")
            stats['wm_retrieved'] = max_working_memories

        # 2. LTM Semantic Search
        semantic_concepts = []
        if query_embedding:
            semantic_concepts = self.long_term_memory.search_by_embedding(
                query_embedding, limit=max_semantic_concepts
            )
        else:
            semantic_concepts = self.long_term_memory.search_by_text(
                query, limit=max_semantic_concepts
            )

        if semantic_concepts:
            semantic_context = self._format_concepts(semantic_concepts, "Relevant Memories")
            context_parts.append(semantic_context)
            stats['ltm_semantic'] = len(semantic_concepts)

        # 3. Graph Traversal - Related concepts
        graph_concepts = self._get_graph_context(semantic_concepts, max_graph_depth)
        if graph_concepts:
            # Filter out duplicates from semantic search
            seen_ids = {c.id for c in semantic_concepts}
            unique_graph = [c for c in graph_concepts if c.id not in seen_ids]
            if unique_graph:
                graph_context = self._format_concepts(unique_graph[:5], "Related Concepts")
                context_parts.append(graph_context)
                stats['ltm_graph'] = len(unique_graph)

        # 4. High-Importance Memories (always worth surfacing)
        important_memories = self._get_important_memories(threshold=0.7, max_count=3)
        if important_memories:
            # Filter out already-included
            seen_ids = {c.id for c in semantic_concepts + graph_concepts}
            unique_important = [c for c in important_memories if c.id not in seen_ids]
            if unique_important:
                important_context = self._format_concepts(
                    unique_important, "Important Things to Remember"
                )
                context_parts.append(important_context)

        # Combine and estimate tokens (rough: 1 token ≈ 4 chars)
        full_context = "\n\n".join(context_parts)
        stats['total_tokens_estimate'] = len(full_context) // 4

        return full_context, stats

    def _get_working_memory_context(self, limit: int) -> str:
        """Get recent episodes from working memory"""
        episodes = self.working_memory.retrieve(limit=limit)
        if not episodes:
            return ""

        lines = []
        for ep in episodes:
            role = "User" if ep.source == "user" else "Assistant"
            lines.append(f"{role}: {ep.raw_content}")

        return "\n".join(lines)

    def _get_graph_context(
        self,
        seed_concepts: List[Concept],
        depth: int
    ) -> List[Concept]:
        """Traverse graph from seed concepts to find related memories"""
        related = []
        seen = set()

        for concept in seed_concepts:
            neighbors = self.long_term_memory.get_related_concepts(
                concept.id, depth=depth
            )
            for neighbor in neighbors:
                if neighbor.id not in seen and _state_value(neighbor) not in {
                    MemoryState.SUPPRESSED.value,
                    MemoryState.ARCHIVED.value,
                }:
                    seen.add(neighbor.id)
                    related.append(neighbor)

        # Sort by importance
        related.sort(
            key=lambda c: (
                compute_consolidation_score(c),
                c.importance.overall if c.importance else 0,
            ),
            reverse=True,
        )
        return related[:8]

    def _get_important_memories(self, threshold: float, max_count: int) -> List[Concept]:
        """Get high-importance concepts that should always be remembered"""
        important = []
        all_concepts = self.long_term_memory.get_all_concepts(include_suppressed=False)

        for concept in all_concepts:
            if concept.importance and concept.importance.overall >= threshold:
                important.append(concept)

        # Sort by importance, then recency
        important.sort(
            key=lambda c: (
                c.importance.overall,
                compute_consolidation_score(c),
                c.last_accessed,
            ),
            reverse=True
        )

        return important[:max_count]

    def _format_concepts(self, concepts: List[Concept], section_title: str) -> str:
        """Format a list of concepts for the context prompt"""
        lines = [f"## {section_title}"]

        for concept in concepts:
            desc = concept.description
            imp = f"(importance: {concept.importance.overall:.2f})" if concept.importance else ""
            lines.append(f"- {desc} {imp}")

        return "\n".join(lines)

    def get_memory_stats(self) -> Dict:
        """Get current memory statistics for display"""
        ltm_stats = self.long_term_memory.get_stats()

        return {
            'working_memory_size': self.working_memory.size(),
            'working_memory_capacity': self.working_memory.capacity,
            'long_term_concepts': ltm_stats.get('total_concepts', 0),
            'long_term_relations': ltm_stats.get('total_relations', 0),
            'suppressed_count': ltm_stats.get('suppressed_count', 0),
            'working_memory_full': self.working_memory.is_full()
        }
