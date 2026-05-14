"""
Memory API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core.models import Episode, Concept, ImportanceVector, MemoryState, ConceptType
from ..core.encoder import MeaningEncoder
from ..core.value_tagger import ValueTagger
from ..core.working_memory import WorkingMemory
from ..core.long_term_memory import LongTermMemory
from ..llm import LLMExtractor

router = APIRouter(prefix="/memory", tags=["memory"])

# Global instances (initialized at startup)
encoder: Optional[MeaningEncoder] = None
value_tagger: Optional[ValueTagger] = None
working_memory: Optional[WorkingMemory] = None
long_term_memory: Optional[LongTermMemory] = None
llm_extractor: Optional[LLMExtractor] = None


def init_memory_components():
    """Initialize memory components with LLM"""
    global encoder, value_tagger, working_memory, long_term_memory, llm_extractor

    # Initialize LLM extractor (llama3.2 works reliably for text extraction)
    llm_extractor = LLMExtractor(model="llama3.2:latest")
    health = llm_extractor.health_check()
    print(f"LLM Status: {health}")

    # Initialize encoder with LLM
    encoder = MeaningEncoder(llm=llm_extractor)
    value_tagger = ValueTagger()
    working_memory = WorkingMemory()
    long_term_memory = LongTermMemory()

    # Try to load existing memory from DB
    try:
        long_term_memory.load_from_db()
    except Exception as e:
        print(f"Could not load from DB: {e}")


class MemoryInput(BaseModel):
    """Input for storing a memory"""
    text: str = Field(..., description="Text to encode into memory")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    source: str = Field(default="user", description="Source of input (user, assistant, system)")


class MemoryResponse(BaseModel):
    """Response after storing memory"""
    success: bool
    concepts_created: int
    episode_id: str
    message: str


class QueryInput(BaseModel):
    """Query input"""
    query: str = Field(..., description="Query text")
    limit: int = Field(default=5, description="Max results to return")


class QueryResponse(BaseModel):
    """Query response"""
    results: List[Dict[str, Any]]
    memory_stats: Dict[str, Any]


class SleepStatus(BaseModel):
    """Sleep status response"""
    should_sleep: bool
    entropy: float
    conflict: float
    working_memory_size: int
    last_sleep: Optional[datetime] = None


@router.post("/", response_model=MemoryResponse)
async def store_memory(input: MemoryInput):
    """
    Store new information in memory.
    Extracts concepts, assigns importance, and stores in working memory.
    """
    global encoder, value_tagger, working_memory, long_term_memory

    try:
        # 1. Extract concepts from text
        concepts = encoder.extract(input.text)

        # 2. Tag with importance values
        for concept in concepts:
            concept.importance = value_tagger.tag(concept, input.context)
            value_tagger.update_history(concept)

        # 3. Create episode in working memory
        episode = Episode(
            concept_ids=[c.id for c in concepts],
            raw_content=input.text,
            context=input.context or {},
            importance=concepts[0].importance if concepts else ImportanceVector(),
            source=input.source
        )
        working_memory.store(episode)

        # 4. Also add concepts to long-term memory
        for concept in concepts:
            long_term_memory.add_concept(concept)

        return MemoryResponse(
            success=True,
            concepts_created=len(concepts),
            episode_id=episode.id,
            message=f"Stored {len(concepts)} concepts from: {input.text[:50]}..."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=QueryResponse)
async def query_memory(query: str, limit: int = 5):
    """
    Query memory for relevant information.
    """
    global long_term_memory, working_memory

    try:
        # Search long-term memory
        ltm_results = long_term_memory.search_by_text(query, limit=limit)

        # Also check working memory
        wm_episodes = working_memory.retrieve(query, limit=limit)

        # Combine results
        results = []

        for concept in ltm_results:
            results.append({
                'type': 'concept',
                'id': concept.id,
                'description': concept.description,
                'concept_type': concept.type.value,
                'importance': concept.importance.overall,
                'created_at': concept.created_at.isoformat(),
                'source': 'long_term'
            })

        for episode in wm_episodes:
            results.append({
                'type': 'episode',
                'id': episode.id,
                'raw_content': episode.raw_content,
                'timestamp': episode.timestamp.isoformat(),
                'importance': episode.importance.overall,
                'source': 'working'
            })

        # Sort by importance
        results.sort(key=lambda x: x.get('importance', 0), reverse=True)

        return QueryResponse(
            results=results[:limit],
            memory_stats={
                'total_concepts': len(long_term_memory.graph.nodes()),
                'working_memory_size': working_memory.size(),
                'suppressed_count': sum(
                    1 for n in long_term_memory.graph.nodes()
                    if long_term_memory.graph.nodes[n].get('state') == 'suppressed'
                )
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get memory statistics"""
    global long_term_memory, working_memory

    try:
        ltm_stats = long_term_memory.get_stats()
        return {
            'long_term': ltm_stats,
            'working_memory_size': working_memory.size(),
            'working_memory_capacity': working_memory.capacity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/working")
async def clear_working_memory():
    """Clear working memory (used during sleep)"""
    global working_memory

    try:
        working_memory.clear()
        return {'success': True, 'message': 'Working memory cleared'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate")
async def consolidate_to_long_term():
    """
    Manually trigger consolidation from working to long-term memory.
    """
    global working_memory, long_term_memory

    try:
        consolidated_count = 0

        # Get all episodes from working memory
        episodes = working_memory.get_all()

        for episode in episodes:
            # Get concepts from long-term memory
            for concept_id in episode.concept_ids:
                concept = long_term_memory.get_concept(concept_id)
                if concept:
                    # Update importance based on episode
                    concept.importance = episode.importance
                    concept.strength += 0.1  # Strengthen
                    consolidated_count += 1

        return {
            'success': True,
            'consolidated': consolidated_count,
            'episodes_processed': len(episodes)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))