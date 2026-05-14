"""
SleepAI Core Data Models
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import uuid

from .time_utils import utc_now


class ConceptType(str, Enum):
    """Types of concepts in memory"""
    PERSON = "person"
    PREFERENCE = "preference"
    FACT = "fact"
    EVENT = "event"
    RELATION = "relation"
    OBJECT = "object"
    LOCATION = "location"
    ABSTRACT = "abstract"


class PredicateType(str, Enum):
    """Types of relations between concepts"""
    HAS_PROPERTY = "has_property"
    PREFERS = "prefers"
    LOCATED_AT = "located_at"
    PART_OF = "part_of"
    CAUSED_BY = "caused_by"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    REFERENTIAL = "referential"
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"
    SIMILAR_TO = "similar_to"
    EXPERIENCED = "experienced"


class MemoryState(str, Enum):
    """State of memory in sleep cycle"""
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"
    ARCHIVED = "archived"
    SUPPRESSED = "suppressed"


class ImportanceVector(BaseModel):
    """Multi-dimensional importance signal for concepts"""
    novelty: float = Field(default=0.5, ge=0, le=1)
    emotional: float = Field(default=0.0, ge=-1, le=1)
    task_relevance: float = Field(default=0.5, ge=0, le=1)
    repetition: float = Field(default=0.5, ge=0, le=1)

    @property
    def overall(self) -> float:
        return (
            self.novelty * 0.30 +
            (self.emotional + 1) / 2 * 0.20 +
            self.task_relevance * 0.35 +
            self.repetition * 0.15
        )


class EncodeIntensity(str, Enum):
    """How strongly a memory trace is encoded"""
    STRONG = "strong"    # High salience — durable, one-shot capable
    NORMAL = "normal"   # Medium salience — standard encoding
    WEAK = "weak"       # Low salience — minimal trace, prone to fast decay
    SKIP = "skip"       # Near-zero value — buffered only, not durable


class Concept(BaseModel):
    """A semantic concept in memory"""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ConceptType
    description: str
    embedding: Optional[List[float]] = None
    importance: ImportanceVector = Field(default_factory=ImportanceVector)
    state: MemoryState = Field(default=MemoryState.ACTIVE)
    created_at: datetime = Field(default_factory=utc_now)
    last_accessed: datetime = Field(default_factory=utc_now)
    access_count: int = 0
    strength: float = 1.0

    # HME Phase 1 fields
    encode_intensity: EncodeIntensity = Field(default=EncodeIntensity.NORMAL)
    salience_score: float = Field(default=0.5, ge=0, le=1)
    grasp_score: float = Field(default=0.5, ge=0, le=1)
    prediction_error: float = Field(default=0.0, ge=0, le=1)
    retention_score: float = Field(default=0.5, ge=0, le=1)
    consolidation_score: float = Field(default=0.5, ge=0, le=1)
    rehearsal_count: int = 0
    activation_count: int = 0
    association_density: float = Field(default=0.0, ge=0, le=1)
    decay_rate: float = Field(default=0.01, ge=0)
    confidence: float = Field(default=0.5, ge=0, le=1)
    schema_overlap: float = Field(default=0.0, ge=0, le=1)
    version_parent: Optional[str] = None
    version_root: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    is_current_version: bool = True
    context_tags: Dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """A relation between two concepts"""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str  # Concept that the relation starts from
    predicate: PredicateType
    object_id: str  # Concept that the relation points to
    strength: float = 1.0
    created_at: datetime = Field(default_factory=utc_now)
    bidirectional: bool = False


class Episode(BaseModel):
    """An episodic memory entry (working memory)"""
    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=utc_now)
    concept_ids: List[str] = []
    raw_content: str
    context: Dict[str, Any] = {}
    importance: ImportanceVector = Field(default_factory=ImportanceVector)
    state: MemoryState = MemoryState.ACTIVE
    source: str = "user"

    # HME Phase 1 fields
    encode_intensity: EncodeIntensity = Field(default=EncodeIntensity.NORMAL)
    salience_score: float = Field(default=0.5, ge=0, le=1)
    grasp_score: float = Field(default=0.5, ge=0, le=1)
    prediction_error: float = Field(default=0.0, ge=0, le=1)
    who: Optional[str] = None
    what: Optional[str] = None
    when_: Optional[str] = Field(default=None, alias="when")
    where_: Optional[str] = Field(default=None, alias="where")
    why: Optional[str] = None
    certainty: float = Field(default=0.5, ge=0, le=1)
    interlocutor: Optional[str] = None
    task_context: Optional[str] = None


class EventSchema(BaseModel):
    """Structured event frame extracted from an episode"""
    who: str
    what: str
    when: Optional[str] = None
    where: Optional[str] = None
    why: Optional[str] = None
    source: str = "user"
    certainty: float = Field(default=0.5, ge=0, le=1)
    salience: float = 0.5
    grasp: float = 0.5
    is_contradiction: bool = False
    event_key: Optional[str] = None
    raw_episode_id: Optional[str] = None


class SleepCycle(BaseModel):
    """Record of a sleep cycle"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    end_time: Optional[datetime] = None
    nrem_duration: float = 0.0  # seconds
    rem_duration: float = 0.0
    memories_consolidated: int = 0
    memories_forgotten: int = 0
    dreams_generated: List[str] = []


class MemoryStats(BaseModel):
    """Statistics about memory state"""
    total_concepts: int = 0
    total_relations: int = 0
    working_memory_size: int = 0
    avg_importance: float = 0.0
    suppressed_count: int = 0
    archived_count: int = 0
    versioned_count: int = 0
    last_sleep: Optional[datetime] = None


class EncodeResult(BaseModel):
    """Result of the AttentionGate encode decision"""
    should_encode: bool
    intensity: EncodeIntensity
    salience: float
    grasp: float
    prediction_error: float
    reason: str
    noise_penalty: float = 0.0
    schema_overlap: float = 0.0


class AttentionGateResult(BaseModel):
    """Full output from AttentionGate evaluation"""
    encode_result: EncodeResult
    novelty_component: float
    task_component: float
    emotional_component: float
    repetition_component: float
    prediction_component: float
    noise_component: float


class GraspResult(BaseModel):
    """Result of grasp score computation"""
    grasp_score: float
    schema_overlap: float
    clarity: float
    cognitive_load: float
    one_shot_capable: bool
