"""
Multi-Agent Memory Sync Protocol
Agents share dreams and important memories during sleep
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict

from ..core.models import Concept, ImportanceVector


class DreamPacket(BaseModel):
    """A packet of dream-generated insights to share with other agents"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    timestamp: str
    dream_concepts: List[Dict]  # Concepts generated during REM
    novel_connections: List[Dict]  # New relations discovered
    emotional_tone: str

class MemoryExport(BaseModel):
    """Export important memories for cross-agent sync"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    session_id: str
    high_importance_concepts: List[Dict]  # Concepts with importance > 0.7
    shared_episodes: List[Dict]  # Key conversation moments
    sleep_stats: Dict

class SyncMessage(BaseModel):
    """Message format for agent-to-agent communication"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sender_id: str
    receiver_id: str
    message_type: str  # "dream_share", "memory_request", "consensus_check"
    payload: Dict
    timestamp: str

class AgentIdentity(BaseModel):
    """Identity of a SleepAI agent in the network"""
    agent_id: str
    name: str
    capabilities: List[str]
    memory_fingerprint: str  # Hash of current memory state
    last_sync: Optional[str] = None
