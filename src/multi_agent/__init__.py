"""
SleepAI Multi-Agent System
Collaborative memory across multiple SleepAI instances
"""
from .protocol import DreamPacket, MemoryExport, SyncMessage, AgentIdentity
from .sync import AgentSyncHub, AgentSwarm

__all__ = ['DreamPacket', 'MemoryExport', 'SyncMessage', 'AgentIdentity', 
           'AgentSyncHub', 'AgentSwarm']