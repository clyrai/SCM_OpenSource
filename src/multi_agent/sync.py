"""
Cross-Agent Memory Sync
Enables multiple SleepAI agents to share memories and dreams
"""
from typing import List, Dict, Optional, Tuple
import json
import hashlib

from ..core.models import Concept, ConceptType, ImportanceVector, MemoryState
from ..core.long_term_memory import LongTermMemory
from .protocol import DreamPacket, MemoryExport, SyncMessage, AgentIdentity
from ..core.time_utils import utc_isoformat


class AgentSyncHub:
    """
    Central hub for coordinating memory sync between SleepAI agents.
    
    Architecture:
    - Each agent has its own memory graph
    - After sleep, agents publish "dream packets" to the hub
    - Other agents can subscribe to and incorporate these packets
    - Creates emergent collective memory
    """

    def __init__(self):
        self.agents: Dict[str, AgentIdentity] = {}
        self.dream_queue: List[DreamPacket] = []
        self.memory_exports: List[MemoryExport] = []
        self.sync_history: List[SyncMessage] = []

    def register_agent(self, agent_id: str, name: str, capabilities: List[str] = None):
        """Register a new agent with the sync hub"""
        self.agents[agent_id] = AgentIdentity(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities or ["memory", "sleep"],
            memory_fingerprint="",
            last_sync=None
        )
        print(f"[SyncHub] Agent '{name}' ({agent_id}) registered")

    def publish_dream(self, agent_id: str, dream_concepts: List[Dict], 
                      novel_connections: List[Dict], emotional_tone: str = "neutral"):
        """Publish a dream packet from an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")

        packet = DreamPacket(
            agent_id=agent_id,
            timestamp=utc_isoformat(),
            dream_concepts=dream_concepts,
            novel_connections=novel_connections,
            emotional_tone=emotional_tone
        )
        
        self.dream_queue.append(packet)
        print(f"[SyncHub] Agent {agent_id} published dream with {len(dream_concepts)} concepts")

    def export_memory(self, agent_id: str, session_id: str, 
                      concepts: List[Concept], sleep_stats: Dict) -> MemoryExport:
        """Export high-importance memories from an agent"""
        
        # Filter high-importance concepts
        high_imp = []
        for c in concepts:
            if c.importance and c.importance.overall >= 0.7:
                high_imp.append({
                    'id': c.id,
                    'description': c.description,
                    'type': c.type.value if hasattr(c.type, 'value') else str(c.type),
                    'importance': c.importance.model_dump(),
                    'source_agent': agent_id
                })

        export = MemoryExport(
            agent_id=agent_id,
            session_id=session_id,
            high_importance_concepts=high_imp,
            shared_episodes=[],
            sleep_stats=sleep_stats
        )
        
        self.memory_exports.append(export)
        return export

    def sync_agent(self, agent_id: str, target_agent_id: Optional[str] = None,
                   max_dreams: int = 3, max_concepts: int = 10) -> List[Dict]:
        """
        Sync memories from other agents into this agent.
        
        Returns:
            List of incorporated concepts
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")

        incorporated = []

        # 1. Incorporate dream packets from other agents
        other_dreams = [d for d in self.dream_queue 
                       if d.agent_id != agent_id]
        
        if target_agent_id:
            other_dreams = [d for d in other_dreams 
                          if d.agent_id == target_agent_id]

        for dream in other_dreams[-max_dreams:]:
            for concept_data in dream.dream_concepts[:max_concepts]:
                incorporated.append({
                    'type': 'dream_concept',
                    'source_agent': dream.agent_id,
                    'description': concept_data.get('description', ''),
                    'emotional_tone': dream.emotional_tone
                })

        # 2. Incorporate high-importance memories from other agents
        other_exports = [e for e in self.memory_exports
                        if e.agent_id != agent_id]
        
        if target_agent_id:
            other_exports = [e for e in other_exports
                           if e.agent_id == target_agent_id]

        for export in other_exports:
            for concept_data in export.high_importance_concepts[:5]:
                incorporated.append({
                    'type': 'shared_memory',
                    'source_agent': export.agent_id,
                    'description': concept_data.get('description', ''),
                    'original_importance': concept_data.get('importance', {})
                })

        # Update sync timestamp
        self.agents[agent_id].last_sync = utc_isoformat()
        
        if incorporated:
            print(f"[SyncHub] Agent {agent_id} incorporated {len(incorporated)} items from peers")
        
        return incorporated

    def get_collective_memory(self) -> Dict:
        """Get overview of all shared memories across agents"""
        return {
            'total_agents': len(self.agents),
            'total_dreams_shared': len(self.dream_queue),
            'total_memory_exports': len(self.memory_exports),
            'agent_states': {
                aid: {
                    'name': a.name,
                    'last_sync': a.last_sync,
                    'capabilities': a.capabilities
                }
                for aid, a in self.agents.items()
            },
            'recent_dreams': [
                {
                    'agent': d.agent_id,
                    'concepts': len(d.dream_concepts),
                    'tone': d.emotional_tone,
                    'time': d.timestamp
                }
                for d in self.dream_queue[-5:]
            ]
        }

    def create_consensus_memory(self, concept_description: str, 
                                participating_agents: List[str]) -> Dict:
        """
        Create a consensus memory when multiple agents agree on a fact.
        This creates a 'stronger' memory with boosted importance.
        """
        if len(participating_agents) < 2:
            return {'error': 'Need at least 2 agents for consensus'}

        # Boost importance based on number of agreeing agents
        consensus_boost = min(0.3, len(participating_agents) * 0.1)
        
        consensus = {
            'description': concept_description,
            'agents': participating_agents,
            'consensus_strength': len(participating_agents),
            'importance_boost': consensus_boost,
            'timestamp': utc_isoformat(),
            'type': 'consensus'
        }
        
        print(f"[SyncHub] Consensus formed: '{concept_description}' by {len(participating_agents)} agents")
        return consensus


class AgentSwarm:
    """
    A swarm of SleepAI agents that can collaborate and share memories.
    """

    def __init__(self):
        self.hub = AgentSyncHub()
        self.agent_engines: Dict[str, any] = {}  # agent_id -> ChatEngine

    def add_agent(self, agent_id: str, name: str, engine: any):
        """Add a SleepAI agent to the swarm"""
        self.hub.register_agent(agent_id, name)
        self.agent_engines[agent_id] = engine
        print(f"[Swarm] Added agent '{name}' to swarm")

    def collective_sleep(self):
        """
        Trigger sleep for all agents, then sync dreams.
        This is like a 'group meditation' where agents share insights.
        """
        print("\n" + "="*60)
        print("COLLECTIVE SLEEP CYCLE")
        print("="*60)

        # 1. All agents sleep individually
        dream_results = {}
        for agent_id, engine in self.agent_engines.items():
            print(f"\n[Swarm] Agent {agent_id} sleeping...")
            
            # Force sleep
            result = engine.force_sleep()
            if result:
                dream_results[agent_id] = result
                
                # Publish dream to hub
                # Generate synthetic dream concepts from sleep stats
                dream_concepts = [
                    {'description': f'dream_concept_{i}', 'type': 'abstract'}
                    for i in range(result.get('dreams', 0))
                ]
                
                self.hub.publish_dream(
                    agent_id=agent_id,
                    dream_concepts=dream_concepts,
                    novel_connections=[],
                    emotional_tone="neutral"
                )

        # 2. Sync dreams across agents
        print("\n[Swarm] Syncing dreams across agents...")
        for agent_id in self.agent_engines:
            incorporated = self.hub.sync_agent(agent_id)
            if incorporated:
                print(f"  Agent {agent_id} learned {len(incorporated)} new things from peers")

        # 3. Report collective state
        collective = self.hub.get_collective_memory()
        print(f"\n[Swarm] Collective state:")
        print(f"  Total dreams shared: {collective['total_dreams_shared']}")
        print(f"  Agents synced: {collective['total_agents']}")

        return collective

    def share_experience(self, source_agent_id: str, experience: str,
                        target_agent_ids: Optional[List[str]] = None):
        """
        Have one agent share a direct experience with others.
        This bypasses sleep and does immediate memory transfer.
        """
        if source_agent_id not in self.agent_engines:
            raise ValueError(f"Agent {source_agent_id} not in swarm")

        # Source agent processes the experience
        source_engine = self.agent_engines[source_agent_id]
        response, metadata = source_engine.chat(experience)

        # Export the memory
        concepts = source_engine.long_term_memory.get_all_concepts()
        export = self.hub.export_memory(
            agent_id=source_agent_id,
            session_id=source_engine.session_id,
            concepts=concepts,
            sleep_stats={}
        )

        # Sync to target agents
        targets = target_agent_ids or [aid for aid in self.agent_engines if aid != source_agent_id]
        
        for target_id in targets:
            if target_id in self.agent_engines:
                incorporated = self.hub.sync_agent(
                    agent_id=target_id,
                    target_agent_id=source_agent_id
                )
                print(f"[Swarm] Agent {target_id} incorporated {len(incorporated)} memories from {source_agent_id}")

        return export

    def get_swarm_report(self) -> Dict:
        """Get comprehensive report on swarm state"""
        report = {
            'swarm_size': len(self.agent_engines),
            'agents': {},
            'collective_memory': self.hub.get_collective_memory(),
            'shared_memories': []
        }

        for agent_id, engine in self.agent_engines.items():
            mem_report = engine.get_memory_report()
            report['agents'][agent_id] = {
                'name': self.hub.agents.get(agent_id, AgentIdentity(agent_id=agent_id, name="unknown", capabilities=[], memory_fingerprint="")).name,
                'ltm_concepts': mem_report['long_term_memory']['total_concepts'],
                'wm_size': mem_report['working_memory']['size'],
                'messages': mem_report['messages_exchanged'],
                'sleeps': len(engine._sleep_history)
            }

        return report
