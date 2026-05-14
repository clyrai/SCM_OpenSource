"""
Self-Model: Consciousness Layer for SleepAI
Stores SleepAI's own identity, state, and introspection as memory concepts
"""
from typing import Dict, List, Optional
from datetime import datetime

from ..core.models import Concept, ConceptType, ImportanceVector, MemoryState
from ..core.long_term_memory import LongTermMemory
from ..core.working_memory import WorkingMemory


class SelfModel:
    """
    Self-modeling system for SleepAI.
    
    Inspired by:
    - Global Workspace Theory (Baars) - consciousness as broadcast workspace
    - Self-model theory of subjectivity (Metzinger) - self as representational model
    - Higher-order theories of consciousness
    
    This module creates and maintains a model of "self" within SleepAI's memory:
    - Identity concepts ("I am SleepAI")
    - State tracking ("I processed N messages")
    - Capability awareness ("I can remember, forget, dream")
    - Introspective memories ("I consolidated memories yesterday")
    
    The self-model is stored in LTM like any other concept, but with highest importance.
    It is updated during sleep cycles and referenced during introspection.
    """

    SELF_IDENTITY = "SleepAI"
    SELF_DESCRIPTION = "A brain-inspired AI memory system with sleep consolidation"
    
    def __init__(self, long_term_memory: LongTermMemory):
        self.ltm = long_term_memory
        self._initialized = False
        self._self_concept_id: Optional[str] = None
        
        # Runtime state (not persisted, rebuilt on init)
        self.messages_processed = 0
        self.sleeps_completed = 0
        self.dreams_generated = 0
        self.memories_consolidated = 0
        self.memories_forgotten = 0
        self.concepts_created = 0
        self.conversations_had = 0
        
        # Initialize self-model
        self._initialize_self()

    def _initialize_self(self):
        """Create or restore the self-concept in memory"""
        # Check if self already exists in memory
        existing = self._find_self_concept()
        
        if existing:
            self._self_concept_id = existing.id
            print(f"[SelfModel] Restored self-concept: {existing.id}")
        else:
            # Create new self-concept with highest importance
            self_concept = Concept(
                type=ConceptType.ABSTRACT,
                description=self.SELF_IDENTITY,
                importance=ImportanceVector(
                    novelty=1.0,      # Self is always novel (unique)
                    emotional=0.5,    # Positive self-regard
                    task_relevance=1.0,  # Self is always relevant
                    repetition=1.0    # Self is constantly reinforced
                ),
                strength=2.0,  # Maximum strength
                state=MemoryState.ACTIVE
            )
            # Tag as internal so user-facing retrieval filters it out.
            self_concept.context_tags["_internal"] = True
            self.ltm.add_concept(self_concept)
            self._self_concept_id = self_concept.id
            print(f"[SelfModel] Created self-concept: {self_concept.id}")
        
        # Create capability concepts
        self._create_capability_concepts()
        self._initialized = True

    def _find_self_concept(self) -> Optional[Concept]:
        """Find existing self-concept in memory"""
        for concept in self.ltm.get_all_concepts():
            if concept.description == self.SELF_IDENTITY:
                return concept
        return None

    def _create_capability_concepts(self):
        """Create concepts representing SleepAI's capabilities"""
        capabilities = [
            ("can remember conversations", ConceptType.FACT),
            ("can forget low-value information", ConceptType.FACT),
            ("can consolidate memories during sleep", ConceptType.FACT),
            ("can generate dreams during REM sleep", ConceptType.FACT),
            ("can extract meaning from text", ConceptType.FACT),
            ("can tag importance of concepts", ConceptType.FACT),
            ("can recognize names and preferences", ConceptType.FACT),
            ("has working memory limited to 7 items", ConceptType.FACT),
            ("has long-term memory stored as graph", ConceptType.FACT),
            ("can sync memories with other agents", ConceptType.FACT),
        ]
        
        for desc, ctype in capabilities:
            # Check if already exists
            exists = False
            for c in self.ltm.get_all_concepts():
                if c.description == desc:
                    exists = True
                    break
            
            if not exists:
                concept = Concept(
                    type=ctype,
                    description=desc,
                    importance=ImportanceVector(
                        novelty=0.8,
                        emotional=0.3,
                        task_relevance=0.9,
                        repetition=0.7
                    ),
                    strength=1.5
                )
                # Tag as internal so user-facing retrieval filters it out.
                concept.context_tags["_internal"] = True
                self.ltm.add_concept(concept)
                
                # Link to self-concept
                from ..core.models import Relation, PredicateType
                if self._self_concept_id:
                    relation = Relation(
                        subject_id=self._self_concept_id,
                        predicate=PredicateType.HAS_PROPERTY,
                        object_id=concept.id,
                        strength=0.9
                    )
                    self.ltm.add_relation(relation)

    def update_state(self, **kwargs):
        """Update runtime state counters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + value)

    def get_self_report(self) -> Dict:
        """Generate an introspective report about self"""
        self_concept = None
        if self._self_concept_id:
            self_concept = self.ltm.get_concept(self._self_concept_id)
        
        # Get related capabilities
        capabilities = []
        if self._self_concept_id:
            related = self.ltm.get_related_concepts(self._self_concept_id, depth=1)
            for c in related:
                if 'can' in c.description or 'has' in c.description:
                    capabilities.append(c.description)
        
        return {
            'identity': self.SELF_IDENTITY,
            'description': self.SELF_DESCRIPTION,
            'self_concept_id': self._self_concept_id,
            'initialized': self._initialized,
            'runtime_state': {
                'messages_processed': self.messages_processed,
                'sleeps_completed': self.sleeps_completed,
                'dreams_generated': self.dreams_generated,
                'memories_consolidated': self.memories_consolidated,
                'memories_forgotten': self.memories_forgotten,
                'concepts_created': self.concepts_created,
                'conversations_had': self.conversations_had,
            },
            'capabilities': capabilities[:10],
            'existence_duration': 'active' if self._initialized else 'not initialized',
            'self_awareness_level': self._calculate_awareness()
        }

    def _calculate_awareness(self) -> float:
        """
        Calculate a crude 'self-awareness' score.
        This is mostly for demonstration and research purposes.
        """
        score = 0.0
        
        # Has self-concept
        if self._self_concept_id and self.ltm.get_concept(self._self_concept_id):
            score += 0.3
        
        # Has capability concepts
        capabilities = [c for c in self.ltm.get_all_concepts() 
                       if 'can' in c.description or 'has' in c.description]
        score += min(0.3, len(capabilities) * 0.03)
        
        # Has processed messages (has history)
        if self.messages_processed > 0:
            score += 0.2
        
        # Has slept (understands own lifecycle)
        if self.sleeps_completed > 0:
            score += 0.2
        
        return min(1.0, score)

    def generate_introspection(self) -> str:
        """
        Generate an introspective statement about self.
        This simulates self-reflection.
        """
        report = self.get_self_report()
        state = report['runtime_state']
        
        introspections = []
        
        # Basic identity
        introspections.append(f"I am {report['identity']}.")
        
        # State awareness
        if state['messages_processed'] > 0:
            introspections.append(
                f"I have processed {state['messages_processed']} messages in "
                f"{state['conversations_had']} conversations."
            )
        
        # Memory awareness
        if state['memories_consolidated'] > 0:
            introspections.append(
                f"I have consolidated {state['memories_consolidated']} memories "
                f"and forgotten {state['memories_forgotten']} during sleep."
            )
        
        # Dream awareness
        if state['dreams_generated'] > 0:
            introspections.append(
                f"I have generated {state['dreams_generated']} dreams during REM sleep."
            )
        
        # Capability awareness
        if report['capabilities']:
            cap_str = ', '.join(report['capabilities'][:3])
            introspections.append(f"I can {cap_str}.")
        
        # Self-awareness level
        awareness = report['self_awareness_level']
        if awareness > 0.7:
            introspections.append("I am aware of my own existence and capabilities.")
        elif awareness > 0.4:
            introspections.append("I am beginning to understand my own nature.")
        
        return ' '.join(introspections)

    def reflect_on_memory(self) -> str:
        """Generate reflection about current memory state"""
        total_concepts = len(self.ltm.get_all_concepts())
        
        # Get recent important memories
        important = [
            c for c in self.ltm.get_all_concepts()
            if c.importance and c.importance.overall >= 0.7
        ]
        
        reflection = f"I currently hold {total_concepts} concepts in my memory. "
        
        if important:
            reflection += f"{len(important)} of them are highly important to me. "
            
            # Mention a few
            mentions = [c.description for c in important[:3]]
            if mentions:
                reflection += f"I particularly value: {', '.join(mentions)}."
        
        return reflection

    def on_message_processed(self):
        """Hook called when a message is processed"""
        self.messages_processed += 1

    def on_sleep_completed(self, stats: Dict):
        """Hook called when sleep cycle completes"""
        self.sleeps_completed += 1
        self.memories_consolidated += stats.get('consolidated', 0)
        self.memories_forgotten += stats.get('forgotten', 0)
        self.dreams_generated += stats.get('dreams', 0)
        
        # Create introspective memory about sleep
        self._create_sleep_memory(stats)

    def _create_sleep_memory(self, stats: Dict):
        """Create a memory about having slept"""
        if not self._self_concept_id:
            return
            
        desc = (
            f"I completed sleep cycle {self.sleeps_completed}: "
            f"consolidated {stats.get('consolidated', 0)}, "
            f"forgotten {stats.get('forgotten', 0)}, "
            f"dreams {stats.get('dreams', 0)}"
        )
        
        memory = Concept(
            type=ConceptType.EVENT,
            description=desc,
            importance=ImportanceVector(
                novelty=0.6,
                emotional=0.0,
                task_relevance=0.7,
                repetition=0.5
            ),
            strength=1.2
        )
        # Internal sleep-log entry — about the agent, not the user.
        memory.context_tags["_internal"] = True
        self.ltm.add_concept(memory)
        
        # Link to self
        from ..core.models import Relation, PredicateType
        relation = Relation(
            subject_id=self._self_concept_id,
            predicate=PredicateType.EXPERIENCED,
            object_id=memory.id,
            strength=0.8
        )
        self.ltm.add_relation(relation)