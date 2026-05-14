"""
Chat Engine: Core conversational loop for SleepAI
Orchestrates memory, LLM, and sleep cycle during conversation
"""
import threading
import time
import json
import re
from typing import Dict, List, Optional, Tuple

from ..core.models import (
    Concept,
    Episode,
    ImportanceVector,
    MemoryState,
    EncodeIntensity,
    PredicateType,
    Relation,
)
from ..core.encoder import MeaningEncoder
from ..core.value_tagger import ValueTagger
from ..core.working_memory import WorkingMemory
from ..core.long_term_memory import LongTermMemory
from ..core.event_compiler import EventCompiler
from ..core.association_binder import AssociationBinder
from ..sleep.sleep_cycle import SleepCycleOrchestrator
from ..sleep.trigger import SleepTrigger
from ..llm import LLMExtractor
from ..consciousness.self_model import SelfModel
from ..core.attention_gate import AttentionGate
from ..core.config import HME_ENABLED
from ..core.profiles import get_runtime_profile, normalize_profile_name
from .memory_retriever import MemoryRetriever
from ..retrieval.spreading_activation import SpreadingActivationRetriever
from ..retrieval.hypothesis_ranker import HypothesisRanker
from ..core.time_utils import ensure_utc, utc_isoformat, utc_now


class ChatEngine:
    """
    The conversational brain of SleepAI.

    Flow:
    1. User speaks → extract concepts → store in WM
    2. Retrieve relevant memories (WM + LTM)
    3. Build context-augmented prompt
    4. Generate response via LLM
    5. Extract concepts from response → store in WM
    6. Check if sleep should trigger
    7. Return response to user
    """

    SYSTEM_PERSONA = """You are SleepAI, a memory-enabled AI assistant. You have a brain-inspired memory system that lets you remember conversations, facts, and preferences.

When you recall memories, reference them naturally in conversation. If someone told you their name, use it. If they mentioned a preference, respect it. Be warm, conversational, and sound like a thoughtful person rather than a system report.

You are currently in a conversation. Here are relevant memories that might help you respond:"""

    def __init__(
        self,
        llm: Optional[LLMExtractor] = None,
        encoder: Optional[MeaningEncoder] = None,
        value_tagger: Optional[ValueTagger] = None,
        working_memory: Optional[WorkingMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        sleep_orchestrator: Optional[SleepCycleOrchestrator] = None,
        enable_auto_sleep: Optional[bool] = None,
        sleep_check_interval: Optional[int] = None,
        session_id: str = "default",
        profile: str = "chatbot",
        sandbox_mode: bool = False,
        enable_persistence: Optional[bool] = None,
        cross_session_pool: Optional["object"] = None,  # CrossSessionMemoryPool
    ):
        from ..core.config import AUTO_SLEEP_ENABLED, AUTO_SLEEP_INTERVAL, ENABLE_SESSION_PERSISTENCE, DEFAULT_SESSION_ID

        self.profile = normalize_profile_name(profile)
        profile_cfg = get_runtime_profile(self.profile)
        self.sandbox_mode = bool(sandbox_mode)

        self.llm = llm or LLMExtractor()
        self.encoder = encoder or MeaningEncoder(llm=self.llm)
        self.value_tagger = value_tagger or ValueTagger()
        self.working_memory = working_memory or WorkingMemory(
            capacity=profile_cfg.working_memory_capacity
        )
        self.long_term_memory = long_term_memory or LongTermMemory(
            persist=not self.sandbox_mode
        )
        if hasattr(self.long_term_memory, "set_persistence"):
            self.long_term_memory.set_persistence(not self.sandbox_mode)

        if sleep_orchestrator is None:
            trigger = SleepTrigger(**profile_cfg.trigger)
            self.sleep_orchestrator = SleepCycleOrchestrator(trigger=trigger)
        else:
            self.sleep_orchestrator = sleep_orchestrator

        self.memory_retriever = MemoryRetriever(
            working_memory=self.working_memory,
            long_term_memory=self.long_term_memory
        )

        default_auto_sleep = profile_cfg.auto_sleep_enabled if profile_cfg else AUTO_SLEEP_ENABLED
        default_sleep_interval = profile_cfg.sleep_check_interval if profile_cfg else AUTO_SLEEP_INTERVAL
        self.enable_auto_sleep = enable_auto_sleep if enable_auto_sleep is not None else default_auto_sleep
        self.sleep_check_interval = sleep_check_interval if sleep_check_interval is not None else default_sleep_interval
        self._message_count = 0
        self._turns_since_micro_sleep = 0
        self._sleep_history: List[Dict] = []
        self._conversation_start = utc_now()
        self.session_id = session_id or DEFAULT_SESSION_ID
        configured_persistence = ENABLE_SESSION_PERSISTENCE if enable_persistence is None else bool(enable_persistence)
        self._enable_persistence = configured_persistence and not self.sandbox_mode
        self._user_history: List[str] = []

        self._load_session()

        self.self_model = SelfModel(long_term_memory=self.long_term_memory)

        self._attention_gate: Optional[AttentionGate] = None
        self._hme_enabled = HME_ENABLED
        if self._hme_enabled:
            self._attention_gate = AttentionGate()
            self._event_compiler = EventCompiler()
            self._association_binder = AssociationBinder()
            self._spreading_activation = SpreadingActivationRetriever(
                working_memory=self.working_memory,
                long_term_memory=self.long_term_memory,
                encoder=self.encoder,
                vector_index=getattr(self.long_term_memory, "vector_index", None),
            )
            self._hypothesis_ranker = HypothesisRanker()
        else:
            self._event_compiler = None
            self._association_binder = None
            self._spreading_activation = None
            self._hypothesis_ranker = None
        self._prior_concepts: List[Concept] = []
        self._event_history = []

        # Phase 7: cross-session memory pool. None = legacy (current-session
        # WM only). When set, sleep cycles will pull recent prior-session
        # episodes to consolidate across temporal boundaries.
        self.cross_session_pool = cross_session_pool
        if self.cross_session_pool is None:
            try:
                from ..core.config import (
                    CROSS_SESSION_POOL_ENABLED,
                    CROSS_SESSION_POOL_LOOKBACK_HOURS,
                    CROSS_SESSION_POOL_MAX_SESSIONS,
                    CROSS_SESSION_POOL_MAX_EPISODES_PER_SESSION,
                    CROSS_SESSION_POOL_MAX_TOTAL_BORROWED,
                )
                if CROSS_SESSION_POOL_ENABLED and not self.sandbox_mode:
                    from ..core.cross_session_pool import (
                        CrossSessionMemoryPool,
                        CrossSessionPoolConfig,
                    )
                    from ..core.sqlite_db import get_memory
                    self.cross_session_pool = CrossSessionMemoryPool(
                        current_session_id=self.session_id,
                        sqlite_factory=get_memory,
                        current_profile=self.profile,
                        config=CrossSessionPoolConfig(
                            enabled=True,
                            max_sessions=CROSS_SESSION_POOL_MAX_SESSIONS,
                            lookback_hours=CROSS_SESSION_POOL_LOOKBACK_HOURS,
                            max_episodes_per_session=CROSS_SESSION_POOL_MAX_EPISODES_PER_SESSION,
                            max_total_borrowed=CROSS_SESSION_POOL_MAX_TOTAL_BORROWED,
                        ),
                    )
            except Exception:
                # Defensive: never block ChatEngine init on optional pool.
                self.cross_session_pool = None

    def _load_session(self):
        """Load memory from previous session if exists"""
        if not self._enable_persistence:
            return

        try:
            from ..core.sqlite_db import get_memory
            sqlite = get_memory()
            meta = sqlite.load_session_meta(self.session_id)

            # Always rehydrate concepts/episodes if they exist in SQLite,
            # regardless of whether session_meta is present. session_meta
            # is just a "last seen" record, not a permission gate.
            concept_rows = sqlite.get_all_concepts_raw()
            episode_rows = sqlite.get_all_episodes_raw()
            relation_rows = sqlite.get_all_relations_raw()

            if not (concept_rows or episode_rows):
                return  # truly fresh — nothing to load

            print(f"[ChatEngine] Resuming session '{self.session_id}'...")

            if meta:
                self._message_count = meta.get('message_count', 0)

            for row in concept_rows:
                try:
                    concept = self.long_term_memory._concept_from_record(row)
                    self.long_term_memory._sync_concept(concept)
                except Exception as e:
                    print(f"[ChatEngine] Failed to restore concept: {e}")

            for row in episode_rows[-self.working_memory.capacity:]:
                try:
                    from ..core.models import Episode
                    episode = Episode(
                        id=row['id'],
                        timestamp=ensure_utc(row['timestamp']) or utc_now(),
                        concept_ids=json.loads(row['concept_ids']) if row['concept_ids'] else [],
                        raw_content=row['raw_content'],
                        importance=ImportanceVector(),
                        source=row.get('source', 'user')
                    )
                    self.working_memory.store(episode)
                    if episode.source == "user" and episode.raw_content:
                        self._user_history.append(episode.raw_content)
                except Exception as e:
                    print(f"[ChatEngine] Failed to restore episode: {e}")

            for row in relation_rows:
                try:
                    if row['subject_id'] in self.long_term_memory.graph and row['object_id'] in self.long_term_memory.graph:
                        self.long_term_memory.graph.add_edge(
                            row['subject_id'],
                            row['object_id'],
                            predicate=row.get('predicate', 'related_to'),
                            strength=row.get('strength', 1.0),
                            id=row.get('id'),
                        )
                except Exception as e:
                    print(f"[ChatEngine] Failed to restore relation: {e}")

            # Restore sleep history so wake-summary sees prior cycles.
            try:
                sleep_rows = sqlite.get_sleep_records_since()
                self._sleep_history = sleep_rows
            except Exception as e:
                print(f"[ChatEngine] Failed to restore sleep history: {e}")

            # Rebuild the vector index from the restored concept embeddings.
            # _sync_concept already populates the index incrementally during
            # load, but a full rebuild is cheaper than n incremental adds
            # for the 100s-of-concepts range — and avoids the matrix being
            # repeatedly resized via vstack during cold start.
            try:
                vidx = getattr(self.long_term_memory, "vector_index", None)
                if vidx is not None:
                    pairs = []
                    for c in self.long_term_memory.get_all_concepts(include_suppressed=False):
                        emb = getattr(c, "embedding", None)
                        if emb:
                            pairs.append((c.id, emb))
                    vidx.rebuild(pairs)
            except Exception as e:
                print(f"[ChatEngine] Failed to rebuild vector index: {e}")

            indexed = (
                self.long_term_memory.vector_index.size()
                if getattr(self.long_term_memory, "vector_index", None) is not None
                else 0
            )
            print(
                f"[ChatEngine] Restored {len(concept_rows)} concepts, "
                f"{len(episode_rows)} episodes, "
                f"{len(getattr(self, '_sleep_history', []))} sleep cycles, "
                f"{indexed} vectors indexed"
            )

        except Exception as e:
            print(f"[ChatEngine] No previous session found: {e}")

    def save_session(self):
        """Save current session to database"""
        if not self._enable_persistence:
            return True

        try:
            from ..core.sqlite_db import get_memory
            sqlite = get_memory()

            # Save session metadata
            sqlite.save_session_meta(
                session_id=self.session_id,
                message_count=self._message_count,
                total_concepts=len(self.long_term_memory.get_all_concepts()),
                total_sleeps=len(self._sleep_history)
            )

            print(f"[ChatEngine] Session '{self.session_id}' saved.")
            return True
        except Exception as e:
            print(f"[ChatEngine] Failed to save session: {e}")
            return False

    def chat(
        self,
        user_message: str,
        *,
        force_versioning: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Process a user message and generate a response.

        Args:
            user_message: text from the user.
            force_versioning: when True, every user-source concept goes
                through `LongTermMemory.add_concept(allow_versioning=True)`
                regardless of whether the keyword-based contradiction
                detector fired. Used by the public `add_memory` tool —
                a user explicitly volunteering a fact should always
                supersede prior conflicting facts of the same type
                (e.g., "I am Saish" supersedes a stored "Person: Alex"
                even though there's no "actually no" wording).

        Returns:
            Tuple of (assistant_response, metadata_dict)
        """
        metadata = {
            'user_concepts': 0,
            'response_concepts': 0,
            'memories_retrieved': 0,
            'sleep_triggered': False,
            'sleep_stats': None,
            'latency_ms': 0
        }

        start_time = time.time()

        # Step 1: Extract concepts from user message
        user_concepts = self._extract_and_store(
            user_message, source="user", force_versioning=force_versioning,
        )
        metadata['user_concepts'] = len(user_concepts)

        # Step 2: Retrieve relevant memories
        query_embedding = None
        if user_concepts and user_concepts[0].embedding:
            query_embedding = user_concepts[0].embedding

        if self._hme_enabled and self._spreading_activation is not None:
            memory_context, retrieval_stats = self._retrieve_hme(
                user_message, query_embedding
            )
        else:
            memory_context, retrieval_stats = self.memory_retriever.retrieve_context(
                query=user_message,
                query_embedding=query_embedding
            )
        metadata['memories_retrieved'] = (
            retrieval_stats.get('wm_retrieved', 0) +
            retrieval_stats.get('ltm_semantic', 0) +
            retrieval_stats.get('ltm_graph', 0) +
            retrieval_stats.get('total_concepts_activated', 0)
        )

        # Step 2.5: Check for introspection queries
        is_introspection = self._is_introspection_query(user_message)
        if is_introspection:
            introspection = self.self_model.generate_introspection()
            memory_context += f"\n\n## Self-Awareness\n{introspection}"

        # Step 3: Build prompt with memory context
        prompt = self._build_prompt(user_message, memory_context)

        # Update self-model state
        self.self_model.on_message_processed()
        self.self_model.concepts_created += len(user_concepts)
        if 'introspection' in user_message.lower() or 'aware' in user_message.lower():
            self.self_model.conversations_had += 1

        # Step 4: Generate response via LLM
        try:
            response = self.llm._chat(prompt, num_predict=512)
        except Exception as e:
            print(f"[ChatEngine] LLM generation failed: {e}")
            response = self._generate_fallback_response(user_message, memory_context)

        # Step 5: Store assistant response as episode (without LLM extraction to save time)
        assistant_episode = Episode(
            concept_ids=[],
            raw_content=response,
            importance=ImportanceVector(task_relevance=0.5),
            source="assistant"
        )
        self.working_memory.store(assistant_episode)
        metadata['response_concepts'] = 0

        # Step 6: Auto-sleep check
        self._message_count += 1
        self._turns_since_micro_sleep += 1
        if self.enable_auto_sleep and self._message_count % self.sleep_check_interval == 0:
            sleep_result = self._check_and_trigger_sleep()
            if sleep_result:
                metadata['sleep_triggered'] = True
                metadata['sleep_stats'] = sleep_result

        # Step 7: Track latency
        metadata['latency_ms'] = round((time.time() - start_time) * 1000, 1)

        return response, metadata

    def _extract_and_store(
        self,
        text: str,
        source: str = "user",
        *,
        force_versioning: bool = False,
    ) -> List[Concept]:
        """
        Extract concepts from text and store in memory system.
        Returns the extracted concepts.
        """
        concepts = self.encoder.extract(text)
        if not concepts:
            return []

        interlocutor = self._resolve_interlocutor_tag(text=text, source=source)
        task_context = "conversation"

        prior_descriptions = [c.description for c in self._prior_concepts[-20:]]

        for concept in concepts:
            concept.importance = self.value_tagger.tag(concept)
            concept.context_tags.update({
                "session_id": self.session_id,
                "person": interlocutor,
                "source": source,
                "task": task_context,
            })

            if self._attention_gate is not None:
                encode_result = self._attention_gate.evaluate(
                    text=text,
                    importance=concept.importance,
                    concept=concept,
                    prior_concepts=prior_descriptions,
                    context={"source": source},
                )

                concept.encode_intensity = encode_result.intensity
                concept.salience_score = encode_result.salience
                concept.grasp_score = encode_result.grasp
                concept.prediction_error = encode_result.prediction_error

        if self._attention_gate is not None:
            episode_intensity = concepts[0].encode_intensity
            episode_salience = concepts[0].salience_score
            episode_grasp = concepts[0].grasp_score
            episode_prediction_error = concepts[0].prediction_error
        else:
            episode_intensity = EncodeIntensity.NORMAL
            episode_salience = concepts[0].salience_score
            episode_grasp = concepts[0].grasp_score
            episode_prediction_error = concepts[0].prediction_error

        episode = Episode(
            concept_ids=[c.id for c in concepts],
            raw_content=text,
            importance=concepts[0].importance,
            source=source,
        )
        episode.interlocutor = interlocutor
        episode.task_context = task_context
        episode.encode_intensity = episode_intensity
        episode.salience_score = episode_salience
        episode.grasp_score = episode_grasp
        episode.prediction_error = episode_prediction_error

        event = None
        if self._hme_enabled and source == "user" and self._event_compiler is not None:
            event = self._event_compiler.compile_episode(
                episode,
                interlocutor=interlocutor,
                task_context=task_context,
            )
            episode.who = event.who
            episode.what = event.what
            episode.when_ = event.when
            episode.where_ = event.where
            episode.why = event.why
            episode.certainty = event.certainty
            episode.interlocutor = interlocutor
            episode.task_context = task_context
            episode.context["event_key"] = event.event_key
            episode.context["is_duplicate_event"] = self._event_compiler.is_duplicate(
                event, self._event_history[-50:]
            )
            episode.context["is_contradiction"] = event.is_contradiction
            episode.context["versioning_enabled"] = bool(event.is_contradiction)

        self.working_memory.store(episode)

        # Phase 7: persist episode with session_id so the cross-session memory
        # pool can read it back on future sleep cycles. Only persist when the
        # session is durable (not sandbox / persistence disabled). Failure
        # here must never block the chat path.
        if self._enable_persistence and not self.sandbox_mode:
            try:
                # Tag context with session_id so cross-session pool can attribute it
                if isinstance(episode.context, dict):
                    episode.context.setdefault("session_id", self.session_id)
                from ..core.sqlite_db import get_memory
                get_memory().save_episode(episode, session_id=self.session_id)
                # Auto-save session metadata so _load_session() can rehydrate
                # on the next process start. Without this, every restart looks
                # like a fresh user — which kills the cross-session demo.
                get_memory().save_session_meta(
                    session_id=self.session_id,
                    message_count=self._message_count,
                    total_concepts=len(self.long_term_memory.get_all_concepts()),
                    total_sleeps=len(self._sleep_history),
                )
            except Exception as exc:
                # Defensive: never block chat ingestion on persistence
                print(f"[ChatEngine] episode persistence failed: {exc}")

        if source == "user" and text.strip():
            self._user_history.append(text.strip())
            if len(self._user_history) > 500:
                self._user_history = self._user_history[-500:]

        durable_concepts: List[Concept] = []
        for concept in concepts:
            if self._attention_gate is not None:
                if concept.encode_intensity == EncodeIntensity.SKIP:
                    continue

            try:
                if not concept.embedding and hasattr(self.encoder, '_get_embedding'):
                    concept.embedding = self.encoder._get_embedding(concept.description)
                concept = self.long_term_memory.add_concept(
                    concept,
                    context_tags=concept.context_tags,
                    # Allow versioning when EITHER: (a) the explicit
                    # contradiction-keyword detector fired ("actually
                    # no", "changed my mind", etc), OR (b) the caller
                    # explicitly requested it via force_versioning.
                    # Direct add_memory tool calls set force_versioning
                    # because explicit "remember this" should always
                    # supersede conflicting prior facts of the same
                    # type. Without this, the agent stores "Person:
                    # Saish" alongside an old "Person: Alex" instead
                    # of superseding it.
                    allow_versioning=(
                        force_versioning
                        or bool(event and event.is_contradiction and source == "user")
                    ) and source == "user",
                )
                self._prior_concepts.append(concept)
                durable_concepts.append(concept)
            except Exception as e:
                print(f"[ChatEngine] Failed to store concept: {e}")

        if len(self._prior_concepts) > 200:
            self._prior_concepts = self._prior_concepts[-200:]

        if (
            self._hme_enabled
            and source == "user"
            and self._event_compiler is not None
            and self._association_binder is not None
            and durable_concepts
            and event is not None
        ):
            bindable_concepts = [
                c
                for c in durable_concepts
                if c.encode_intensity in (EncodeIntensity.STRONG, EncodeIntensity.NORMAL)
            ]

            if not episode.context.get("is_duplicate_event") and bindable_concepts:
                assoc_stats = self._association_binder.bind_event(
                    event=event,
                    event_concepts=bindable_concepts,
                    long_term_memory=self.long_term_memory,
                )
                episode.context["association_stats"] = assoc_stats
                self._event_history.append(event)
                if len(self._event_history) > 500:
                    self._event_history = self._event_history[-500:]

            episode.context["versioned_concepts"] = sum(1 for c in durable_concepts if c.version_parent)

        return concepts

    def _retrieve_hme(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
    ) -> Tuple[str, Dict]:
        """
        Phase 3 HME retrieval: cue-driven associative recall via spreading activation.

        1. Extract cues from query.
        2. Select seed concepts from WM + LTM by token overlap.
        3. Propagate activation across the association graph (bounded steps).
        4. Gate by context relevance (session, person, recency).
        5. Rank hypotheses and format for prompt.
        """
        recent_episodes = self.working_memory.retrieve(limit=5)
        recent_user = next((ep for ep in recent_episodes if ep.source == "user"), None)
        context_tags = {
            "session_id": self.session_id,
            "person": getattr(recent_user, 'interlocutor', None) if recent_user else None,
        }

        graph_activated, sa_stats = self._spreading_activation.retrieve(
            query=query,
            context_tags=context_tags,
        )

        # Channel 1: graph retrieval via spreading activation.
        graph_activated = [
            c for c in graph_activated
            if not (isinstance(getattr(c, "context_tags", None), dict)
                    and c.context_tags.get("_internal"))
        ]

        # Channel 2: vector semantic retrieval (query embedding -> ANN/cosine).
        semantic_concepts = self._semantic_channel_candidates(
            query=query,
            query_embedding=query_embedding,
            limit=8,
        )

        # Channel 3: lexical retrieval.
        lexical_concepts = self._lexical_channel_candidates(query=query, limit=8)

        # Channel 4: exact slot/entity match.
        exact_concepts = self._exact_channel_candidates(query=query, limit=8)

        channels = {
            "graph": graph_activated,
            "semantic": semantic_concepts,
            "lexical": lexical_concepts,
            "exact": exact_concepts,
        }

        activated, fused_scores, channel_votes = self._fuse_hybrid_channels(
            channels=channels,
            context_tags=context_tags,
            exact_match_ids={c.id for c in exact_concepts},
        )

        # Keep prompt payload bounded.
        max_candidates = max(12, self._hypothesis_ranker.max_hypotheses * 2)
        activated = activated[:max_candidates]
        active_ids = {c.id for c in activated}
        activation_map = self._normalize_scores(
            {cid: score for cid, score in fused_scores.items() if cid in active_ids}
        )

        hypothesis_set = self._hypothesis_ranker.rank(
            activated_concepts=activated,
            activation_map=activation_map,
            context_tags=context_tags,
        )

        memory_context = self._hypothesis_ranker.format_context(
            hypothesis_set,
            section_title="Retrieved Memories",
            include_evidence=True,
        )

        confidence_value = hypothesis_set.confidence.value
        if confidence_value in {"low", "none"}:
            memory_context += (
                "\n\n## Retrieval Guidance\n"
                "Confidence is low. Ask one concise clarifying question before "
                "asserting uncertain facts. If uncertainty remains, say so explicitly."
            )

        top_agreement = channel_votes.get(activated[0].id, 0) if activated else 0
        retrieval_stats = {
            'wm_retrieved': len(recent_episodes),
            'ltm_semantic': len(semantic_concepts),
            'ltm_graph': len(graph_activated),
            'ltm_lexical': len(lexical_concepts),
            'ltm_exact': len(exact_concepts),
            'total_concepts_activated': len(activated),
            'seeds': sa_stats.get('seeds', 0),
            'fusion_mode': 'weighted_rrf',
            'channel_count': 4,
            'top_channel_agreement': top_agreement,
            'hypothesis_count': len(hypothesis_set.hypotheses),
            'hypothesis_ensemble': hypothesis_set.ensemble_score,
            'hypothesis_confidence': confidence_value,
            'coverage': hypothesis_set.coverage,
            'confidence_decision': (
                'clarify_or_bound_uncertainty'
                if confidence_value in {"low", "none"}
                else 'answer'
            ),
        }
        return memory_context, retrieval_stats

    def _semantic_channel_candidates(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        limit: int = 8,
    ) -> List[Concept]:
        embedding = query_embedding
        if embedding is None and hasattr(self.encoder, "_get_embedding"):
            try:
                embedding = self.encoder._get_embedding(query, mode="query")
            except Exception:
                try:
                    embedding = self.encoder._get_embedding(query)
                except Exception:
                    embedding = None
        if not embedding:
            return []
        try:
            return self.long_term_memory.search_by_embedding(embedding, limit=limit)
        except Exception:
            return []

    def _query_terms(self, query: str) -> List[str]:
        tokens = re.findall(r"\b[\w'-]+\b", (query or "").lower())
        stop = {
            "the", "and", "for", "with", "that", "this", "what", "when",
            "where", "which", "about", "from", "into", "your", "have",
            "has", "had", "would", "could", "should", "tell", "please",
            "show", "give", "know", "want", "need", "like",
        }
        return [t for t in tokens if len(t) >= 3 and t not in stop]

    def _lexical_channel_candidates(self, query: str, limit: int = 8) -> List[Concept]:
        terms = self._query_terms(query)
        query_lower = (query or "").strip().lower()
        if not terms and query_lower:
            return self.long_term_memory.search_by_text(query, limit=limit)

        term_set = set(terms)
        scored: List[Tuple[Concept, float]] = []
        for concept in self.long_term_memory.get_all_concepts(include_suppressed=False):
            if not getattr(concept, "is_current_version", True):
                continue
            tags = concept.context_tags or {}
            blob = " ".join(
                [
                    concept.description or "",
                    tags.get("original_description", "") if isinstance(tags.get("original_description"), str) else "",
                ]
            ).lower()
            desc_tokens = set(re.findall(r"\b[\w'-]+\b", blob))
            overlap = len(term_set & desc_tokens) if term_set else 0
            phrase_hit = 1.0 if query_lower and query_lower in blob else 0.0
            if overlap == 0 and phrase_hit == 0.0:
                continue
            overlap_score = overlap / max(1, len(term_set))
            importance = concept.importance.overall if concept.importance else 0.0
            scored.append((concept, overlap_score + 0.2 * phrase_hit + 0.1 * importance))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in scored[:limit]]

    def _exact_channel_candidates(self, query: str, limit: int = 8) -> List[Concept]:
        terms = self._query_terms(query)
        if not terms:
            return []
        term_set = set(terms)
        query_lower = (query or "").strip().lower()
        scored: List[Tuple[Concept, float]] = []
        for concept in self.long_term_memory.get_all_concepts(include_suppressed=False):
            if not getattr(concept, "is_current_version", True):
                continue
            tags = concept.context_tags or {}
            blob = " ".join(
                [
                    concept.description or "",
                    tags.get("original_description", "") if isinstance(tags.get("original_description"), str) else "",
                ]
            ).lower()
            desc_tokens = set(re.findall(r"\b[\w'-]+\b", blob))
            overlap = term_set & desc_tokens
            if not overlap:
                continue
            precision = len(overlap) / len(term_set)
            phrase_bonus = 0.25 if query_lower and query_lower in blob else 0.0
            if precision < 0.45 and phrase_bonus == 0.0:
                continue
            scored.append((concept, precision + phrase_bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in scored[:limit]]

    def _fuse_hybrid_channels(
        self,
        channels: Dict[str, List[Concept]],
        context_tags: Dict[str, Optional[str]],
        exact_match_ids: Optional[set] = None,
    ) -> Tuple[List[Concept], Dict[str, float], Dict[str, int]]:
        exact_match_ids = exact_match_ids or set()
        channel_weights = {
            "graph": 1.00,
            "semantic": 0.90,
            "lexical": 0.95,
            "exact": 1.20,
        }
        rrf_k = 50.0

        concept_by_id: Dict[str, Concept] = {}
        fused_scores: Dict[str, float] = {}
        channel_votes: Dict[str, int] = {}

        for channel, concepts in channels.items():
            weight = channel_weights.get(channel, 1.0)
            for rank, concept in enumerate(concepts, start=1):
                if concept is None:
                    continue
                if not getattr(concept, "is_current_version", True):
                    continue
                tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
                if tags.get("_internal"):
                    continue
                concept_by_id[concept.id] = concept
                fused_scores[concept.id] = fused_scores.get(concept.id, 0.0) + (
                    weight / (rrf_k + rank)
                )
                channel_votes[concept.id] = channel_votes.get(concept.id, 0) + 1

        for concept_id, concept in concept_by_id.items():
            symbolic = 0.0
            if concept_id in exact_match_ids:
                symbolic += 0.12
            symbolic += 0.06 * self._context_match_signal(concept, context_tags)
            symbolic += 0.05 * self._recency_signal(concept.last_accessed)
            symbolic += 0.05 * self._source_confidence_signal(concept)
            if not getattr(concept, "is_current_version", True):
                symbolic -= 0.20
            fused_scores[concept_id] = fused_scores.get(concept_id, 0.0) + symbolic

        ordered_ids = sorted(
            concept_by_id.keys(),
            key=lambda cid: (
                fused_scores.get(cid, 0.0),
                channel_votes.get(cid, 0),
                concept_by_id[cid].importance.overall if concept_by_id[cid].importance else 0.0,
            ),
            reverse=True,
        )
        ordered = [concept_by_id[cid] for cid in ordered_ids]
        return ordered, fused_scores, channel_votes

    def _context_match_signal(
        self,
        concept: Concept,
        context_tags: Dict[str, Optional[str]],
    ) -> float:
        if not context_tags:
            return 0.0
        tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
        score = 0.0

        ctx_session = context_tags.get("session_id")
        concept_session = tags.get("session_id")
        if isinstance(ctx_session, str) and isinstance(concept_session, str):
            score += 1.0 if ctx_session == concept_session else -0.4

        ctx_person = context_tags.get("person")
        concept_person = tags.get("person")
        if isinstance(ctx_person, str) and isinstance(concept_person, str):
            score += 0.8 if ctx_person.lower() == concept_person.lower() else -0.3

        return max(0.0, min(1.0, score))

    def _source_confidence_signal(self, concept: Concept) -> float:
        priors = {
            "profile": 1.0,
            "explicit_profile": 1.0,
            "inferred": 0.75,
            "assistant_inferred": 0.70,
            "chat": 0.55,
            "assistant": 0.55,
            "curiosity": 0.45,
            "noisy_chat": 0.35,
        }
        tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
        raw = tags.get("source") or tags.get("memory_source") or tags.get("origin")
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            for key, prior in priors.items():
                if key in lowered:
                    return prior
        return priors["chat"]

    def _recency_signal(self, last_accessed) -> float:
        if last_accessed is None:
            return 0.4
        try:
            ts = ensure_utc(last_accessed) if hasattr(last_accessed, "tzinfo") else last_accessed
            age_s = max(0.0, (utc_now() - ts).total_seconds())
        except Exception:
            return 0.4

        if age_s <= 3600:
            return 1.0
        week = 7 * 24 * 3600
        if age_s >= week:
            return 0.2
        return 1.0 - ((age_s - 3600) / (week - 3600)) * 0.8

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        lo = min(scores.values())
        hi = max(scores.values())
        span = hi - lo
        if span <= 1e-9:
            return {k: 1.0 for k in scores}
        return {
            k: max(0.0, min(1.0, (v - lo) / span))
            for k, v in scores.items()
        }

    def _build_prompt(self, user_message: str, memory_context: str) -> str:
        """Build the full prompt with system persona, memory context, and user message"""
        parts = [self.SYSTEM_PERSONA]

        if memory_context:
            parts.append(memory_context)
        else:
            parts.append("(No prior memories for this conversation yet)")

        parts.append(f"\n## Current Message\nUser: {user_message}\n\nRespond naturally as SleepAI:")

        return "\n\n".join(parts)

    def _generate_fallback_response(self, user_message: str, memory_context: str = "") -> str:
        """
        Deterministic response path when LLM backends are unavailable.
        Keeps chat useful in offline/test environments.
        """
        msg = (user_message or "").strip()
        q = msg.lower()

        if self._is_introspection_query(msg):
            return self.self_model.generate_introspection()

        profile = self._infer_user_profile()

        if "name" in q and ("what" in q or "my" in q):
            if profile.get("name"):
                return self._humanize_memory_answer(
                    lead_in="If I have it right",
                    detail=f"your name is {profile['name']}",
                )
            return "I don't quite have your name yet, but tell me once and I'll remember it."

        if ("where" in q and "live" in q) or "location" in q:
            if profile.get("location"):
                return self._humanize_memory_answer(
                    lead_in="If I have it right",
                    detail=f"you live in {profile['location']}",
                )
            return "I don't have your location yet, but tell me once and I'll keep it straight."

        if ("what do i do" in q) or ("where do i work" in q) or ("profession" in q):
            if profile.get("profession"):
                return self._humanize_memory_answer(
                    lead_in="If I have it right",
                    detail=f"you work as {profile['profession']}",
                )
            return "I don't know your profession yet, but tell me once and I'll remember it."

        if (
            ("what do i like" in q)
            or ("what do i prefer" in q)
            or ("preferences" in q)
            or ("preference" in q)
            or ("favorite" in q)
        ):
            prefs = profile.get("preferences", [])
            if prefs:
                if len(prefs) == 1:
                    return self._humanize_memory_answer(
                        lead_in="You seem to prefer",
                        detail=prefs[0],
                    )

                latest = prefs[-1]
                earlier = prefs[0]
                if earlier.lower() == latest.lower():
                    return self._humanize_memory_answer(
                        lead_in="You seem to prefer",
                        detail=latest,
                    )

                return (
                    f"You first mentioned {earlier}, then updated that to {latest}, "
                    f"so I'd go with {latest} for now."
                )
            return "I don't have your preferences yet, but tell me what you like and I'll keep track."

        remembered = profile.get("name") or profile.get("location") or profile.get("profession")
        if remembered:
            details = []
            if profile.get("name"):
                details.append(f"name is {profile['name']}")
            if profile.get("location"):
                details.append(f"location is {profile['location']}")
            if profile.get("profession"):
                details.append(f"profession is {profile['profession']}")
            return "Here's what I've got so far: " + ", ".join(details) + "."

        if memory_context.strip():
            return "I'm still getting the full picture, so keep going and I'll hold onto the important bits."
        return "Got it, I'll remember that."

    def _humanize_memory_answer(self, lead_in: str, detail: str, tail: str = "") -> str:
        """
        Turn memory answers into a more natural human-style reply.
        """
        clause = lead_in.rstrip().rstrip(",")
        if tail:
            return f"{clause}, {detail}, {tail.strip()}."
        return f"{clause}, {detail}."

    def _infer_user_profile(self) -> Dict[str, object]:
        """
        Build a lightweight user profile from stored concepts and user messages.
        """
        concepts = self.long_term_memory.get_all_concepts(include_suppressed=False)
        text_candidates = [c.description for c in concepts if c.description]
        text_candidates.extend(self._user_history)

        name = None
        location = None
        profession = None
        preferences: List[str] = []

        for text in text_candidates:
            if not name:
                name = self._extract_name(text)
            if not location:
                location = self._extract_location(text)
            if not profession:
                profession = self._extract_profession(text)
            pref = self._extract_preference(text)
            if pref and pref.lower() not in {p.lower() for p in preferences}:
                preferences.append(pref)

        return {
            "name": name,
            "location": location,
            "profession": profession,
            "preferences": preferences,
        }

    def _extract_name(self, text: str) -> Optional[str]:
        person_match = re.search(r"\bPerson:\s*([A-Za-z][a-zA-Z'-]*)", text)
        if person_match:
            return person_match.group(1)

        msg_match = re.search(
            r"(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z'-]*)",
            text,
            flags=re.IGNORECASE,
        )
        if msg_match:
            return msg_match.group(1)
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:i live in|live in)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _extract_profession(self, text: str) -> Optional[str]:
        patterns = [
            r"i work as\s+(?:a|an)\s+([^.!?]+)",
            r"i work at\s+[^.!?]+\s+as\s+(?:a|an)\s+([^.!?]+)",
            r"i am\s+(?:a|an)\s+([^.!?]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_preference(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:Preference:\s*|i prefer|i love|i like|i enjoy|my favorite\s+\w+\s+is)\s*([^.!?]+)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        pref = self._normalize_preference_text(match.group(1))
        return pref if pref else None

    def _resolve_interlocutor_tag(self, text: str, source: str) -> str:
        """
        Resolve stable person tag used by context-gated retrieval.
        """
        if source != "user":
            return source
        return "user"

    def _collect_relations_from_ltm(self) -> List[Relation]:
        """Snapshot relation edges from the LTM graph."""
        relations: List[Relation] = []
        for subject_id, object_id, data in self.long_term_memory.graph.edges(data=True):
            predicate_raw = data.get("predicate", PredicateType.RELATED_TO.value)
            if hasattr(predicate_raw, "value"):
                predicate_raw = predicate_raw.value
            try:
                predicate = PredicateType(predicate_raw)
            except Exception:
                predicate = PredicateType.RELATED_TO

            relation_kwargs = {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "strength": float(data.get("strength", 1.0)),
                "bidirectional": bool(data.get("bidirectional", False)),
            }
            relation_id = data.get("id")
            if relation_id:
                relation_kwargs["id"] = relation_id
            relations.append(Relation(**relation_kwargs))
        return relations

    def _normalize_preference_text(self, text: str) -> str:
        """
        Remove trailing filler words so preferences read like a real human update.
        """
        cleaned = re.sub(
            r"\s+(?:right now|for now|currently|at the moment|now)\s*$",
            "",
            (text or "").strip(),
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" .,!?:;")

    def _apply_sleep_updates(self, prior_concepts: List[Concept], stats: Dict) -> Dict[str, int]:
        """
        Sync sleep cycle outputs back into LTM graph/cache/persistence.
        """
        updated_concepts = stats.get("updated_concepts")
        updated_relations = stats.get("updated_relations")
        retired_concepts = stats.get("retired_concepts") or []

        if updated_concepts is None:
            return {"synced_concepts": 0, "removed_concepts": 0, "synced_relations": 0}

        prior_ids = {c.id for c in prior_concepts}
        surviving_ids = set()
        for concept in list(updated_concepts) + list(retired_concepts):
            surviving_ids.add(concept.id)
            self.long_term_memory.add_concept(concept)

        removed_ids = prior_ids - surviving_ids
        for concept_id in removed_ids:
            if concept_id in self.long_term_memory._concept_cache:
                del self.long_term_memory._concept_cache[concept_id]
            if concept_id in self.long_term_memory.graph:
                self.long_term_memory.graph.remove_node(concept_id)

        synced_relations = 0
        if updated_relations is not None:
            for relation in updated_relations:
                if relation.subject_id not in self.long_term_memory.graph:
                    continue
                if relation.object_id not in self.long_term_memory.graph:
                    continue
                self.long_term_memory.add_relation(relation)
                synced_relations += 1

        return {
            "synced_concepts": len(updated_concepts),
            "removed_concepts": len(removed_ids),
            "synced_relations": synced_relations,
        }

    def _gather_cross_session_episodes(self) -> List[Episode]:
        """
        Phase 7: ask the cross-session memory pool for prior-session episodes
        to feed into the current sleep cycle. Returns [] when the pool is
        absent or disabled. Never raises — sleep must succeed even if the
        pool fails.
        """
        if self.cross_session_pool is None:
            return []
        try:
            borrowed = self.cross_session_pool.gather()
            if borrowed:
                # Surface diagnostics on the engine so wake-summary can see them.
                stats = getattr(self.cross_session_pool, "stats_dict", lambda: {})()
                self._last_cross_session_pool_stats = stats
            return borrowed or []
        except Exception as exc:
            print(f"[ChatEngine] cross-session pool failed: {exc}")
            return []

    def _check_and_trigger_sleep(self) -> Optional[Dict]:
        """
        Check if sleep should trigger and run consolidation if so.
        Returns sleep stats if sleep occurred, None otherwise.
        """
        # Get all active concepts and relations
        concepts = self.long_term_memory.get_all_concepts(include_suppressed=False)
        relations = self._collect_relations_from_ltm()

        mode, reason, _ = self.sleep_orchestrator.select_sleep_mode(
            concepts=concepts,
            relations=relations,
            turns_since_micro=self._turns_since_micro_sleep,
            session_turns=self._message_count,
        )

        if mode is None:
            return None

        # Phase 6 fix: cooldown gate. Stops the auto-sleep storm that happens
        # when a uniform-salience encoder saturates entropy and triggers
        # deep-sleep on every check. Each sleep MODE has its own minimum
        # turn-count between firings.
        from ..core.config import (
            DEEP_SLEEP_COOLDOWN_TURNS,
            MICRO_SLEEP_COOLDOWN_TURNS,
        )
        last_turn = getattr(self, "_last_sleep_turn", {})
        cooldown_required = (
            DEEP_SLEEP_COOLDOWN_TURNS if mode == "deep" else MICRO_SLEEP_COOLDOWN_TURNS
        )
        last = last_turn.get(mode, -10**6)
        if self._message_count - last < cooldown_required:
            return None

        print(f"[ChatEngine] Auto-sleep triggered ({mode}): {reason}")

        # Get episodes from working memory + cross-session pool (Phase 7) for consolidation
        episodes = list(self.working_memory.get_all())
        episodes.extend(self._gather_cross_session_episodes())

        # Run sleep cycle
        try:
            success, cycle, stats = self.sleep_orchestrator.begin_sleep_cycle(
                concepts=concepts,
                relations=relations,
                episodes=episodes,
                mode=mode,
                force=True,
                session_turns=self._message_count,
                turns_since_micro=self._turns_since_micro_sleep,
            )

            if success:
                sync_stats = self._apply_sleep_updates(concepts, stats)
                sleep_record = {
                    'timestamp': utc_isoformat(),
                    'mode': mode,
                    'reason': reason,
                    'consolidated': cycle.memories_consolidated,
                    'forgotten': cycle.memories_forgotten,
                    'dreams': len(cycle.dreams_generated),
                    'duration': cycle.nrem_duration + cycle.rem_duration,
                    'synced_concepts': sync_stats.get('synced_concepts', 0),
                    'synced_relations': sync_stats.get('synced_relations', 0),
                }
                self._sleep_history.append(sleep_record)

                # Deep sleep is disruptive enough to clear WM; micro-sleep keeps context live.
                if mode == "deep":
                    self.working_memory.clear()
                    self.self_model.on_sleep_completed({
                        'consolidated': cycle.memories_consolidated,
                        'forgotten': cycle.memories_forgotten,
                        'dreams': len(cycle.dreams_generated),
                    })

                # Phase 6 fix: record turn so the cooldown gate can use it.
                if not hasattr(self, "_last_sleep_turn"):
                    self._last_sleep_turn = {}
                self._last_sleep_turn[mode] = self._message_count

                self._turns_since_micro_sleep = 0
                return sleep_record

        except Exception as e:
            print(f"[ChatEngine] Sleep cycle failed: {e}")

        return None

    def force_sleep(self, mode: str = "deep") -> Optional[Dict]:
        """Manually trigger sleep consolidation"""
        concepts = self.long_term_memory.get_all_concepts(include_suppressed=False)
        relations = self._collect_relations_from_ltm()
        episodes = list(self.working_memory.get_all())
        # Phase 7: augment with prior-session episodes if cross-session pool active.
        episodes.extend(self._gather_cross_session_episodes())

        try:
            success, cycle, stats = self.sleep_orchestrator.begin_sleep_cycle(
                concepts=concepts,
                relations=relations,
                episodes=episodes,
                force=True,
                mode=mode,
                session_turns=self._message_count,
                turns_since_micro=self._turns_since_micro_sleep,
            )

            if success:
                sync_stats = self._apply_sleep_updates(concepts, stats)

                if mode == "deep":
                    self.self_model.on_sleep_completed({
                        'consolidated': cycle.memories_consolidated,
                        'forgotten': cycle.memories_forgotten,
                        'dreams': len(cycle.dreams_generated),
                    })
                    self.working_memory.clear()
                self._turns_since_micro_sleep = 0

                # Phase 7: record forced sleeps in history too so the IdleLearner
                # daemon and wake-summary endpoint can see the full picture.
                # Previously only auto-triggered sleeps were recorded, which made
                # autonomous sleep cycles invisible to introspection.
                sleep_record = {
                    'timestamp': utc_isoformat(),
                    'mode': mode,
                    'reason': 'forced',
                    'consolidated': cycle.memories_consolidated,
                    'forgotten': cycle.memories_forgotten,
                    'dreams': len(cycle.dreams_generated),
                    'duration': cycle.nrem_duration + cycle.rem_duration,
                    'synced_concepts': sync_stats.get('synced_concepts', 0),
                    'synced_relations': sync_stats.get('synced_relations', 0),
                }
                self._sleep_history.append(sleep_record)
                # Persist so wake-summary still sees the cycle after restart.
                if self._enable_persistence and not self.sandbox_mode:
                    try:
                        from ..core.sqlite_db import get_memory
                        get_memory().save_sleep_record(sleep_record)
                    except Exception as exc:
                        print(f"[ChatEngine] sleep_history persistence failed: {exc}")

                return {
                    'timestamp': utc_isoformat(),
                    'mode': mode,
                    'consolidated': cycle.memories_consolidated,
                    'forgotten': cycle.memories_forgotten,
                    'dreams': len(cycle.dreams_generated),
                    'nrem_duration': round(cycle.nrem_duration, 2),
                    'rem_duration': round(cycle.rem_duration, 2),
                    'synced_concepts': sync_stats.get('synced_concepts', 0),
                    'synced_relations': sync_stats.get('synced_relations', 0),
                }
        except Exception as e:
            print(f"[ChatEngine] Force sleep failed: {e}")

        return None

    def _is_introspection_query(self, message: str) -> bool:
        """Check if user is asking about SleepAI's self/awareness"""
        introspection_keywords = [
            'who are you', 'what are you', 'your name',
            'do you know', 'are you aware', 'self aware',
            'conscious', 'think about yourself', 'introspect',
            'your memory', 'your dreams', 'your existence',
            'do you remember', 'your purpose', 'yourself'
        ]
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in introspection_keywords)

    def get_memory_report(self) -> Dict:
        """Get a full report of current memory state"""
        ltm_stats = self.long_term_memory.get_stats()
        wm_episodes = self.working_memory.get_all()

        # Get readiness for sleep
        readiness = self.sleep_orchestrator.get_sleep_readiness(
            self.long_term_memory.get_all_concepts(include_suppressed=False),
            []
        )

        report = {
            'conversation_duration_minutes': round(
                (utc_now() - ensure_utc(self._conversation_start)).total_seconds() / 60, 1
            ),
            'runtime': {
                'profile': self.profile,
                'sandbox_mode': self.sandbox_mode,
                'auto_sleep_enabled': self.enable_auto_sleep,
                'sleep_check_interval': self.sleep_check_interval,
                'persistence_enabled': self._enable_persistence,
            },
            'messages_exchanged': self._message_count,
            'working_memory': {
                'size': self.working_memory.size(),
                'capacity': self.working_memory.capacity,
                'is_full': self.working_memory.is_full(),
                'recent_episodes': [
                    {'content': ep.raw_content[:60], 'source': ep.source}
                    for ep in wm_episodes[-3:]
                ]
            },
            'long_term_memory': {
                'total_concepts': ltm_stats.get('total_concepts', 0),
                'total_relations': ltm_stats.get('total_relations', 0),
                'suppressed': ltm_stats.get('suppressed_count', 0)
            },
            'sleep_readiness': readiness,
            'sleep_history': self._sleep_history[-5:],
            'total_sleeps': len(self._sleep_history)
        }

        # Add self-model / consciousness layer
        if hasattr(self, 'self_model') and self.self_model:
            report['self_model'] = self.self_model.get_self_report()

        return report

    def export_memory(
        self,
        include_suppressed: bool = True,
        include_superseded: bool = True,
    ) -> Dict:
        """Export session memory graph for backup/migration."""
        payload = self.long_term_memory.export_memory(
            include_suppressed=include_suppressed,
            include_superseded=include_superseded,
        )
        payload["session_id"] = self.session_id
        payload["profile"] = self.profile
        payload["sandbox_mode"] = self.sandbox_mode
        payload["sleep_history"] = list(self._sleep_history)
        return payload

    def import_memory(
        self,
        payload: Dict,
        replace_existing: bool = False,
    ) -> Dict[str, int]:
        """Import memory graph payload into current engine session."""
        stats = self.long_term_memory.import_memory(
            payload=payload,
            replace_existing=replace_existing,
            persist_import=not self.sandbox_mode,
        )
        # Ensure self-model pointer is valid after replace/import operations.
        self.self_model._initialize_self()
        return stats

    def reset_memory(self, clear_persistence: bool = False) -> None:
        """Reset in-memory state for this engine/session."""
        self.working_memory.clear()
        self.long_term_memory.clear(clear_persistence=clear_persistence and not self.sandbox_mode)
        self._sleep_history.clear()
        self._message_count = 0
        self._turns_since_micro_sleep = 0
        self._user_history.clear()
        self.self_model = SelfModel(long_term_memory=self.long_term_memory)
