"""
LongTermMemory: Cortical-equivalent stable, persistent memory
"""
from typing import List, Optional, Dict, Tuple, Any
import networkx as nx
import json

from .models import Concept, Relation, MemoryState, ImportanceVector, PredicateType, ConceptType
from .database import ConceptModel, RelationModel, get_session
from .sqlite_db import get_memory
from .config import ASSOCIATION_MAX_EDGES_PER_CONCEPT
from .config import CONTRADICTION_VERSION_SIMILARITY, CONTRADICTION_VERSION_CONTEXT_BONUS
from .memory_scoring import compute_consolidation_score, refresh_consolidation_score
from .time_utils import ensure_utc, utc_now, utc_isoformat


class LongTermMemory:
    """
    Cortical-equivalent: Stable, compressed, persistent semantic graph.
    Stores concepts and relations as a graph structure.
    """

    def __init__(self, persist: bool = True, vector_index: Optional[Any] = None):
        self.graph = nx.DiGraph()  # Directed graph for concept relations
        self._concept_cache = {}  # Local cache for fast access
        self._persist_enabled = bool(persist)
        self._sqlite = get_memory() if self._persist_enabled else None
        self._use_postgres = bool(self._persist_enabled)  # Try PostgreSQL first
        # Pluggable vector index. Resolution order:
        #   1. caller-provided `vector_index` argument
        #   2. SCM_VECTOR_BACKEND=pgvector → PgvectorIndex (production)
        #   3. default → InMemoryVectorIndex (numpy, single-server)
        # If pgvector is requested but unavailable (no DB, no extension,
        # missing optional deps), fall back to in-memory with a WARNING
        # so a misconfigured deploy doesn't silently lose ANN.
        if vector_index is None:
            vector_index = self._resolve_default_vector_index()
        self.vector_index = vector_index

    @staticmethod
    def _resolve_default_vector_index():
        """Pick a VectorIndex backend based on env config."""
        import os
        backend = (os.environ.get("SCM_VECTOR_BACKEND") or "").lower().strip()
        if backend == "pgvector":
            try:
                from ..retrieval.pgvector_index import PgvectorIndex
                from .config import EMBEDDING_DIM
                # Honor SCM_EMBEDDING_DIM override; otherwise use the
                # config-declared default (384 for MiniLM, 768 for nomic-
                # embed-text). Mismatches with stored vectors will raise
                # at first insert — better than silently corrupting.
                dim = int(os.environ.get("SCM_EMBEDDING_DIM", EMBEDDING_DIM))
                return PgvectorIndex(dim=dim)
            except Exception as e:
                print(
                    f"[LTM] SCM_VECTOR_BACKEND=pgvector requested but unavailable "
                    f"({type(e).__name__}: {e}); falling back to InMemoryVectorIndex"
                )
        from ..retrieval.vector_index import InMemoryVectorIndex
        return InMemoryVectorIndex()

    def set_persistence(self, enabled: bool) -> None:
        """Enable/disable persistence at runtime (used by sandbox sessions)."""
        self._persist_enabled = bool(enabled)
        if self._persist_enabled and self._sqlite is None:
            self._sqlite = get_memory()
            self._use_postgres = True
        if not self._persist_enabled:
            self._use_postgres = False

    def add_concept(
        self,
        concept: Concept,
        context_tags: Optional[Dict[str, Any]] = None,
        allow_versioning: bool = False,
        version_threshold: float = CONTRADICTION_VERSION_SIMILARITY,
    ) -> Concept:
        """Add a concept to long-term memory."""
        concept = self._prepare_concept(concept)
        if context_tags:
            concept.context_tags.update(context_tags)

        if allow_versioning:
            candidate = self._find_version_candidate(
                concept=concept,
                context_tags=concept.context_tags,
                similarity_threshold=version_threshold,
            )
            if candidate is not None:
                self._supersede_concept(candidate, concept)

        self._sync_concept(concept)
        self._persist_concept(concept)
        return concept

    def _prepare_concept(self, concept: Concept) -> Concept:
        """Normalize version metadata before persistence."""
        now = utc_now()
        if getattr(concept, "valid_from", None) is None:
            concept.valid_from = now
        if getattr(concept, "version_root", None) is None:
            concept.version_root = getattr(concept, "version_parent", None) or concept.id
        if getattr(concept, "is_current_version", None) is None:
            concept.is_current_version = True
        if getattr(concept, "state", None) is None:
            concept.state = MemoryState.ACTIVE
        refresh_consolidation_score(concept, now=now)
        return concept

    def _sync_concept(self, concept: Concept) -> None:
        """Update graph/cache with the latest concept state."""
        self.graph.add_node(concept.id, **concept.model_dump(exclude={'id'}))
        self._concept_cache[concept.id] = concept
        # Keep the vector index in sync. The encoder may not have produced
        # an embedding yet (heuristic path) — in that case, skip silently
        # and the concept becomes index-eligible whenever a backfill runs.
        if self.vector_index is not None and getattr(concept, "embedding", None):
            try:
                self.vector_index.add(concept.id, concept.embedding)
            except Exception:
                pass

    @staticmethod
    def _record_value(record: Any, key: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)

    def _concept_from_record(self, record: Any) -> Concept:
        """Construct a Concept from a DB row or ORM record."""
        concept_type = self._record_value(record, "type", ConceptType.FACT.value)
        if hasattr(concept_type, "value"):
            concept_type = concept_type.value

        valid_from = self._record_value(record, "valid_from")
        if isinstance(valid_from, str):
            valid_from = ensure_utc(valid_from)
        valid_to = self._record_value(record, "valid_to")
        if isinstance(valid_to, str):
            valid_to = ensure_utc(valid_to)
        context_tags = self._record_value(record, "context_tags", {}) or {}
        if isinstance(context_tags, str):
            try:
                context_tags = json.loads(context_tags)
            except Exception:
                context_tags = {}

        state_raw = self._record_value(record, "state", MemoryState.ACTIVE.value)
        if hasattr(state_raw, "value"):
            state_raw = state_raw.value

        created_at = ensure_utc(self._record_value(record, "created_at", utc_now())) or utc_now()
        last_accessed = ensure_utc(self._record_value(record, "last_accessed", utc_now())) or utc_now()

        concept = Concept(
            id=self._record_value(record, "id"),
            type=ConceptType(concept_type) if concept_type in [t.value for t in ConceptType] else ConceptType.FACT,
            description=self._record_value(record, "description", ""),
            embedding=json.loads(self._record_value(record, "embedding")) if self._record_value(record, "embedding") else None,
            importance=ImportanceVector(
                novelty=float(self._record_value(record, "novelty", 0.5)),
                emotional=float(self._record_value(record, "emotional", 0.0)),
                task_relevance=float(self._record_value(record, "task_relevance", 0.5)),
                repetition=float(self._record_value(record, "repetition", 0.5)),
            ),
            state=MemoryState(state_raw) if state_raw in [s.value for s in MemoryState] else MemoryState.ACTIVE,
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=int(self._record_value(record, "access_count", 0) or 0),
            strength=float(self._record_value(record, "strength", 1.0) or 1.0),
            retention_score=float(self._record_value(record, "retention_score", 0.5) or 0.5),
            consolidation_score=float(self._record_value(record, "consolidation_score", 0.5) or 0.5),
            rehearsal_count=int(self._record_value(record, "rehearsal_count", 0) or 0),
            activation_count=int(self._record_value(record, "activation_count", 0) or 0),
            association_density=float(self._record_value(record, "association_density", 0.0) or 0.0),
            decay_rate=float(self._record_value(record, "decay_rate", 0.01) or 0.01),
            confidence=float(self._record_value(record, "confidence", 0.5) or 0.5),
            schema_overlap=float(self._record_value(record, "schema_overlap", 0.0) or 0.0),
            version_parent=self._record_value(record, "version_parent"),
            version_root=self._record_value(record, "version_root"),
            valid_from=valid_from,
            valid_to=valid_to,
            is_current_version=bool(self._record_value(record, "is_current_version", 1)),
            context_tags=context_tags,
        )
        refresh_consolidation_score(concept)
        return concept

    def _concept_similarity(self, left: Concept, right: Concept) -> float:
        """Similarity between two concepts for version matching."""
        if left.embedding and right.embedding and len(left.embedding) == len(right.embedding):
            return self._cosine_similarity(left.embedding, right.embedding)
        return self._token_overlap(left.description, right.description)

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        tokens_a = {t for t in (text_a or "").lower().split() if t}
        tokens_b = {t for t in (text_b or "").lower().split() if t}
        if not tokens_a or not tokens_b:
            return 0.0
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _recency_boost(concept: Concept) -> float:
        """Recent concepts are more likely to be the active version."""
        if not concept.last_accessed:
            return 0.0
        age_seconds = max(0.0, (utc_now() - ensure_utc(concept.last_accessed)).total_seconds())
        if age_seconds <= 3600:
            return 1.0
        if age_seconds >= 86400:
            return 0.2
        return 1.0 - ((age_seconds - 3600) / (23 * 3600))

    def _context_version_bonus(self, concept: Concept, context_tags: Dict[str, Any]) -> float:
        """Small bonus when a candidate shares the same session/person context."""
        if not context_tags:
            return 0.0

        bonus = 0.0
        concept_tags = concept.context_tags or {}

        if context_tags.get("session_id") and concept_tags.get("session_id"):
            if context_tags["session_id"] == concept_tags["session_id"]:
                bonus += 0.10

        person = context_tags.get("person")
        if person and concept_tags.get("person"):
            if str(person).lower() == str(concept_tags["person"]).lower():
                bonus += CONTRADICTION_VERSION_CONTEXT_BONUS

        task = context_tags.get("task")
        if task and concept_tags.get("task"):
            if str(task).lower() == str(concept_tags["task"]).lower():
                bonus += 0.05

        return min(0.25, bonus)

    def _find_version_candidate(
        self,
        concept: Concept,
        context_tags: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = CONTRADICTION_VERSION_SIMILARITY,
    ) -> Optional[Concept]:
        """Find the most likely previous version for a contradictory concept."""
        context_tags = context_tags or {}
        concept_type = concept.type.value if hasattr(concept.type, "value") else str(concept.type)
        versionable_types = {
            t.value for t in {
                ConceptType.PREFERENCE,
                ConceptType.FACT,
                ConceptType.EVENT,
                ConceptType.LOCATION,
                ConceptType.ABSTRACT,
                ConceptType.PERSON,
            }
        }

        if concept_type not in versionable_types:
            return None

        candidates: Dict[str, Tuple[Concept, float]] = {}
        if concept.embedding:
            for candidate in self.search_by_embedding(concept.embedding, limit=10, include_history=True):
                score = self._concept_similarity(concept, candidate)
                score += self._context_version_bonus(candidate, context_tags)
                candidates[candidate.id] = (candidate, min(1.0, score))

        for candidate in self.search_by_text(concept.description, limit=10, include_history=True):
            score = self._concept_similarity(concept, candidate)
            score += self._context_version_bonus(candidate, context_tags)
            existing = candidates.get(candidate.id)
            if existing is None or score > existing[1]:
                candidates[candidate.id] = (candidate, min(1.0, score))

        if not candidates:
            for candidate in self.get_all_concepts(include_suppressed=False, include_superseded=True):
                score = self._concept_similarity(concept, candidate)
                score += self._context_version_bonus(candidate, context_tags)
                if score >= 0.55:
                    candidates[candidate.id] = (candidate, min(1.0, score))

        best: Optional[Tuple[Concept, float]] = None
        for candidate, score in candidates.values():
            if candidate.id == concept.id:
                continue
            candidate_type = candidate.type.value if hasattr(candidate.type, "value") else str(candidate.type)
            if candidate_type != concept_type:
                continue
            if score < similarity_threshold:
                continue
            if best is None:
                best = (candidate, score)
                continue
            if score > best[1]:
                best = (candidate, score)
                continue
            if score == best[1] and self._recency_boost(candidate) > self._recency_boost(best[0]):
                best = (candidate, score)

        return best[0] if best else None

    def _sync_version_lineage(self, previous: Concept, successor: Concept) -> None:
        """Mark the old concept as superseded and link it to the new version."""
        now = utc_now()
        previous.valid_to = now
        previous.is_current_version = False
        previous.state = MemoryState.ARCHIVED
        previous.context_tags = previous.context_tags or {}
        previous.context_tags["superseded_by"] = successor.id
        previous.context_tags["superseded_at"] = now.isoformat()
        if previous.version_root is None:
            previous.version_root = previous.id
        previous.context_tags["version_root"] = previous.version_root
        successor.version_parent = previous.id
        successor.version_root = previous.version_root or previous.id
        successor.valid_from = now
        successor.valid_to = None
        successor.is_current_version = True
        successor.context_tags = successor.context_tags or {}
        successor.context_tags["version_parent"] = previous.id
        successor.context_tags["version_root"] = successor.version_root

        self._sync_concept(previous)
        self._persist_concept(previous)

        contradiction = Relation(
            subject_id=previous.id,
            predicate=PredicateType.CONTRADICTS,
            object_id=successor.id,
            strength=1.0,
            bidirectional=False,
        )
        self.add_relation(contradiction)

    def _supersede_concept(self, previous: Concept, successor: Concept) -> None:
        """Version-safe update path used when a contradiction is detected."""
        self._sync_version_lineage(previous, successor)

    def add_relation(self, relation: Relation) -> Relation:
        """Add a relation between concepts"""
        self.graph.add_edge(
            relation.subject_id,
            relation.object_id,
            predicate=relation.predicate,
            strength=relation.strength,
            id=relation.id
        )

        self._persist_relation(relation)
        self._refresh_association_density([relation.subject_id, relation.object_id])
        return relation

    def _refresh_association_density(self, concept_ids: Optional[List[str]] = None) -> None:
        """Refresh association density from the current graph degree."""
        if concept_ids is None:
            concept_ids = list(self.graph.nodes())

        for concept_id in concept_ids:
            if concept_id not in self.graph:
                continue

            concept = self.get_concept(concept_id)
            if concept is None:
                continue

            degree = self.graph.out_degree(concept_id) + self.graph.in_degree(concept_id)
            density = round(
                min(1.0, degree / max(1.0, ASSOCIATION_MAX_EDGES_PER_CONCEPT * 2.0)),
                4,
            )
            concept.association_density = density
            refresh_consolidation_score(concept)

            self._concept_cache[concept_id] = concept
            if concept_id in self.graph.nodes:
                self.graph.nodes[concept_id]["association_density"] = density
                self.graph.nodes[concept_id]["consolidation_score"] = concept.consolidation_score

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieve a concept by ID"""
        # Check cache first
        if concept_id in self._concept_cache:
            return self._concept_cache[concept_id]

        # Load from PostgreSQL first if enabled.
        if self._use_postgres:
            try:
                session = get_session()
                try:
                    db_concept = session.query(ConceptModel).filter(
                        ConceptModel.id == concept_id
                    ).first()

                    if db_concept:
                        concept = self._concept_from_record(db_concept)
                        self._concept_cache[concept_id] = concept
                        return concept
                finally:
                    session.close()
            except Exception as e:
                self._use_postgres = False
                print(f"[LTM] PostgreSQL concept lookup failed, using SQLite: {e}")

        # SQLite fallback lookup.
        if self._sqlite is not None:
            row = self._sqlite.get_concept(concept_id)
            if row:
                concept = self._concept_from_record(row)
                self._concept_cache[concept_id] = concept
                return concept

        return None

    def get_related_concepts(
        self,
        concept_id: str,
        depth: int = 1,
        include_history: bool = False,
    ) -> List[Concept]:
        """Get concepts related to given concept"""
        related = []
        if concept_id not in self.graph:
            return related
        try:
            for node_id in nx.single_source_shortest_path_length(
                self.graph, concept_id, cutoff=depth
            ):
                if node_id != concept_id:
                    node_data = self.graph.nodes.get(node_id, {})
                    if not include_history:
                        if node_data.get('state') in {
                            MemoryState.SUPPRESSED.value,
                            MemoryState.ARCHIVED.value,
                        }:
                            continue
                        if not bool(node_data.get('is_current_version', True)):
                            continue
                    concept = self.get_concept(node_id)
                    if concept:
                        related.append(concept)
        except (nx.NetworkXError, nx.NetworkXException):
            pass

        related.sort(
            key=lambda c: (
                compute_consolidation_score(c),
                c.importance.overall if c.importance else 0.0,
            ),
            reverse=True,
        )
        return related

    def get_all_relations(self, include_history: bool = False) -> List[Relation]:
        """Return relation objects from the graph."""
        relations: List[Relation] = []
        for subject_id, object_id, data in self.graph.edges(data=True):
            if not include_history:
                subject = self.graph.nodes.get(subject_id, {})
                object_ = self.graph.nodes.get(object_id, {})
                if subject.get("state") in {
                    MemoryState.SUPPRESSED.value,
                    MemoryState.ARCHIVED.value,
                } or object_.get("state") in {
                    MemoryState.SUPPRESSED.value,
                    MemoryState.ARCHIVED.value,
                }:
                    continue
                if not bool(subject.get("is_current_version", True)) or not bool(object_.get("is_current_version", True)):
                    continue

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

    def search_by_embedding(
        self,
        query_embedding: List[float],
        limit: int = 5,
        include_history: bool = False,
    ) -> List[Concept]:
        """Search concepts by embedding similarity"""
        # Simple cosine similarity on cached concepts
        candidates = []

        for concept in self.get_all_concepts(
            include_suppressed=False,
            include_superseded=include_history,
        ):
            if concept.embedding:
                similarity = self._cosine_similarity(query_embedding, concept.embedding)
                candidates.append((
                    concept,
                    similarity,
                    compute_consolidation_score(concept),
                    concept.importance.overall if concept.importance else 0.0,
                ))

        # Sort by similarity and return top N
        candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        return [c[0] for c in candidates[:limit]]

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        include_history: bool = False,
    ) -> List[Concept]:
        """Search concepts by text description.

        Phase 6 paraphrase rewrites concept.description on consolidation but
        preserves the original text in context_tags["original_description"].
        We match against BOTH so verbatim entity strings (employer names,
        allergies, etc.) remain retrievable after deep-sleep.
        """
        query_lower = query.lower()
        matches = []

        for concept in self.get_all_concepts(
            include_suppressed=False,
            include_superseded=include_history,
        ):
            blobs: List[str] = [concept.description or ""]
            tags = concept.context_tags or {}
            orig = tags.get("original_description")
            if isinstance(orig, str) and orig:
                blobs.append(orig)
            if any(query_lower in b.lower() for b in blobs):
                matches.append((
                    concept,
                    concept.importance.overall,
                    compute_consolidation_score(concept),
                ))

        # Sort by relevance score
        matches.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [m[0] for m in matches[:limit]]

    def update_concept(self, concept_id: str, updates: Dict) -> bool:
        """Update a concept's properties"""
        concept = self.get_concept(concept_id)
        if not concept:
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(concept, key):
                setattr(concept, key, value)

        # Update last accessed
        concept.last_accessed = utc_now()
        concept.access_count += 1
        refresh_consolidation_score(concept)

        # Update graph node
        self.graph.nodes[concept_id].update(concept.model_dump(exclude={'id'}))

        # Persist
        self._persist_concept(concept)

        return True

    def update_importance(self, concept_id: str, importance_vec) -> bool:
        """Update concept's importance vector"""
        concept = self.get_concept(concept_id)
        if concept:
            concept.importance = importance_vec
            refresh_consolidation_score(concept)
            self._persist_concept(concept)
            return True
        return False

    def remove_concept(self, concept_id: str, soft: bool = True) -> bool:
        """Remove concept from memory"""
        if concept_id in self._concept_cache:
            del self._concept_cache[concept_id]

        if soft:
            # Soft delete - mark as suppressed. Keep the vector index entry
            # so the concept remains discoverable; retrieval filters
            # suppressed/superseded concepts at the layer above.
            if concept_id in self.graph:
                self.graph.nodes[concept_id]['state'] = MemoryState.SUPPRESSED.value
        else:
            # Hard delete
            self.graph.remove_node(concept_id)
            if self.vector_index is not None:
                try:
                    self.vector_index.remove(concept_id)
                except Exception:
                    pass

        return True

    def get_all_concepts(
        self,
        include_suppressed: bool = False,
        include_superseded: bool = False,
    ) -> List[Concept]:
        """Get all concepts in memory"""
        concepts = []

        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if not include_suppressed and node_data.get('state') == MemoryState.SUPPRESSED.value:
                continue
            if not include_superseded:
                if node_data.get('state') == MemoryState.ARCHIVED.value:
                    continue
                if not bool(node_data.get('is_current_version', True)):
                    continue
            concept = self.get_concept(node_id)
            if concept:
                concepts.append(concept)

        return concepts

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        total = len(self.graph.nodes())
        suppressed = sum(
            1 for n in self.graph.nodes()
            if self.graph.nodes[n].get('state') == MemoryState.SUPPRESSED.value
        )
        archived = sum(
            1 for n in self.graph.nodes()
            if self.graph.nodes[n].get('state') == MemoryState.ARCHIVED.value
        )
        versioned = sum(
            1 for n in self.graph.nodes()
            if self.graph.nodes[n].get('version_parent')
        )

        return {
            'total_concepts': total,
            'total_relations': len(self.graph.edges()),
            'suppressed_count': suppressed,
            'archived_count': archived,
            'versioned_count': versioned,
        }

    @staticmethod
    def _iso_or_none(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            normalized = ensure_utc(value)
            if normalized is not None:
                return normalized.isoformat()
        except Exception:
            pass
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    def _lineage_sort_key(self, concept: Concept) -> Tuple[float, float, str]:
        valid_from = ensure_utc(getattr(concept, "valid_from", None))
        created_at = ensure_utc(getattr(concept, "created_at", None))
        valid_ts = valid_from.timestamp() if valid_from is not None else 0.0
        created_ts = created_at.timestamp() if created_at is not None else 0.0
        return (valid_ts, created_ts, concept.id)

    def get_lineage(self, concept_id: str) -> Dict[str, Any]:
        """Return version lineage + contradiction edges for one concept."""
        target = self.get_concept(concept_id)
        if target is None:
            return {}

        root_id = getattr(target, "version_root", None) or target.id
        lineage_map: Dict[str, Concept] = {}

        for concept in self.get_all_concepts(
            include_suppressed=True,
            include_superseded=True,
        ):
            concept_root = getattr(concept, "version_root", None) or concept.id
            if concept.id == concept_id or concept_root == root_id:
                lineage_map[concept.id] = concept

        if target.id not in lineage_map:
            lineage_map[target.id] = target

        ordered = sorted(lineage_map.values(), key=self._lineage_sort_key)
        current = next((c for c in ordered if getattr(c, "is_current_version", False)), target)

        versions: List[Dict[str, Any]] = []
        for concept in ordered:
            tags = concept.context_tags if isinstance(concept.context_tags, dict) else {}
            versions.append({
                "id": concept.id,
                "type": concept.type.value if hasattr(concept.type, "value") else str(concept.type),
                "description": concept.description,
                "state": concept.state.value if hasattr(concept.state, "value") else str(concept.state),
                "version_root": getattr(concept, "version_root", None) or concept.id,
                "version_parent": getattr(concept, "version_parent", None),
                "is_current_version": bool(getattr(concept, "is_current_version", True)),
                "valid_from": self._iso_or_none(getattr(concept, "valid_from", None)),
                "valid_to": self._iso_or_none(getattr(concept, "valid_to", None)),
                "created_at": self._iso_or_none(getattr(concept, "created_at", None)),
                "last_accessed": self._iso_or_none(getattr(concept, "last_accessed", None)),
                "superseded_by": tags.get("superseded_by"),
                "superseded_at": tags.get("superseded_at"),
                "provenance": {
                    "source": tags.get("source"),
                    "session_id": tags.get("session_id"),
                    "person": tags.get("person"),
                    "task": tags.get("task"),
                },
            })

        lineage_ids = set(lineage_map.keys())
        conflicts: List[Dict[str, Any]] = []
        for relation in self.get_all_relations(include_history=True):
            predicate = relation.predicate.value if hasattr(relation.predicate, "value") else str(relation.predicate)
            if predicate != PredicateType.CONTRADICTS.value:
                continue
            if relation.subject_id not in lineage_ids or relation.object_id not in lineage_ids:
                continue
            conflicts.append({
                "relation_id": relation.id,
                "from": relation.subject_id,
                "to": relation.object_id,
                "predicate": predicate,
                "strength": float(relation.strength),
                "created_at": self._iso_or_none(getattr(relation, "created_at", None)),
            })

        return {
            "memory_id": concept_id,
            "version_root": root_id,
            "current_id": current.id,
            "version_count": len(versions),
            "conflict_count": len(conflicts),
            "versions": versions,
            "conflicts": conflicts,
        }

    def _persist_concept(self, concept: Concept):
        """Save concept to database (PostgreSQL -> SQLite fallback)"""
        if not self._persist_enabled:
            return

        if self._use_postgres:
            try:
                session = get_session()
                try:
                    db_concept = session.query(ConceptModel).filter(
                        ConceptModel.id == concept.id
                    ).first()

                    importance_dict = concept.importance.model_dump()

                    if db_concept:
                        db_concept.type = concept.type.value if hasattr(concept.type, 'value') else concept.type
                        db_concept.description = concept.description
                        db_concept.embedding = json.dumps(concept.embedding) if concept.embedding else None
                        db_concept.novelty = importance_dict.get('novelty', 0.5)
                        db_concept.emotional = importance_dict.get('emotional', 0.0)
                        db_concept.task_relevance = importance_dict.get('task_relevance', 0.5)
                        db_concept.repetition = importance_dict.get('repetition', 0.5)
                        db_concept.importance_score = concept.importance.overall
                        db_concept.last_accessed = concept.last_accessed
                        db_concept.access_count = concept.access_count
                        db_concept.strength = concept.strength
                        db_concept.state = concept.state.value if hasattr(concept, 'state') else 'active'
                        db_concept.version_root = getattr(concept, 'version_root', None)
                        db_concept.version_parent = getattr(concept, 'version_parent', None)
                        db_concept.valid_from = getattr(concept, 'valid_from', None)
                        db_concept.valid_to = getattr(concept, 'valid_to', None)
                        db_concept.is_current_version = 1 if getattr(concept, 'is_current_version', True) else 0
                        db_concept.context_tags = json.dumps(getattr(concept, "context_tags", {}) or {}, default=str)
                    else:
                        db_concept = ConceptModel(
                            id=concept.id,
                            type=concept.type.value,
                            description=concept.description,
                            embedding=json.dumps(concept.embedding) if concept.embedding else None,
                            novelty=importance_dict.get('novelty', 0.5),
                            emotional=importance_dict.get('emotional', 0.0),
                            task_relevance=importance_dict.get('task_relevance', 0.5),
                            repetition=importance_dict.get('repetition', 0.5),
                            importance_score=concept.importance.overall,
                            created_at=concept.created_at,
                            last_accessed=concept.last_accessed,
                            access_count=concept.access_count,
                            strength=concept.strength,
                            state=concept.state.value if hasattr(concept, 'state') else 'active',
                            version_root=getattr(concept, 'version_root', None),
                            version_parent=getattr(concept, 'version_parent', None),
                            valid_from=getattr(concept, 'valid_from', None),
                            valid_to=getattr(concept, 'valid_to', None),
                            is_current_version=1 if getattr(concept, 'is_current_version', True) else 0,
                            context_tags=json.dumps(getattr(concept, "context_tags", {}) or {}, default=str),
                        )
                        session.add(db_concept)

                    session.commit()
                    return
                except Exception as e:
                    session.rollback()
                    self._use_postgres = False
                    print(f"[LTM] PostgreSQL failed, switching to SQLite: {e}")
                finally:
                    session.close()
            except Exception as e:
                self._use_postgres = False
                print(f"[LTM] PostgreSQL unavailable, using SQLite: {e}")

        # SQLite fallback
        if self._sqlite is not None:
            self._sqlite.save_concept(concept)

    def _persist_relation(self, relation: Relation):
        """Save relation to database (PostgreSQL -> SQLite fallback)"""
        if not self._persist_enabled:
            return

        if self._use_postgres:
            try:
                session = get_session()
                try:
                    db_relation = RelationModel(
                        id=relation.id,
                        subject_id=relation.subject_id,
                        predicate=relation.predicate,
                        object_id=relation.object_id,
                        strength=relation.strength,
                        created_at=relation.created_at,
                        bidirectional=relation.bidirectional
                    )
                    session.add(db_relation)
                    session.commit()
                    return
                except Exception as e:
                    session.rollback()
                    self._use_postgres = False
                    print(f"[LTM] PostgreSQL failed, switching to SQLite: {e}")
                finally:
                    session.close()
            except Exception as e:
                self._use_postgres = False
                print(f"[LTM] PostgreSQL unavailable, using SQLite: {e}")

        if self._sqlite is not None:
            self._sqlite.save_relation(relation)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Defensively returns 0.0 when shapes mismatch (mixed-vintage data
        where some concepts were embedded by a 384-dim model and others
        by a 768-dim model). Without this guard the entire sleep cycle
        crashes on the first mismatched pair.
        """
        if not vec1 or not vec2:
            return 0.0
        import numpy as np

        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)
        if v1.shape != v2.shape:
            return 0.0
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def load_from_db(self):
        """Load all concepts and relations from database into graph"""
        if not self._persist_enabled:
            return

        if self._use_postgres:
            try:
                session = get_session()
                try:
                    # Load concepts
                    db_concepts = session.query(ConceptModel).all()
                    for db_concept in db_concepts:
                        self._sync_concept(self._concept_from_record(db_concept))

                    # Load relations
                    db_relations = session.query(RelationModel).all()
                    for db_relation in db_relations:
                        relation = Relation(
                            id=db_relation.id,
                            subject_id=db_relation.subject_id,
                            predicate=db_relation.predicate,
                            object_id=db_relation.object_id,
                            strength=db_relation.strength,
                            created_at=db_relation.created_at,
                            bidirectional=db_relation.bidirectional
                        )
                        self.add_relation(relation)
                    return
                finally:
                    session.close()
            except Exception as e:
                self._use_postgres = False
                print(f"[LTM] PostgreSQL load failed, using SQLite: {e}")

        # SQLite fallback
        if self._sqlite is not None:
            for row in self._sqlite.get_all_concepts_raw():
                concept = self._concept_from_record(row)
                self._sync_concept(concept)

            for row in self._sqlite.get_all_relations_raw():
                relation = Relation(
                    id=row['id'],
                    subject_id=row['subject_id'],
                    predicate=row['predicate'],
                    object_id=row['object_id'],
                    strength=row.get('strength', 1.0),
                    created_at=ensure_utc(row['created_at']) or utc_now(),
                    bidirectional=bool(row.get('bidirectional', 0)),
                )
                self.graph.add_edge(
                    relation.subject_id,
                    relation.object_id,
                    predicate=relation.predicate,
                    strength=relation.strength,
                    id=relation.id,
                )

    def clear(self, clear_persistence: bool = False) -> None:
        """Clear in-memory graph/cache and optionally persistent concept/relation rows."""
        self.graph.clear()
        self._concept_cache.clear()
        if clear_persistence:
            self._clear_persistence_store()

    def _clear_persistence_store(self) -> None:
        """Delete persisted concept/relation rows from active backends."""
        if not self._persist_enabled:
            return

        if self._use_postgres:
            try:
                session = get_session()
                try:
                    session.query(RelationModel).delete()
                    session.query(ConceptModel).delete()
                    session.commit()
                    return
                except Exception as exc:
                    session.rollback()
                    self._use_postgres = False
                    print(f"[LTM] PostgreSQL clear failed, switching to SQLite: {exc}")
                finally:
                    session.close()
            except Exception as exc:
                self._use_postgres = False
                print(f"[LTM] PostgreSQL clear unavailable, using SQLite: {exc}")

        if self._sqlite is not None:
            self._sqlite.clear_concepts_relations()

    def export_memory(
        self,
        include_suppressed: bool = True,
        include_superseded: bool = True,
    ) -> Dict[str, Any]:
        """
        Serialize the memory graph into a JSON-safe payload.
        Useful for backup, migration, and cold-start bootstrapping.
        """
        concepts = self.get_all_concepts(
            include_suppressed=include_suppressed,
            include_superseded=include_superseded,
        )
        relations = self.get_all_relations(include_history=include_superseded)

        concept_payload = [concept.model_dump(mode="json") for concept in concepts]
        relation_payload = [relation.model_dump(mode="json") for relation in relations]

        return {
            "schema_version": "scm-memory-export-v1",
            "exported_at_utc": utc_isoformat(),
            "counts": {
                "concepts": len(concept_payload),
                "relations": len(relation_payload),
            },
            "concepts": concept_payload,
            "relations": relation_payload,
        }

    def import_memory(
        self,
        payload: Dict[str, Any],
        replace_existing: bool = False,
        persist_import: bool = True,
    ) -> Dict[str, int]:
        """
        Ingest a serialized memory payload produced by export_memory().
        """
        concepts_payload = payload.get("concepts") if isinstance(payload, dict) else None
        relations_payload = payload.get("relations") if isinstance(payload, dict) else None
        if not isinstance(concepts_payload, list):
            raise ValueError("Invalid memory payload: 'concepts' must be a list")
        if relations_payload is None:
            relations_payload = []
        if not isinstance(relations_payload, list):
            raise ValueError("Invalid memory payload: 'relations' must be a list")

        imported_concepts = 0
        imported_relations = 0
        skipped_relations = 0

        if replace_existing:
            self.clear(clear_persistence=persist_import)

        previous_persistence = self._persist_enabled
        if not persist_import:
            self.set_persistence(False)

        try:
            for concept_data in concepts_payload:
                if not isinstance(concept_data, dict):
                    continue
                concept = Concept(**concept_data)
                concept = self._prepare_concept(concept)
                self._sync_concept(concept)
                self._persist_concept(concept)
                imported_concepts += 1

            for relation_data in relations_payload:
                if not isinstance(relation_data, dict):
                    continue
                relation = Relation(**relation_data)
                if relation.subject_id not in self.graph or relation.object_id not in self.graph:
                    skipped_relations += 1
                    continue
                self.add_relation(relation)
                imported_relations += 1

        finally:
            if not persist_import:
                self.set_persistence(previous_persistence)

        return {
            "concepts_imported": imported_concepts,
            "relations_imported": imported_relations,
            "relations_skipped": skipped_relations,
        }
