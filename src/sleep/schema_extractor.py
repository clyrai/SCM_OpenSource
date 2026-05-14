"""
Schema extraction during REM sleep — Phase 7.

Default REM dreaming generates novel concept sequences. That is useful but
shallow. To match human REM, the agent must also detect *recurring patterns*
across episodes and abstract them into schema concepts:

  - "Caroline went to the support group" appears in 5 episodes on different
    dates → schema: "Caroline attends a recurring support group routine."
  - "I work at GreenLeaf Cafe" + "left GreenLeaf Cafe" → trajectory schema:
    "User changed employers (former: GreenLeaf Cafe)."
  - Two entities that consistently appear together across episodes →
    co-occurrence schema.

These schemas are stored as ABSTRACT-type Concepts with a `_schema=True` flag
in context_tags, and a list of source-episode IDs so the wake summary can
explain why each schema was formed.

This module is fully heuristic; no LLM call required. It works on the rolling
multi-session episode pool from M2, so patterns can span days. Production
deployments may add an LLM-backed paraphraser to clean up the schema text,
similar to how Phase 6's sleep-time paraphrase works.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import linguistic_resources as lr
from ..core.models import Concept, ConceptType, Episode, ImportanceVector, MemoryState
from ..core.time_utils import ensure_utc, utc_now


# ─── Configuration ──────────────────────────────────────────────────────────


@dataclass
class SchemaExtractorConfig:
    """Knobs controlling pattern detection sensitivity."""

    # Master switch
    enabled: bool = True

    # An entity must appear in at least this many distinct episodes to count
    # as a "recurring topic." Default 3 — two appearances is coincidence,
    # three is pattern.
    min_repetitions: int = 3

    # Two entities must co-occur in at least this many distinct episodes
    # before we emit a co-occurrence schema. Default 2.
    cooccurrence_min: int = 2

    # Hard ceiling on schemas emitted per cycle. Prevents pattern detection
    # from flooding LTM on a chatty session.
    max_schemas_per_cycle: int = 12

    # Cap how many "recent" episodes to look at, even if more were borrowed.
    max_episodes_window: int = 200

    # Time window in hours within which timestamps are considered "regular."
    # If episodes about the same entity occur within +/- this hour-range over
    # multiple weeks, a temporal schema is emitted. 0 disables temporal pass.
    temporal_window_hours: float = 24.0

    # Minimum entity length (in characters) to even consider as a candidate.
    # If 0, falls back to whatever the linguistic resources file specifies.
    min_entity_length: int = 0

    # Stoplist of entity-like tokens we should never schema on.
    # If left empty (default), pulled from linguistic_resources at runtime
    # so it tracks the locale config instead of being hardcoded here.
    entity_stoplist: Set[str] = field(default_factory=set)


# ─── Output record ──────────────────────────────────────────────────────────


@dataclass
class ExtractedSchema:
    """One detected pattern, ready to be inserted as a Concept."""

    schema_type: str  # "recurring_topic" | "cooccurrence" | "trajectory"
    description: str  # human-readable text of the pattern
    source_episode_ids: List[str] = field(default_factory=list)
    source_session_ids: List[str] = field(default_factory=list)
    occurrence_count: int = 0
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_concept(self) -> Concept:
        """Convert to a durable Concept that can be stored in LTM.

        IMPORTANT: the Concept ID is derived deterministically from the
        schema's content (type + sorted entities). This makes schema
        creation IDEMPOTENT across multiple sleep cycles: re-detecting the
        same pattern overwrites the same node instead of inserting a
        duplicate. Without this, every sleep would multiply schema concepts
        in LTM and bloat memory exponentially over a long-running session.
        """
        import hashlib
        sig = "|".join([
            self.schema_type,
            ":".join(sorted(e.lower() for e in self.entities)),
        ])
        digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()
        # Format as a UUID-shaped string so downstream code that assumes
        # UUID syntax stays happy.
        stable_id = f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

        importance = ImportanceVector(
            novelty=0.7,            # schemas are typically novel insights
            emotional=0.0,
            task_relevance=0.65,    # generally useful
            repetition=min(1.0, self.occurrence_count / 10.0),
        )
        concept = Concept(
            id=stable_id,
            type=ConceptType.ABSTRACT,
            description=self.description,
            importance=importance,
            state=MemoryState.ACTIVE,
            strength=1.2,            # slightly stronger than default
            confidence=self.confidence,
            salience_score=0.65,     # above the protect-floor by default
        )
        # Mark provenance and schema flag in context_tags so the rest of the
        # system can identify schema concepts (e.g. wake summary, retrieval).
        concept.context_tags.update({
            "_schema": True,
            "schema_type": self.schema_type,
            "source_episodes": list(self.source_episode_ids),
            "source_sessions": list(self.source_session_ids),
            "entities": list(self.entities),
            "occurrence_count": self.occurrence_count,
        })
        return concept


# ─── The extractor ─────────────────────────────────────────────────────────


_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b")  # locale-agnostic, unused for entities


class SchemaExtractor:
    """
    Heuristic pattern detector that emits schema-level Concepts during REM.

    Pipeline per call:
      1. Window the input episodes (most recent N)
      2. Tokenize each into normalized entity sets
      3. Repetition pass: entities appearing in >= min_repetitions episodes
      4. Co-occurrence pass: entity pairs appearing in >= cooccurrence_min episodes
      5. Trajectory pass: detect "from X / to Y" or "stopped X / started Y" pairs
      6. Optional temporal pass: regular cadence detection
      7. Cap, dedupe, return
    """

    def __init__(self, config: Optional[SchemaExtractorConfig] = None):
        self.config = config or SchemaExtractorConfig()
        # Last-call diagnostics for the wake-summary endpoint
        self.last_stats: Dict[str, Any] = {}
        # Compile linguistic resources once per instance. Failures here are
        # surfaced loudly because operating without them is silently broken.
        self._named_entity_re = lr.compile_named_entity_regex()
        self._trajectory_patterns = lr.compile_trajectory_patterns()
        # Resolve effective stoplist + min length: prefer per-instance config,
        # fall back to the locale-defined defaults.
        self._effective_stoplist: Set[str] = (
            set(s.lower() for s in self.config.entity_stoplist)
            if self.config.entity_stoplist
            else lr.get_stopwords("schema_extractor")
        )
        ed = lr.get_entity_detection()
        self._effective_min_entity_length: int = (
            self.config.min_entity_length
            if self.config.min_entity_length > 0
            else int(ed["min_entity_length"])
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(
        self,
        episodes: List[Episode],
        existing_concepts: Optional[List[Concept]] = None,
        now: Optional[datetime] = None,
    ) -> List[ExtractedSchema]:
        """
        Detect schemas across the given episodes. Returns a list of
        `ExtractedSchema` objects sorted by confidence descending. Caller
        is responsible for inserting them into LTM (typically via
        `Concept` conversion + `LongTermMemory.add_concept`).
        """
        stats = {
            "episodes_in": len(episodes),
            "episodes_windowed": 0,
            "candidate_entities": 0,
            "schemas_emitted": 0,
            "by_type": {},
        }
        self.last_stats = stats

        if not self.config.enabled or not episodes:
            return []

        # 1. Window
        window = self._window_episodes(episodes)
        stats["episodes_windowed"] = len(window)

        # 2. Tokenize
        ep_entities = self._tokenize_episodes(window)
        candidate_universe: Set[str] = set()
        for ents in ep_entities.values():
            candidate_universe.update(ents)
        stats["candidate_entities"] = len(candidate_universe)

        schemas: List[ExtractedSchema] = []

        # 3. Repetition pass
        schemas.extend(self._repetition_schemas(ep_entities, window))

        # 4. Co-occurrence pass
        schemas.extend(self._cooccurrence_schemas(ep_entities, window))

        # 5. Trajectory pass
        schemas.extend(self._trajectory_schemas(window))

        # 6. Temporal pass (optional)
        if self.config.temporal_window_hours > 0:
            schemas.extend(self._temporal_schemas(ep_entities, window))

        # 7. Dedupe + cap + sort
        schemas = self._dedupe_and_cap(schemas)

        stats["schemas_emitted"] = len(schemas)
        by_type: Dict[str, int] = defaultdict(int)
        for s in schemas:
            by_type[s.schema_type] += 1
        stats["by_type"] = dict(by_type)
        return schemas

    # ── Pipeline stages ────────────────────────────────────────────────────

    def _window_episodes(self, episodes: List[Episode]) -> List[Episode]:
        """Cap to the most recent N episodes."""
        if len(episodes) <= self.config.max_episodes_window:
            return list(episodes)
        # Sort by timestamp desc and take head
        return sorted(
            episodes,
            key=lambda e: ensure_utc(e.timestamp) or datetime.min,
            reverse=True,
        )[: self.config.max_episodes_window]

    def _tokenize_episodes(
        self, episodes: List[Episode]
    ) -> Dict[str, Set[str]]:
        """Map episode_id → set of normalized entity tokens (proper nouns)."""
        out: Dict[str, Set[str]] = {}
        stop = self._effective_stoplist
        min_len = self._effective_min_entity_length
        for ep in episodes:
            text = ep.raw_content or ""
            entities = set()
            for m in self._named_entity_re.finditer(text):
                tok = m.group(1).strip()
                if len(tok) < min_len:
                    continue
                low = tok.lower()
                if low in stop:
                    continue
                entities.add(tok)
            out[ep.id] = entities
        return out

    def _repetition_schemas(
        self,
        ep_entities: Dict[str, Set[str]],
        episodes: List[Episode],
    ) -> List[ExtractedSchema]:
        """Entities appearing in >= min_repetitions distinct episodes."""
        ep_index = {ep.id: ep for ep in episodes}
        # entity → set of episode_ids where it appears
        ent_to_eps: Dict[str, List[str]] = defaultdict(list)
        for ep_id, entities in ep_entities.items():
            for ent in entities:
                ent_to_eps[ent].append(ep_id)

        schemas: List[ExtractedSchema] = []
        for ent, ep_ids in ent_to_eps.items():
            if len(ep_ids) < self.config.min_repetitions:
                continue
            sources = sorted(set(ep_ids))
            session_ids = sorted({
                self._session_of(ep_index.get(eid)) for eid in sources
                if self._session_of(ep_index.get(eid))
            })
            confidence = min(0.95, 0.4 + 0.1 * len(sources))
            schema = ExtractedSchema(
                schema_type="recurring_topic",
                description=(
                    f"{ent} is a recurring topic in conversation "
                    f"({len(sources)} mentions across "
                    f"{len(session_ids) or 1} session"
                    f"{'s' if (len(session_ids) or 1) != 1 else ''})."
                ),
                source_episode_ids=sources,
                source_session_ids=session_ids,
                occurrence_count=len(sources),
                entities=[ent],
                confidence=confidence,
            )
            schemas.append(schema)
        return schemas

    def _cooccurrence_schemas(
        self,
        ep_entities: Dict[str, Set[str]],
        episodes: List[Episode],
    ) -> List[ExtractedSchema]:
        """Pairs of entities that appear together in >= cooccurrence_min episodes."""
        ep_index = {ep.id: ep for ep in episodes}
        pair_to_eps: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for ep_id, entities in ep_entities.items():
            ents = sorted(entities)
            if len(ents) < 2:
                continue
            for i, a in enumerate(ents):
                for b in ents[i + 1 :]:
                    pair_to_eps[(a, b)].append(ep_id)

        schemas: List[ExtractedSchema] = []
        for (a, b), ep_ids in pair_to_eps.items():
            if len(ep_ids) < self.config.cooccurrence_min:
                continue
            sources = sorted(set(ep_ids))
            session_ids = sorted({
                self._session_of(ep_index.get(eid)) for eid in sources
                if self._session_of(ep_index.get(eid))
            })
            confidence = min(0.90, 0.35 + 0.1 * len(sources))
            schema = ExtractedSchema(
                schema_type="cooccurrence",
                description=(
                    f"{a} and {b} are linked: they appear together in "
                    f"{len(sources)} conversation"
                    f"{'s' if len(sources) != 1 else ''}."
                ),
                source_episode_ids=sources,
                source_session_ids=session_ids,
                occurrence_count=len(sources),
                entities=[a, b],
                confidence=confidence,
            )
            schemas.append(schema)
        return schemas

    def _trajectory_schemas(self, episodes: List[Episode]) -> List[ExtractedSchema]:
        """
        Detect explicit state changes using locale-defined trajectory keywords
        loaded from the linguistic_resources file (NOT hardcoded English).
        """
        # Find episodes that are clearly trajectory turns
        trajectory_eps: List[Tuple[Episode, str]] = []  # (episode, kind)
        for ep in episodes:
            text = ep.raw_content or ""
            for pat, kind in self._trajectory_patterns:
                if pat.search(text):
                    trajectory_eps.append((ep, kind))
                    break

        if len(trajectory_eps) < 2:
            return []

        # Cluster by shared entity
        # Map entity → list of (episode, kind)
        ent_to_signals: Dict[str, List[Tuple[Episode, str]]] = defaultdict(list)
        for ep, kind in trajectory_eps:
            ents = self._extract_entities(ep.raw_content)
            for ent in ents:
                ent_to_signals[ent].append((ep, kind))

        schemas: List[ExtractedSchema] = []
        for ent, signals in ent_to_signals.items():
            if len(signals) < 2:
                continue
            # Need at least one "transition" or "past_state" signal to call it a trajectory
            kinds = {kind for _, kind in signals}
            if not (kinds & {"transition", "past_state"}):
                continue
            ep_ids = sorted({ep.id for ep, _ in signals})
            session_ids = sorted({
                self._session_of(ep) for ep, _ in signals
                if self._session_of(ep)
            })
            schemas.append(ExtractedSchema(
                schema_type="trajectory",
                description=(
                    f"State change involving {ent}: detected in "
                    f"{len(ep_ids)} episode(s) with transition language."
                ),
                source_episode_ids=ep_ids,
                source_session_ids=session_ids,
                occurrence_count=len(ep_ids),
                entities=[ent],
                confidence=min(0.85, 0.4 + 0.1 * len(ep_ids)),
            ))
        return schemas

    def _temporal_schemas(
        self,
        ep_entities: Dict[str, Set[str]],
        episodes: List[Episode],
    ) -> List[ExtractedSchema]:
        """
        Detect entities that recur on a regular cadence across multiple days.

        Cheap version: for each repeat-entity, look at the timestamps of its
        episodes; if the gaps between consecutive mentions cluster around a
        common interval (within +/- temporal_window_hours), call it a
        cadence schema.
        """
        ep_index = {ep.id: ep for ep in episodes}
        ent_to_times: Dict[str, List[datetime]] = defaultdict(list)
        for ep_id, entities in ep_entities.items():
            ep = ep_index.get(ep_id)
            if not ep:
                continue
            ts = ensure_utc(ep.timestamp)
            if not ts:
                continue
            for ent in entities:
                ent_to_times[ent].append(ts)

        schemas: List[ExtractedSchema] = []
        tol = timedelta(hours=self.config.temporal_window_hours)
        for ent, times in ent_to_times.items():
            if len(times) < 3:
                continue
            sorted_ts = sorted(times)
            gaps = [
                (sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
                for i in range(len(sorted_ts) - 1)
            ]
            if not gaps:
                continue
            avg_gap = sum(gaps) / len(gaps)
            tolerated = max(1.0, tol.total_seconds())
            consistent = all(abs(g - avg_gap) <= tolerated for g in gaps)
            if not consistent:
                continue
            avg_days = avg_gap / 86400.0
            cadence = self._cadence_label(avg_days)
            ep_ids = sorted({
                ep.id for ep_id, ents in ep_entities.items()
                if ent in ents and (ep := ep_index.get(ep_id)) is not None
            })
            session_ids = sorted({
                self._session_of(ep_index.get(eid))
                for eid in ep_ids if self._session_of(ep_index.get(eid))
            })
            # For weekly cadence, attach the modal day-of-week so the
            # description carries enough context to be recognizable
            # ("running recurs weekly on tuesdays" is more useful than
            # "running recurs weekly").
            day_phrase = ""
            if cadence == "weekly":
                day_counts = Counter(t.strftime("%A").lower() for t in sorted_ts)
                modal_day, modal_count = day_counts.most_common(1)[0]
                if modal_count >= max(2, len(sorted_ts) // 2):
                    day_phrase = f" on {modal_day}s"
            schemas.append(ExtractedSchema(
                schema_type="temporal_cadence",
                description=(
                    f"{ent} recurs on a {cadence} cadence{day_phrase} "
                    f"(observed {len(ep_ids)} times)."
                ),
                source_episode_ids=ep_ids,
                source_session_ids=session_ids,
                occurrence_count=len(ep_ids),
                entities=[ent],
                confidence=0.55,
            ))
        return schemas

    @staticmethod
    def _cadence_label(avg_days: float) -> str:
        if avg_days < 1.5:
            return "daily"
        if avg_days < 4.0:
            return "every-few-days"
        if avg_days < 9.0:
            return "weekly"
        if avg_days < 17.0:
            return "biweekly"
        if avg_days < 35.0:
            return "monthly"
        return "occasional"

    def _dedupe_and_cap(self, schemas: List[ExtractedSchema]) -> List[ExtractedSchema]:
        """Remove duplicate descriptions, sort by confidence, apply cap."""
        seen: Set[str] = set()
        deduped: List[ExtractedSchema] = []
        for s in schemas:
            key = s.description.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)
        deduped.sort(key=lambda s: (s.confidence, s.occurrence_count), reverse=True)
        return deduped[: self.config.max_schemas_per_cycle]

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _session_of(episode: Optional[Episode]) -> Optional[str]:
        if not episode or not isinstance(episode.context, dict):
            return None
        return (
            episode.context.get("_origin_session")
            or episode.context.get("session_id")
        )

    def _extract_entities(self, text: str) -> Set[str]:
        stop = self._effective_stoplist
        min_len = self._effective_min_entity_length
        return {
            m.group(1) for m in self._named_entity_re.finditer(text or "")
            if len(m.group(1)) >= min_len and m.group(1).lower() not in stop
        }
