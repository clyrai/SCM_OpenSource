"""
Curiosity Engine — Phase 7's A-Z autonomous learning feature.

While the agent sleeps, it doesn't only consolidate what the user said. It
also identifies *what it doesn't know* about topics the user keeps bringing
up, and (with permission) reaches into configured knowledge sources to
fill those gaps. The result: an agent that grows beyond explicit instruction.

Two-phase pipeline:

  1. Gap detection — find named entities that appear in episodes but have
     no descriptive concept in long-term memory. Sort by frequency.
  2. Gap filling   — for each gap, query configured sources in priority
     order. The first source to produce a brief wins. The brief is stored
     as a new Concept tagged `_curiosity=True` for provenance.

Privacy model:
  - Disabled by default. Set `CURIOSITY_ENGINE_ENABLED=true` to opt in.
  - Sources are explicit. The default ships with `StaticDictionarySource`
    (no network, no filesystem) and `LocalDocsSource` (user-supplied folder).
    Network-based sources like Wikipedia must be wired in by the user.
  - Hard caps on gaps-per-cycle, brief length, and cost prevent runaway use.
  - Every filled gap is audited in concept context_tags and surfaced in the
    wake summary so the user always sees what the agent learned.

The engine is heuristic and bounded. It is not a substitute for retrieval
or for the LLM — it's a way to give the agent a notion of self-directed
study during downtime, like a student reviewing notes before class.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core import linguistic_resources as lr
from ..core.models import Concept, ConceptType, Episode, ImportanceVector, MemoryState
from ..core.time_utils import utc_isoformat, utc_now


# ─── Configuration ──────────────────────────────────────────────────────────


@dataclass
class CuriosityConfig:
    """Sensitivity + safety knobs for the engine."""

    enabled: bool = False

    # An entity must appear in at least this many distinct episodes to count
    # as a real gap. Lower = more aggressive learning.
    min_occurrences: int = 2

    # Max gaps to fill per sleep cycle. Caps cost and prevents runaway behavior.
    max_gaps_per_cycle: int = 3

    # Hard cap on the length (chars) of a single ingested brief.
    max_brief_chars: int = 280

    # Stoplist of entity-likes we should never look up. Empty by default →
    # falls back to the locale-defined stoplist for "curiosity_engine"
    # loaded from linguistic_resources. Override per-instance to customize.
    entity_stoplist: Set[str] = field(default_factory=set)

    # Minimum entity length (chars) before we even consider it a candidate.
    # 0 = use the locale default from linguistic_resources.
    min_entity_length: int = 0


# ─── Records ────────────────────────────────────────────────────────────────


@dataclass
class KnowledgeGap:
    """One identified gap, ready to be looked up."""
    entity: str
    occurrence_count: int = 0
    seen_in_episode_ids: List[str] = field(default_factory=list)
    seen_in_session_ids: List[str] = field(default_factory=list)
    has_existing_concept: bool = False


@dataclass
class FilledGap:
    """Result of filling one gap from one source."""
    entity: str
    source_name: str
    brief: str
    fetched_at: datetime = field(default_factory=utc_now)
    concept: Optional[Concept] = None


# ─── Source interface ──────────────────────────────────────────────────────


class CuriositySource(ABC):
    """Abstract base class for any knowledge source."""

    name: str = "abstract"

    @abstractmethod
    def lookup(self, entity: str) -> Optional[str]:
        """
        Return a short, factual brief about the entity, or None if the source
        has no answer. Implementations should be defensive: never raise, never
        block longer than a couple of seconds.
        """
        ...

    def is_available(self) -> bool:
        """Sources can advertise availability (e.g., file exists, network up)."""
        return True


# ─── Built-in sources ──────────────────────────────────────────────────────


class StaticDictionarySource(CuriositySource):
    """
    Pre-loaded mapping of entity → brief. The simplest possible source.
    Useful for:
      - Tests (deterministic, no I/O)
      - User-supplied glossaries (drop a JSON file into the agent)
      - Default ship-with-the-package coverage (common tools, places, etc.)
    """

    name = "static_dictionary"

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self._map: Dict[str, str] = {
            (k or "").strip().lower(): (v or "")
            for k, v in (mapping or {}).items()
            if k and v
        }

    @classmethod
    def from_json(cls, path: Path) -> "StaticDictionarySource":
        try:
            data = json.loads(Path(path).read_text())
            if isinstance(data, dict):
                return cls(data)
        except Exception:
            pass
        return cls({})

    def lookup(self, entity: str) -> Optional[str]:
        if not entity:
            return None
        return self._map.get(entity.strip().lower())


class LLMSource(CuriositySource):
    """
    LLM-backed knowledge source. Asks an OpenAI-compatible / Ollama LLM for
    a one-sentence brief on the entity. This is the source that turns the
    curiosity engine from "looks things up in your glossary" into "actually
    generates new knowledge during sleep."

    Privacy / safety:
      - Opt-in only. Caller must explicitly construct + pass the source.
      - Each lookup costs one short LLM call (≤ 64 output tokens).
      - The brief is bounded by `max_brief_chars` upstream.
      - Failures are silent — the engine falls through to the next source.

    The actual prompt is loaded from linguistic_resources at init time so
    locales / domain-tuning can override it without code changes.
    """

    name = "llm"

    DEFAULT_PROMPT = (
        "Write ONE concise factual sentence (max 30 words) describing what "
        "{entity} is. If you don't know with confidence, reply exactly: "
        "UNKNOWN. Do not speculate."
    )

    def __init__(self, llm, prompt_template: Optional[str] = None, max_tokens: int = 64):
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.max_tokens = int(max_tokens)

    def is_available(self) -> bool:
        return self.llm is not None and hasattr(self.llm, "_chat")

    def lookup(self, entity: str) -> Optional[str]:
        if not self.is_available() or not entity:
            return None
        prompt = self.prompt_template.format(entity=entity)
        try:
            out = self.llm._chat(prompt, num_predict=self.max_tokens)
        except Exception:
            return None
        text = (out or "").strip().strip('"').strip("'")
        if not text:
            return None
        # If the LLM admits ignorance, treat as miss so the next source can try.
        if text.upper().startswith("UNKNOWN"):
            return None
        # Also reject hallucination tells — empty / single-token answers.
        if len(text) < 8:
            return None
        return text


class LocalDocsSource(CuriositySource):
    """
    Searches a folder of plain-text / markdown notes for the first paragraph
    that mentions the entity. Pure local, zero-network, zero-API.

    The user points the engine at a folder of their own notes (e.g.,
    ~/Documents/Notes). Privacy: only files inside that folder are read.
    """

    name = "local_docs"

    SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".org")

    def __init__(self, folder: Path, max_files_scanned: int = 200, max_chars: int = 280):
        self.folder = Path(folder).expanduser() if folder else None
        self.max_files_scanned = max_files_scanned
        self.max_chars = max_chars

    def is_available(self) -> bool:
        return bool(self.folder and self.folder.exists() and self.folder.is_dir())

    def lookup(self, entity: str) -> Optional[str]:
        if not self.is_available():
            return None
        needle = entity.strip().lower()
        if not needle:
            return None
        try:
            scanned = 0
            for path in self._iter_doc_files():
                if scanned >= self.max_files_scanned:
                    break
                scanned += 1
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                lower = text.lower()
                idx = lower.find(needle)
                if idx < 0:
                    continue
                # Pull the paragraph containing the hit.
                snippet = self._paragraph_around(text, idx)
                if snippet:
                    return snippet[: self.max_chars]
        except Exception:
            return None
        return None

    def _iter_doc_files(self):
        for path in self.folder.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in self.SUFFIXES:
                yield path

    @staticmethod
    def _paragraph_around(text: str, idx: int) -> str:
        # Find paragraph boundaries (\n\n) around idx
        before = text.rfind("\n\n", 0, idx)
        after = text.find("\n\n", idx)
        start = before + 2 if before >= 0 else 0
        end = after if after >= 0 else len(text)
        return text[start:end].strip()


# ─── The engine ────────────────────────────────────────────────────────────


class CuriosityEngine:
    """Orchestrates gap detection and filling during deep-sleep."""

    def __init__(
        self,
        sources: Optional[List[CuriositySource]] = None,
        config: Optional[CuriosityConfig] = None,
    ):
        self.sources = sources or []
        self.config = config or CuriosityConfig()
        # Last-call diagnostics for the wake summary
        self.last_stats: Dict[str, Any] = {}
        # Resolve effective stoplist + entity regex from linguistic_resources
        # at init time so reloads take effect on next instance.
        self._effective_stoplist: Set[str] = (
            set(s.lower() for s in self.config.entity_stoplist)
            if self.config.entity_stoplist
            else lr.get_stopwords("curiosity_engine")
        )
        ed = lr.get_entity_detection()
        self._effective_min_entity_length: int = (
            self.config.min_entity_length
            if self.config.min_entity_length > 0
            else int(ed["min_entity_length"])
        )
        self._named_entity_re = lr.compile_named_entity_regex()

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        episodes: List[Episode],
        existing_concepts: List[Concept],
    ) -> List[FilledGap]:
        """
        Detect gaps and fill them. Returns the list of successfully filled
        gaps; each one carries a Concept that's ready to be inserted into LTM.
        Always safe to call — disabled or misconfigured returns [].
        """
        stats: Dict[str, Any] = {
            "enabled": self.config.enabled,
            "sources": [s.name for s in self.sources],
            "candidates": 0,
            "gaps_detected": 0,
            "gaps_filled": 0,
            "filled_entities": [],
            "lookup_failures": [],
        }
        self.last_stats = stats

        if not self.config.enabled or not self.sources or not episodes:
            return []

        all_gaps = self.detect_gaps(episodes, existing_concepts)
        stats["candidates"] = len(all_gaps)

        # Try candidates in priority order until we hit max_gaps_per_cycle OR
        # exhaust the list. Lookup failures don't waste a slot — they just
        # cause the next candidate to be tried.
        filled: List[FilledGap] = []
        attempted = 0
        # Hard ceiling on total lookup attempts to bound source-call costs even
        # when many lookups miss. Default: 4× the fill target.
        max_attempts = self.config.max_gaps_per_cycle * 4
        for gap in all_gaps:
            if len(filled) >= self.config.max_gaps_per_cycle:
                break
            if attempted >= max_attempts:
                break
            attempted += 1
            result = self._fill_one(gap)
            if result is None:
                stats["lookup_failures"].append(gap.entity)
                continue
            filled.append(result)
            stats["filled_entities"].append(gap.entity)
        stats["gaps_detected"] = attempted
        stats["gaps_filled"] = len(filled)
        return filled

    # ── Gap detection ──────────────────────────────────────────────────────

    def detect_gaps(
        self,
        episodes: List[Episode],
        existing_concepts: List[Concept],
    ) -> List[KnowledgeGap]:
        """
        An entity is a 'gap' if:
          - It appears in >= min_occurrences distinct episodes
          - No existing LTM concept *defines* it (see _is_definitional)
          - It is not on the stoplist
        Returned list is sorted by occurrence_count desc.

        Note on the "definitional" filter: a concept like "GreenLeaf is a
        coffee shop" defines the entity GreenLeaf and should block curiosity.
        A concept like "Caroline mentioned GreenLeaf" merely references the
        entity — it does NOT block curiosity, because the agent doesn't
        actually understand what GreenLeaf is yet.
        """
        # Set of entities the curiosity engine has already learned about
        # (any prior cycle's curiosity-filled concept counts as "known").
        already_learned: Set[str] = set()
        # Set of entities described by a non-curiosity, non-schema
        # definitional concept. We deliberately exclude our own meta-concepts
        # (schemas, curiosity briefs) so they don't block future passes:
        # a schema like "Datadog is a recurring topic" looks definitional
        # to a naive regex but is meta-information about *the pattern*, not
        # about Datadog itself.
        defined_entities: Set[str] = set()
        for c in existing_concepts:
            tags = c.context_tags if isinstance(c.context_tags, dict) else {}
            if tags.get("_curiosity"):
                ent = tags.get("curiosity_entity")
                if ent:
                    already_learned.add(ent.lower())
                continue
            if tags.get("_schema"):
                # Skip our own meta-concepts when scanning for definitions.
                continue
            # Definitional check: entity appears at start AND followed by
            # "is/was/means/refers to". Conservative — better to fill an
            # extra gap than to skip a real one.
            for ent in self._extract_entities(c.description):
                if self._is_definitional(c.description, ent):
                    defined_entities.add(ent.lower())

        # Count entity occurrences across episodes
        ent_counts: Dict[str, KnowledgeGap] = {}
        for ep in episodes:
            text = ep.raw_content or ""
            session_id = self._session_of(ep)
            for ent in self._extract_entities(text):
                low = ent.lower()
                if low in self._effective_stoplist:
                    continue
                if low in already_learned or low in defined_entities:
                    continue
                gap = ent_counts.setdefault(low, KnowledgeGap(entity=ent))
                gap.occurrence_count += 1
                if ep.id not in gap.seen_in_episode_ids:
                    gap.seen_in_episode_ids.append(ep.id)
                if session_id and session_id not in gap.seen_in_session_ids:
                    gap.seen_in_session_ids.append(session_id)

        gaps = [
            g for g in ent_counts.values()
            if g.occurrence_count >= self.config.min_occurrences
        ]
        gaps.sort(key=lambda g: g.occurrence_count, reverse=True)
        return gaps

    @staticmethod
    def _is_definitional(description: str, entity: str) -> bool:
        """
        True if `description` looks like a real definition of `entity`.
        Pattern + thresholds come from linguistic_resources, NOT hardcoded.
        """
        if not description or not entity:
            return False
        cfg = lr.get_definitional_pattern()
        if len(description) < int(cfg["min_description_chars"]):
            return False
        head = description[: int(cfg["head_chars"])]
        try:
            pattern = lr.compile_definitional_regex(entity)
        except Exception:
            return False
        return bool(pattern.search(head))

    # ── Gap filling ────────────────────────────────────────────────────────

    def _fill_one(self, gap: KnowledgeGap) -> Optional[FilledGap]:
        """Try each source in priority order; first hit wins."""
        for src in self.sources:
            try:
                if not src.is_available():
                    continue
                brief = src.lookup(gap.entity)
            except Exception:
                continue
            if not brief:
                continue
            brief = brief.strip()
            if len(brief) < 8:
                continue
            brief = brief[: self.config.max_brief_chars]
            concept = self._brief_to_concept(gap, brief, src.name)
            return FilledGap(
                entity=gap.entity,
                source_name=src.name,
                brief=brief,
                concept=concept,
            )
        return None

    @staticmethod
    def _brief_to_concept(
        gap: KnowledgeGap,
        brief: str,
        source_name: str,
    ) -> Concept:
        """Build a Concept ready to be inserted into LTM by DeepSleep sync."""
        c = Concept(
            type=ConceptType.FACT,
            description=brief,
            importance=ImportanceVector(
                novelty=0.7,
                emotional=0.0,
                task_relevance=0.6,
                repetition=0.0,
            ),
            state=MemoryState.ACTIVE,
            strength=1.1,
            confidence=0.6,
            salience_score=0.6,  # above the protect-floor by default
        )
        c.context_tags.update({
            "_curiosity": True,
            "curiosity_entity": gap.entity,
            "curiosity_source": source_name,
            "fetched_at": utc_isoformat(utc_now()),
            "source_episodes": list(gap.seen_in_episode_ids),
            "source_sessions": list(gap.seen_in_session_ids),
            "occurrence_count": gap.occurrence_count,
        })
        return c

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> Set[str]:
        out = set()
        min_len = self._effective_min_entity_length
        for m in self._named_entity_re.finditer(text or ""):
            tok = m.group(1)
            if len(tok) >= min_len:
                out.add(tok)
        return out

    @staticmethod
    def _session_of(ep: Episode) -> Optional[str]:
        if not isinstance(ep.context, dict):
            return None
        return ep.context.get("_origin_session") or ep.context.get("session_id")
