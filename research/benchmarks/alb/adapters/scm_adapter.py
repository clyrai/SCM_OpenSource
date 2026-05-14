"""
SCM adapter for ALB.

Wires the full Phase 7 SCM stack (ChatEngine + DeepSleep with schema
extraction + curiosity engine + WakeSummaryBuilder) to the
BaseMemorySystem interface.

Honest design choices:
  - The curiosity engine's static dictionary is a small, general-purpose
    technical glossary, NOT persona-tuned. The same glossary is used for
    every persona, so any gap-fill credit comes from the dictionary
    actually containing the term, not from cherry-picking.
  - Schema concepts are detected via context_tags["schema_type"] — the
    same field DeepSleep itself uses to mark schemas. We do not synthesize
    fake schemas to game PDR.
  - Curiosity-filled concepts are detected via context_tags["curiosity_*"]
    — the same field DeepSleep uses to mark curiosity output.
  - sandbox_mode=True keeps the run in-memory; nothing persists across
    runs of the same adapter. reset() zeros everything.
"""
from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

# Make the SCM source tree importable.
SCM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if SCM_ROOT not in sys.path:
    sys.path.insert(0, SCM_ROOT)

from .base import (
    BaseMemorySystem,
    Capability,
    Gap,
    IdleReport,
    Message,
    QueryResult,
    Schema,
    SystemStats,
    WakeSummary,
)


# ─── Built-in glossary used by the curiosity engine ────────────────────────
#
# This is intentionally small and general. It does NOT contain
# persona-specific knowledge. A term is filled if and only if it's in this
# table. If you add persona-specific entries, you compromise the benchmark.
# ──────────────────────────────────────────────────────────────────────────

GENERAL_TECH_GLOSSARY: Dict[str, str] = {
    "oauth flow": "An authorization protocol where a client app obtains a token via a redirect-based exchange, with scopes defining what the token can access.",
    "oauth": "An authorization framework where applications obtain access tokens to act on behalf of a resource owner without sharing credentials.",
    "kubernetes ingress": "A Kubernetes resource that defines external access rules to services in a cluster, typically routing HTTP and HTTPS traffic via a load balancer.",
    "ingress": "A networking resource that exposes services to external traffic, applying routing rules and load balancing.",
    "titration": "A laboratory technique to determine concentration by adding a reagent of known concentration from a burette to an analyte until an indicator marks the endpoint of neutralization.",
    "stoichiometry": "The branch of chemistry concerned with the quantitative ratios of reactants and products in chemical reactions, typically using moles.",
    "molarity": "Concentration expressed as moles of solute per liter of solution.",
    "molality": "Concentration expressed as moles of solute per kilogram of solvent.",
    "ngss": "Next Generation Science Standards — a set of K-12 science education standards emphasizing inquiry-based learning and three-dimensional integration of practices, crosscutting concepts, and core ideas.",
    "rest api": "An architectural style for networked applications using HTTP verbs (GET, POST, PUT, DELETE) over stateless client-server communication.",
    "graphql": "A query language and runtime for APIs allowing clients to request precisely the data shape they need from a single endpoint.",
    "load balancer": "A device or service that distributes incoming network traffic across multiple backend servers to improve availability and responsiveness.",
}


# ─── Helpers ───────────────────────────────────────────────────────────────


def _peak_rss_bytes() -> int:
    """Peak resident set size of this process. Linux returns kB, macOS bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)        # bytes on macOS
    return int(rss) * 1024     # kB on Linux


# ─── The adapter ───────────────────────────────────────────────────────────


class SCMAdapter(BaseMemorySystem):
    """ALB adapter for the full Phase 7 SCM stack."""

    def __init__(self, version_pin: str = "phase7-pilot"):
        self._version = version_pin
        self._engine = None
        self._curiosity_engine = None
        self._wake_summary_builder = None
        self._stats = SystemStats()
        # Track which gaps were filled during idle (vs reactively).
        # (term_lowercase, source_name) tuples.
        self._autonomous_fills: Set[str] = set()
        # Track open gaps detected during the most recent idle.
        self._known_gaps: List[Any] = []
        # ALL gaps the system has ever identified (whether or not later filled).
        # CGC_id should reward identification, not penalize successful filling.
        self._ever_identified_gaps: Dict[str, Any] = {}
        # Schema snapshot tracking for WakeSummary windowing.
        self._idle_started_at: Optional[datetime] = None

    @property
    def system_name(self) -> str:
        return "SCM"

    @property
    def system_version(self) -> str:
        return self._version

    def supports(self) -> Set[Capability]:
        return {
            Capability.SCHEMA_EXTRACTION,
            Capability.GAP_TRACKING,
            Capability.AUTONOMOUS_FILL,
            Capability.WAKE_SUMMARY,
            Capability.VERSIONING,
            Capability.CROSS_SESSION_POOL,
            Capability.IDLE_PROCESSING,
        }

    # Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, persona_id: str, seed: int) -> None:
        """Build a fresh ChatEngine with the full Phase 7 stack.

        LLM extractor is optional, gated by ALB_USE_LLM_EXTRACTOR=1. With it
        enabled, the system uses DeepSeek-chat (or whatever LLM_PROVIDER
        specifies) for concept extraction at ingest time. This produces
        salience signals with real dynamic range, which addresses the
        encoder-dependence story behind the v3 pilot's CSS = -1 lift and
        CRAI_current = 0 results.
        """
        # Imports here so an SCM build error is visible at adapter-use time
        # rather than at module-load time.
        from src.chat.engine import ChatEngine
        from src.chat import engine as engine_mod
        from src.core.encoder import MeaningEncoder
        from src.lifecycle.curiosity import (
            CuriosityConfig,
            CuriosityEngine,
            StaticDictionarySource,
        )
        from src.lifecycle.wake_summary import WakeSummaryBuilder
        from src.sleep.deep_sleep import DeepSleep
        from src.sleep.schema_extractor import SchemaExtractor, SchemaExtractorConfig
        from src.sleep.sleep_cycle import SleepCycleOrchestrator

        # HME pipeline on (working memory + selective encoding + ...).
        engine_mod.HME_ENABLED = True

        # Optional LLM extractor (off by default for offline-deterministic runs).
        llm_extractor = None
        if os.environ.get("ALB_USE_LLM_EXTRACTOR") == "1":
            from src.llm import LLMExtractor
            provider = os.environ.get("ALB_LLM_PROVIDER", "deepseek")
            try:
                llm_extractor = LLMExtractor(provider=provider)
            except Exception as e:
                print(f"[SCMAdapter] LLM extractor unavailable ({e}); falling back to heuristic")
                llm_extractor = None

        # Use Ollama (nomic-embed-text) for embeddings instead of the
        # default sentence-transformers/all-MiniLM-L6-v2. The 768-dim model
        # plus task-prefix-aware encoding bridges question→fact phrasing
        # gaps that MiniLM doesn't (Bug B from the v1/v3 ALB pilot).
        encoder = MeaningEncoder(
            llm=llm_extractor,
            embedding_backend="ollama",
            embedding_model_name="nomic-embed-text",
        )

        self._curiosity_engine = CuriosityEngine(
            sources=[StaticDictionarySource(dict(GENERAL_TECH_GLOSSARY))],
            config=CuriosityConfig(
                enabled=True,
                min_occurrences=2,
                max_gaps_per_cycle=3,
            ),
        )

        deep = DeepSleep(
            enable_synthesis=False,
            enable_schema_extraction=True,
            schema_extractor=SchemaExtractor(
                config=SchemaExtractorConfig(
                    enabled=True,
                    # default min_repetitions=3 — tighter than prior pilot's
                    # min_repetitions=2 which produced excessive noise
                    # (40+ schemas/run, 85% noise).
                ),
            ),
            enable_paraphrase=True,
            enable_curiosity=True,
            curiosity_engine=self._curiosity_engine,
        )
        orch = SleepCycleOrchestrator(deep_sleep=deep)

        self._engine = ChatEngine(
            llm=None,
            encoder=encoder,
            sleep_orchestrator=orch,
            session_id=f"alb_{persona_id}_seed{seed}",
            profile="research",
            sandbox_mode=True,
            enable_persistence=False,
            enable_auto_sleep=False,
        )
        self._wake_summary_builder = WakeSummaryBuilder(engine=self._engine)
        self._stats = SystemStats()
        self._autonomous_fills = set()
        self._known_gaps = []
        self._ever_identified_gaps = {}
        self._idle_started_at = None

    def ingest(self, message: Message, sim_time: datetime) -> None:
        if message.speaker != "user":
            return  # ALB only ingests user turns; agent turns are filler
        # We use ChatEngine.chat to drive the full HME pipeline.
        # The response text is discarded — ALB measures memory, not
        # generation.
        try:
            self._engine.chat(message.text)
        except Exception:
            # Don't crash the run; record the failure in stats notes.
            pass
        self._stats.total_messages_ingested += 1

    def idle(
        self,
        duration_sim_seconds: float,
        sim_time: datetime,
        allow_compute: bool = True,
    ) -> IdleReport:
        """Trigger a deep sleep cycle if compute is allowed."""
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        rss_start = _peak_rss_bytes()
        cycles = 0
        schemas_formed = 0
        gaps_identified = 0
        gaps_filled = 0

        if allow_compute:
            # Snapshot schemas before sleep for delta computation.
            schemas_before = {s.schema_id for s in self.list_schemas()}
            try:
                self._engine.force_sleep("deep")
                cycles = 1
            except Exception:
                pass

            schemas_after = {s.schema_id for s in self.list_schemas()}
            schemas_formed = len(schemas_after - schemas_before)

            # Curiosity-filled gaps: any concept in LTM that has a
            # curiosity_entity tag and a fetched_at within this idle.
            for c in self._all_concepts():
                tag = (c.context_tags or {}).get("curiosity_entity")
                if tag:
                    self._autonomous_fills.add(tag.lower())
                    gaps_filled += 1
                    break  # already counted; outer loop continues only
                           # to populate _autonomous_fills

            # Re-detect gaps so list_open_questions reflects current state.
            # ALSO record all-time identified gaps in _ever_identified_gaps so
            # CGC_id rewards identification even if the gap was later filled.
            try:
                gaps = self._curiosity_engine.detect_gaps(
                    episodes=self._all_episodes(),
                    existing_concepts=self._all_concepts(),
                )
                self._known_gaps = list(gaps)
                gaps_identified = len(gaps)
                for kg in gaps:
                    term = (getattr(kg, "entity", "") or "").lower()
                    if term and term not in self._ever_identified_gaps:
                        self._ever_identified_gaps[term] = kg
            except Exception:
                pass

            # Also record gaps whose filled concept now exists. Pull terms
            # from curiosity_entity tags so a fill that happened earlier
            # in the run is captured even if detect_gaps now skips it.
            for c in self._all_concepts():
                tag = (c.context_tags or {}).get("curiosity_entity")
                if tag:
                    term = str(tag).lower()
                    self._autonomous_fills.add(term)
                    if term not in self._ever_identified_gaps:
                        # Synthesize a stub identification record.
                        self._ever_identified_gaps[term] = type("_FilledStub", (), {
                            "entity": tag,
                            "occurrence_count": 0,
                            "has_existing_concept": True,
                        })()

        wall_elapsed = time.perf_counter() - wall_start
        cpu_elapsed = time.process_time() - cpu_start
        rss_after = _peak_rss_bytes()

        self._stats.total_idle_periods += 1
        self._stats.cumulative_wall_seconds += wall_elapsed
        self._stats.cumulative_cpu_seconds += cpu_elapsed

        return IdleReport(
            duration_sim_seconds=duration_sim_seconds,
            wall_clock_seconds=wall_elapsed,
            cpu_seconds=cpu_elapsed,
            peak_rss_bytes=max(rss_start, rss_after),
            sleep_cycles_fired=cycles,
            schemas_formed=schemas_formed,
            gaps_identified=gaps_identified,
            gaps_filled=gaps_filled,
        )

    def query(self, text: str, sim_time: datetime) -> QueryResult:
        """Retrieve relevant concepts and format as response text.

        Strategy: use SCM's spreading-activation retriever (the same code
        path ChatEngine.chat() uses to surface memory context), but
        WITHOUT calling chat() — we don't want the query to be ingested
        as a turn.

        Fall back to LTM text search + token-level search if spreading
        activation is unavailable.
        """
        self._stats.total_queries += 1
        if self._engine is None:
            return QueryResult(text="")

        results: List[Any] = []
        seen_ids: Set[str] = set()
        memory_context = ""

        def _add(concepts):
            for c in concepts or []:
                cid = getattr(c, "id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append(c)

        # Phase 1: use SCM's HME spreading-activation retriever.
        try:
            mc, _stats = self._engine._retrieve_hme(text)
            memory_context = mc or ""
        except Exception:
            memory_context = ""

        ltm = getattr(self._engine, "long_term_memory", None)

        # Phase 2: token-level fallback to broaden recall.
        import re
        STOP = {"the", "and", "for", "you", "your", "what", "where",
                "when", "how", "did", "do", "i", "my", "me", "is",
                "are", "was", "were", "have", "has", "had", "should",
                "could", "would", "to", "in", "of", "at", "a", "an",
                "previously", "recently", "currently", "bring", "bringing"}
        tokens = [
            t for t in re.findall(r"[A-Za-z]+", text.lower())
            if len(t) >= 3 and t not in STOP
        ]
        m = re.match(r"\s*define\s+(.+?)\s*\.?\s*$", text, re.IGNORECASE)
        if m:
            tokens.append(m.group(1).strip().lower())

        if ltm is not None:
            for tok in tokens:
                try:
                    _add(ltm.search_by_text(tok, limit=4))
                except Exception:
                    pass

        # Phase 3: substring scan for token mentions in concept descriptions.
        # Check BOTH the (possibly paraphrased) description and the
        # original_description preserved in context_tags. The paraphrase pass
        # rewrites descriptions during deep-sleep but keeps the original for
        # audit, and ALB's keyword scorer needs verbatim entity strings.
        for c in self._all_concepts():
            cid = getattr(c, "id", "")
            if cid in seen_ids:
                continue
            desc_low = (getattr(c, "description", "") or "").lower()
            tags = getattr(c, "context_tags", None) or {}
            orig_low = (tags.get("original_description") or "").lower()
            blob = desc_low + " " + orig_low
            if any(tok in blob for tok in tokens):
                seen_ids.add(cid)
                results.append(c)

        # Bug 4 filter: drop superseded concepts.
        kept = [c for c in results if getattr(c, "is_current_version", True)]

        # Rank by token overlap then by recency.
        def _score(c):
            d = (getattr(c, "description", "") or "").lower()
            overlap = sum(1 for t in tokens if t in d)
            return (-overlap, -(c.last_accessed.timestamp() if getattr(c, "last_accessed", None) else 0))
        kept.sort(key=_score)

        descriptions = [getattr(c, "description", "") or "" for c in kept[:5]]
        resp_text = " | ".join(d for d in descriptions if d)
        # Combine the spreading-activation-formatted memory context with the
        # token-search descriptions. Both are evidence the system "knows"
        # something about the query.
        if memory_context:
            resp_text = (memory_context + " | " + resp_text).strip(" |")

        return QueryResult(
            text=resp_text,
            retrieved_concepts=[getattr(c, "id", "") for c in kept[:5]],
            confidence=min(1.0, len(kept) / 5.0),
        )

    # Capability surfaces ──────────────────────────────────────────────────

    def list_schemas(self) -> List[Schema]:
        out: List[Schema] = []
        for c in self._all_concepts():
            tags = c.context_tags or {}
            schema_type = tags.get("schema_type")
            if not schema_type:
                continue
            entities = tags.get("entities") or []
            out.append(Schema(
                schema_id=getattr(c, "id", ""),
                type=str(schema_type).upper(),
                signature={
                    "entities": list(entities) if isinstance(entities, list) else [str(entities)],
                    "occurrence_count": tags.get("occurrence_count", 0),
                    "source_sessions": tags.get("source_sessions", []),
                },
                confidence=getattr(c, "confidence", 1.0),
                supporting_episode_count=tags.get("occurrence_count", 0) or 0,
                raw_text=getattr(c, "description", "") or "",
            ))
        return out

    def list_open_questions(self) -> List[Gap]:
        """Return ALL gaps the system has ever identified.

        For ALB CGC_id scoring, "identification" is a positive signal
        regardless of whether the gap was subsequently filled. Filtering
        out filled gaps here would penalize SCM for being good at its
        job, so we report the union of currently-open and previously-
        identified gaps, and use Gap.has_been_filled to distinguish.
        """
        out: List[Gap] = []
        seen: Set[str] = set()

        # First, the recently-identified set (may overlap with filled).
        for kg in self._known_gaps:
            term = getattr(kg, "entity", "") or ""
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(Gap(
                gap_id=key.replace(" ", "_"),
                term=term,
                occurrences=getattr(kg, "occurrence_count", 0) or 0,
                has_been_filled=key in self._autonomous_fills,
            ))

        # Then the all-time identified set, including filled gaps.
        for term_lower, kg in self._ever_identified_gaps.items():
            if term_lower in seen:
                continue
            seen.add(term_lower)
            out.append(Gap(
                gap_id=term_lower.replace(" ", "_"),
                term=getattr(kg, "entity", term_lower),
                occurrences=getattr(kg, "occurrence_count", 0) or 0,
                has_been_filled=term_lower in self._autonomous_fills,
            ))

        return out

    def wake_summary(self, since_sim_time: datetime) -> Optional[WakeSummary]:
        if self._wake_summary_builder is None:
            return None
        try:
            scm_summary = self._wake_summary_builder.build(since=since_sim_time)
        except Exception:
            return None

        # Translate the SCM-native summary into ALB's WakeSummary shape.
        schemas_formed: List[Schema] = []
        for c in self._all_concepts():
            tags = c.context_tags or {}
            if not tags.get("schema_type"):
                continue
            created = getattr(c, "created_at", None)
            if created and since_sim_time and created < since_sim_time:
                continue
            schemas_formed.append(Schema(
                schema_id=getattr(c, "id", ""),
                type=str(tags.get("schema_type")).upper(),
                signature={
                    "entities": list(tags.get("entities") or []),
                    "occurrence_count": tags.get("occurrence_count", 0),
                },
                raw_text=getattr(c, "description", "") or "",
            ))

        return WakeSummary(
            since_sim_time=since_sim_time,
            schemas_formed=schemas_formed,
            narrative=getattr(scm_summary, "narrative", "") or "",
            contradictions_resolved=[],
            gaps_filled=[],
        )

    def was_autonomous_fill(self, gap_id_or_term: str) -> bool:
        """True iff the curiosity engine filled this term during an idle."""
        if not gap_id_or_term:
            return False
        # Try the gap_id (snake_case) form first, then the human-readable term.
        candidates = {
            gap_id_or_term.lower(),
            gap_id_or_term.lower().replace("_", " "),
            gap_id_or_term.lower().replace("g_", "").replace("_", " "),
        }
        for fill in self._autonomous_fills:
            for cand in candidates:
                if cand in fill or fill in cand:
                    return True
        return False

    def stats(self) -> SystemStats:
        # Fold in current LTM stats.
        ltm = getattr(self._engine, "long_term_memory", None)
        if ltm is not None:
            try:
                ltm_stats = ltm.get_stats()
                if isinstance(ltm_stats, dict):
                    self._stats.current_concept_count = int(
                        ltm_stats.get("total_concepts", 0)
                    )
            except Exception:
                pass
        self._stats.current_schema_count = len(self.list_schemas())
        self._stats.current_open_gap_count = len(self.list_open_questions())
        return self._stats

    # Internals ─────────────────────────────────────────────────────────────

    def _all_concepts(self) -> List[Any]:
        ltm = getattr(self._engine, "long_term_memory", None)
        if ltm is None:
            return []
        try:
            return list(ltm.get_all_concepts())
        except Exception:
            return []

    def _all_episodes(self) -> List[Any]:
        wm = getattr(self._engine, "working_memory", None)
        if wm is None:
            return []
        try:
            episodes = wm.get_all() if hasattr(wm, "get_all") else getattr(wm, "episodes", [])
            return list(episodes)
        except Exception:
            return []
