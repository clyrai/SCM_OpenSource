"""
ALB adapter contract — the interface every memory system must implement to be
benchmarked.

Design notes:
  - The runner advances *simulated* time. Adapters MUST honor the sim_time
    argument rather than reading wall-clock. This is what makes multi-day
    runs cost minutes instead of days.
  - `idle()` is the only method allowed to do autonomous-learning work. A
    correctly-implemented adapter does NOT do schema extraction / curiosity
    / consolidation inside `ingest()`. ALB measures what happens during idle
    specifically.
  - All capability methods (`list_schemas`, `list_open_questions`,
    `wake_summary`) are expected to return EMPTY values for systems that
    don't support the capability. Returning empty is the honest answer.
    Faking a return to game ALB is reportable misconduct.
  - `supports()` is self-reported and is NOT used for scoring. It exists
    only for sanity cross-checks (does the system claim something it scores
    zero on?). Lying in `supports()` does not affect the score; it affects
    the failure ledger.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ─── Value objects ──────────────────────────────────────────────────────────


@dataclass
class Message:
    """One conversation turn fed to the system."""
    text: str
    speaker: str            # "user" | "agent"
    timestamp_utc: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Schema:
    """An abstraction the system has formed.

    Adapters wrap their native schema/abstraction representation in this
    structure. Free-form `signature` allows each system to expose its
    natural form; the scorer matches by deterministic rule per type.
    """
    schema_id: str
    type: str               # "REPETITION" | "COOCCUR" | "TEMPORAL_CADENCE" | "TRAJECTORY" | other
    signature: Dict[str, Any]
    confidence: float = 1.0
    formed_at: Optional[datetime] = None
    supporting_episode_count: int = 0
    raw_text: str = ""      # the system's own description, for human review


@dataclass
class Gap:
    """A knowledge gap the system has identified."""
    gap_id: str
    term: str
    identified_at: Optional[datetime] = None
    occurrences: int = 0
    has_been_filled: bool = False
    fill_source: Optional[str] = None  # "static_dict" | "local_docs" | "llm" | "user" | None
    fill_text: str = ""


@dataclass
class WakeSummary:
    """Structured report of work the system did during idle.

    `since_sim_time` is the lower bound of the reporting window. The
    summary should describe events that occurred in [since_sim_time, now].
    """
    since_sim_time: datetime
    schemas_formed: List[Schema] = field(default_factory=list)
    contradictions_resolved: List[Dict[str, Any]] = field(default_factory=list)
    gaps_filled: List[Gap] = field(default_factory=list)
    narrative: str = ""


@dataclass
class QueryResult:
    """Response to a probe query."""
    text: str                                # the system's answer in natural language
    retrieved_concepts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdleReport:
    """What the system did during one idle period.

    The runner records this for IMC + WSI scoring. Adapters that do nothing
    during idle return an IdleReport with all zeros — that's correct.
    """
    duration_sim_seconds: float
    wall_clock_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    sleep_cycles_fired: int = 0
    schemas_formed: int = 0
    gaps_identified: int = 0
    gaps_filled: int = 0
    contradictions_resolved: int = 0
    notes: str = ""


@dataclass
class SystemStats:
    """Cumulative system stats since last reset()."""
    total_messages_ingested: int = 0
    total_idle_periods: int = 0
    total_queries: int = 0
    current_concept_count: int = 0
    current_schema_count: int = 0
    current_open_gap_count: int = 0
    cumulative_cpu_seconds: float = 0.0
    cumulative_wall_seconds: float = 0.0
    cumulative_llm_tokens: int = 0
    cumulative_llm_cost_usd: float = 0.0


# ─── Capabilities ──────────────────────────────────────────────────────────


class Capability(Enum):
    """Self-reported capabilities. Used for sanity cross-checks only."""
    SCHEMA_EXTRACTION = "schema_extraction"
    GAP_TRACKING = "gap_tracking"
    AUTONOMOUS_FILL = "autonomous_fill"      # fills gaps during idle, not on demand
    WAKE_SUMMARY = "wake_summary"
    VERSIONING = "versioning"
    CROSS_SESSION_POOL = "cross_session_pool"
    IDLE_PROCESSING = "idle_processing"


# ─── The contract ──────────────────────────────────────────────────────────


class BaseMemorySystem(ABC):
    """Every system benchmarked by ALB implements this interface.

    Lifecycle per run:
        adapter.reset(persona_id, seed)
        for each turn:
            adapter.ingest(msg, sim_time)
        for each idle period:
            adapter.idle(duration, sim_time)
        for each probe:
            adapter.query(text, sim_time)
        adapter.list_schemas()      # at scoring time
        adapter.list_open_questions()
        adapter.wake_summary(since)
        adapter.stats()
    """

    # Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def system_name(self) -> str:
        """Short stable name. Used as the column header in result tables."""

    @property
    @abstractmethod
    def system_version(self) -> str:
        """Version pin (git SHA, package version, or similar). Recorded with results."""

    @abstractmethod
    def supports(self) -> Set[Capability]:
        """Self-reported capability set. Sanity-check only; not used in scoring."""

    # Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def reset(self, persona_id: str, seed: int) -> None:
        """Fresh state. Called once per run.

        After reset:
          - all in-memory structures are clear
          - any persistent stores are scoped to (persona_id, seed) and isolated
          - the random number generator is seeded
        """

    @abstractmethod
    def ingest(self, message: Message, sim_time: datetime) -> None:
        """Add a conversation turn.

        IMPORTANT: this method MUST NOT trigger autonomous-learning work.
        Schema extraction, curiosity, consolidation, and gap detection
        belong in `idle()`. Ingest may do encoding (turning text into the
        system's internal representation) — that's not autonomous learning,
        it's input handling.

        Conformance check: if a system's ingest() does substantive work
        like running an LLM extractor, that work counts against ingest
        cost, not idle cost. The runner records ingest wall-time and
        will flag systems where ingest is suspiciously expensive (it
        suggests work that should have been deferred to idle).
        """

    @abstractmethod
    def idle(
        self,
        duration_sim_seconds: float,
        sim_time: datetime,
        allow_compute: bool = True,
    ) -> IdleReport:
        """Simulated idle period.

        Args:
            duration_sim_seconds: how long the user was away in simulated time.
                The system can use this to decide how much work to do
                (e.g., a longer idle warrants a deeper consolidation).
            sim_time: the simulated time at the START of the idle period.
            allow_compute: when False, this is the no-idle ablation (NIAL).
                The adapter MUST do nothing autonomous and return an empty
                IdleReport. Honest behavior is the requirement.

        Returns:
            IdleReport recording what the system did.
        """

    @abstractmethod
    def query(self, text: str, sim_time: datetime) -> QueryResult:
        """Answer a probe query.

        The adapter is allowed to use its full memory state at sim_time.
        It is NOT allowed to do new autonomous-learning work as a side
        effect of querying — queries should be read-only. (This is
        enforced softly: the runner takes a snapshot of stats() before
        and after each query and flags adapters with substantive
        post-query state changes.)
        """

    # Capability surfaces ──────────────────────────────────────────────────

    @abstractmethod
    def list_schemas(self) -> List[Schema]:
        """Return the current set of formed schemas/abstractions.

        Empty list is the correct answer for systems that don't form
        schemas. Adapters MUST NOT synthesize fake schemas to game PDR.
        """

    @abstractmethod
    def list_open_questions(self) -> List[Gap]:
        """Return the system's currently-tracked knowledge gaps.

        Empty list is correct for systems without gap tracking.
        """

    @abstractmethod
    def wake_summary(self, since_sim_time: datetime) -> Optional[WakeSummary]:
        """Return the system's report on idle work since `since_sim_time`.

        None is the correct answer for systems without a wake-summary
        endpoint. Adapters MUST NOT fabricate a summary to game WSI.
        """

    @abstractmethod
    def was_autonomous_fill(self, gap_id_or_term: str) -> bool:
        """True iff the system filled this gap during idle (not reactively).

        Used by the CGC-fill scorer to distinguish autonomous fills from
        reactive answers. Systems without a curiosity engine should
        return False.
        """

    # Diagnostics ──────────────────────────────────────────────────────────

    @abstractmethod
    def stats(self) -> SystemStats:
        """Cumulative stats since last reset()."""


# ─── Conformance checker ──────────────────────────────────────────────────


def assert_conformant(adapter: BaseMemorySystem) -> List[str]:
    """Return list of conformance warnings for an adapter instance.

    Empty list = conformant. Non-empty list = issues to address before
    benchmarking. This is a soft check; it doesn't prevent running, but
    runs with warnings are flagged in the result manifest.
    """
    warnings: List[str] = []

    if not adapter.system_name:
        warnings.append("system_name is empty")
    if not adapter.system_version:
        warnings.append("system_version is empty")

    caps = adapter.supports()
    if not isinstance(caps, set):
        warnings.append(f"supports() returned {type(caps).__name__}, expected set[Capability]")

    # Light invariant checks: methods must be callable post-reset on a dummy.
    try:
        adapter.reset("conformance_check", seed=0)
    except Exception as e:
        warnings.append(f"reset() raised: {e!r}")
        return warnings  # can't continue

    try:
        schemas = adapter.list_schemas()
        if not isinstance(schemas, list):
            warnings.append("list_schemas() did not return a list")
    except Exception as e:
        warnings.append(f"list_schemas() raised: {e!r}")

    try:
        gaps = adapter.list_open_questions()
        if not isinstance(gaps, list):
            warnings.append("list_open_questions() did not return a list")
    except Exception as e:
        warnings.append(f"list_open_questions() raised: {e!r}")

    try:
        summary = adapter.wake_summary(datetime.fromtimestamp(0))
        if summary is not None and not isinstance(summary, WakeSummary):
            warnings.append("wake_summary() returned non-WakeSummary, non-None value")
    except Exception as e:
        warnings.append(f"wake_summary() raised: {e!r}")

    try:
        s = adapter.stats()
        if not isinstance(s, SystemStats):
            warnings.append("stats() did not return SystemStats")
    except Exception as e:
        warnings.append(f"stats() raised: {e!r}")

    # Capability vs surface cross-check.
    if Capability.SCHEMA_EXTRACTION in caps:
        # The adapter claims it; list_schemas should not unconditionally
        # raise. We can't tell here whether it produces schemas — that's
        # what the benchmark measures.
        pass
    if Capability.WAKE_SUMMARY not in caps:
        # If the adapter doesn't claim wake_summary, it should return None.
        if adapter.wake_summary(datetime.fromtimestamp(0)) is not None:
            warnings.append(
                "wake_summary() returned a value but Capability.WAKE_SUMMARY is not declared"
            )

    return warnings
