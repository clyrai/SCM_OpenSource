"""
SleepCycle: Orchestrates the full sleep consolidation process
Coordinates wake -> NREM -> REM -> wake cycle
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time

from ..core.models import Concept, Episode, Relation, SleepCycle, MemoryState
from ..core.config import (
    CURIOSITY_DICTIONARY_PATH,
    CURIOSITY_ENGINE_ENABLED,
    CURIOSITY_LLM_SOURCE_ENABLED,
    CURIOSITY_LOCAL_DOCS_FOLDER,
    CURIOSITY_MAX_BRIEF_CHARS,
    CURIOSITY_MAX_GAPS_PER_CYCLE,
    CURIOSITY_MIN_OCCURRENCES,
    DEEP_SLEEP_ENABLE_PARAPHRASE,
    DEEP_SLEEP_ENABLE_SCHEMAS,
    MICRO_SLEEP_ENABLED,
    SCHEMA_COOCCURRENCE_MIN,
    SCHEMA_MAX_EPISODES_WINDOW,
    SCHEMA_MAX_PER_CYCLE,
    SCHEMA_MIN_REPETITIONS,
    SCHEMA_TEMPORAL_WINDOW_HOURS,
)
from .schema_extractor import SchemaExtractor, SchemaExtractorConfig


def _build_curiosity_engine():
    """Construct a CuriosityEngine from env vars. Returns None when disabled."""
    if not CURIOSITY_ENGINE_ENABLED:
        return None
    try:
        from pathlib import Path
        from ..lifecycle.curiosity import (
            CuriosityConfig,
            CuriosityEngine,
            LLMSource,
            LocalDocsSource,
            StaticDictionarySource,
        )
        sources = []
        # Static dictionary first (fastest, deterministic, free)
        if CURIOSITY_DICTIONARY_PATH:
            sources.append(StaticDictionarySource.from_json(Path(CURIOSITY_DICTIONARY_PATH)))
        # Local docs folder second (broader coverage, free)
        if CURIOSITY_LOCAL_DOCS_FOLDER:
            sources.append(LocalDocsSource(folder=Path(CURIOSITY_LOCAL_DOCS_FOLDER)))
        # LLM source last (autonomous knowledge, costs API tokens)
        if CURIOSITY_LLM_SOURCE_ENABLED:
            try:
                from ..llm import LLMExtractor
                sources.append(LLMSource(llm=LLMExtractor()))
            except Exception:
                pass
        if not sources:
            return None
        return CuriosityEngine(
            sources=sources,
            config=CuriosityConfig(
                enabled=True,
                min_occurrences=CURIOSITY_MIN_OCCURRENCES,
                max_gaps_per_cycle=CURIOSITY_MAX_GAPS_PER_CYCLE,
                max_brief_chars=CURIOSITY_MAX_BRIEF_CHARS,
            ),
        )
    except Exception:
        return None
from .trigger import SleepTrigger
from .nrem import NREMConsolidation
from .rem import REMDreaming
from .forgetting import ForgettingModule
from .micro_sleep import MicroSleep
from .deep_sleep import DeepSleep
from ..core.time_utils import ensure_utc, utc_now


def _state_value(concept: Concept) -> str:
    state = getattr(concept, "state", MemoryState.ACTIVE.value)
    return state.value if hasattr(state, "value") else str(state)


class SleepCycleOrchestrator:
    """
    Orchestrates the complete sleep cycle.

    Biological inspiration:
    - Sleep alternates between NREM and REM stages (~90 min cycles in humans)
    - NREM: Memory consolidation and synaptic downscaling
    - REM: Emotional processing and creative integration
    - Wake: New memories are encoded

    This orchestrator:
    1. Checks if sleep should begin (SleepTrigger)
    2. Performs NREM consolidation (Hebbian + downscaling)
    3. Performs REM dreaming (generative replay)
    4. Applies forgetting (removes low-value memories)
    5. Returns to wake state with updated memory
    """

    def __init__(
        self,
        trigger: Optional[SleepTrigger] = None,
        nrem: Optional[NREMConsolidation] = None,
        rem: Optional[REMDreaming] = None,
        forgetting: Optional[ForgettingModule] = None,
        micro_sleep: Optional[MicroSleep] = None,
        deep_sleep: Optional[DeepSleep] = None,
    ):
        self.trigger = trigger or SleepTrigger()
        self.nrem = nrem or NREMConsolidation()
        self.rem = rem or REMDreaming()
        self.forgetting = forgetting or ForgettingModule()
        self.micro_sleep = micro_sleep or MicroSleep()
        _curiosity = _build_curiosity_engine()
        self.deep_sleep = deep_sleep or DeepSleep(
            nrem=self.nrem,
            rem=self.rem,
            forgetting=self.forgetting,
            enable_paraphrase=DEEP_SLEEP_ENABLE_PARAPHRASE,
            enable_schema_extraction=DEEP_SLEEP_ENABLE_SCHEMAS,
            schema_extractor=SchemaExtractor(
                config=SchemaExtractorConfig(
                    enabled=DEEP_SLEEP_ENABLE_SCHEMAS,
                    min_repetitions=SCHEMA_MIN_REPETITIONS,
                    cooccurrence_min=SCHEMA_COOCCURRENCE_MIN,
                    max_schemas_per_cycle=SCHEMA_MAX_PER_CYCLE,
                    max_episodes_window=SCHEMA_MAX_EPISODES_WINDOW,
                    temporal_window_hours=SCHEMA_TEMPORAL_WINDOW_HOURS,
                ),
            ),
            enable_curiosity=(_curiosity is not None),
            curiosity_engine=_curiosity,
        )
        self.micro_sleep_enabled = MICRO_SLEEP_ENABLED

        self._current_cycle: Optional[SleepCycle] = None
        self._last_wake_time: datetime = utc_now()
        self._turns_since_micro_sleep = 0

    def begin_sleep_cycle(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        episodes: List[Episode],
        force: bool = False,
        mode: Optional[str] = None,
        session_turns: int = 0,
        turns_since_micro: Optional[int] = None,
    ) -> Tuple[bool, SleepCycle, Dict]:
        """
        Begin a complete sleep cycle.

        Args:
            force: If True, bypass trigger check and force sleep

        Returns:
            Tuple of (success, sleep_cycle_record, detailed_stats)
        """
        mode_normalized = mode.lower().strip() if mode else None
        if mode_normalized not in {None, "micro", "deep"}:
            return False, SleepCycle(start_time=utc_now()), {
                "reason": f"Unknown sleep mode: {mode}",
                "skipped": True,
            }

        resolved_mode = mode_normalized
        reason = "Forced sleep" if force else "Sleep not required"

        if resolved_mode is None:
            # Legacy default behavior: deep cycle using existing trigger.
            if not force:
                time_since_sleep = self.trigger.time_since_last_sleep()
                should_sleep, reason = self.trigger.should_sleep(
                    concepts=concepts,
                    relations=relations,
                    recent_conflicts=0,
                    time_since_last_sleep=time_since_sleep,
                )
                if not should_sleep:
                    self._turns_since_micro_sleep += 1
                    return False, SleepCycle(start_time=utc_now()), {
                        "reason": reason,
                        "skipped": True,
                    }
            resolved_mode = "deep"
            reason = reason if force else f"Legacy trigger: {reason}"
        elif resolved_mode == "micro":
            if not self.micro_sleep_enabled:
                self._turns_since_micro_sleep += 1
                return False, SleepCycle(start_time=utc_now()), {
                    "reason": "MicroSleep disabled via config",
                    "skipped": True,
                }
            if not force:
                turn_count = (
                    turns_since_micro
                    if turns_since_micro is not None
                    else self._turns_since_micro_sleep
                )
                should_sleep, reason = self.trigger.should_micro_sleep(
                    concepts=concepts,
                    relations=relations,
                    turns_since_last_micro=turn_count,
                )
                if not should_sleep:
                    self._turns_since_micro_sleep += 1
                    return False, SleepCycle(start_time=utc_now()), {
                        "reason": reason,
                        "skipped": True,
                    }
            reason = "Forced micro sleep" if force else reason
        else:
            if not force:
                deep_idle = self.trigger.time_since_last_deep_sleep()
                should_sleep, reason = self.trigger.should_deep_sleep(
                    concepts=concepts,
                    relations=relations,
                    time_since_last_deep=deep_idle,
                    session_turns=session_turns,
                )
                if not should_sleep:
                    self._turns_since_micro_sleep += 1
                    return False, SleepCycle(start_time=utc_now()), {
                        "reason": reason,
                        "skipped": True,
                    }
            reason = "Forced deep sleep" if force else reason

        self._current_cycle = SleepCycle(start_time=utc_now())
        cycle_stats = {
            'mode': resolved_mode,
            'trigger_reason': reason,
            'nrem': {},
            'rem': {},
            'forgetting': {},
            'micro': {},
            'dreams': []
        }
        if resolved_mode == "micro":
            phase_start = time.time()
            updated_concepts, updated_relations, micro_stats = self.micro_sleep.run(
                concepts=concepts,
                relations=relations,
                episodes=episodes,
            )
            duration = time.time() - phase_start

            cycle_stats['micro'] = micro_stats
            cycle_stats['updated_concepts'] = updated_concepts
            cycle_stats['updated_relations'] = updated_relations
            cycle_stats['forgotten_ids'] = list(micro_stats.get("merged_ids", []))

            self._current_cycle.nrem_duration = duration
            self._current_cycle.rem_duration = 0.0
            self._current_cycle.end_time = utc_now()
            self._current_cycle.memories_consolidated = (
                micro_stats.get("replayed", 0) + micro_stats.get("relations_reinforced", 0)
            )
            self._current_cycle.memories_forgotten = micro_stats.get("duplicates_merged", 0)
            self._current_cycle.dreams_generated = []

            self.trigger.record_sleep(self._current_cycle.end_time, mode="micro")
            self._turns_since_micro_sleep = 0
            self._last_wake_time = utc_now()
            return True, self._current_cycle, cycle_stats

        phase_start = time.time()
        updated_concepts, updated_relations, deep_stats = self.deep_sleep.run(
            concepts=concepts,
            relations=relations,
            episodes=episodes,
        )
        total_duration = time.time() - phase_start

        cycle_stats['nrem'] = deep_stats.get('nrem', {})
        cycle_stats['rem'] = deep_stats.get('rem', {})
        cycle_stats['forgetting'] = deep_stats.get('forgetting', {})
        cycle_stats['dreams'] = deep_stats.get('dreams', [])
        cycle_stats['deep'] = deep_stats
        cycle_stats['updated_concepts'] = updated_concepts
        cycle_stats['updated_relations'] = updated_relations
        cycle_stats['forgotten_ids'] = deep_stats.get('forgotten_ids', [])
        cycle_stats['retired_ids'] = deep_stats.get('retired_ids', [])
        cycle_stats['retired_concepts'] = deep_stats.get('retired_concepts', [])

        # Keep legacy fields populated while routing through DeepSleep module.
        self._current_cycle.nrem_duration = total_duration * 0.7
        self._current_cycle.rem_duration = total_duration * 0.3
        self._current_cycle.end_time = utc_now()
        self._current_cycle.memories_consolidated = cycle_stats['nrem'].get('consolidated_to_ltm', 0)
        self._current_cycle.memories_forgotten = cycle_stats['forgetting'].get('forgotten', 0)
        self._current_cycle.dreams_generated = [d.get('id', '') for d in cycle_stats['dreams']]

        self.trigger.record_sleep(self._current_cycle.end_time, mode="deep")
        self._turns_since_micro_sleep = 0
        self._last_wake_time = utc_now()

        return True, self._current_cycle, cycle_stats

    def _run_nrem_phase(
        self,
        concepts: List[Concept],
        episodes: List[Episode],
        existing_relations: List[Relation]
    ) -> Dict:
        """
        Run NREM (slow-wave sleep) consolidation phase.

        Returns:
            Stats dictionary from NREM consolidation
        """
        active_concepts = [
            c for c in concepts
            if _state_value(c) not in {
                MemoryState.SUPPRESSED.value,
                MemoryState.ARCHIVED.value,
            }
        ]

        updated_concepts, new_relations, stats = self.nrem.consolidate(
            concepts=active_concepts,
            episodes=episodes,
            existing_relations=existing_relations
        )

        return {
            **stats,
            'new_relations_count': len(new_relations),
            'updated_concepts_count': len(updated_concepts)
        }

    def _run_rem_phase(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        episodes: List[Episode]
    ) -> Dict:
        """
        Run REM (dreaming) phase.

        Returns:
            Dictionary with dreams and stats
        """
        recent_episodes = episodes[-10:] if len(episodes) > 10 else episodes

        dreams, new_relations, stats = self.rem.dream(
            concepts=concepts,
            relations=relations,
            recent_episodes=recent_episodes
        )

        return {
            'dreams': dreams,
            'stats': stats,
            'new_relations': new_relations
        }

    def _apply_forgetting(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        cycle_stats: Dict
    ) -> Tuple[List[Concept], List[Relation]]:
        """
        Apply forgetting to concepts and relations.

        Returns:
            Tuple of (updated_concepts, updated_relations)
        """
        conflict_pairs = self._extract_conflict_pairs(relations)

        forgotten_ids, preserved_ids, forgetting_stats = self.forgetting.evaluate_forgetting(
            concepts=concepts,
            conflict_pairs=conflict_pairs
        )

        cycle_stats['forgetting'] = forgetting_stats

        updated_concepts = [
            c for c in concepts
            if c.id not in forgotten_ids
        ]

        updated_relations = [
            r for r in relations
            if r.subject_id not in forgotten_ids and r.object_id not in forgotten_ids
        ]

        return updated_concepts, updated_relations

    def _extract_conflict_pairs(self, relations: List[Relation]) -> List[Tuple[str, str]]:
        """Extract conflicting concept pairs from relations"""
        conflicts = []

        for relation in relations:
            if hasattr(relation, 'predicate'):
                predicate_val = relation.predicate
                if isinstance(predicate_val, str):
                    if 'contradict' in predicate_val.lower() or 'conflict' in predicate_val.lower():
                        conflicts.append((relation.subject_id, relation.object_id))

        return conflicts

    def select_sleep_mode(
        self,
        concepts: List[Concept],
        relations: List[Relation],
        turns_since_micro: int = 0,
        session_turns: int = 0,
    ) -> Tuple[Optional[str], str, Dict]:
        """
        Decide whether to run micro-sleep or deep-sleep.
        Returns:
            (mode, reason, stats) where mode is "micro", "deep", or None.
        """
        if not concepts:
            stats = self.trigger.get_trigger_stats(concepts, relations)
            stats["selected_mode"] = None
            return None, "No concepts to consolidate", stats

        trigger_stats = self.trigger.get_trigger_stats(concepts, relations)

        deep_idle = self.trigger.time_since_last_deep_sleep()
        should_deep, deep_reason = self.trigger.should_deep_sleep(
            concepts=concepts,
            relations=relations,
            time_since_last_deep=deep_idle,
            session_turns=session_turns,
        )
        if should_deep:
            trigger_stats["selected_mode"] = "deep"
            trigger_stats["deep_reason"] = deep_reason
            return "deep", deep_reason, trigger_stats

        if self.micro_sleep_enabled:
            should_micro, micro_reason = self.trigger.should_micro_sleep(
                concepts=concepts,
                relations=relations,
                turns_since_last_micro=turns_since_micro,
            )
            if should_micro:
                trigger_stats["selected_mode"] = "micro"
                trigger_stats["micro_reason"] = micro_reason
                return "micro", micro_reason, trigger_stats

        trigger_stats["selected_mode"] = None
        return None, "Sleep not needed", trigger_stats

    def check_should_sleep(
        self,
        concepts: List[Concept],
        relations: List[Relation]
    ) -> Tuple[bool, str, Dict]:
        """
        Check if sleep should be initiated.

        Returns:
            Tuple of (should_sleep, reason, trigger_stats)
        """
        mode, reason, trigger_stats = self.select_sleep_mode(concepts, relations)
        if mode is not None:
            return True, f"{mode} sleep: {reason}", trigger_stats

        # Backward-compatible check path.
        time_since = self.trigger.time_since_last_sleep()
        should_sleep, legacy_reason = self.trigger.should_sleep(
            concepts=concepts,
            relations=relations,
            time_since_last_sleep=time_since,
        )
        trigger_stats["legacy_reason"] = legacy_reason
        return should_sleep, legacy_reason, trigger_stats

    def get_sleep_readiness(self, concepts: List[Concept], relations: List[Relation]) -> Dict:
        """
        Get detailed sleep readiness metrics.

        Returns:
            Dictionary with readiness components
        """
        trigger_stats = self.trigger.get_trigger_stats(concepts, relations)

        forgetting_stats = self.forgetting.get_forgetting_stats(concepts)

        time_since = self.trigger.time_since_last_sleep()
        mode, reason, _ = self.select_sleep_mode(concepts, relations)

        return {
            'entropy': trigger_stats['current_entropy'],
            'entropy_threshold': trigger_stats['entropy_threshold'],
            'conflict_density': trigger_stats['conflict_density'],
            'conflict_threshold': trigger_stats['conflict_threshold'],
            'time_since_sleep': time_since,
            'time_since_micro_sleep': self.trigger.time_since_last_micro_sleep(),
            'time_since_deep_sleep': self.trigger.time_since_last_deep_sleep(),
            'max_interval': self.trigger.max_interval,
            'forgettable_count': forgetting_stats.get('forgettable', 0),
            'total_concepts': forgetting_stats.get('total_concepts', 0),
            'should_sleep': trigger_stats['should_sleep'],
            'suggested_mode': mode,
            'sleep_reason': reason,
        }

    def get_last_cycle(self) -> Optional[SleepCycle]:
        """Get the last completed sleep cycle"""
        return self._current_cycle

    def get_current_state(self) -> Dict:
        """Get current orchestrator state"""
        return {
            'last_wake_time': self._last_wake_time.isoformat() if self._last_wake_time else None,
            'current_cycle_active': self._current_cycle is not None,
            'turns_since_micro_sleep': self._turns_since_micro_sleep,
            'last_micro_sleep_seconds': self.trigger.time_since_last_micro_sleep(),
            'last_deep_sleep_seconds': self.trigger.time_since_last_deep_sleep(),
            'time_since_last_wake': (utc_now() - ensure_utc(self._last_wake_time)).total_seconds()
                if self._last_wake_time else None
        }
