"""
Sleep API: Endpoints for sleep cycle management
"""
from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from ..core.long_term_memory import LongTermMemory
from ..core.working_memory import WorkingMemory
from ..core.models import Concept, Episode, Relation, SleepCycle
from .sleep_cycle import SleepCycleOrchestrator


router = APIRouter(prefix="/sleep", tags=["sleep"])


class SleepTriggerResponse(BaseModel):
    should_sleep: bool
    reason: str
    readiness: Dict


class SleepCycleResponse(BaseModel):
    success: bool
    mode: Optional[str] = None
    cycle_id: Optional[str] = None
    trigger_reason: Optional[str] = None
    nrem_stats: Dict
    rem_stats: Dict
    forgetting_stats: Dict
    dreams: list
    memories_consolidated: int
    memories_forgotten: int
    duration_seconds: float


class ReadinessResponse(BaseModel):
    entropy: float
    entropy_threshold: float
    conflict_density: float
    conflict_threshold: float
    time_since_sleep: Optional[float]
    max_interval: int
    forgettable_count: int
    total_concepts: int
    should_sleep: bool


orchestrator = SleepCycleOrchestrator()


def get_memory_components():
    """Get memory components from app state"""
    from fastapi import Request
    return None


@router.get("/trigger", response_model=SleepTriggerResponse)
async def check_sleep_trigger(
    concepts: list = None,
    relations: list = None
) -> SleepTriggerResponse:
    """
    Check if sleep should be triggered.

    Returns sleep trigger decision and detailed readiness metrics.
    """
    if not concepts or not relations:
        return SleepTriggerResponse(
            should_sleep=False,
            reason="No memory data provided",
            readiness={}
        )

    should_sleep, reason, trigger_stats = orchestrator.check_should_sleep(
        concepts=concepts,
        relations=relations
    )

    readiness = orchestrator.get_sleep_readiness(concepts, relations)

    return SleepTriggerResponse(
        should_sleep=should_sleep,
        reason=reason,
        readiness=readiness
    )


@router.post("/cycle", response_model=SleepCycleResponse)
async def begin_sleep_cycle(
    concepts: list,
    relations: list,
    episodes: list,
    mode: str = "deep",
) -> SleepCycleResponse:
    """
    Begin a complete sleep cycle.

    Executes:
    - NREM consolidation (Hebbian + downscaling)
    - REM dreaming (generative replay)
    - Forgetting (value-based removal)

    Returns detailed stats including dreams generated.
    """
    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts provided")

    success, cycle_record, cycle_stats = orchestrator.begin_sleep_cycle(
        concepts=concepts,
        relations=relations,
        episodes=episodes,
        mode=mode
    )

    if not success and cycle_stats.get('skipped'):
        raise HTTPException(
            status_code=200,
            detail={
                "success": False,
                "reason": cycle_stats.get('reason', 'Sleep not needed'),
                "skipped": True
            }
        )

    duration = 0.0
    if cycle_record.end_time and cycle_record.start_time:
        duration = (cycle_record.end_time - cycle_record.start_time).total_seconds()

    return SleepCycleResponse(
        success=True,
        mode=cycle_stats.get('mode'),
        cycle_id=cycle_record.id,
        trigger_reason=cycle_stats.get('trigger_reason'),
        nrem_stats=cycle_stats.get('nrem', {}),
        rem_stats=cycle_stats.get('rem', {}),
        forgetting_stats=cycle_stats.get('forgetting', {}),
        dreams=cycle_stats.get('dreams', []),
        memories_consolidated=cycle_record.memories_consolidated,
        memories_forgotten=cycle_record.memories_forgotten,
        duration_seconds=duration
    )


@router.get("/readiness", response_model=ReadinessResponse)
async def get_sleep_readiness(
    concepts: list,
    relations: list
) -> ReadinessResponse:
    """
    Get detailed sleep readiness metrics.

    Shows entropy, conflict density, time since last sleep,
    and how many concepts are candidates for forgetting.
    """
    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts provided")

    readiness = orchestrator.get_sleep_readiness(concepts, relations)

    return ReadinessResponse(**readiness)


@router.get("/cycle/latest")
async def get_latest_sleep_cycle() -> Optional[SleepCycle]:
    """
    Get the most recent sleep cycle record.
    """
    return orchestrator.get_last_cycle()


@router.get("/state")
async def get_orchestrator_state() -> Dict:
    """
    Get current sleep orchestrator state.

    Returns time since last wake, current cycle status, etc.
    """
    return orchestrator.get_current_state()
