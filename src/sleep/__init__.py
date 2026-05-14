"""
SleepAI Sleep Module
Brain-inspired sleep consolidation with NREM/REM cycles
"""
from .trigger import SleepTrigger
from .nrem import NREMConsolidation
from .rem import REMDreaming
from .forgetting import ForgettingModule
from .forgetting_dynamics import ForgettingDynamics
from .micro_sleep import MicroSleep
from .deep_sleep import DeepSleep
from .sleep_cycle import SleepCycleOrchestrator

__all__ = [
    'SleepTrigger',
    'NREMConsolidation',
    'REMDreaming',
    'ForgettingModule',
    'ForgettingDynamics',
    'MicroSleep',
    'DeepSleep',
    'SleepCycleOrchestrator'
]
