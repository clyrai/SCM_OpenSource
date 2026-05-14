"""
Lifecycle: autonomous background processes that give the agent a continuous
existence beyond user-driven API calls.

Phase 7 modules:
  - idle_learner:  background daemon that runs sleep cycles during user idle time
  - wake_summary:  builds the user-visible "while you were away..." report
"""

from .idle_learner import IdleLearner, IdleLearnerConfig
from .wake_summary import WakeInsight, WakeSummary, WakeSummaryBuilder
from .curiosity import (
    CuriosityConfig,
    CuriosityEngine,
    CuriositySource,
    FilledGap,
    KnowledgeGap,
    LLMSource,
    LocalDocsSource,
    StaticDictionarySource,
)
from .lifecycle_policy import (
    AlwaysAllowPolicy,
    BatteryPolicy,
    CompositePolicy,
    CPULoadPolicy,
    LifecyclePolicy,
    PolicyDecision,
    build_default_policy_from_config,
)
from .state_store import IdleLearnerStateStore

__all__ = [
    "IdleLearner",
    "IdleLearnerConfig",
    "IdleLearnerStateStore",
    "WakeInsight",
    "WakeSummary",
    "WakeSummaryBuilder",
    "CuriosityConfig",
    "CuriosityEngine",
    "CuriositySource",
    "FilledGap",
    "KnowledgeGap",
    "LLMSource",
    "LocalDocsSource",
    "StaticDictionarySource",
    "AlwaysAllowPolicy",
    "BatteryPolicy",
    "CompositePolicy",
    "CPULoadPolicy",
    "LifecyclePolicy",
    "PolicyDecision",
    "build_default_policy_from_config",
]
