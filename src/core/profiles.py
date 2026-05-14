"""
Runtime profiles for productized SCM deployments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .config import (
    AUTO_SLEEP_ENABLED,
    AUTO_SLEEP_INTERVAL,
    WORKING_MEMORY_CAPACITY,
    SLEEP_ENTROPY_THRESHOLD,
    SLEEP_CONFLICT_THRESHOLD,
    SLEEP_INTERVAL_MAX,
    MICRO_SLEEP_INTERVAL_TURNS,
    MICRO_SLEEP_ENTROPY_THRESHOLD,
    DEEP_SLEEP_MIN_IDLE_SECONDS,
    DEEP_SLEEP_SESSION_TURNS,
    DEEP_SLEEP_PRESSURE_THRESHOLD,
)


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    description: str
    working_memory_capacity: int
    auto_sleep_enabled: bool
    sleep_check_interval: int
    trigger: Dict[str, Any]


RUNTIME_PROFILES: Dict[str, RuntimeProfile] = {
    "chatbot": RuntimeProfile(
        name="chatbot",
        description="Balanced conversational memory profile for user-facing bots.",
        working_memory_capacity=max(7, WORKING_MEMORY_CAPACITY),
        auto_sleep_enabled=True if AUTO_SLEEP_ENABLED else True,
        sleep_check_interval=min(max(3, AUTO_SLEEP_INTERVAL), 6),
        trigger={
            "entropy_threshold": SLEEP_ENTROPY_THRESHOLD,
            "conflict_threshold": SLEEP_CONFLICT_THRESHOLD,
            "max_interval": SLEEP_INTERVAL_MAX,
            "micro_interval_turns": MICRO_SLEEP_INTERVAL_TURNS,
            "micro_entropy_threshold": MICRO_SLEEP_ENTROPY_THRESHOLD,
            "deep_min_idle_seconds": DEEP_SLEEP_MIN_IDLE_SECONDS,
            "deep_session_turns": DEEP_SLEEP_SESSION_TURNS,
            "deep_pressure_threshold": DEEP_SLEEP_PRESSURE_THRESHOLD,
        },
    ),
    "agent": RuntimeProfile(
        name="agent",
        description="Higher-throughput profile for autonomous/multi-step agents.",
        working_memory_capacity=max(10, WORKING_MEMORY_CAPACITY),
        auto_sleep_enabled=True,
        sleep_check_interval=3,
        trigger={
            "entropy_threshold": max(0.78, SLEEP_ENTROPY_THRESHOLD - 0.08),
            "conflict_threshold": max(0.2, SLEEP_CONFLICT_THRESHOLD - 0.05),
            "max_interval": min(SLEEP_INTERVAL_MAX, 1800),
            "micro_interval_turns": max(2, MICRO_SLEEP_INTERVAL_TURNS - 1),
            "micro_entropy_threshold": max(0.72, MICRO_SLEEP_ENTROPY_THRESHOLD - 0.05),
            "deep_min_idle_seconds": min(DEEP_SLEEP_MIN_IDLE_SECONDS, 480),
            "deep_session_turns": max(12, DEEP_SLEEP_SESSION_TURNS - 8),
            "deep_pressure_threshold": max(0.82, DEEP_SLEEP_PRESSURE_THRESHOLD - 0.06),
        },
    ),
    "research": RuntimeProfile(
        name="research",
        description="Retention-first profile for experiments and long-context studies.",
        working_memory_capacity=max(14, WORKING_MEMORY_CAPACITY),
        auto_sleep_enabled=True,
        sleep_check_interval=6,
        trigger={
            "entropy_threshold": min(0.97, SLEEP_ENTROPY_THRESHOLD + 0.05),
            "conflict_threshold": min(0.45, SLEEP_CONFLICT_THRESHOLD + 0.05),
            "max_interval": max(SLEEP_INTERVAL_MAX, 5400),
            "micro_interval_turns": max(4, MICRO_SLEEP_INTERVAL_TURNS),
            "micro_entropy_threshold": min(0.9, MICRO_SLEEP_ENTROPY_THRESHOLD + 0.04),
            "deep_min_idle_seconds": max(DEEP_SLEEP_MIN_IDLE_SECONDS, 1500),
            "deep_session_turns": max(DEEP_SLEEP_SESSION_TURNS, 36),
            "deep_pressure_threshold": min(0.98, DEEP_SLEEP_PRESSURE_THRESHOLD + 0.03),
        },
    ),
}

DEFAULT_PROFILE = "chatbot"


def normalize_profile_name(profile: str | None) -> str:
    value = (profile or DEFAULT_PROFILE).strip().lower()
    if value not in RUNTIME_PROFILES:
        return DEFAULT_PROFILE
    return value


def get_runtime_profile(profile: str | None) -> RuntimeProfile:
    return RUNTIME_PROFILES[normalize_profile_name(profile)]


def list_runtime_profiles() -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for key, preset in RUNTIME_PROFILES.items():
        payload[key] = {
            "name": preset.name,
            "description": preset.description,
            "working_memory_capacity": preset.working_memory_capacity,
            "auto_sleep_enabled": preset.auto_sleep_enabled,
            "sleep_check_interval": preset.sleep_check_interval,
            "trigger": dict(preset.trigger),
        }
    return payload
