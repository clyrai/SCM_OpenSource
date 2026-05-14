"""
Python SDK wrapper for SCM runtime.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

from src.chat.engine import ChatEngine
from src.core.profiles import list_runtime_profiles, normalize_profile_name
from src.core.time_utils import utc_now
from src.lifecycle import WakeSummary, WakeSummaryBuilder


class SCMEngine:
    """
    High-level SDK wrapper around ChatEngine.

    Example:
        engine = SCMEngine(session_id="demo", profile="chatbot", sandbox=True)
        response, meta = engine.message("My name is Alice")
    """

    def __init__(
        self,
        session_id: str = "default",
        profile: str = "chatbot",
        sandbox: bool = False,
        auto_sleep: Optional[bool] = None,
        sleep_check_interval: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.profile = normalize_profile_name(profile)
        self.sandbox = bool(sandbox)
        self._engine = ChatEngine(
            session_id=self.session_id,
            profile=self.profile,
            sandbox_mode=self.sandbox,
            enable_persistence=not self.sandbox,
            enable_auto_sleep=auto_sleep,
            sleep_check_interval=sleep_check_interval,
        )

    @property
    def engine(self) -> ChatEngine:
        return self._engine

    def message(self, text: str) -> Tuple[str, Dict[str, Any]]:
        return self._engine.chat(text)

    def sleep(self, mode: str = "deep") -> Optional[Dict[str, Any]]:
        return self._engine.force_sleep(mode=mode)

    def memory_report(self) -> Dict[str, Any]:
        return self._engine.get_memory_report()

    def wake_summary(
        self,
        since_hours: Optional[float] = None,
        max_insights: int = 6,
        include_diagnostics: bool = False,
    ) -> WakeSummary:
        since = None
        if since_hours is not None:
            since = utc_now() - timedelta(hours=since_hours)
        return WakeSummaryBuilder(self._engine).build(
            since=since,
            max_insights=max_insights,
            include_diagnostics=include_diagnostics,
        )

    def export_memory(
        self,
        include_suppressed: bool = True,
        include_superseded: bool = True,
    ) -> Dict[str, Any]:
        return self._engine.export_memory(
            include_suppressed=include_suppressed,
            include_superseded=include_superseded,
        )

    def import_memory(self, payload: Dict[str, Any], replace_existing: bool = False) -> Dict[str, int]:
        return self._engine.import_memory(payload=payload, replace_existing=replace_existing)

    def reset(self, clear_persistence: bool = False) -> None:
        self._engine.reset_memory(clear_persistence=clear_persistence)

    def close(self) -> bool:
        return self._engine.save_session()


def list_profiles() -> Dict[str, Dict[str, Any]]:
    return list_runtime_profiles()
