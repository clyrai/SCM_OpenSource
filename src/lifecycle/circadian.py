"""Timezone-aware nightly sleep scheduling.

Single source of truth for "should this user sleep right now?" — used by
both the public /v1/* MCP server sweeper and the /demo router.

Replaces the legacy fixed-idle-timer model (`SCM_IDLE_THRESHOLD_SEC`).
The pitch is "memory that works like yours"; humans don't sleep every
N seconds of inactivity, they sleep once per night at a configured
bedtime in their local timezone. This module makes SCM behave the same.

Public surface:
    parse_hhmm(s)            → minute-of-day or None
    is_in_window(now, s, e)  → bool
    resolve_tz(name)         → tzinfo
    should_fire(config, now_utc) → bool
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, Dict, Optional


def resolve_tz(name: str) -> tzinfo:
    """Return a tzinfo for an IANA name (e.g. 'Europe/Lisbon').
    Falls back to UTC silently when zoneinfo can't resolve the name —
    keeping the scheduler robust against typos rather than crashing."""
    if not name:
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def parse_hhmm(s: str) -> Optional[int]:
    """Parse 'HH:MM' into minute-of-day (0..1439). None on bad input."""
    if not s:
        return None
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h = int(parts[0]); m = int(parts[1])
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h * 60 + m


def is_in_window(now_local: datetime, start_min: int, end_min: int) -> bool:
    """Whether `now_local`'s minute-of-day is inside [start, end), wrapping
    midnight when start > end. Zero-length window means "disabled"."""
    cur = now_local.hour * 60 + now_local.minute
    if start_min == end_min:
        return False
    if start_min < end_min:
        return start_min <= cur < end_min
    return cur >= start_min or cur < end_min


def _window_anchor_date(now_local: datetime, start_min: int, end_min: int) -> date:
    """Return the date that the *current* sleep window started on.

    For a wrap-around window (23:00→07:00), 02:00 belongs to the night
    that started yesterday, not today. Used to enforce once-per-night
    firing without the scheduler double-firing across midnight.
    """
    cur_min = now_local.hour * 60 + now_local.minute
    if start_min > end_min and cur_min < end_min:
        return (now_local - timedelta(days=1)).date()
    return now_local.date()


def should_fire(config: Dict[str, Any], now_utc: Optional[datetime] = None) -> bool:
    """Decide whether this user should run a deep-sleep cycle right now.

    Args:
        config: dict from `SQLiteMemory.get_user_sleep_config(...)`.
                Expects keys: enabled, timezone, sleep_start, sleep_end,
                last_sleep_at (ISO string or None).
        now_utc: override for testing. Defaults to wall clock.

    Returns True iff:
        • config.enabled
        • valid HH:MM in start/end
        • current local time is inside the configured window
        • last_sleep_at falls in a different window-anchor-date than now
    """
    if not config.get("enabled", True):
        return False
    start_min = parse_hhmm(config.get("sleep_start", ""))
    end_min = parse_hhmm(config.get("sleep_end", ""))
    if start_min is None or end_min is None:
        return False

    tz = resolve_tz(config.get("timezone", "UTC"))
    now_utc = now_utc or datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)

    if not is_in_window(now_local, start_min, end_min):
        return False

    cur_anchor = _window_anchor_date(now_local, start_min, end_min)
    last_iso = config.get("last_sleep_at")
    if last_iso:
        try:
            from ..core.time_utils import ensure_utc
            last_utc = ensure_utc(last_iso)
            if last_utc is not None:
                last_local = last_utc.astimezone(tz)
                last_anchor = _window_anchor_date(last_local, start_min, end_min)
                if last_anchor == cur_anchor:
                    return False  # already slept this night
        except Exception:
            pass

    return True
