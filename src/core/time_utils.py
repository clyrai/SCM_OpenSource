"""
Shared UTC time helpers.
"""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime | str | None) -> datetime | None:
    """Normalize a datetime-like value to an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def utc_isoformat(value: datetime | str | None = None) -> str:
    """Return an ISO-8601 string in UTC."""
    normalized = ensure_utc(value) if value is not None else utc_now()
    return normalized.isoformat()
