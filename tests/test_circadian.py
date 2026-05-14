"""Tests for the circadian sleep scheduling primitive (v0.7.7+)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.lifecycle.circadian import (
    is_in_window, parse_hhmm, resolve_tz, should_fire,
)


# ── parse_hhmm ──────────────────────────────────────────────────────────


def test_parse_hhmm_valid():
    assert parse_hhmm("00:00") == 0
    assert parse_hhmm("07:30") == 450
    assert parse_hhmm("23:59") == 23 * 60 + 59


def test_parse_hhmm_invalid():
    assert parse_hhmm("") is None
    assert parse_hhmm("25:00") is None
    assert parse_hhmm("12:60") is None
    assert parse_hhmm("nope") is None
    assert parse_hhmm("12") is None


# ── is_in_window ────────────────────────────────────────────────────────


def _at(h: int, m: int = 0) -> datetime:
    return datetime(2026, 5, 5, h, m, tzinfo=timezone.utc)


def test_window_no_wrap():
    # 22:00 → 06:00 wraps; 09:00 → 17:00 doesn't.
    s, e = parse_hhmm("09:00"), parse_hhmm("17:00")
    assert is_in_window(_at(8, 59), s, e) is False
    assert is_in_window(_at(9, 0), s, e) is True
    assert is_in_window(_at(13, 30), s, e) is True
    assert is_in_window(_at(17, 0), s, e) is False


def test_window_wraps_midnight():
    # Classic 23:00 → 07:00 night
    s, e = parse_hhmm("23:00"), parse_hhmm("07:00")
    assert is_in_window(_at(22, 30), s, e) is False
    assert is_in_window(_at(23, 0), s, e) is True
    assert is_in_window(_at(2, 0), s, e) is True
    assert is_in_window(_at(6, 59), s, e) is True
    assert is_in_window(_at(7, 0), s, e) is False
    assert is_in_window(_at(9, 0), s, e) is False


def test_window_zero_length_disabled():
    s, e = parse_hhmm("12:00"), parse_hhmm("12:00")
    assert is_in_window(_at(12, 0), s, e) is False
    assert is_in_window(_at(0, 0), s, e) is False


# ── should_fire ─────────────────────────────────────────────────────────


def _cfg(**overrides) -> dict:
    base = {
        "enabled": True,
        "timezone": "UTC",
        "sleep_start": "23:00",
        "sleep_end": "07:00",
        "last_sleep_at": None,
    }
    base.update(overrides)
    return base


def test_disabled_never_fires():
    now = _at(23, 30)
    assert should_fire(_cfg(enabled=False), now_utc=now) is False


def test_outside_window_doesnt_fire():
    now = _at(15, 0)  # afternoon
    assert should_fire(_cfg(), now_utc=now) is False


def test_inside_window_fires_when_never_slept():
    now = _at(23, 30)
    assert should_fire(_cfg(last_sleep_at=None), now_utc=now) is True


def test_inside_window_doesnt_re_fire_same_night():
    """Wrap-around guard: if we slept at 23:00 last night, then 02:00 the
    same night must NOT trigger again."""
    now = _at(2, 0)  # 2am, still in the window that opened at 23:00 yesterday
    last_sleep_iso = (now - timedelta(hours=3)).isoformat()  # ~23:00 yesterday
    cfg = _cfg(last_sleep_at=last_sleep_iso)
    assert should_fire(cfg, now_utc=now) is False


def test_fires_again_next_night():
    """If we slept on night N, night N+1 should fire fresh."""
    now = _at(23, 30)
    # Pretend last sleep was 24h ago, same time-of-day.
    last_sleep = (now - timedelta(hours=24)).isoformat()
    cfg = _cfg(last_sleep_at=last_sleep)
    assert should_fire(cfg, now_utc=now) is True


def test_invalid_hhmm_doesnt_fire():
    cfg = _cfg(sleep_start="not-a-time")
    assert should_fire(cfg, now_utc=_at(23, 30)) is False


def test_user_in_lisbon_at_2300_local():
    """A Lisbon user at 23:00 their time = 22:00 UTC during winter / 22:00 UTC.
    The scheduler must look at LOCAL time, not UTC."""
    # 2026-05-05 22:00 UTC is 23:00 in Europe/Lisbon (BST equivalent).
    now_utc = datetime(2026, 5, 5, 22, 0, tzinfo=timezone.utc)
    cfg = _cfg(timezone="Europe/Lisbon")
    assert should_fire(cfg, now_utc=now_utc) is True


def test_unknown_timezone_falls_back_to_utc():
    """resolve_tz silently returns UTC for unknown names — keeps the
    scheduler running rather than crashing on a typo."""
    cfg = _cfg(timezone="Bogus/Nowhere")
    # 23:30 UTC, treated as UTC because tz couldn't resolve
    assert should_fire(cfg, now_utc=_at(23, 30)) is True


# ── resolve_tz sanity ───────────────────────────────────────────────────


def test_resolve_tz_known_name():
    tz = resolve_tz("Europe/Lisbon")
    assert tz is not None
    # tz attribute should exist on a datetime localized to it
    now = datetime.now(timezone.utc).astimezone(tz)
    assert now.tzinfo is tz


def test_resolve_tz_blank_returns_utc():
    assert resolve_tz("") is timezone.utc
