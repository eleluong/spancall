"""User-local clock resolution (docs/remediation_plan.md §3 D3).

The template API stays deterministic: ``build_input(..., now: date)`` takes the
place-holder date and never reads a wall clock or config. The *caller* (router /
resolver / probe) resolves the user-local date from the tenant's IANA timezone
and passes it as ``now``. A per-session override is free — pass a different
``now``.

Correctness floor: the fast path never guesses a timezone. If no timezone is
known, ``local_today`` is not called (the caller decides: server-local fallback
is a policy choice, not a silent default here).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

_UTC = timezone.utc


def local_today(zone: str, utc_now: Optional[datetime] = None) -> date:
    """Return the user-local calendar date for an IANA ``zone`` name.

    Args:
        zone: IANA timezone name (e.g. ``"Asia/Saigon"``, ``"America/New_York"``).
        utc_now: UTC datetime to anchor; defaults to ``datetime.now(timezone.utc)``.

    Raises:
        ZoneInfoNotFoundError (or ``pytz.UnknownTimeZoneError``): unknown IANA zone
            — the caller must treat this as an abstain ("the fast path never
            guesses"), never as a silent UTC fallback.

    DST note: ``datetime.astimezone`` resolves the zone's active UTC offset, so a
    spring-forward/fall-back boundary (e.g. 02:30 on a DST-transition day) maps
    to the correct local date. Unit tests cover boundary and DST cases.
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover - Python < 3.9
        raise
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    local = utc_now.astimezone(ZoneInfo(zone))
    return local.date()


def tz_offset_minutes(zone: str, at: datetime) -> int:
    """Current UTC offset (minutes) of ``zone`` at the given instant.

    Used by tests to assert the boundary behavior without mocking ZoneInfo.
    """
    from zoneinfo import ZoneInfo

    return int(at.astimezone(ZoneInfo(zone)).utcoffset().total_seconds() // 60)