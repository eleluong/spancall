"""Day-boundary and DST tests for the D3 user-local clock (spancall/infer/clock.py).

Contract under test (D3):
- The template API stays deterministic: ``build_input(..., now: date)`` never reads
  a wall clock or config. The *caller* resolves the user-local date via
  ``local_today(zone, utc_now)`` and passes it as ``now``.
- The fast path never guesses a timezone: unknown IANA zone raises so the caller
  can abstain on date params — never a silent UTC/server-local fallback.
- ``datetime.astimezone`` resolves the zone's active UTC offset, so DST
  spring-forward/fall-back boundaries map to the correct local date.

All expected dates/offsets below were verified against zoneinfo at authoring time
(2026-10-27), not guessed.
"""

import unittest
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfoNotFoundError

from spancall.infer.clock import local_today, tz_offset_minutes

_UTC = timezone.utc


def _utc(y, mo, d, h=0, mi=0, s=0, us=0):
    return datetime(y, mo, d, h, mi, s, us, tzinfo=_UTC)


class LocalTodayTest(unittest.TestCase):
    """`local_today` maps a UTC instant to the user-local calendar date."""

    def test_same_instant_three_calendar_dates(self):
        # 2026-10-27T10:00Z — a single instant spans three calendar dates because
        # of the ±14h/±12h extreme offsets. A caller that ignores the timezone
        # emits the wrong `today` with high confidence at day boundaries (P4).
        utc_now = _utc(2026, 10, 27, 10, 0, 0)
        self.assertEqual(local_today("Pacific/Kiritimati", utc_now), date(2026, 10, 28))  # UTC+14: already tomorrow
        self.assertEqual(local_today("UTC", utc_now), date(2026, 10, 27))
        self.assertEqual(local_today("Asia/Saigon", utc_now), date(2026, 10, 27))  # UTC+7
        self.assertEqual(local_today("Etc/GMT+12", utc_now), date(2026, 10, 26))  # UTC-12: still yesterday

    def test_negative_offset_zone_name_convention(self):
        # IANA fixed-offset zones are inverted: Etc/GMT+12 == UTC-12, Etc/GMT-14 ==
        # UTC+14. The test above pins the convention so a future refactor cannot
        # silently flip it (cf. Pacific/Kiritimati being UTC+14).
        self.assertEqual(
            local_today("Etc/GMT-14", _utc(2026, 10, 27, 10, 0, 0)), date(2026, 10, 28)
        )

    def test_minutes_before_and_after_midnight_flip_date(self):
        # Kiritimati crosses midnight at 2026-10-27T10:00Z.
        just_before = _utc(2026, 10, 27, 9, 59, 59)
        just_after = _utc(2026, 10, 27, 10, 0, 0)
        self.assertEqual(local_today("Pacific/Kiritimati", just_before), date(2026, 10, 27))
        self.assertEqual(local_today("Pacific/Kiritimati", just_after), date(2026, 10, 28))

    def test_dst_spring_forward_date_flip(self):
        # America/Havana springs forward 2026-03-08T05:00Z; local jumps
        # 23:59 -> 01:00, so the *calendar date* flips at the transition.
        before = _utc(2026, 3, 8, 4, 59, 59)
        after = _utc(2026, 3, 8, 5, 0, 0)
        self.assertEqual(local_today("America/Havana", before), date(2026, 3, 7))
        self.assertEqual(local_today("America/Havana", after), date(2026, 3, 8))

    def test_dst_spring_forward_date_stable_after_midnight(self):
        # America/New_York springs forward at 02:00 local (07:00Z) — well after
        # midnight, so the date is stable across the transition.
        before = _utc(2026, 3, 8, 6, 59, 59)
        after = _utc(2026, 3, 8, 7, 0, 0)
        self.assertEqual(local_today("America/New_York", before), date(2026, 3, 8))
        self.assertEqual(local_today("America/New_York", after), date(2026, 3, 8))

    def test_dst_fall_back_repeats_hour_date_stable(self):
        # America/Havana falls back 2026-11-01T05:00Z; local 00:59 -> 00:00
        # (midnight repeats). The date stays 2026-11-01 across the ambiguity.
        before = _utc(2026, 11, 1, 4, 59, 59)
        after = _utc(2026, 11, 1, 5, 0, 0)
        self.assertEqual(local_today("America/Havana", before), date(2026, 11, 1))
        self.assertEqual(local_today("America/Havana", after), date(2026, 11, 1))

    def test_unknown_timezone_raises_no_silent_fallback(self):
        # Correctness floor (D3): never guess a timezone. The caller must abstain
        # on date params, so an unknown IANA name must raise — not default to UTC.
        with self.assertRaises(ZoneInfoNotFoundError):
            local_today("Not/AZone", _utc(2026, 10, 27, 10, 0, 0))

    def test_defaults_to_current_utc_date(self):
        now = datetime.now(timezone.utc)
        self.assertEqual(local_today("UTC"), now.date())


class TzOffsetMinutesTest(unittest.TestCase):
    """`tz_offset_minutes` exposes the active offset so boundary tests are explicit."""

    def test_dst_spring_forward_offset_change(self):
        # NY: EST (-300) -> EDT (-240) at 2026-03-08T07:00Z.
        self.assertEqual(tz_offset_minutes("America/New_York", _utc(2026, 3, 8, 6, 59, 59)), -300)
        self.assertEqual(tz_offset_minutes("America/New_York", _utc(2026, 3, 8, 7, 0, 0)), -240)

    def test_dst_fall_back_offset_change(self):
        # NY: EDT (-240) -> EST (-300) at 2026-11-01T06:00Z.
        self.assertEqual(tz_offset_minutes("America/New_York", _utc(2026, 11, 1, 5, 59, 59)), -240)
        self.assertEqual(tz_offset_minutes("America/New_York", _utc(2026, 11, 1, 6, 0, 0)), -300)

    def test_fixed_offset_zones(self):
        self.assertEqual(tz_offset_minutes("Etc/GMT+12", _utc(2026, 10, 27, 10, 0, 0)), -720)
        self.assertEqual(tz_offset_minutes("Asia/Saigon", _utc(2026, 10, 27, 10, 0, 0)), 420)


class CallerContractTest(unittest.TestCase):
    """The caller-facing path: resolve `now` once and reuse the deterministic template."""

    def test_resolved_then_supplied_to_template_is_invariant(self):
        # Per-session override is free because the template API never reads the
        # clock: `now=local_today(zone, utc_now)` is just a date to `build_input`.
        from span_template.template import build_time_window

        utc_now = _utc(2026, 10, 27, 10, 0, 0)  # NY is still 10-27 here
        ny_today = local_today("America/New_York", utc_now)
        self.assertEqual(ny_today, date(2026, 10, 27))

        # Minimal deterministic window: what the template emits for "today" and
        # the D1 idiom "tomorrow" must be a pure function of the caller-resolved
        # `now` — the same date the clock resolved for the actual instant.
        window = build_time_window(ny_today, window_days=(), weekdays=(), idioms=("tomorrow",))
        by_key = {e.key: e.iso for e in window}
        self.assertEqual(by_key["today"], "2026-10-27")
        self.assertEqual(by_key["tomorrow"], "2026-10-28")


if __name__ == "__main__":
    unittest.main()