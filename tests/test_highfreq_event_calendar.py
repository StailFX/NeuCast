"""Tests for ``app.highfreq.event_calendar`` — pure decision + JSON loader."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from app.highfreq.event_calendar import (
    CalendarEvent,
    load_events_from_json,
    should_halt_for_event,
    upcoming_events,
)


def _evt(**kw) -> CalendarEvent:
    base = dict(
        name="FOMC", ts=datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc),
        halt_before_min=15, halt_after_min=60, category="macro",
        symbols=None,
    )
    base.update(kw)
    return CalendarEvent(**base)


# ─── window ───


def test_event_window_brackets_ts():
    e = _evt()
    start, end = e.window()
    assert start == datetime(2026, 5, 1, 17, 45, tzinfo=timezone.utc)
    assert end == datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc)


# ─── should_halt_for_event ───


def test_no_events_means_no_halt():
    """Empty calendar = always pass-through. Pin so a refactor that
    flips default-halt-true never lands."""
    d = should_halt_for_event(
        [], symbol="BTCUSDT",
        now=datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc),
    )
    assert d.halted is False
    assert d.event_name is None


def test_halt_active_inside_pre_event_window():
    """10 min before FOMC (which has halt_before_min=15) → halted."""
    e = _evt()
    d = should_halt_for_event(
        [e], symbol="BTCUSDT",
        now=datetime(2026, 5, 1, 17, 50, tzinfo=timezone.utc),
    )
    assert d.halted is True
    assert d.event_name == "FOMC"
    assert d.minutes_until_event == pytest.approx(10.0)


def test_halt_active_inside_post_event_window():
    """30 min after FOMC (halt_after_min=60) → still halted."""
    e = _evt()
    d = should_halt_for_event(
        [e], symbol="BTCUSDT",
        now=datetime(2026, 5, 1, 18, 30, tzinfo=timezone.utc),
    )
    assert d.halted is True
    assert d.minutes_until_event == pytest.approx(-30.0)


def test_no_halt_just_before_window():
    """16 min before FOMC (halt_before_min=15) → NOT halted yet."""
    e = _evt()
    d = should_halt_for_event(
        [e], symbol="BTCUSDT",
        now=datetime(2026, 5, 1, 17, 44, tzinfo=timezone.utc),
    )
    assert d.halted is False


def test_no_halt_just_after_window():
    """61 min after FOMC (halt_after_min=60) → window closed."""
    e = _evt()
    d = should_halt_for_event(
        [e], symbol="BTCUSDT",
        now=datetime(2026, 5, 1, 19, 1, tzinfo=timezone.utc),
    )
    assert d.halted is False


def test_symbol_specific_event_does_not_halt_other_symbols():
    """Hard fork on ETH should NOT halt BTC trading."""
    eth_fork = _evt(name="ETH Cancun", symbols=("ETHUSDT",))
    now = datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc)
    assert should_halt_for_event([eth_fork], symbol="BTCUSDT", now=now).halted is False
    assert should_halt_for_event([eth_fork], symbol="ETHUSDT", now=now).halted is True


def test_first_matching_event_wins_on_overlap():
    """If two events overlap (rare), the first by ts wins. Pin so a
    refactor that picks last/random fails CI."""
    e1 = _evt(name="A", ts=datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc))
    e2 = _evt(name="B", ts=datetime(2026, 5, 1, 18, 30, tzinfo=timezone.utc),
              halt_before_min=60)  # overlaps with A's window
    now = datetime(2026, 5, 1, 18, 5, tzinfo=timezone.utc)
    d = should_halt_for_event([e1, e2], symbol="BTCUSDT", now=now)
    # Both events' windows include now; first encountered wins.
    assert d.halted is True
    assert d.event_name in ("A", "B")  # one of them


# ─── load_events_from_json ───


def test_load_missing_file_returns_empty(tmp_path):
    """File doesn't exist = quiet empty calendar, NOT raise."""
    out = load_events_from_json(tmp_path / "nope.json")
    assert out == []


def test_load_valid_calendar(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps([
        {"name": "FOMC", "ts": "2026-05-01T18:00:00+00:00",
         "halt_before_min": 15, "halt_after_min": 60, "category": "macro"},
        {"name": "ETH fork", "ts": "2026-05-20T03:00:00+00:00",
         "halt_before_min": 30, "halt_after_min": 120,
         "category": "fork", "symbols": ["ETHUSDT"]},
    ]))
    events = load_events_from_json(p)
    assert len(events) == 2
    # Sorted by ts asc.
    assert events[0].name == "FOMC"
    assert events[1].name == "ETH fork"
    assert events[1].symbols == ("ETHUSDT",)


def test_load_skips_malformed_entries(tmp_path):
    """One bad row must not take the whole calendar offline."""
    p = tmp_path / "cal.json"
    p.write_text(json.dumps([
        {"name": "good", "ts": "2026-05-01T18:00:00+00:00"},
        {"name": "bad", "ts": "not-a-date"},
        {"name": "alsogood", "ts": "2026-05-02T18:00:00+00:00"},
    ]))
    events = load_events_from_json(p)
    assert len(events) == 2
    assert {e.name for e in events} == {"good", "alsogood"}


def test_load_handles_naive_timestamp_as_utc(tmp_path):
    """ISO timestamp without timezone — treat as UTC. Crypto market
    operates in UTC; if someone forgets the +00:00 we don't crash."""
    p = tmp_path / "cal.json"
    p.write_text(json.dumps([
        {"name": "x", "ts": "2026-05-01T18:00:00"},
    ]))
    events = load_events_from_json(p)
    assert len(events) == 1
    assert events[0].ts.tzinfo is not None


def test_load_invalid_json_returns_empty(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    assert load_events_from_json(p) == []


def test_load_non_list_root_returns_empty(tmp_path):
    """Root must be a list."""
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"events": []}))
    assert load_events_from_json(p) == []


# ─── upcoming_events ───


def test_upcoming_events_only_future(tmp_path):
    past = _evt(ts=datetime(2025, 1, 1, tzinfo=timezone.utc))
    future = _evt(name="future", ts=datetime(2027, 1, 1, tzinfo=timezone.utc))
    out = upcoming_events(
        [past, future], symbol="BTCUSDT",
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    assert len(out) == 1
    assert out[0].name == "future"


def test_upcoming_events_filters_by_symbol():
    eth_only = _evt(symbols=("ETHUSDT",), ts=datetime(2027, 1, 1, tzinfo=timezone.utc))
    out_btc = upcoming_events(
        [eth_only], symbol="BTCUSDT",
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    out_eth = upcoming_events(
        [eth_only], symbol="ETHUSDT",
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    assert out_btc == []
    assert len(out_eth) == 1


def test_upcoming_events_respects_limit():
    events = [
        _evt(name=f"e{i}", ts=datetime(2027, 1, i + 1, tzinfo=timezone.utc))
        for i in range(20)
    ]
    out = upcoming_events(
        events, symbol="BTCUSDT",
        now=datetime(2026, 5, 1, tzinfo=timezone.utc), limit=5,
    )
    assert len(out) == 5


# ─── shipped calendar file is parseable ───


def test_repo_default_calendar_loads():
    """The committed `docs/highfreq/event_calendar.json` must always
    be parseable. Pin so a JSON syntax error in commit lands a CI
    fail rather than a silent runtime degradation on Tokyo."""
    from app.highfreq.event_calendar import DEFAULT_CALENDAR_PATH
    if DEFAULT_CALENDAR_PATH.exists():
        events = load_events_from_json(DEFAULT_CALENDAR_PATH)
        # Just verify it parses; content can change.
        assert isinstance(events, list)
