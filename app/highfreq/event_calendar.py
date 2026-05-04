"""Event-aware halt — pause trading around scheduled high-vol events.

Why this exists
===============

The directional model is fit on **regular regime** data — minute-by-
minute order-flow at typical Binance Spot vol. Around scheduled
**discontinuities** (FOMC decisions, CPI prints, ETF approvals, hard
forks), 1-minute realized vol can jump 10-50× and the moves are
narrative-driven, not microstructure-driven. The model has no signal
in those windows; trading them is a coin flip with magnified stakes.

Industry-standard solution: **event-aware halt**. A static calendar of
known high-vol events; the trader pauses entries 15-30 minutes before
and 30-60 minutes after each event. Existing positions still close
on time-stop normally — we don't strand capital, we just don't open
new exposure into a known coin flip.

Defence-grade narrative:
> "I do NOT use news parsing for directional signals — empirical
> studies show 1-minute crypto moves are dominated by order flow, not
> headlines. But I DO use a curated event calendar as a **risk
> filter**: model trained on regular regime, halts trading around
> scheduled discontinuities."

This is the cheapest, most reliable form of "news integration" — no
NLP, no API costs, no latency, no false positives from sarcasm.

Calendar source
---------------

Static JSON file at ``docs/highfreq/event_calendar.json`` with
hand-curated events. Format::

    [
      {"name": "FOMC decision",     "ts": "2026-05-01T18:00:00+00:00",
       "halt_before_min": 15, "halt_after_min": 60,
       "category": "macro"},
      {"name": "US CPI print",      "ts": "2026-05-13T12:30:00+00:00",
       "halt_before_min": 15, "halt_after_min": 45,
       "category": "macro"},
      {"name": "ETH Cancun fork",   "ts": "2026-05-20T03:00:00+00:00",
       "halt_before_min": 30, "halt_after_min": 120,
       "category": "fork", "symbols": ["ETHUSDT"]}
    ]

* ``ts`` ISO-8601 with timezone (UTC convention).
* ``halt_before_min`` / ``halt_after_min`` — minutes around the event
  to pause new entries.
* ``category`` — free-form (macro / regulator / fork / etc.) for log
  filtering.
* ``symbols`` (optional) — restrict the halt to a subset; absent ⇒
  applies to ALL symbols.

The operator updates the JSON manually — typical maintenance is
~10 events/month. Runner reads it on each tick (cheap: < 1 ms for
hundreds of entries).

Pure logic
----------

:func:`should_halt_for_event` is pure: takes the calendar + current
time + symbol, returns ``EventHaltDecision``. Tests pin the boundary
conditions (start of window, end of window, off-symbol).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


DEFAULT_CALENDAR_PATH = Path(__file__).resolve().parents[2] / "docs" / "highfreq" / "event_calendar.json"


@dataclass(frozen=True)
class CalendarEvent:
    name: str
    ts: datetime  # UTC-aware
    halt_before_min: int
    halt_after_min: int
    category: str = "uncategorised"
    symbols: tuple[str, ...] | None = None  # None ⇒ applies to all symbols

    def applies_to(self, symbol: str) -> bool:
        if self.symbols is None:
            return True
        return symbol.upper() in self.symbols

    def window(self) -> tuple[datetime, datetime]:
        """[start, end] of the halt window (inclusive)."""
        return (
            self.ts - timedelta(minutes=int(self.halt_before_min)),
            self.ts + timedelta(minutes=int(self.halt_after_min)),
        )


@dataclass(frozen=True)
class EventHaltDecision:
    """JSON-friendly: what the runner / endpoint surfaces."""
    halted: bool
    event_name: str | None
    event_ts_iso: str | None
    minutes_until_event: float | None     # negative when AFTER event
    halt_window_end_iso: str | None
    category: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────


_EVENTS_CACHE: dict[str, tuple[float, list["CalendarEvent"]]] = {}


def load_events_from_json(path: Path | str) -> list[CalendarEvent]:
    """Load + validate the calendar JSON. Returns events sorted by ts.

    Malformed entries are skipped with a WARNING — one bad row should
    NOT take the whole calendar offline.

    Code-review Perf-low (2026-05-04): mtime-cached. The paper-trader
    runner used to re-read + re-parse this file every minute tick;
    the file is updated maybe once a week. Cache key: (path, mtime).
    """
    p = Path(path)
    if not p.exists():
        logger.info("event_calendar: file not found at %s — running without", p)
        # Invalidate any prior cache entry so a future writeback is
        # picked up cleanly rather than masked by a stale "empty" cache.
        _EVENTS_CACHE.pop(str(p), None)
        return []

    try:
        st = p.stat()
    except OSError:
        return []
    cache_key = str(p)
    cached = _EVENTS_CACHE.get(cache_key)
    if cached is not None and cached[0] == st.st_mtime:
        return cached[1]

    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("event_calendar: failed to parse %s: %s", p, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("event_calendar: expected JSON list at %s, got %s", p, type(raw))
        return []

    events: list[CalendarEvent] = []
    for i, e in enumerate(raw):
        try:
            ts = datetime.fromisoformat(e["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            symbols = e.get("symbols")
            symbols_tuple = tuple(s.upper() for s in symbols) if symbols else None
            events.append(CalendarEvent(
                name=str(e["name"]),
                ts=ts,
                halt_before_min=int(e.get("halt_before_min", 15)),
                halt_after_min=int(e.get("halt_after_min", 45)),
                category=str(e.get("category", "uncategorised")),
                symbols=symbols_tuple,
            ))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "event_calendar: skipping malformed entry #%d in %s: %s",
                i, p, exc,
            )
    events.sort(key=lambda x: x.ts)
    # Cache the parsed-and-sorted list keyed by (path, mtime).
    _EVENTS_CACHE[cache_key] = (st.st_mtime, events)
    return events


def _clear_events_cache_for_tests() -> None:
    """Test helper — drops the mtime cache so each test sees a
    deterministic fresh-cache state. Production callers never need
    this; the cache is correctness-invariant under normal flow."""
    _EVENTS_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────
# Pure decision
# ─────────────────────────────────────────────────────────────────────


def should_halt_for_event(
    events: Sequence[CalendarEvent],
    *,
    symbol: str,
    now: datetime | None = None,
) -> EventHaltDecision:
    """Pure: decide if ``symbol`` should be halted right now.

    The "active" event is the FIRST one whose window contains ``now``
    AND that applies to this symbol (multi-event overlap is rare and
    the operator should fix the calendar; we don't try to be clever).
    """
    now = now if now is not None else datetime.now(tz=timezone.utc)

    for ev in events:
        start, end = ev.window()
        if not (start <= now <= end):
            continue
        if not ev.applies_to(symbol):
            continue
        delta_min = (ev.ts - now).total_seconds() / 60.0
        return EventHaltDecision(
            halted=True,
            event_name=ev.name,
            event_ts_iso=ev.ts.isoformat(),
            minutes_until_event=delta_min,
            halt_window_end_iso=end.isoformat(),
            category=ev.category,
        )

    return EventHaltDecision(
        halted=False,
        event_name=None,
        event_ts_iso=None,
        minutes_until_event=None,
        halt_window_end_iso=None,
        category=None,
    )


def upcoming_events(
    events: Sequence[CalendarEvent],
    *,
    symbol: str,
    now: datetime | None = None,
    limit: int = 10,
) -> list[CalendarEvent]:
    """Return up to ``limit`` future events that apply to ``symbol``.
    Used by the /events Telegram command to show the schedule."""
    now = now if now is not None else datetime.now(tz=timezone.utc)
    out = []
    for ev in events:
        if ev.ts <= now:
            continue
        if not ev.applies_to(symbol):
            continue
        out.append(ev)
        if len(out) >= limit:
            break
    return out
