"""Tests for ``app.highfreq.realized_accuracy`` — rolling realized
directional accuracy of the paper trader.

Two layers:

1. **Pure logic** (``compute_rolling_accuracy_from_rows``) — takes a
   list of :class:`TradeRow` and produces a snapshot. No DB, no async.
   These pin the math: hit/miss rules, halt_close exclusion, tie
   handling, empty-window behaviour.

2. **Sync DB layer** (``fetch_rolling_accuracy_sync``) — exercised
   with a fake SQLAlchemy-like session (``execute(...).fetchall()``)
   so we don't need a real Postgres but we DO assert the SQL shape
   and the row mapping.

The async path (``fetch_rolling_accuracy``) is structurally identical
— same row mapping, same compute call — and is exercised in
production by the paper-trader runner; we don't duplicate the asyncpg
mock here because (a) it's >2x test overhead for zero new coverage
and (b) the runner's smoke test on the VPS is the integration test.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.highfreq.realized_accuracy import (
    DEFAULT_WINDOW_SIZES,
    RealizedAccuracy,
    TradeRow,
    compute_rolling_accuracy_from_rows,
    fetch_rolling_accuracy_sync,
)


# ────────────────────────── fixtures ──────────────────────────


def _row(
    *,
    side: str = "long",
    entry: float = 77_000.0,
    exit: float = 77_010.0,
    reason: str = "time_stop",
    minutes_ago: int = 1,
    proba_up: float = 0.62,
    symbol: str = "BTCUSDT",
) -> TradeRow:
    """Compact builder for hand-crafted fixtures."""
    base = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
    return TradeRow(
        symbol=symbol, side=side,
        entry_price=entry, exit_price=exit,
        exit_reason=reason,
        exit_ts=base - timedelta(minutes=minutes_ago),
        entry_prob_up=proba_up,
    )


# ────────────────────── pure-logic tests ──────────────────────


def test_compute_returns_none_accuracy_on_empty_rows():
    """Cold start (no trades closed yet) — endpoint must render "—",
    NOT "0.00 accuracy". Pin this contract."""
    snap = compute_rolling_accuracy_from_rows([], symbol="BTCUSDT", window=100)
    assert snap.n_trades_total == 0
    assert snap.n_eligible == 0
    assert snap.n_correct == 0
    assert snap.accuracy is None
    assert snap.avg_predicted_proba_up is None


def test_compute_long_correct_when_exit_above_entry():
    rows = [_row(side="long", entry=77_000, exit=77_010)]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_eligible == 1
    assert snap.n_correct == 1
    assert snap.accuracy == 1.0


def test_compute_long_wrong_when_exit_below_entry():
    rows = [_row(side="long", entry=77_000, exit=76_990)]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_correct == 0
    assert snap.accuracy == 0.0


def test_compute_short_correct_when_exit_below_entry():
    rows = [_row(side="short", entry=77_000, exit=76_990)]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_correct == 1
    assert snap.accuracy == 1.0


def test_compute_short_wrong_when_exit_above_entry():
    rows = [_row(side="short", entry=77_000, exit=77_010)]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_correct == 0
    assert snap.accuracy == 0.0


def test_compute_ties_count_as_misses():
    """Tied bar — exit == entry — is a miss. The rationale (no
    profitable move materialised) is in the module docstring; this
    test pins it so a future "let's count ties as half-correct"
    refactor fails CI immediately."""
    rows = [_row(side="long", entry=77_000.0, exit=77_000.0)]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_correct == 0
    assert snap.accuracy == 0.0


def test_compute_excludes_halt_close_from_denominator():
    """halt_close exits are dropped — they're forced by risk caps,
    not a fair test of model directional skill. n_trades_total
    still counts them so the operator can see "5 trades but only 3
    eligible because 2 were halt_close"."""
    rows = [
        _row(side="long", entry=77_000, exit=77_010, reason="time_stop"),
        _row(side="long", entry=77_000, exit=76_990, reason="halt_close"),
        _row(side="long", entry=77_000, exit=77_010, reason="halt_close"),
    ]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_trades_total == 3
    assert snap.n_eligible == 1
    assert snap.n_correct == 1
    assert snap.accuracy == 1.0


def test_compute_returns_none_when_only_halt_close_in_window():
    """If every trade in the window is halt_close, sample is empty —
    accuracy must be None (not 0/0 div-by-zero)."""
    rows = [
        _row(side="long", reason="halt_close"),
        _row(side="short", reason="halt_close"),
    ]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_trades_total == 2
    assert snap.n_eligible == 0
    assert snap.accuracy is None


def test_compute_aggregates_mixed_window_correctly():
    """7 trades: 4 long-correct, 1 long-wrong, 1 short-correct, 1 halt.
    Eligible = 6, correct = 5, accuracy = 5/6."""
    rows = [
        _row(side="long",  entry=77_000, exit=77_010, minutes_ago=1),
        _row(side="long",  entry=77_000, exit=77_005, minutes_ago=2),
        _row(side="long",  entry=77_000, exit=77_002, minutes_ago=3),
        _row(side="long",  entry=77_000, exit=77_001, minutes_ago=4),
        _row(side="long",  entry=77_000, exit=76_990, minutes_ago=5),  # wrong
        _row(side="short", entry=77_000, exit=76_990, minutes_ago=6),
        _row(side="long",  entry=77_000, exit=76_990,
             reason="halt_close",                    minutes_ago=7),
    ]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_trades_total == 7
    assert snap.n_eligible == 6
    assert snap.n_correct == 5
    assert snap.accuracy == pytest.approx(5 / 6)


def test_compute_returns_avg_predicted_proba():
    """avg_predicted_proba_up averages over eligible trades only.
    Validates that the proba and accuracy are computed on the SAME
    sample — important so the UI can correlate "trader was 0.65
    confident on the up-call → was right 60% of time"."""
    rows = [
        _row(side="long", entry=77_000, exit=77_010, proba_up=0.7,
             minutes_ago=1),
        _row(side="long", entry=77_000, exit=77_010, proba_up=0.6,
             minutes_ago=2),
        # halt_close — must NOT contribute to the proba average.
        _row(side="long", entry=77_000, exit=76_990, proba_up=0.99,
             reason="halt_close", minutes_ago=3),
    ]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    assert snap.n_eligible == 2
    assert snap.avg_predicted_proba_up == pytest.approx(0.65)


def test_compute_records_earliest_and_latest_exit_ts():
    """Sample boundaries — useful for the UI to show "based on trades
    from <time> to <time>". These come from eligible trades only."""
    rows = [
        _row(side="long", minutes_ago=10),
        _row(side="long", minutes_ago=20),
        _row(side="long", minutes_ago=30),
    ]
    snap = compute_rolling_accuracy_from_rows(rows, symbol="BTCUSDT", window=100)
    # earliest is the OLDEST trade's exit_ts (30 min ago), latest is
    # the most recent (10 min ago).
    assert snap.earliest_exit_ts is not None
    assert snap.latest_exit_ts is not None
    assert snap.latest_exit_ts > snap.earliest_exit_ts


def test_compute_raises_on_unknown_side():
    """Defence: a row with an unexpected ``side`` would silently report
    0% accuracy if ``_is_directional_hit`` defaulted to ``False``. We
    raise instead so a DB CHECK violation gets noticed loudly."""
    bad = TradeRow(
        symbol="BTCUSDT", side="???",
        entry_price=77_000, exit_price=77_010,
        exit_reason="time_stop",
        exit_ts=datetime(2026, 4, 27, tzinfo=timezone.utc),
        entry_prob_up=0.5,
    )
    with pytest.raises(ValueError, match="unknown side"):
        compute_rolling_accuracy_from_rows([bad], symbol="BTCUSDT", window=100)


def test_default_window_sizes_constant():
    """Pin the windows so the Prometheus gauge label space doesn't
    drift unexpectedly. Grafana panels query specific window labels
    — changing this forces a dashboard update too."""
    assert DEFAULT_WINDOW_SIZES == (50, 100)


def test_realized_accuracy_to_dict_is_json_safe():
    """The endpoint embeds ``snap.to_dict()`` directly. Pin the keys
    so a future field rename breaks tests instead of breaking the UI
    silently."""
    snap = RealizedAccuracy(
        symbol="BTCUSDT", window=100,
        n_trades_total=10, n_eligible=8, n_correct=5,
        accuracy=5 / 8, avg_predicted_proba_up=0.6,
        earliest_exit_ts="2026-04-27T11:00:00+00:00",
        latest_exit_ts="2026-04-27T12:00:00+00:00",
    )
    d = snap.to_dict()
    assert set(d.keys()) == {
        "symbol", "window",
        "n_trades_total", "n_eligible", "n_correct",
        "accuracy", "avg_predicted_proba_up",
        "earliest_exit_ts", "latest_exit_ts",
    }


# ──────────────── sync DB-layer test ────────────────


class _FakeRow:
    """Mimic SQLAlchemy Row (attribute access)."""
    def __init__(self, **kw): self.__dict__.update(kw)


class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def __iter__(self): return iter(self._rows)


class _FakeSession:
    """Records executed SQL + params, returns canned rows."""
    def __init__(self, rows):
        self._rows = rows
        self.last_sql_text: str | None = None
        self.last_params: dict[str, Any] | None = None

    def execute(self, sql, params):
        # SQLAlchemy `text()` exposes its string at .text in SA 2.x —
        # we don't pin the API, just stash for asserts.
        self.last_sql_text = str(getattr(sql, "text", sql))
        self.last_params = params
        return _FakeResult(self._rows)


def test_fetch_rolling_accuracy_sync_runs_correct_sql():
    """Pin the DB query: filtered by symbol, ordered by exit_ts DESC,
    LIMITed by window. A regression here would silently make the
    rolling accuracy use ALL trades instead of the latest N."""
    session = _FakeSession(rows=[
        _FakeRow(
            symbol="BTCUSDT", side="long",
            entry_price=77_000.0, exit_price=77_010.0,
            exit_reason="time_stop",
            exit_ts=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
            entry_prob_up=0.6,
        ),
    ])
    snap = fetch_rolling_accuracy_sync(session, symbol="BTCUSDT", window=50)

    assert "SELECT" in session.last_sql_text.upper()
    assert "paper_trades" in session.last_sql_text
    assert "exit_ts DESC" in session.last_sql_text
    assert "LIMIT" in session.last_sql_text.upper()
    assert session.last_params == {"symbol": "BTCUSDT", "window": 50}
    assert snap.n_eligible == 1
    assert snap.accuracy == 1.0


def test_fetch_rolling_accuracy_sync_raises_on_zero_window():
    """Zero / negative windows are operator errors — fail loudly so a
    bad ?window=0 query parameter doesn't silently return 0/0=NaN."""
    session = _FakeSession(rows=[])
    with pytest.raises(ValueError, match="window must be positive"):
        fetch_rolling_accuracy_sync(session, symbol="BTCUSDT", window=0)
