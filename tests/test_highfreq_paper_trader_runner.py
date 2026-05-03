"""Tests for ``app.highfreq.paper_trader_runner`` — Phase C runtime glue.

The runner orchestrates four already-tested layers (feature pipeline,
predictor, paper_trader, asyncpg writer). Its OWN surface is small:

* Pure timing helpers (``next_minute_boundary``, ``sleep_until_next_tick``,
  ``bar_close_ts_for``) — sync, easy to pin.
* ``process_one_tick`` — async, but its decision tree is the
  interesting bit. Tested with mocked pool + stub predictor + real
  trader so we exercise the integration without spinning up Postgres.
* ``write_paper_trade`` / ``fetch_recent_seconds`` — async DB I/O.
  Tested by mocking ``asyncpg.Pool.acquire``.
* ``run_loop`` — the loop itself. One iteration test using injectable
  ``sleep_fn`` and ``now_fn`` so we don't wait real wall-clock minutes.

We deliberately do NOT spin up a real asyncpg connection or the actual
ingest — those are integration concerns covered by deploying to Tokyo
and watching the journal. Unit tests here cover all the branch logic.

Async tests use ``asyncio.run`` inside sync test functions to avoid a
new ``pytest-asyncio`` dependency.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.highfreq.feature_pipeline import FEATURE_COLUMNS
from app.highfreq.paper_trader import PaperTrade, PaperTrader, PaperTraderConfig
from app.highfreq.paper_trader_runner import (
    BAR_GRACE_SECONDS,
    LOOKBACK_SECONDS,
    _ShutdownFlag,
    bar_close_ts_for,
    fetch_recent_seconds,
    next_minute_boundary,
    process_one_tick,
    run_loop,
    sleep_until_next_tick,
    write_paper_trade,
)

UTC = timezone.utc


# ───────────────────────── helpers ─────────────────────────


def _ts(year: int = 2026, month: int = 4, day: int = 26, hour: int = 12,
        minute: int = 34, second: int = 0, micro: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=UTC)


def _seconds_frame(n_minutes: int = 3, seconds_per_minute: int = 60,
                   start: str = "2026-04-26 12:32:00") -> pd.DataFrame:
    """Per-second frame matching highfreq_ofi_1s schema. Same shape the
    ingest writes to Postgres."""
    rng = np.random.default_rng(0)
    rows = []
    t0 = pd.Timestamp(start, tz="UTC")
    for m in range(n_minutes):
        mid = 78_000.0 + m * 5.0  # gentle drift so mp_close varies
        for s in range(seconds_per_minute):
            ts = t0 + pd.Timedelta(minutes=m, seconds=s)
            rows.append({
                "ts": ts,
                "symbol": "BTCUSDT",
                "ofi": float(rng.normal(0.0, 0.5)),
                "microprice": float(mid + rng.normal(0.0, 1.0)),
                "depth_imb": float(rng.uniform(-0.2, 0.2)),
                "spread_bps": float(rng.uniform(0.5, 1.5)),
                "trade_imb": float(rng.normal(0.0, 0.001)),
                "n_updates": 10,
            })
    return pd.DataFrame(rows)


class _StubPredictor:
    """Minimal predictor stand-in — controllable by the test."""

    def __init__(self, *, has_model: bool = True, prob_up: Optional[float] = 0.7,
                 calibrated: bool = True) -> None:
        self._has_model = has_model
        self._prob_up = prob_up
        self._calibrated = calibrated
        self.predict_calls = 0
        self.last_features: Optional[pd.Series] = None

    def status(self) -> Any:
        s = MagicMock()
        s.has_model = self._has_model
        s.is_calibrated = self._calibrated
        s.model_age_seconds = 123.0
        return s

    def predict(self, features) -> Optional[float]:
        self.predict_calls += 1
        self.last_features = features
        return self._prob_up


def _mock_pool_returning(rows: list[dict]) -> MagicMock:
    """Mock asyncpg pool whose ``acquire().__aenter__()`` yields a conn
    whose ``fetch`` returns ``rows`` (each as an asyncpg-ish dict)."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.fetchval = AsyncMock(return_value=42)  # pretend INSERT returned id=42

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


# ───────────────────────── pure timing helpers ─────────────────────────


def test_next_minute_boundary_in_middle_of_minute():
    """12:34:17 → 12:35:00."""
    assert next_minute_boundary(_ts(minute=34, second=17)) == _ts(minute=35, second=0)


def test_next_minute_boundary_exactly_on_minute_advances_one():
    """12:34:00 (microsec=0) → 12:35:00 — never zero-sleep."""
    assert next_minute_boundary(_ts(minute=34, second=0)) == _ts(minute=35, second=0)


def test_next_minute_boundary_with_microseconds():
    """12:34:59.999999 → 12:35:00."""
    assert next_minute_boundary(_ts(minute=34, second=59, micro=999_999)) == \
        _ts(minute=35, second=0)


def test_sleep_until_next_tick_includes_grace_period():
    """At 12:34:17, next minute is 43s away + 1.5s grace = 44.5s."""
    s = sleep_until_next_tick(_ts(minute=34, second=17))
    assert s == pytest.approx(43.0 + BAR_GRACE_SECONDS, abs=0.01)


def test_sleep_until_next_tick_never_zero():
    """Even at the boundary, we sleep at least the grace period."""
    s = sleep_until_next_tick(_ts(minute=34, second=0))
    assert s >= 60.0  # full minute + grace, not zero


def test_bar_close_ts_for_floors_to_minute():
    """When firing at 12:35:01.5, the bar we score is the one that
    closed at 12:35:00."""
    assert bar_close_ts_for(_ts(minute=35, second=1, micro=500_000)) == \
        _ts(minute=35, second=0)


# ───────────────────────── _ShutdownFlag ─────────────────────────


def test_shutdown_flag_starts_unset():
    assert _ShutdownFlag().is_set() is False


def test_shutdown_flag_set_then_is_set():
    f = _ShutdownFlag()
    f.set()
    assert f.is_set() is True


# ───────────────────────── fetch_recent_seconds ─────────────────────────


def test_fetch_recent_seconds_empty_returns_empty_df_with_correct_columns():
    """When DB has no rows, runner gets an EMPTY frame with the right
    column shape — NOT None — so build_latest_feature_row can fail
    cleanly via its own None-return path."""
    pool = _mock_pool_returning([])
    df = asyncio.run(fetch_recent_seconds(pool, "BTCUSDT"))
    assert df.empty
    assert list(df.columns) == [
        "ts", "symbol", "ofi", "microprice", "depth_imb",
        "spread_bps", "trade_imb", "n_updates",
    ]


def test_fetch_recent_seconds_returns_populated_df():
    rows = _seconds_frame(n_minutes=2).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    df = asyncio.run(fetch_recent_seconds(pool, "BTCUSDT"))
    assert not df.empty
    assert len(df) == 120  # 2 minutes × 60s
    assert (df["symbol"] == "BTCUSDT").all()


# ──────────────────── fetch_recent_futures_seconds (T.23.b) ───────────────


def test_fetch_recent_futures_seconds_empty_returns_typed_empty_frame():
    """Cold start: futures table has no rows in the lookback window.
    Helper MUST return an empty DataFrame with the FULL 10-col shape
    (incl. mark_price + funding_rate) so the v3 inference builder's
    zero-fill path engages cleanly."""
    from app.highfreq.paper_trader_runner import fetch_recent_futures_seconds
    pool = _mock_pool_returning([])
    df = asyncio.run(fetch_recent_futures_seconds(pool, "BTCUSDT"))
    assert df.empty
    assert list(df.columns) == [
        "ts", "symbol", "ofi", "microprice", "depth_imb",
        "spread_bps", "trade_imb", "n_updates",
        "mark_price", "funding_rate",
    ]


def test_fetch_recent_futures_seconds_handles_db_error_returns_empty():
    """If the futures table doesn't exist (early deploy / cold cluster)
    or the funding poller is offline, we MUST log + return empty frame
    rather than crash the runner. Cold-start safety: missing futures
    just degrades to v1-style features via the v3 zero-fill path."""
    from app.highfreq.paper_trader_runner import fetch_recent_futures_seconds
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=Exception("table missing"))
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    df = asyncio.run(fetch_recent_futures_seconds(pool, "BTCUSDT"))
    # Empty frame with the full schema — runner can pass it to v3
    # builder which will zero-fill.
    assert df.empty
    assert "mark_price" in df.columns
    assert "funding_rate" in df.columns


def test_fetch_recent_futures_seconds_returns_populated_df():
    """Happy path: futures rows present, helper returns them with
    the 10-col schema including mark_price + funding_rate."""
    from app.highfreq.paper_trader_runner import fetch_recent_futures_seconds
    rows = []
    t0 = pd.Timestamp("2026-05-04T00:00:00Z")
    for s in range(30):
        rows.append({
            "ts": t0 + pd.Timedelta(seconds=s),
            "symbol": "BTCUSDT",
            "ofi": 0.1, "microprice": 79000.5,
            "depth_imb": 0.05, "spread_bps": 0.9,
            "trade_imb": 0.01, "n_updates": 5,
            "mark_price": 79002.0, "funding_rate": 0.0001,
        })
    pool = _mock_pool_returning(rows)
    df = asyncio.run(fetch_recent_futures_seconds(pool, "BTCUSDT"))
    assert not df.empty
    assert len(df) == 30
    assert df["mark_price"].iloc[0] == pytest.approx(79002.0)
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)


# ───────────────────────── write_paper_trade ─────────────────────────


def test_write_paper_trade_inserts_with_returning_id():
    trade = PaperTrade(
        symbol="BTCUSDT", side="long", qty=0.001,
        entry_ts=_ts(minute=10), entry_price=78_000.0, entry_prob_up=0.7,
        exit_ts=_ts(minute=11), exit_price=78_010.0,
        exit_reason="time_stop",
        fee_paid_total_usd=0.117, pnl_usd=0.005, pnl_bps=0.6,
        model_version="v1",
    )
    pool = _mock_pool_returning([])  # rows unused; fetchval mocked → 42

    new_id = asyncio.run(write_paper_trade(pool, trade))
    assert new_id == 42

    # Check the SQL params were constructed with all 13 fields in the right order.
    conn_mock = pool.acquire.return_value.__aenter__.return_value
    conn_mock.fetchval.assert_awaited_once()
    args = conn_mock.fetchval.await_args.args
    # arg[0] is the SQL string; arg[1..] are the parameters.
    assert "INSERT INTO paper_trades" in args[0]
    assert args[1:] == (
        "BTCUSDT", "long", 0.001,
        trade.entry_ts, 78_000.0, 0.7,
        trade.exit_ts, 78_010.0, "time_stop",
        0.117, 0.005, 0.6, "v1",
    )


# ───────────────────────── process_one_tick: branches ─────────────────────────


def test_process_one_tick_returns_none_when_no_model():
    """Cold-start contract: predictor.has_model=False → log + skip,
    NEVER call DB or trader."""
    pool = _mock_pool_returning([])
    predictor = _StubPredictor(has_model=False)
    trader = PaperTrader("BTCUSDT")

    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert out is None
    assert predictor.predict_calls == 0
    pool.acquire.assert_not_called()


def test_process_one_tick_returns_none_when_db_empty():
    pool = _mock_pool_returning([])
    predictor = _StubPredictor(has_model=True)
    trader = PaperTrader("BTCUSDT")

    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert out is None
    assert predictor.predict_calls == 0  # never reached


def test_process_one_tick_returns_none_when_feature_row_unbuildable():
    """Single in-flight minute (no complete prior minute) → feat=None → skip."""
    # 30 seconds in a single minute = current/in-flight only.
    rows = _seconds_frame(n_minutes=1, seconds_per_minute=30).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    predictor = _StubPredictor(has_model=True)
    trader = PaperTrader("BTCUSDT")

    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert out is None
    assert predictor.predict_calls == 0


def test_process_one_tick_returns_none_when_predict_returns_none():
    """Defensive: predictor.has_model=True but predict() returned None
    (rare, but matches our predictor's failure-keeps-old-model contract)."""
    rows = _seconds_frame(n_minutes=3).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    predictor = _StubPredictor(has_model=True, prob_up=None)
    trader = PaperTrader("BTCUSDT")

    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert out is None


def test_process_one_tick_neutral_signal_no_trade_returned():
    """Signal in the [0.45, 0.55] neutral band → trader.on_bar_close returns
    None (no trade closed, no entry opened)."""
    rows = _seconds_frame(n_minutes=3).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    predictor = _StubPredictor(has_model=True, prob_up=0.5)  # neutral
    trader = PaperTrader("BTCUSDT")

    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert out is None
    # predictor WAS called this time (we got past the no-model gate).
    assert predictor.predict_calls == 1


def test_process_one_tick_strong_signal_opens_position_no_trade_yet():
    """First bar with strong signal → trader OPENS position but returns
    None (no trade row written until close on time-stop)."""
    rows = _seconds_frame(n_minutes=3).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    predictor = _StubPredictor(has_model=True, prob_up=0.85)
    trader = PaperTrader("BTCUSDT")

    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert out is None
    assert trader.state.open_position is not None
    assert trader.state.open_position.side == "long"


def test_process_one_tick_closes_open_position_and_writes_trade():
    """T=35: open. T=36 (one minute later): close on time-stop → trade
    returned + INSERT executed + new id returned."""
    rows = _seconds_frame(n_minutes=3).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    predictor = _StubPredictor(has_model=True, prob_up=0.85)
    trader = PaperTrader("BTCUSDT")

    # Bar 1: open the position.
    asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert trader.state.open_position is not None

    # Bar 2: should close on time-stop (elapsed = 1 minute = horizon).
    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=36),
    ))
    assert out is not None
    assert isinstance(out, PaperTrade)
    assert out.exit_reason == "time_stop"

    # write_paper_trade was awaited.
    conn_mock = pool.acquire.return_value.__aenter__.return_value
    assert conn_mock.fetchval.await_count == 1


def test_process_one_tick_skips_when_uncalibrated():
    """Calibration gate: has_model=True but is_calibrated=False → trader
    short-circuits with SKIP_NOT_CALIBRATED. Returns None, no trade."""
    rows = _seconds_frame(n_minutes=3).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    predictor = _StubPredictor(has_model=True, prob_up=0.85, calibrated=False)
    trader = PaperTrader("BTCUSDT", config=PaperTraderConfig(require_calibrated=True))

    out = asyncio.run(process_one_tick(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", now=_ts(minute=35),
    ))
    assert out is None
    assert trader.state.open_position is None


# ───────────────────────── run_loop ─────────────────────────


def test_run_loop_one_iteration_then_shutdown():
    """Drive a single iteration end-to-end with injectable now_fn / sleep_fn,
    then trigger shutdown. Validates the loop's wiring without waiting
    real minutes."""
    pool = _mock_pool_returning([])
    predictor = _StubPredictor(has_model=False)  # idle path = simplest
    trader = PaperTrader("BTCUSDT")
    shutdown = _ShutdownFlag()

    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)
        # After the first sleep, set shutdown so the loop exits before
        # re-iterating (keeps the test bounded).
        shutdown.set()

    fixed_now = _ts(minute=34, second=17)

    asyncio.run(run_loop(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", shutdown=shutdown,
        sleep_fn=fake_sleep, now_fn=lambda: fixed_now,
    ))

    # Slept once before processing the tick.
    assert len(sleep_calls) == 1
    # Sleep was for the next-tick + grace.
    assert sleep_calls[0] == pytest.approx(43.0 + BAR_GRACE_SECONDS, abs=0.01)


def test_run_loop_force_closes_open_position_on_shutdown():
    """If a position is open when shutdown fires, run_loop must call
    trader.force_close + persist the resulting trade."""
    rows = _seconds_frame(n_minutes=3).to_dict(orient="records")
    pool = _mock_pool_returning(rows)
    predictor = _StubPredictor(has_model=True, prob_up=0.85)
    trader = PaperTrader("BTCUSDT")
    shutdown = _ShutdownFlag()

    # Pre-open a position on the trader (simulate a prior tick).
    trader.on_bar_close(
        ts=_ts(minute=30), microprice=100.0, prob_up=0.85, calibrated=True,
    )
    assert trader.state.open_position is not None

    async def fake_sleep(_seconds):
        # Trigger shutdown immediately so the loop exits before processing.
        shutdown.set()

    asyncio.run(run_loop(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", shutdown=shutdown,
        sleep_fn=fake_sleep, now_fn=lambda: _ts(minute=35),
    ))

    # force_close should have flattened the position.
    assert trader.state.open_position is None

    # And one trade should have been INSERTed (the forced close).
    conn_mock = pool.acquire.return_value.__aenter__.return_value
    assert conn_mock.fetchval.await_count == 1


def test_run_loop_swallows_exceptions_and_continues():
    """A single bad tick (e.g. transient DB error) must NOT kill the
    loop — log + continue. We verify by raising on the first tick and
    confirming the loop reaches the SECOND sleep call (i.e. didn't crash
    out of the while-body)."""
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=RuntimeError("transient DB error"))

    predictor = _StubPredictor(has_model=True)
    trader = PaperTrader("BTCUSDT")
    shutdown = _ShutdownFlag()

    sleep_count = {"n": 0}

    async def fake_sleep(_seconds):
        sleep_count["n"] += 1
        # Set shutdown on the SECOND sleep so we get one full
        # iteration + the start of a second iteration's sleep before
        # the loop checks shutdown.is_set() and breaks.
        if sleep_count["n"] >= 2:
            shutdown.set()

    asyncio.run(run_loop(
        pool=pool, predictor=predictor, trader=trader,
        symbol="BTCUSDT", shutdown=shutdown,
        sleep_fn=fake_sleep, now_fn=lambda: _ts(minute=35),
    ))

    # Reaching the second sleep call proves the first iteration's
    # exception was swallowed — the loop didn't bail.
    assert sleep_count["n"] == 2
    # The first iteration attempted DB acquire (and failed). The second
    # sleep set shutdown, so the loop exited before the second tick.
    assert pool.acquire.call_count == 1
