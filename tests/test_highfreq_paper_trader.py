"""Tests for ``app.highfreq.paper_trader`` — Phase C state machine.

The paper trader is sim-only by ADR-005 — no real orders are placed —
so the entire surface is unit-testable without any network or DB stubs.
What we DO need to pin down rigorously:

* **Pure helpers** (``fee_per_side``, ``compute_qty_for_notional``,
  ``compute_pnl``, ``decide_entry_side``). These run on every bar and a
  bug here silently corrupts every paper P&L number we'll show on the
  defense demo. No defaults, no clever fallbacks — exact arithmetic.
* **State machine**. ``on_bar_close`` is a 7-step decision tree (rollover
  → time-stop close → halt check → position-open guard → calibration
  gate → bad-price guard → entry decision). Each branch must land in
  ``_NoOpReason`` correctly so the UI's "last action" badge tells the
  truth.
* **Risk caps**. Three kill switches (consecutive losses, daily loss,
  one position at a time). A bug that lets a runaway loop accumulate
  positions defeats the entire safety story we're selling on защита.
* **Day rollover**. UTC midnight resets daily counters. Loss-streak
  halts persist across days (regime-shift catch); daily-loss halts
  clear (bounded-loss-per-window contract).

Tests are synchronous and do not touch the filesystem — the trader is
pure logic over ``(ts, microprice, prob_up, calibrated)`` tuples. The
runner that drives it is tested separately under ``test_highfreq_paper_trader_runner.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, cast

import pytest

from app.highfreq.paper_trader import (
    PaperPosition,
    PaperTrade,
    PaperTrader,
    PaperTraderConfig,
    RiskCaps,
    _NoOpReason,
    compute_pnl,
    compute_qty_for_notional,
    decide_entry_side,
    fee_per_side,
)


# ───────────────────────── helpers ─────────────────────────


UTC = timezone.utc


def _ts(minute: int = 0, *, hour: int = 0, day: int = 25) -> datetime:
    """Build an aware UTC timestamp for 2026-04-{day} {hour}:{minute}:00."""
    return datetime(2026, 4, day, hour, minute, 0, tzinfo=UTC)


def _bar(
    trader: PaperTrader,
    *,
    ts: datetime,
    microprice: float = 100.0,
    prob_up: float = 0.7,
    calibrated: bool = True,
    model_version: str = "test-v1",
) -> Optional[PaperTrade]:
    """Compact wrapper around ``trader.on_bar_close`` for table-driven tests."""
    return trader.on_bar_close(
        ts=ts,
        microprice=microprice,
        prob_up=prob_up,
        calibrated=calibrated,
        model_version=model_version,
    )


# ───────────────────────── fee_per_side ─────────────────────────


def test_fee_per_side_basic_arithmetic():
    """7.5 bps on $1000 notional = $0.75."""
    # qty=10 @ price=100 → notional $1000.
    assert fee_per_side(qty=10.0, price=100.0, maker_fee_bps=7.5) == pytest.approx(0.75)


def test_fee_per_side_zero_when_qty_zero():
    assert fee_per_side(qty=0.0, price=100.0, maker_fee_bps=7.5) == 0.0


def test_fee_per_side_zero_when_price_zero():
    """A bad price upstream must not produce phantom fees that look real."""
    assert fee_per_side(qty=1.0, price=0.0, maker_fee_bps=7.5) == 0.0


def test_fee_per_side_zero_when_negative_bps():
    """Negative bps would imply a rebate — not our model. Defensive zero."""
    assert fee_per_side(qty=1.0, price=100.0, maker_fee_bps=-1.0) == 0.0


# ───────────────────────── compute_qty_for_notional ─────────────────────────


def test_compute_qty_for_notional_basic():
    """$100 / $50_000 = 0.002 BTC."""
    assert compute_qty_for_notional(notional_usd=100.0, price=50_000.0) == pytest.approx(
        0.002,
    )


def test_compute_qty_for_notional_zero_on_bad_price():
    assert compute_qty_for_notional(notional_usd=100.0, price=0.0) == 0.0


def test_compute_qty_for_notional_zero_on_negative_notional():
    assert compute_qty_for_notional(notional_usd=-1.0, price=100.0) == 0.0


# ───────────────────────── compute_pnl ─────────────────────────


def test_compute_pnl_long_winning_zero_fee():
    """Long: entry 100 → exit 110, qty 1, no fees → +$10, +1000 bps."""
    net, bps = compute_pnl(
        side="long", entry_price=100.0, exit_price=110.0, qty=1.0,
        maker_fee_bps_per_side=0.0,
    )
    assert net == pytest.approx(10.0)
    assert bps == pytest.approx(1000.0)  # 10/100 * 10_000


def test_compute_pnl_long_losing_with_fees():
    """Long: entry 100, exit 99, qty 1, 7.5 bps/side.
    Gross = -$1; fees use the price OF EACH LEG (not symmetric):
    fee_entry = 1 * 100 * 7.5e-4 = $0.075
    fee_exit  = 1 * 99  * 7.5e-4 = $0.07425
    net = -1 - 0.075 - 0.07425 = -1.14925
    """
    net, bps = compute_pnl(
        side="long", entry_price=100.0, exit_price=99.0, qty=1.0,
        maker_fee_bps_per_side=7.5,
    )
    assert net == pytest.approx(-1.14925)
    assert bps == pytest.approx(-114.925)  # -1.14925/100 * 10_000


def test_compute_pnl_short_winning():
    """Short: entry 110, exit 100, qty 1 → gross +$10 (sold high, bought back low)."""
    net, bps = compute_pnl(
        side="short", entry_price=110.0, exit_price=100.0, qty=1.0,
        maker_fee_bps_per_side=0.0,
    )
    assert net == pytest.approx(10.0)
    # bps normalised by ENTRY notional (110), not exit.
    assert bps == pytest.approx(10.0 / 110.0 * 1e4)


def test_compute_pnl_short_losing_with_fees():
    """Short: entry 100, exit 101, qty 1, 7.5 bps/side → -$1.15."""
    net, _ = compute_pnl(
        side="short", entry_price=100.0, exit_price=101.0, qty=1.0,
        maker_fee_bps_per_side=7.5,
    )
    # Fees use exit_price=101 for the closing leg, not symmetric to long.
    fee_entry = 1.0 * 100.0 * 7.5e-4  # 0.075
    fee_exit = 1.0 * 101.0 * 7.5e-4   # 0.07575
    expected_net = -1.0 - fee_entry - fee_exit  # -1.15075
    assert net == pytest.approx(expected_net)


def test_compute_pnl_zero_on_bad_inputs():
    """Defensive: a NaN/zero price upstream must not produce P&L noise."""
    assert compute_pnl("long", 0.0, 100.0, 1.0, 7.5) == (0.0, 0.0)
    assert compute_pnl("long", 100.0, 0.0, 1.0, 7.5) == (0.0, 0.0)
    assert compute_pnl("long", 100.0, 100.0, 0.0, 7.5) == (0.0, 0.0)


def test_compute_pnl_unknown_side_raises():
    """Defensive: a typo in the runner must surface loudly, not be a silent zero."""
    with pytest.raises(ValueError, match="unknown side"):
        compute_pnl(
            side=cast(Any, "sideways"),
            entry_price=100.0, exit_price=100.0, qty=1.0, maker_fee_bps_per_side=0.0,
        )


def test_compute_pnl_round_trip_fee_cost_matches_15_bps():
    """Round-trip cost on ANY zero-move trade must be 2 × maker_fee_bps_per_side.
    This is the bar Tier 3 has to clear (~15 bp on Binance Spot+BNB).
    """
    net, bps = compute_pnl(
        side="long", entry_price=100.0, exit_price=100.0, qty=1.0,
        maker_fee_bps_per_side=7.5,
    )
    # No price move → net = 0 - fee_entry - fee_exit = -2 × fee.
    assert net == pytest.approx(-0.15)
    assert bps == pytest.approx(-15.0)


# ───────────────────────── decide_entry_side ─────────────────────────


def test_decide_entry_side_long_at_threshold():
    """0.55 (= entry_long_threshold) must trigger long — predicate is ``>=``."""
    cfg = PaperTraderConfig()
    assert decide_entry_side(0.55, cfg) == "long"


def test_decide_entry_side_short_at_threshold():
    """0.45 (= entry_short_threshold) must trigger short — predicate is ``<=``."""
    cfg = PaperTraderConfig()
    assert decide_entry_side(0.45, cfg) == "short"


def test_decide_entry_side_neutral_band_returns_none():
    cfg = PaperTraderConfig()
    for p in (0.46, 0.50, 0.54):
        assert decide_entry_side(p, cfg) is None, f"prob={p} should be neutral"


def test_decide_entry_side_garbage_returns_none():
    """NaN, negative, >1 — all bail. Don't open a trade on garbage signal."""
    cfg = PaperTraderConfig()
    assert decide_entry_side(-0.1, cfg) is None
    assert decide_entry_side(1.1, cfg) is None
    assert decide_entry_side(float("nan"), cfg) is None


def test_decide_entry_side_respects_custom_thresholds():
    """Tightening thresholds (e.g. 0.6 / 0.4 after grid-search) must take effect."""
    cfg = PaperTraderConfig(entry_long_threshold=0.6, entry_short_threshold=0.4)
    assert decide_entry_side(0.55, cfg) is None  # would have opened on default 0.55
    assert decide_entry_side(0.6, cfg) == "long"
    assert decide_entry_side(0.4, cfg) == "short"


# ───────────────────────── PaperPosition.elapsed_minutes ─────────────────────────


def test_position_elapsed_minutes_basic():
    pos = PaperPosition(
        symbol="BTCUSDT", side="long", qty=0.001,
        entry_ts=_ts(minute=10),
        entry_price=100.0, entry_prob_up=0.7, fee_paid_entry=0.0,
        model_version="v1",
    )
    assert pos.elapsed_minutes(_ts(minute=11)) == pytest.approx(1.0)
    assert pos.elapsed_minutes(_ts(minute=10, hour=1)) == pytest.approx(60.0)


def test_position_elapsed_minutes_floors_at_zero():
    """A bar timestamp BEFORE entry (clock skew) must not flip elapsed negative."""
    pos = PaperPosition(
        symbol="BTCUSDT", side="long", qty=0.001,
        entry_ts=_ts(minute=10),
        entry_price=100.0, entry_prob_up=0.7, fee_paid_entry=0.0,
        model_version="v1",
    )
    assert pos.elapsed_minutes(_ts(minute=5)) == 0.0


# ───────────────────────── PaperTrade.to_dict ─────────────────────────


def test_paper_trade_to_dict_serialises_datetimes():
    """asyncpg writer keeps datetimes aware; the JSON endpoint must not."""
    trade = PaperTrade(
        symbol="BTCUSDT", side="long", qty=0.001,
        entry_ts=_ts(minute=0), entry_price=100.0, entry_prob_up=0.7,
        exit_ts=_ts(minute=1), exit_price=100.5,
        exit_reason="time_stop",
        fee_paid_total_usd=0.15, pnl_usd=0.35, pnl_bps=35.0,
        model_version="v1",
    )
    d = trade.to_dict()
    # Must round-trip through json.dumps without TypeError.
    json.dumps(d)
    # ISO-format datetimes (not raw datetime objects).
    assert isinstance(d["entry_ts"], str)
    assert isinstance(d["exit_ts"], str)
    assert d["entry_ts"].startswith("2026-04-25T00:00:00")
    assert d["exit_ts"].startswith("2026-04-25T00:01:00")


# ───────────────────────── on_bar_close: happy path ─────────────────────────


def test_open_long_when_signal_strong():
    """prob_up >= 0.55 + calibrated → opens long, returns None (no trade row yet)."""
    trader = PaperTrader("BTCUSDT")
    out = _bar(trader, ts=_ts(minute=0), microprice=78_000.0, prob_up=0.7)
    assert out is None  # nothing CLOSED on this bar
    assert trader.state.open_position is not None
    assert trader.state.open_position.side == "long"
    assert trader.state.open_position.entry_price == 78_000.0
    assert trader.state.last_no_op == _NoOpReason.OK_OPENED


def test_open_short_when_signal_low():
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=78_000.0, prob_up=0.20)
    assert trader.state.open_position is not None
    assert trader.state.open_position.side == "short"
    assert trader.state.last_no_op == _NoOpReason.OK_OPENED


def test_neutral_signal_no_position_no_op():
    """0.50 lands in the neutral band — no position opens, last_no_op set."""
    trader = PaperTrader("BTCUSDT")
    out = _bar(trader, ts=_ts(minute=0), microprice=78_000.0, prob_up=0.50)
    assert out is None
    assert trader.state.open_position is None
    assert trader.state.last_no_op == _NoOpReason.SKIP_NEUTRAL_SIGNAL


def test_close_after_horizon_returns_trade():
    """Open at T=0, fire at T=1 (horizon=1) → trade returned, P&L computed."""
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)
    assert trader.state.open_position is not None

    # Bar T+1 with neutral prob so we don't immediately re-open.
    out = _bar(trader, ts=_ts(minute=1), microprice=101.0, prob_up=0.50)

    assert out is not None
    assert isinstance(out, PaperTrade)
    assert out.side == "long"
    assert out.entry_price == 100.0
    assert out.exit_price == 101.0
    assert out.exit_reason == "time_stop"
    # Long winning trade: gross +$1 on qty=1 at 7.5 bps/side fees.
    assert out.pnl_usd == pytest.approx(1.0 - 0.075 - 0.07575)
    # No new position opened (signal was neutral).
    assert trader.state.open_position is None


def test_close_then_immediately_open_new_on_same_bar():
    """Bar T+1: close existing trade AND open a new one on the same call.
    The returned trade is the CLOSED one; new position lives in state."""
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)

    out = _bar(trader, ts=_ts(minute=1), microprice=101.0, prob_up=0.7)

    assert out is not None  # the close
    assert out.side == "long"
    # And a new position opened at the close bar.
    assert trader.state.open_position is not None
    assert trader.state.open_position.entry_price == 101.0
    # last_no_op is OK_CLOSED (the close took precedence in labeling).
    assert trader.state.last_no_op == _NoOpReason.OK_CLOSED


def test_full_round_trip_long_winning_pnl_matches_compute_pnl():
    """End-to-end P&L must equal what the pure helper computes — proves
    the state machine doesn't quietly miscount fees or drop terms."""
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)
    out = _bar(trader, ts=_ts(minute=1), microprice=110.0, prob_up=0.5)

    assert out is not None
    expected_net, expected_bps = compute_pnl(
        side="long", entry_price=100.0, exit_price=110.0,
        qty=trader.config.max_qty_usd / 100.0,
        maker_fee_bps_per_side=trader.config.maker_fee_bps_per_side,
    )
    assert out.pnl_usd == pytest.approx(expected_net)
    assert out.pnl_bps == pytest.approx(expected_bps)


# ───────────────────────── on_bar_close: skip branches ─────────────────────────


def test_skip_when_not_calibrated():
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), prob_up=0.7, calibrated=False)
    assert trader.state.open_position is None
    assert trader.state.last_no_op == _NoOpReason.SKIP_NOT_CALIBRATED


def test_skip_when_position_already_open():
    """We opened at T=0; T=0:30s (within horizon) sees position-open guard.
    Note: with horizon_minutes=1 and ts at T=0:30, elapsed=0.5 < 1 → no close,
    then guard fires."""
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)
    assert trader.state.open_position is not None

    # Half a minute later — same minute floor, but elapsed < horizon.
    out = trader.on_bar_close(
        ts=_ts(minute=0).replace(second=30),
        microprice=100.5, prob_up=0.7, calibrated=True,
    )
    assert out is None
    assert trader.state.last_no_op == _NoOpReason.SKIP_POSITION_OPEN


def test_skip_when_microprice_zero():
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=0.0, prob_up=0.7)
    assert trader.state.open_position is None
    assert trader.state.last_no_op == _NoOpReason.SKIP_BAD_PRICE


def test_skip_when_microprice_negative():
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=-1.0, prob_up=0.7)
    assert trader.state.open_position is None
    assert trader.state.last_no_op == _NoOpReason.SKIP_BAD_PRICE


def test_can_open_when_uncalibrated_if_overridden():
    """``require_calibrated=False`` knob unblocks for backtest sweeps."""
    cfg = PaperTraderConfig(require_calibrated=False)
    trader = PaperTrader("BTCUSDT", config=cfg)
    _bar(trader, ts=_ts(minute=0), prob_up=0.7, calibrated=False)
    assert trader.state.open_position is not None


# ───────────────────────── risk caps ─────────────────────────


def _drive_n_losing_longs(
    trader: PaperTrader, n: int, *, entry_minute_start: int = 0,
) -> list[PaperTrade]:
    """Open and close ``n`` losing long trades back-to-back.

    Bar cadence: bar_i opens, bar_{i+1} closes (and opens i+1, etc.).
    Prices march down 1 USD per bar so every long is a loss.

    Returns the list of closed trades for assertions.
    """
    trades: list[PaperTrade] = []
    # Bar 0 opens trade #1.
    _bar(trader, ts=_ts(minute=entry_minute_start), microprice=100.0, prob_up=0.7)
    # Subsequent bars close+open until we've closed n trades.
    for i in range(1, n + 1):
        out = _bar(
            trader,
            ts=_ts(minute=entry_minute_start + i),
            microprice=100.0 - float(i),  # 99, 98, 97, ...
            prob_up=0.7,  # keep re-opening
        )
        assert out is not None, f"bar {i} should have closed a trade"
        trades.append(out)
    return trades


def test_loss_streak_halt_engages_at_cap():
    """5 consecutive losses → halted_reason='loss_streak'."""
    # Disable daily cap so it doesn't fire first.
    trader = PaperTrader(
        "BTCUSDT",
        risk_caps=RiskCaps(max_consecutive_losses=5, max_daily_loss_usd=1e9),
    )
    trades = _drive_n_losing_longs(trader, n=5)

    # All 5 trades must be losses.
    for t in trades:
        assert t.pnl_usd < 0

    assert trader.state.consecutive_losses == 5
    assert trader.state.halted_reason == "loss_streak"


def test_halt_blocks_new_entry_but_status_reflects_it():
    """After halt, next bar with strong signal must NOT open."""
    trader = PaperTrader(
        "BTCUSDT",
        risk_caps=RiskCaps(max_consecutive_losses=5, max_daily_loss_usd=1e9),
    )
    _drive_n_losing_longs(trader, n=5)
    assert trader.state.halted_reason == "loss_streak"
    assert trader.state.open_position is None  # 5th close cleared it

    # Next bar — strong signal but we're halted.
    out = _bar(trader, ts=_ts(minute=10), microprice=100.0, prob_up=0.9)
    assert out is None
    assert trader.state.open_position is None
    assert trader.state.last_no_op == _NoOpReason.SKIP_HALTED_LOSS_STREAK


def test_daily_loss_halt_engages_when_cumulative_below_cap():
    """One large loss exceeding the daily cap → halt='daily_loss'."""
    trader = PaperTrader(
        "BTCUSDT",
        # Cap streak high (won't trigger), tight daily cap.
        risk_caps=RiskCaps(max_consecutive_losses=100, max_daily_loss_usd=1.0),
    )
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)
    out = _bar(trader, ts=_ts(minute=1), microprice=99.0, prob_up=0.5)
    assert out is not None and out.pnl_usd < -1.0

    assert trader.state.halted_reason == "daily_loss"


def test_halt_still_emits_close_for_open_position():
    """Halt engages mid-trade → close must still fire on time-stop, then
    no new entry. We must not strand a position in ``state.open_position``
    just because risk caps engaged."""
    # Pre-load the trader so the FIRST close triggers the halt.
    trader = PaperTrader(
        "BTCUSDT",
        risk_caps=RiskCaps(max_consecutive_losses=2, max_daily_loss_usd=1e9),
    )
    # Drive 2 losses to engage halt right after the 2nd close.
    trades = _drive_n_losing_longs(trader, n=2)
    assert trader.state.halted_reason == "loss_streak"
    # And critically: position cleared, not stranded.
    assert trader.state.open_position is None
    assert all(t.pnl_usd < 0 for t in trades)


def test_one_position_at_a_time_invariant():
    """No matter how many bars fire while we hold, exactly one position exists."""
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)
    # Three more bars within the same minute (sub-horizon).
    for sec in (10, 20, 40):
        trader.on_bar_close(
            ts=_ts(minute=0).replace(second=sec),
            microprice=100.5, prob_up=0.7, calibrated=True,
        )
    assert trader.state.open_position is not None
    assert trader.state.open_position.entry_price == 100.0  # first one stuck


# ───────────────────────── day rollover ─────────────────────────


def test_day_rollover_resets_daily_pnl_and_closed_today():
    """Cross UTC midnight → daily_pnl_usd, closed_today reset."""
    trader = PaperTrader("BTCUSDT")
    # Drive one trade on day 25.
    _bar(trader, ts=_ts(minute=0, hour=23, day=25), microprice=100.0, prob_up=0.7)
    out = _bar(trader, ts=_ts(minute=1, hour=23, day=25), microprice=101.0, prob_up=0.5)
    assert out is not None
    assert trader.state.daily_pnl_usd != 0
    assert len(trader.state.closed_today) == 1

    # Bar at next-day midnight → rollover.
    _bar(trader, ts=_ts(minute=0, hour=0, day=26), microprice=101.0, prob_up=0.5)
    assert trader.state.daily_pnl_usd == 0.0
    assert trader.state.closed_today == []


def test_daily_loss_halt_clears_on_day_rollover():
    """The whole point of the daily-loss cap is bounding a 24h window —
    crossing midnight must un-halt so trading can resume."""
    trader = PaperTrader(
        "BTCUSDT",
        risk_caps=RiskCaps(max_consecutive_losses=100, max_daily_loss_usd=1.0),
    )
    _bar(trader, ts=_ts(minute=0, hour=23, day=25), microprice=100.0, prob_up=0.7)
    _bar(trader, ts=_ts(minute=1, hour=23, day=25), microprice=99.0, prob_up=0.5)
    assert trader.state.halted_reason == "daily_loss"

    # Cross midnight — neutral signal so we don't immediately open.
    _bar(trader, ts=_ts(minute=0, hour=0, day=26), microprice=100.0, prob_up=0.5)
    assert trader.state.halted_reason is None


def test_loss_streak_halt_persists_across_day_rollover():
    """Loss-streak indicates regime shift — should NOT auto-clear at midnight.
    Documented carve-out from the daily-loss behaviour."""
    trader = PaperTrader(
        "BTCUSDT",
        risk_caps=RiskCaps(max_consecutive_losses=2, max_daily_loss_usd=1e9),
    )
    _drive_n_losing_longs(trader, n=2)
    assert trader.state.halted_reason == "loss_streak"

    # Cross midnight (we're at minute 2 of day 25; jump to day 26).
    _bar(trader, ts=_ts(minute=0, hour=0, day=26), microprice=100.0, prob_up=0.5)
    assert trader.state.halted_reason == "loss_streak"


def test_naive_timestamp_treated_as_utc():
    """asyncpg returns aware datetimes, but defensive code in case
    something upstream hands us naive datetimes."""
    trader = PaperTrader("BTCUSDT")
    naive_ts = datetime(2026, 4, 25, 0, 0, 0)  # no tzinfo
    # Must not raise; rollover logic should treat as UTC.
    trader.on_bar_close(
        ts=naive_ts, microprice=100.0, prob_up=0.7, calibrated=True,
    )
    assert trader.state.current_day_utc is not None
    assert trader.state.current_day_utc.tzinfo == UTC


# ───────────────────────── force_close ─────────────────────────


def test_force_close_returns_trade_when_position_open():
    """Used by runner shutdown to flatten before exit."""
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)
    assert trader.state.open_position is not None

    out = trader.force_close(ts=_ts(minute=0).replace(second=30), microprice=100.5)
    assert out is not None
    assert out.exit_reason == "halt_close"
    assert trader.state.open_position is None


def test_force_close_returns_none_when_no_position():
    trader = PaperTrader("BTCUSDT")
    out = trader.force_close(ts=_ts(minute=0), microprice=100.0)
    assert out is None


# ───────────────────────── status ─────────────────────────


def test_status_shape_no_position():
    trader = PaperTrader("BTCUSDT")
    s = trader.status()
    # JSON-serialisable.
    json.dumps(s)
    assert s["symbol"] == "BTCUSDT"
    assert s["open_position"] is None
    assert s["halted"] is False
    assert s["halted_reason"] is None
    assert s["daily_pnl_usd"] == 0.0
    assert s["consecutive_losses"] == 0
    assert s["trades_today"] == 0
    assert s["total_trades"] == 0
    assert s["last_no_op"] is None


def test_status_shape_with_open_position():
    trader = PaperTrader("BTCUSDT")
    _bar(trader, ts=_ts(minute=0), microprice=100.0, prob_up=0.7)
    s = trader.status()
    json.dumps(s)
    assert s["open_position"] is not None
    assert s["open_position"]["side"] == "long"
    assert s["open_position"]["entry_price"] == 100.0
    assert s["last_no_op"] == _NoOpReason.OK_OPENED.value


def test_status_reflects_halt():
    trader = PaperTrader(
        "BTCUSDT",
        risk_caps=RiskCaps(max_consecutive_losses=2, max_daily_loss_usd=1e9),
    )
    _drive_n_losing_longs(trader, n=2)
    s = trader.status()
    assert s["halted"] is True
    assert s["halted_reason"] == "loss_streak"
    assert s["consecutive_losses"] == 2
    assert s["trades_today"] == 2
    assert s["total_trades"] == 2


def test_total_trades_lifetime_counter_does_not_reset_on_rollover():
    """Daily counters reset at midnight; total_trades must not."""
    trader = PaperTrader("BTCUSDT")
    # Day 25: one full round trip.
    _bar(trader, ts=_ts(minute=0, hour=23, day=25), microprice=100.0, prob_up=0.7)
    _bar(trader, ts=_ts(minute=1, hour=23, day=25), microprice=101.0, prob_up=0.5)
    assert trader.state.total_trades == 1

    # Roll to day 26.
    _bar(trader, ts=_ts(minute=0, hour=0, day=26), microprice=101.0, prob_up=0.7)
    _bar(trader, ts=_ts(minute=1, hour=0, day=26), microprice=102.0, prob_up=0.5)

    assert trader.state.total_trades == 2  # not reset
    assert len(trader.state.closed_today) == 1  # but daily IS reset to 1
