"""Pure-function tests for app.highfreq.backtest.

The backtest layer takes a predictions DataFrame and produces a P&L
report — it has no I/O, no external deps beyond pandas/numpy. We test
it on hand-crafted inputs with closed-form expected outputs so any
regression (e.g. a wrong sign on the maker rebate) fails loudly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.highfreq.backtest import (
    FeeModel,
    MINUTES_PER_YEAR,
    StrategyConfig,
    build_trade_ledger,
    fill_rate_sweep,
    max_drawdown_pct,
    run_backtest,
    simulate,
    summarise,
)


# ─────────────────────────── helpers ───────────────────────────

def _predictions(
    probas: list[float],
    returns_bps: list[float],
    start: str = "2026-04-25 00:00:00",
) -> pd.DataFrame:
    """Synthetic predictions frame with controllable signal/outcome."""
    minutes = pd.date_range(start, periods=len(probas), freq="1min", tz="UTC")
    return pd.DataFrame({
        "minute": minutes,
        "y_pred": [int(p >= 0.5) for p in probas],
        "proba": probas,
        "return_bps": returns_bps,
    })


# ───────────────────────── fee model ─────────────────────────

def test_fee_model_per_side_bps_endpoints():
    fm = FeeModel()  # default: maker -1, taker +10
    assert fm.per_side_bps(0.0) == 10.0      # pure taker
    assert fm.per_side_bps(1.0) == -1.0      # pure maker
    # Linear interpolation at 50 %.
    assert fm.per_side_bps(0.5) == pytest.approx(4.5)


def test_fee_model_roundtrip_is_2x_per_side():
    fm = FeeModel()
    for f in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert fm.roundtrip_bps(f) == pytest.approx(2.0 * fm.per_side_bps(f))


def test_fee_model_clamps_out_of_range_fill_rate():
    fm = FeeModel()
    # Out-of-range fill rates must clamp, not extrapolate into rebate land.
    assert fm.per_side_bps(-0.5) == 10.0
    assert fm.per_side_bps(1.5) == -1.0


# ───────────────────────── trade ledger ─────────────────────────

def test_build_trade_ledger_long_signal_correct_pnl():
    """proba=0.9 → long; +5 bp gross return; 50 % fill → ~+0.5 bp net."""
    preds = _predictions([0.9], [5.0])
    cfg = StrategyConfig(notional_per_trade=10_000.0)
    ledger = build_trade_ledger(preds, strategy=cfg, maker_fill_rate=0.5)
    assert len(ledger) == 1
    assert ledger.loc[0, "direction"] == 1
    assert ledger.loc[0, "gross_bps"] == 5.0
    # roundtrip @ f=0.5: 2 * (0.5*-1 + 0.5*10) = 9 bps
    assert ledger.loc[0, "fee_bps"] == pytest.approx(9.0)
    assert ledger.loc[0, "net_bps"] == pytest.approx(-4.0)
    assert ledger.loc[0, "net_dollars"] == pytest.approx(-4.0)  # -4 bp on $10k
    assert not ledger.loc[0, "is_win"]


def test_build_trade_ledger_short_signal_inverts_return():
    """proba=0.1 → short; market goes -7 bp; short profits +7 bp gross."""
    preds = _predictions([0.1], [-7.0])
    cfg = StrategyConfig()
    ledger = build_trade_ledger(preds, strategy=cfg, maker_fill_rate=1.0)
    assert ledger.loc[0, "direction"] == -1
    assert ledger.loc[0, "gross_bps"] == 7.0
    # 100 % maker → roundtrip = -2 bp (you EARN fees)
    assert ledger.loc[0, "fee_bps"] == pytest.approx(-2.0)
    assert ledger.loc[0, "net_bps"] == pytest.approx(9.0)


def test_build_trade_ledger_threshold_filters_low_conviction():
    probas = [0.55, 0.50, 0.45, 0.30, 0.70]
    preds = _predictions(probas, [10.0, 10.0, 10.0, 10.0, 10.0])
    cfg = StrategyConfig(confidence_threshold=0.6)
    ledger = build_trade_ledger(preds, strategy=cfg, maker_fill_rate=0.0)
    # threshold=0.6 → only proba ≥ 0.6 (long) or ≤ 0.4 (short) trade.
    # 0.55 → flat, 0.50 → flat, 0.45 → flat, 0.30 → short, 0.70 → long.
    assert len(ledger) == 2
    assert set(ledger["direction"]) == {-1, 1}


def test_build_trade_ledger_preserves_chronological_order():
    # Use big enough gross returns to dominate fees so equity is positive,
    # and ALSO assert the deeper invariant: equity == cumsum(net_dollars).
    preds = _predictions([0.9, 0.9, 0.9], [50.0, 60.0, 70.0])
    # Shuffle input on purpose.
    preds = preds.iloc[[2, 0, 1]].reset_index(drop=True)
    ledger = build_trade_ledger(
        preds, strategy=StrategyConfig(), maker_fill_rate=0.5,
    )
    minutes = ledger["minute"].astype("int64").to_numpy()
    assert (minutes == np.sort(minutes)).all()
    # Equity curve cumulates in chronological order — the cumulative sum
    # invariant holds for ANY signs of returns.
    expected_equity = ledger["net_dollars"].cumsum().to_numpy()
    actual_equity = ledger["equity_dollars"].to_numpy()
    assert np.allclose(actual_equity, expected_equity)


def test_build_trade_ledger_empty_when_all_flat():
    preds = _predictions([0.5, 0.5, 0.5], [1.0, 2.0, 3.0])
    cfg = StrategyConfig(confidence_threshold=0.6)
    ledger = build_trade_ledger(preds, strategy=cfg, maker_fill_rate=0.5)
    assert ledger.empty
    expected_columns = {
        "minute", "direction", "gross_bps", "fee_bps", "net_bps",
        "net_dollars", "is_win", "equity_dollars",
    }
    assert expected_columns.issubset(ledger.columns)


def test_build_trade_ledger_raises_on_missing_columns():
    bad = pd.DataFrame({"minute": [0], "proba": [0.9]})  # no return_bps
    with pytest.raises(ValueError, match="return_bps"):
        build_trade_ledger(
            bad, strategy=StrategyConfig(), maker_fill_rate=0.5,
        )


# ───────────────────────── max drawdown ─────────────────────────

def test_max_drawdown_monotonic_curve_is_zero():
    eq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert max_drawdown_pct(eq) == 0.0


def test_max_drawdown_known_drop():
    # Peak at 100, trough at 75 → 25 % drawdown.
    eq = np.array([100.0, 100.0, 75.0, 80.0])
    assert max_drawdown_pct(eq) == pytest.approx(-0.25)


def test_max_drawdown_empty_is_zero():
    assert max_drawdown_pct(np.array([])) == 0.0


# ───────────────────────── summarise ─────────────────────────

def test_summarise_known_winrate_and_pnl():
    """3 wins of +10 bp, 2 losses of -5 bp at 100 % maker (rebate = -2 bp rt).

    Gross: 3*10 + 2*(-5) = +20 bp. Fees: 5 trades * -2 bp = -10 bp (rebate).
    Net: 20 - (-10) = 30 bp. Net per trade: 6 bp ⇒ wins are +12 bp net,
    losses are -3 bp net. Win-rate stays 3/5 because net of a -5 bp gross
    + 2 bp rebate = -3 bp ⇒ still a loss.
    """
    probas = [0.9] * 5
    rets = [10.0, 10.0, 10.0, -5.0, -5.0]
    preds = _predictions(probas, rets)
    cfg = StrategyConfig()
    ledger = build_trade_ledger(preds, strategy=cfg, maker_fill_rate=1.0)
    rep = summarise(
        ledger, strategy=cfg, maker_fill_rate=1.0, n_signals=len(preds),
    )
    assert rep.n_trades == 5
    assert rep.n_wins == 3 and rep.n_losses == 2
    assert rep.win_rate == pytest.approx(0.6)
    assert rep.gross_pnl_bps == pytest.approx(20.0)
    assert rep.fees_bps == pytest.approx(-10.0)  # 5 × roundtrip -2
    assert rep.net_pnl_bps == pytest.approx(30.0)
    assert rep.net_pnl_dollars == pytest.approx(30.0)  # 30 bp × $10k


def test_summarise_handles_empty_ledger():
    rep = summarise(
        pd.DataFrame(columns=[
            "minute", "direction", "gross_bps", "fee_bps", "net_bps",
            "net_dollars", "is_win", "equity_dollars", "proba",
        ]),
        strategy=StrategyConfig(),
        maker_fill_rate=0.5,
        n_signals=10,
    )
    assert rep.n_trades == 0
    assert rep.n_signals == 10
    assert math.isnan(rep.win_rate)
    assert rep.net_pnl_dollars == 0.0


def test_summarise_sharpe_annualisation():
    """Per-minute Sharpe × √(MINUTES_PER_YEAR) = annualised Sharpe."""
    # 100 trades all returning exactly +5 bp net
    # → mean=5, std=0 → Sharpe undefined.  Add some noise.
    rng = np.random.default_rng(0)
    rets = list(rng.normal(loc=5.0, scale=10.0, size=100))
    preds = _predictions([0.9] * 100, rets)
    cfg = StrategyConfig()
    ledger = build_trade_ledger(preds, strategy=cfg, maker_fill_rate=0.0)
    rep = summarise(
        ledger, strategy=cfg, maker_fill_rate=0.0, n_signals=100,
    )
    expected_ratio = math.sqrt(MINUTES_PER_YEAR)
    assert rep.sharpe_annualised == pytest.approx(
        rep.sharpe_per_minute * expected_ratio, rel=1e-9
    )


# ───────────────────────── simulate / sweep ─────────────────────────

def test_simulate_perfectly_predictive_strategy_is_profitable_at_taker():
    """If the model is right every time, even a pure-taker strategy wins."""
    rng = np.random.default_rng(42)
    # All long, all positive returns of ~50 bp (well above 20 bp roundtrip).
    rets = list(rng.uniform(40.0, 60.0, size=50))
    preds = _predictions([0.9] * 50, rets)
    rep, ledger = simulate(
        preds, strategy=StrategyConfig(), maker_fill_rate=0.0,
    )
    assert rep.win_rate == 1.0
    assert rep.net_pnl_bps > 0
    # Per-trade ~50 - 20 = 30 bp net.
    assert ledger["net_bps"].mean() == pytest.approx(
        np.mean(rets) - 20.0, rel=1e-6
    )


def test_simulate_perfectly_anti_predictive_loses_at_taker():
    """Perfectly wrong predictions guarantee a loss after fees."""
    # Predict long every time, but return is always -10 bp.
    preds = _predictions([0.9] * 30, [-10.0] * 30)
    rep, _ = simulate(
        preds, strategy=StrategyConfig(), maker_fill_rate=0.0,
    )
    # Net = 30 trades × (-10 - 20) bp = -900 bp on $10k notional.
    assert rep.net_pnl_dollars == pytest.approx(-900.0)
    assert rep.win_rate == 0.0


def test_fill_rate_sweep_produces_one_row_per_rate():
    preds = _predictions([0.9, 0.1] * 10, [5.0, -5.0] * 10)
    sweep = fill_rate_sweep(
        preds, strategy=StrategyConfig(),
        fill_rates=(0.0, 0.5, 1.0),
    )
    assert len(sweep) == 3
    assert list(sweep["maker_fill_rate"]) == [0.0, 0.5, 1.0]
    # Higher maker fill rate → cheaper fees → higher net P&L (monotone).
    assert sweep["net_pnl_bps"].is_monotonic_increasing


def test_run_backtest_emits_taker_and_maker_side_by_side():
    """ADR-005: combined report must always show both economic regimes."""
    preds = _predictions([0.9] * 20, [10.0] * 20)
    combined = run_backtest(preds, strategy=StrategyConfig())
    assert combined.taker["maker_fill_rate"] == 0.0
    assert combined.maker_at_50pct["maker_fill_rate"] == 0.5
    assert combined.maker_at_100pct["maker_fill_rate"] == 1.0
    # All trade the same signals.
    assert (
        combined.taker["n_trades"]
        == combined.maker_at_50pct["n_trades"]
        == combined.maker_at_100pct["n_trades"]
        == 20
    )
    # Maker is strictly better than taker at the same gross.
    assert (
        combined.maker_at_100pct["net_pnl_bps"]
        > combined.maker_at_50pct["net_pnl_bps"]
        > combined.taker["net_pnl_bps"]
    )


def test_combined_report_to_json_is_rfc7159():
    preds = _predictions([0.5] * 5, [0.0] * 5)  # all flat → empty ledger
    combined = run_backtest(preds, strategy=StrategyConfig(confidence_threshold=0.6))
    json_str = combined.to_json()
    # Must be parseable by a strict (non-NaN-tolerant) JSON parser.
    import json
    parsed = json.loads(json_str)
    # NaN values (empty win_rate etc.) become null, not the string "NaN".
    assert parsed["taker"]["win_rate"] is None
