"""Tests for ``app.highfreq.feature_pipeline_microstructure_v2`` (T.18.c).

The v2 pipeline extends base microstructure with 4 trade-flow rolling
features. Tests pin:

1. Output schema = 22 cols (18 base + 4 trade-flow).
2. trade_buy_ratio is signed in [-1, 1] and matches the formula.
3. trade_buy_ratio_lag1 is the per-symbol shift-1.
4. 3-bar rolling mean computed per-symbol (no cross-symbol bleed).
5. trade_imb_acceleration = current - lag1.
6. Empty input returns empty df with correct schema.
7. Live inference helper drops in-flight bar + returns 22-col row.
8. Live inference returns None when fewer than 4 bars (need lag/rolling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.highfreq.feature_pipeline_microstructure_v2 import (
    MICROSTRUCTURE_V2_FEATURE_COLUMNS,
    TRADE_FLOW_FEATURE_COLUMNS,
    build_latest_inference_bar_microstructure_v2,
    build_microstructure_v2_features,
)


def _make_minute_df(n: int = 5, *, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Build a synthetic minute-bar frame matching what
    ``aggregate_to_minute`` produces — enough columns for both the
    base v1 build_features and our v2 extension."""
    base_ts = pd.Timestamp("2026-05-03T10:00:00", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({
            "symbol": symbol,
            "minute": base_ts + pd.Timedelta(minutes=i),
            "ofi_sum": rng.normal(0, 1),
            "ofi_mean": rng.normal(0, 0.1),
            "ofi_std": abs(rng.normal(0.5, 0.2)),
            "microprice_open": 50000.0 + i * 10,
            "microprice_close": 50010.0 + i * 10,
            "microprice_high": 50020.0 + i * 10,
            "microprice_low": 50000.0 + i * 10,
            "depth_imb_mean": rng.normal(0, 0.1),
            "depth_imb_std": abs(rng.normal(0.1, 0.05)),
            "depth_imb_last": rng.normal(0, 0.1),
            "spread_bps_mean": abs(rng.normal(0.05, 0.02)),
            "spread_bps_max": abs(rng.normal(0.10, 0.03)),
            "spread_bps_last": abs(rng.normal(0.05, 0.02)),
            # Drive the trade-flow features deterministically:
            #   trade_imb_sum varies per row; abs is consistent.
            "trade_imb_sum": float(i - 2),       # -2, -1, 0, 1, 2
            "trade_imb_abs_sum": 4.0,            # constant denominator
            "n_updates_sum": 100 + i,
        })
    return pd.DataFrame(rows)


def test_v2_schema_is_22_cols():
    """4 new features on top of base 18 = 22 canonical cols."""
    assert len(MICROSTRUCTURE_V2_FEATURE_COLUMNS) == 22
    # The 4 new ones are the suffix.
    assert MICROSTRUCTURE_V2_FEATURE_COLUMNS[-4:] == TRADE_FLOW_FEATURE_COLUMNS


def test_v2_output_columns_match_canonical():
    df = _make_minute_df(n=5)
    out = build_microstructure_v2_features(df)
    assert list(out.columns) == MICROSTRUCTURE_V2_FEATURE_COLUMNS


def test_trade_buy_ratio_formula():
    """trade_buy_ratio = trade_imb_sum / max(trade_imb_abs_sum, eps)
    clipped to [-1, 1]. Synthetic data: trade_imb_sum ∈ {-2,-1,0,1,2},
    abs=4.0 → ratios ∈ {-0.5, -0.25, 0, 0.25, 0.5}."""
    df = _make_minute_df(n=5)
    out = build_microstructure_v2_features(df)
    expected = [-0.5, -0.25, 0.0, 0.25, 0.5]
    np.testing.assert_allclose(
        out["trade_buy_ratio"].to_numpy(), expected, atol=1e-9,
    )


def test_trade_buy_ratio_clipped_to_unit_range():
    """If trade_imb_sum > trade_imb_abs_sum (shouldn't happen but
    defensive guard), the ratio must clip to ±1.0."""
    df = _make_minute_df(n=3)
    df.loc[0, "trade_imb_sum"] = 100.0
    df.loc[0, "trade_imb_abs_sum"] = 1.0  # contrived
    out = build_microstructure_v2_features(df)
    assert out.iloc[0]["trade_buy_ratio"] == 1.0


def test_lag1_ratio_correct():
    """lag1 of trade_buy_ratio: row i should equal trade_buy_ratio
    of row i-1. Row 0's lag1 is fillna 0.0 (no prior row)."""
    df = _make_minute_df(n=5)
    out = build_microstructure_v2_features(df)
    # Ratios from previous test: -0.5, -0.25, 0, 0.25, 0.5
    expected_lag1 = [0.0, -0.5, -0.25, 0.0, 0.25]
    np.testing.assert_allclose(
        out["trade_buy_ratio_lag1"].to_numpy(), expected_lag1, atol=1e-9,
    )


def test_3bar_rolling_avg_correct():
    """3-bar rolling mean of trade_buy_ratio, min_periods=1.
    Series: -0.5, -0.25, 0, 0.25, 0.5
    Rolling: -0.5, -0.375, -0.25, 0.0, 0.25
    """
    df = _make_minute_df(n=5)
    out = build_microstructure_v2_features(df)
    expected = [-0.5, -0.375, -0.25, 0.0, 0.25]
    np.testing.assert_allclose(
        out["trade_buy_ratio_3bar_avg"].to_numpy(), expected, atol=1e-9,
    )


def test_acceleration_is_current_minus_lag1():
    """trade_imb_sum: -2, -1, 0, 1, 2 → diff: 0 (lag1=0), 1, 1, 1, 1."""
    df = _make_minute_df(n=5)
    out = build_microstructure_v2_features(df)
    # Row 0: current - 0 (no lag) = -2 - 0 = -2.
    # Rows 1-4: current - lag1 = always 1 here.
    expected = [-2.0, 1.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(
        out["trade_imb_acceleration"].to_numpy(), expected, atol=1e-9,
    )


def test_per_symbol_lag_isolation():
    """Multi-symbol frame: BTC's lag1 must NOT bleed into ETH's row.
    Build interleaved BTC + ETH bars, verify each symbol's lag1
    is its own previous bar."""
    btc = _make_minute_df(n=3, symbol="BTC")
    eth = _make_minute_df(n=3, symbol="ETH")
    df = pd.concat([btc, eth], ignore_index=True)
    out = build_microstructure_v2_features(df)
    # Sort by symbol for deterministic indexing.
    df_indexed = df.copy()
    df_indexed["_idx"] = range(len(df_indexed))
    out["_orig_idx"] = df_indexed.sort_values(
        ["symbol", "minute"]
    ).reset_index(drop=True)["_idx"].values
    # For both symbols, lag1 of row 0 must be 0 (no prior bar of same
    # symbol). It must NOT be the LAST BTC bar's value, even though
    # BTC and ETH happen to be adjacent in the input frame.
    eth_first_lag1 = out[out["_orig_idx"].isin([3])]["trade_buy_ratio_lag1"].iloc[0]
    assert eth_first_lag1 == 0.0


def test_empty_input_returns_empty_with_22_cols():
    df = pd.DataFrame(columns=[
        "symbol", "minute",
        "ofi_sum", "ofi_mean", "ofi_std",
        "microprice_open", "microprice_close",
        "microprice_high", "microprice_low",
        "depth_imb_mean", "depth_imb_std", "depth_imb_last",
        "spread_bps_mean", "spread_bps_max", "spread_bps_last",
        "trade_imb_sum", "trade_imb_abs_sum",
        "n_updates_sum",
    ])
    out = build_microstructure_v2_features(df)
    assert out.empty
    assert list(out.columns) == MICROSTRUCTURE_V2_FEATURE_COLUMNS


# ───────────────── live inference helper ─────────────────


def _make_seconds_df(n_seconds: int) -> pd.DataFrame:
    """Build a per-second frame matching what
    ``feature_pipeline.aggregate_to_minute`` consumes."""
    base_ts = pd.Timestamp("2026-05-03T10:00:00", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_seconds):
        rows.append({
            "ts": base_ts + pd.Timedelta(seconds=i),
            "symbol": "BTCUSDT",
            "ofi": rng.normal(0, 0.1),
            "microprice": 50000.0 + i * 0.1,
            "depth_imb": rng.normal(0, 0.1),
            "spread_bps": abs(rng.normal(0.05, 0.02)),
            "trade_imb": rng.normal(0, 0.5),
            "n_updates": int(rng.integers(5, 20)),
        })
    return pd.DataFrame(rows)


def test_inference_bar_returns_22col_row():
    """5+ minutes of seconds → drop in-flight current → ≥4 complete
    bars → 22-col feature row + close microprice."""
    df_secs = _make_seconds_df(n_seconds=5 * 60 + 30)  # 5.5 minutes
    out = build_latest_inference_bar_microstructure_v2(df_secs)
    assert out is not None
    feats, close = out
    assert len(feats) == 22
    assert list(feats.index) == MICROSTRUCTURE_V2_FEATURE_COLUMNS
    assert close > 0


def test_inference_bar_returns_none_with_fewer_than_4_bars():
    """3 complete minutes → after in-flight drop, only 2-3 bars →
    can't compute lag1 + 3-bar rolling robustly → None."""
    df_secs = _make_seconds_df(n_seconds=3 * 60 + 30)
    out = build_latest_inference_bar_microstructure_v2(df_secs)
    assert out is None


def test_inference_bar_returns_none_on_empty():
    out = build_latest_inference_bar_microstructure_v2(pd.DataFrame())
    assert out is None
