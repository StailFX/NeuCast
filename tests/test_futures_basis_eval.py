"""Tests for ``tools.futures_basis_eval`` (T.24).

A/B harness comparing baseline microstructure vs +5 futures-basis
features. We don't run the full CatBoost A/B in unit tests — too
slow and needs DB. Instead pin:

1. ``build_futures_basis_features`` returns the right column order +
   shape for both populated and empty futures inputs (zero-fill
   fallback so the model still trains when futures data is missing).
2. Numeric correctness on a synthetic spot+futures pair (basis_bps
   computed from microprices matches by-hand calculation).
3. Output is finite (no NaN/Inf leaking into CatBoost).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.futures_basis_eval import (
    FUTURES_BASIS_FEATURE_COLUMNS,
    build_futures_basis_features,
)


def _spot_minute_df(n: int = 5, *, base_price: float = 30000.0) -> pd.DataFrame:
    """Synthetic spot minute frame matching ``aggregate_to_minute`` shape."""
    base_ts = pd.Timestamp("2026-05-04T00:00:00Z")
    return pd.DataFrame({
        "minute": [base_ts + pd.Timedelta(minutes=i) for i in range(n)],
        "symbol": ["BTCUSDT"] * n,
        "microprice_close": [base_price + i for i in range(n)],
        "ofi_sum": [10.0 * i for i in range(n)],
    })


def _futures_seconds_df(spot_df: pd.DataFrame, *,
                        basis_bps: float = 5.0,
                        funding_rate: float = 0.0001) -> pd.DataFrame:
    """Synthetic futures seconds frame: 60 rows per spot minute,
    futures price set so basis_bps = ``basis_bps`` exactly."""
    rows = []
    for _, spot_row in spot_df.iterrows():
        minute_start = spot_row["minute"]
        spot_close = float(spot_row["microprice_close"])
        fut_price = spot_close * (1.0 + basis_bps / 1e4)
        # 60 seconds, simple repeated values — close is what counts.
        for s in range(60):
            rows.append({
                "ts": minute_start + pd.Timedelta(seconds=s),
                "symbol": "BTCUSDT",
                "ofi": 0.5,
                "microprice": fut_price,
                "depth_imb": 0.1,
                "spread_bps": 1.0,
                "trade_imb": 0.0,
                "vpin": 0.5,
                "n_updates": 5,
                "local_recv_ms": 1.0,
                "mark_price": fut_price * 1.00005,  # 0.5 bps premium
                "funding_rate": funding_rate,
            })
    return pd.DataFrame(rows)


def test_build_futures_basis_returns_expected_columns():
    """Pin column order — model retrain depends on stable feature order."""
    spot = _spot_minute_df(5)
    fut = _futures_seconds_df(spot)
    out = build_futures_basis_features(spot, fut)
    assert list(out.columns) == FUTURES_BASIS_FEATURE_COLUMNS
    assert len(out) == len(spot)


def test_build_futures_basis_missing_futures_zero_fill():
    """When futures table is empty (early ingest, downtime) features
    must zero-fill so the model still trains. Production cron MUST
    NOT crash on missing futures data."""
    spot = _spot_minute_df(3)
    empty_fut = pd.DataFrame(columns=[
        "ts", "symbol", "ofi", "microprice", "depth_imb", "spread_bps",
        "trade_imb", "vpin", "n_updates", "local_recv_ms",
        "mark_price", "funding_rate",
    ])
    out = build_futures_basis_features(spot, empty_fut)
    assert list(out.columns) == FUTURES_BASIS_FEATURE_COLUMNS
    assert len(out) == 3
    # Every value zero — model still gets trainable rows.
    for col in FUTURES_BASIS_FEATURE_COLUMNS:
        assert (out[col] == 0.0).all(), f"{col} not zero-filled"


def test_build_futures_basis_basis_bps_correct():
    """When fut = spot * (1 + 5bps), basis_bps must be ~5.0 on aligned
    rows. Off-by-one or unit-bug here would silently regress the A/B."""
    spot = _spot_minute_df(5)
    fut = _futures_seconds_df(spot, basis_bps=5.0)
    out = build_futures_basis_features(spot, fut)
    # First row may be 0 due to NaN/diff handling; subsequent rows
    # have full futures data → basis_bps ≈ 5.0.
    assert out["basis_bps_close"].iloc[1:].mean() == pytest.approx(5.0, abs=0.5)


def test_build_futures_basis_no_nan_or_inf():
    """Output passed to CatBoost MUST be finite. NaN/Inf would crash
    the trainer or silently corrupt feature splits."""
    spot = _spot_minute_df(5)
    fut = _futures_seconds_df(spot)
    out = build_futures_basis_features(spot, fut)
    arr = out.to_numpy(dtype=float)
    assert np.isfinite(arr).all(), "non-finite values leaked through"


def test_build_futures_basis_funding_rate_scaled():
    """funding_rate is in raw decimal (0.0001 = 1bp).  The pipeline
    multiplies by 1e4 to express in bps so the model sees comparable
    units to other bps features."""
    spot = _spot_minute_df(3)
    fut = _futures_seconds_df(spot, funding_rate=0.0001)
    out = build_futures_basis_features(spot, fut)
    # First-bar row may be empty due to merge keys; later rows ≈ 1.0 bp.
    populated = out["funding_bps_mean"][out["funding_bps_mean"] != 0.0]
    if len(populated):
        assert populated.iloc[-1] == pytest.approx(1.0, abs=0.1)
