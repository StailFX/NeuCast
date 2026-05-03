"""Tests for ``app.highfreq.feature_pipeline_microstructure_v3`` (T.23).

The v3 pipeline = 18 base microstructure cols + 5 futures-basis cols.
We pin:

* Column order is stable (saved-model dispatch reads positionally).
* Cold-start / missing futures data zero-fills the 5 cols cleanly.
* No NaN/Inf leaks into CatBoost from any path.
* The synthetic-fit harness round-trips through the full builder
  without raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.highfreq.feature_pipeline_microstructure_v3 import (
    FUTURES_BASIS_FEATURE_COLUMNS,
    build_futures_basis_block,
    microstructure_v3_feature_columns,
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


def _fut_seconds(spot_df: pd.DataFrame, *,
                 basis_bps: float = 5.0,
                 funding_rate: float = 0.0001) -> pd.DataFrame:
    rows = []
    for _, sr in spot_df.iterrows():
        spot_close = float(sr["microprice_close"])
        fut_price = spot_close * (1.0 + basis_bps / 1e4)
        for s in range(60):
            rows.append({
                "ts": sr["minute"] + pd.Timedelta(seconds=s),
                "symbol": "BTCUSDT",
                "ofi": 0.5,
                "microprice": fut_price,
                "depth_imb": 0.1,
                "spread_bps": 1.0,
                "trade_imb": 0.0,
                "vpin": 0.5,
                "n_updates": 5,
                "local_recv_ms": 1.0,
                "mark_price": fut_price * 1.00005,
                "funding_rate": funding_rate,
            })
    return pd.DataFrame(rows)


# ─────────────────── column contract ───────────────────


def test_v3_feature_columns_is_23():
    """18 base + 5 futures = 23. Pin the count so anyone editing the
    base feature pipeline can't silently shift it."""
    cols = microstructure_v3_feature_columns()
    assert len(cols) == 23


def test_v3_feature_columns_has_all_5_futures_at_end():
    """Order: base first, futures last. Saved CatBoost models depend
    on positional column matching."""
    cols = microstructure_v3_feature_columns()
    assert cols[-5:] == FUTURES_BASIS_FEATURE_COLUMNS


# ─────────────────── futures block ───────────────────


def test_futures_block_returns_expected_columns():
    spot = _spot_minute_df(5)
    fut = _fut_seconds(spot)
    out = build_futures_basis_block(spot, fut)
    assert list(out.columns) == FUTURES_BASIS_FEATURE_COLUMNS
    assert len(out) == 5


def test_futures_block_zero_fills_when_futures_missing():
    """Cold start / ingest gap → must NOT crash. v3 zero-fills the
    5 cols and the model still trains; CatBoost handles a zero-input
    feature gracefully (it just becomes uninformative)."""
    spot = _spot_minute_df(3)
    out = build_futures_basis_block(spot, None)
    assert (out == 0.0).all().all()
    assert list(out.columns) == FUTURES_BASIS_FEATURE_COLUMNS


def test_futures_block_zero_fills_on_empty_df():
    """Edge case — caller passed an empty (but not None) DataFrame."""
    spot = _spot_minute_df(3)
    empty = pd.DataFrame(columns=[
        "ts", "symbol", "ofi", "microprice", "depth_imb", "spread_bps",
        "trade_imb", "vpin", "n_updates", "local_recv_ms",
        "mark_price", "funding_rate",
    ])
    out = build_futures_basis_block(spot, empty)
    assert (out == 0.0).all().all()


def test_futures_block_basis_bps_correct():
    """fut = spot * (1 + 5bps) → basis_bps_close ≈ 5.0 on populated rows.
    Off-by-one or unit bug here would silently regress the +20pp lift."""
    spot = _spot_minute_df(5)
    fut = _fut_seconds(spot, basis_bps=5.0)
    out = build_futures_basis_block(spot, fut)
    assert out["basis_bps_close"].iloc[1:].mean() == pytest.approx(5.0, abs=0.5)


def test_futures_block_no_nan_or_inf():
    """CatBoost-trainable: NaN/Inf would crash the trainer or
    silently corrupt feature splits."""
    spot = _spot_minute_df(5)
    fut = _fut_seconds(spot)
    arr = build_futures_basis_block(spot, fut).to_numpy(dtype=float)
    assert np.isfinite(arr).all()


def test_futures_block_handles_zero_size_spot():
    """No spot bars (rare) → return empty 5-col frame, not crash."""
    spot = _spot_minute_df(0)
    out = build_futures_basis_block(spot, None)
    assert len(out) == 0
    assert list(out.columns) == FUTURES_BASIS_FEATURE_COLUMNS
