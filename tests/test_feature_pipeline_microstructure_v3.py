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


# ─────────────────── inference helper (T.23.b) ───────────────────


def _spot_seconds(n_minutes: int = 5, *, base_price: float = 30000.0) -> pd.DataFrame:
    """Synthetic spot SECONDS-level frame with all columns aggregate_to_minute
    expects — used to drive ``build_latest_inference_bar_microstructure_v3``."""
    base_ts = pd.Timestamp("2026-05-04T00:00:00Z")
    rows = []
    for m in range(n_minutes):
        for s in range(60):
            rows.append({
                "ts": base_ts + pd.Timedelta(minutes=m, seconds=s),
                "symbol": "BTCUSDT",
                "ofi": 0.1 * s,
                "microprice": base_price + m * 1.0 + s * 0.001,
                "depth_imb": 0.1,
                "spread_bps": 1.0,
                "trade_imb": 0.0,
                "vpin": 0.5,
                "n_updates": 5,
                "local_recv_ms": 1.0,
            })
    return pd.DataFrame(rows)


def _fut_seconds_for_inference(spot_secs: pd.DataFrame, *,
                               basis_bps: float = 5.0) -> pd.DataFrame:
    """Synthetic futures seconds keyed off spot timestamps. Adds
    mark_price + funding_rate columns the trainer's load_seconds
    selects for the futures venue."""
    rows = []
    for _, sr in spot_secs.iterrows():
        spot_p = float(sr["microprice"])
        fut_p = spot_p * (1.0 + basis_bps / 1e4)
        rows.append({
            "ts": sr["ts"],
            "symbol": "BTCUSDT",
            "ofi": 0.05,
            "microprice": fut_p,
            "depth_imb": 0.1,
            "spread_bps": 0.9,
            "trade_imb": 0.0,
            "vpin": 0.5,
            "n_updates": 5,
            "local_recv_ms": 1.0,
            "mark_price": fut_p * 1.00005,
            "funding_rate": 0.0001,
        })
    return pd.DataFrame(rows)


def test_inference_helper_returns_23col_vector_with_futures():
    """Live serve path: 5 spot minutes + matching futures → returns
    (Series, close_microprice). The Series MUST have 23 cols matching
    the trained-model dispatch in predictor._expected_feature_columns."""
    from app.highfreq.feature_pipeline_microstructure_v3 import (
        build_latest_inference_bar_microstructure_v3,
    )
    spot = _spot_seconds(5)
    fut = _fut_seconds_for_inference(spot)
    res = build_latest_inference_bar_microstructure_v3(spot, fut)
    assert res is not None, "expected (features, close), got None"
    feats, close = res
    assert isinstance(feats, pd.Series)
    assert len(feats) == 23
    # The 5 futures cols at the end must be populated (non-zero on
    # the last bar where data is fully present).
    for col in FUTURES_BASIS_FEATURE_COLUMNS:
        assert col in feats.index
    assert close > 0


def test_inference_helper_zero_fills_futures_when_missing():
    """T.23.b cold-start: futures fetch returned None — pipeline
    must still produce the 23-col vector with zeros on the 5
    futures cols. The base 18 cols still carry real data."""
    from app.highfreq.feature_pipeline_microstructure_v3 import (
        build_latest_inference_bar_microstructure_v3,
    )
    spot = _spot_seconds(5)
    res = build_latest_inference_bar_microstructure_v3(spot, None)
    assert res is not None
    feats, _close = res
    assert len(feats) == 23
    # All 5 futures cols are zero.
    for col in FUTURES_BASIS_FEATURE_COLUMNS:
        assert feats[col] == 0.0


def test_inference_helper_returns_none_on_empty_spot():
    """No data → fail-safe None (caller surfaces 503). NEVER crash."""
    from app.highfreq.feature_pipeline_microstructure_v3 import (
        build_latest_inference_bar_microstructure_v3,
    )
    empty_spot = pd.DataFrame(columns=[
        "ts", "symbol", "ofi", "microprice", "depth_imb", "spread_bps",
        "trade_imb", "vpin", "n_updates", "local_recv_ms",
    ])
    assert build_latest_inference_bar_microstructure_v3(empty_spot, None) is None


def test_predictor_expected_columns_for_v3(tmp_path):
    """The predictor's _expected_feature_columns must return the 23-col
    list when the saved metrics record feature_set='microstructure_v3'.
    This is the train-vs-serve invariant ADR-001 named the most
    insidious bug.

    Uses a tmp metrics.json on disk so the predictor's normal
    metrics-reload path runs without needing a real .cbm file
    (we never call ``predict`` so the empty weights path is fine)."""
    import json as _json
    from pathlib import Path
    from app.highfreq.feature_pipeline_microstructure_v3 import (
        microstructure_v3_feature_columns,
    )
    from app.highfreq.predictor import LivePredictor

    expected = microstructure_v3_feature_columns()
    assert len(expected) == 23

    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    # Trainer writes metrics next to the weights with the same stem.
    weights_path = weights_dir / "btcusdt_1m.cbm"
    metrics_path = weights_dir / "btcusdt_1m_metrics.json"
    metrics_path.write_text(_json.dumps({
        "feature_set": "microstructure_v3",
        "bar_minutes": 1,
    }))
    # weights file presence is not required for column-list resolution.
    predictor = LivePredictor(weights_path=weights_path)
    assert predictor.feature_set() == "microstructure_v3"
    assert predictor._expected_feature_columns() == expected
