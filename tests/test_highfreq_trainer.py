"""Pure-function tests for app.highfreq.trainer.

Run with::

    pytest tests/test_highfreq_trainer.py -v

These tests exercise the data-transform layer with hand-crafted
synthetic input so we don't depend on a live Postgres or Binance feed.
The DB layer (``load_seconds``) and the CatBoost training loop
(``walk_forward_evaluate``, ``fit_final_model``) are integration paths
covered separately by the smoke run on the VPS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.highfreq.trainer import (
    FEATURE_COLUMNS,
    MIN_SECONDS_PER_MINUTE,
    aggregate_to_minute,
    bootstrap_dir_acc_ci,
    build_features,
    build_target,
    make_supervised,
)


# ───────────────────────────── fixtures ─────────────────────────────

def _seconds_frame(
    n_minutes: int = 5,
    seconds_per_minute: int = 60,
    base_price: float = 77_000.0,
    drift_per_min_bps: float = 2.0,
    symbol: str = "BTCUSDT",
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic per-second frame with known dynamics."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-04-25 00:00:00", tz="UTC")
    for m in range(n_minutes):
        # Microprice walks up by `drift_per_min_bps` bp per minute, plus noise.
        mid_minute = base_price * (1.0 + (m * drift_per_min_bps) / 1e4)
        for s in range(seconds_per_minute):
            ts = start + pd.Timedelta(minutes=m, seconds=s)
            tick = mid_minute + rng.normal(0.0, 1.0)  # $1 noise
            rows.append({
                "ts": ts,
                "symbol": symbol,
                "ofi": float(rng.normal(0.0, 0.5)),
                "microprice": float(tick),
                "depth_imb": float(rng.uniform(-0.2, 0.2)),
                "spread_bps": float(rng.uniform(0.5, 1.5)),
                "trade_imb": float(rng.normal(0.0, 0.001)),
                "vpin": 0.0,
                "n_updates": 10,
                "local_recv_ms": 0,
            })
    return pd.DataFrame(rows)


# ───────────────────────────── tests ─────────────────────────────

def test_aggregate_to_minute_basic_shape():
    df = _seconds_frame(n_minutes=5)
    out = aggregate_to_minute(df)
    assert len(out) == 5
    assert {"minute", "symbol", "seconds_observed",
            "microprice_open", "microprice_close"}.issubset(out.columns)
    # Each minute had 60 seconds of data → seconds_observed == 60.
    assert (out["seconds_observed"] == 60).all()


def test_aggregate_to_minute_drops_sparse_bars():
    """Bars with < MIN_SECONDS_PER_MINUTE coverage must be discarded."""
    df = _seconds_frame(n_minutes=3, seconds_per_minute=60)
    # Surgically remove most rows from the second minute → 5 seconds covered.
    minute_2 = pd.Timestamp("2026-04-25 00:01:00", tz="UTC")
    sparse = df[df["ts"].dt.floor("1min") != minute_2]
    five_secs = df[df["ts"].dt.floor("1min") == minute_2].head(5)
    df = pd.concat([sparse, five_secs]).sort_values("ts").reset_index(drop=True)

    out = aggregate_to_minute(df)
    minutes = set(out["minute"])
    assert minute_2 not in minutes, (
        f"minute with 5 < {MIN_SECONDS_PER_MINUTE} seconds should be dropped"
    )
    assert len(out) == 2


def test_aggregate_to_minute_empty_input():
    empty = pd.DataFrame(columns=[
        "ts", "symbol", "ofi", "microprice", "depth_imb",
        "spread_bps", "trade_imb", "vpin", "n_updates", "local_recv_ms",
    ])
    out = aggregate_to_minute(empty)
    assert out.empty


def test_build_target_forward_shift_and_bps():
    df = _seconds_frame(n_minutes=5, drift_per_min_bps=10.0)
    minute_df = aggregate_to_minute(df)
    out = build_target(minute_df, horizon=1, neutral_band_bps=0.0)

    # 5 minutes → 4 bars have observable forward returns; last bar is NaN.
    assert (out["y"] == -1).sum() == 1
    # Drift is +10 bp/min → return_bps should be ~ +10 ± noise.
    observable = out[out["y"] != -1]
    assert observable["return_bps"].mean() > 5.0
    assert (observable["y"] == 1).all(), "monotonic up drift → y == 1 everywhere"


def test_build_target_neutral_band_flag():
    df = _seconds_frame(n_minutes=4, drift_per_min_bps=0.5)
    minute_df = aggregate_to_minute(df)
    out = build_target(minute_df, horizon=1, neutral_band_bps=2.0)
    # 0.5 bp drift << 2 bp band → most observable bars in neutral band.
    observable = out[out["y"] != -1]
    assert observable["in_neutral_band"].mean() > 0.5


def test_build_features_schema_and_no_nan():
    df = _seconds_frame(n_minutes=3)
    minute_df = aggregate_to_minute(df)
    feats = build_features(minute_df)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == len(minute_df)
    assert feats.notna().all().all(), "build_features must not emit NaN"
    # All numeric.
    assert all(np.issubdtype(t, np.number) for t in feats.dtypes)


def test_build_features_handles_empty():
    empty = pd.DataFrame()
    feats = build_features(empty)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.empty


def test_make_supervised_drops_unobservable_and_neutral():
    df = _seconds_frame(n_minutes=6, drift_per_min_bps=10.0)
    X, y, meta = make_supervised(df)
    # 6 minutes → 5 observable bars. With strong drift none are in neutral
    # band (>1 bp default), so all 5 survive.
    assert len(X) == 5
    assert len(y) == 5
    assert len(meta) == 5
    assert list(X.columns) == FEATURE_COLUMNS
    # y is binary {0, 1} after the `-1` (unobservable) drop.
    assert set(y.unique()).issubset({0, 1})


def test_bootstrap_dir_acc_ci_deterministic():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_pred = y_true.copy()
    # Flip 30 % at random → ~70 % accuracy.
    flip = rng.random(size=500) < 0.3
    y_pred[flip] = 1 - y_pred[flip]

    point, lo, hi = bootstrap_dir_acc_ci(y_true, y_pred, n_resamples=500, seed=42)
    assert 0.65 < point < 0.75
    assert lo < point < hi
    # Deterministic — same seed must reproduce.
    point2, lo2, hi2 = bootstrap_dir_acc_ci(
        y_true, y_pred, n_resamples=500, seed=42,
    )
    assert (point, lo, hi) == (point2, lo2, hi2)


def test_bootstrap_dir_acc_ci_perfect_predictions():
    y = np.array([0, 1, 0, 1, 1, 0])
    point, lo, hi = bootstrap_dir_acc_ci(y, y.copy(), n_resamples=200)
    # Perfect → point is 1.0, CI lower can be anywhere in [<=1, 1].
    assert point == 1.0
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_dir_acc_ci_empty_input():
    point, lo, hi = bootstrap_dir_acc_ci(
        np.array([], dtype=int), np.array([], dtype=int),
    )
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_target_shift_matches_horizon(horizon):
    df = _seconds_frame(n_minutes=5 + horizon, drift_per_min_bps=5.0)
    minute_df = aggregate_to_minute(df)
    out = build_target(minute_df, horizon=horizon, neutral_band_bps=0.0)
    # Last `horizon` rows must be unobservable.
    n_unobs = (out["y"] == -1).sum()
    assert n_unobs == horizon
