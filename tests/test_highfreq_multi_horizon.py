"""Tests for multi-horizon trainer + predictor (release R, 2026-04-29).

Pins the contract that:
* ``weights_path_for_symbol`` returns ``<symbol>_<horizon>m.cbm`` and
  defaults to 1m for backwards compatibility,
* ``get_predictor`` caches per (symbol × horizon) — calling with
  horizon_minutes=1 vs 15 yields different instances,
* ``_resolve_feature_set`` picks microstructure for 1m, long_horizon
  for ≥5m by default, and respects explicit overrides,
* ``WalkForwardConfig`` accepts new ``bar_minutes`` + ``feature_set``
  fields with sane defaults,
* ``walk_forward_evaluate`` scales fold geometry by ``bar_minutes`` so
  a 15-minute model with default config gets reasonable bar counts
  rather than waiting for 1440 bars (15 days of data).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.highfreq.predictor import (
    get_predictor,
    reset_predictor,
    weights_path_for_symbol,
)
from app.highfreq.trainer import (
    WalkForwardConfig,
    _resolve_feature_set,
    _make_supervised_for_feature_set,
)


# ─── weights_path_for_symbol ───


def test_weights_path_default_is_1m_for_backwards_compat():
    """The original single-horizon contract said
    ``weights/highfreq/btcusdt_1m.cbm``. Adding multi-horizon must NOT
    break legacy callers — default ``horizon_minutes=1`` keeps the
    name unchanged."""
    p = weights_path_for_symbol("BTCUSDT")
    assert p.name == "btcusdt_1m.cbm"


def test_weights_path_with_15m_horizon():
    p = weights_path_for_symbol("BTCUSDT", horizon_minutes=15)
    assert p.name == "btcusdt_15m.cbm"


def test_weights_path_with_60m_horizon():
    p = weights_path_for_symbol("BTCUSDT", horizon_minutes=60)
    assert p.name == "btcusdt_60m.cbm"


def test_weights_path_lowercases_symbol():
    """Uppercase symbol input → lowercase filename, same on every horizon."""
    assert weights_path_for_symbol("ETHUSDT", horizon_minutes=15).name == "ethusdt_15m.cbm"
    assert weights_path_for_symbol("BNBUSDT", horizon_minutes=60).name == "bnbusdt_60m.cbm"


def test_weights_path_honours_dir_env(monkeypatch):
    monkeypatch.setenv("HIGHFREQ_WEIGHTS_DIR", "/custom/weights/dir")
    p = weights_path_for_symbol("BTCUSDT", horizon_minutes=15)
    assert str(p) == "/custom/weights/dir/btcusdt_15m.cbm"


# ─── get_predictor caching ───


def test_get_predictor_caches_per_horizon():
    """Same symbol but different horizons MUST produce distinct
    LivePredictor instances — otherwise the 60m runner would overwrite
    the 1m runner's cached model on hot reload."""
    reset_predictor()
    try:
        p1 = get_predictor("BTCUSDT", horizon_minutes=1)
        p15 = get_predictor("BTCUSDT", horizon_minutes=15)
        p60 = get_predictor("BTCUSDT", horizon_minutes=60)
        assert p1 is not p15
        assert p15 is not p60
        assert p1 is not p60
        # Each one points at the right .cbm.
        assert p1.weights_path.name == "btcusdt_1m.cbm"
        assert p15.weights_path.name == "btcusdt_15m.cbm"
        assert p60.weights_path.name == "btcusdt_60m.cbm"
    finally:
        reset_predictor()


def test_get_predictor_cached_returns_same_instance_for_same_horizon():
    reset_predictor()
    try:
        a = get_predictor("BTCUSDT", horizon_minutes=15)
        b = get_predictor("BTCUSDT", horizon_minutes=15)
        assert a is b  # cached
    finally:
        reset_predictor()


def test_get_predictor_default_horizon_is_1m():
    """Backwards compat: legacy code calling get_predictor("BTCUSDT")
    without horizon_minutes must keep getting the 1m predictor."""
    reset_predictor()
    try:
        p = get_predictor("BTCUSDT")
        assert p.weights_path.name == "btcusdt_1m.cbm"
    finally:
        reset_predictor()


# ─── _resolve_feature_set ───


def test_resolve_feature_set_auto_picks_microstructure_for_1m():
    """1-minute bars are the production microstructure regime."""
    assert _resolve_feature_set("auto", bar_minutes=1) == "microstructure"


def test_resolve_feature_set_auto_picks_long_horizon_for_5m_and_up():
    """≥5-minute bars: microstructure decays into noise; OHLC + EMA +
    RSI + Bollinger pipeline empirically dominates (multi-horizon eval
    confirmed this with joint+TA reaching dir_acc 0.61 at 60m)."""
    assert _resolve_feature_set("auto", bar_minutes=5) == "long_horizon"
    assert _resolve_feature_set("auto", bar_minutes=15) == "long_horizon"
    assert _resolve_feature_set("auto", bar_minutes=60) == "long_horizon"


def test_resolve_feature_set_explicit_overrides_auto():
    """Force microstructure on 15m for ablation studies — explicit
    request must beat the auto rule."""
    assert _resolve_feature_set("microstructure", bar_minutes=15) == "microstructure"
    assert _resolve_feature_set("long_horizon", bar_minutes=1) == "long_horizon"


def test_resolve_feature_set_rejects_unknown():
    with pytest.raises(ValueError):
        _resolve_feature_set("transformer", bar_minutes=1)
    with pytest.raises(ValueError):
        _resolve_feature_set("", bar_minutes=1)


# ─── WalkForwardConfig new defaults ───


def test_walk_forward_config_defaults_to_1m_microstructure():
    """Production contract: WalkForwardConfig() with no args produces
    the original 1m + microstructure config. Pinned so a refactor
    that flips defaults to long_horizon doesn't silently change
    deployed behaviour."""
    cfg = WalkForwardConfig()
    assert cfg.bar_minutes == 1
    assert cfg.feature_set == "auto"
    # Sample-weight + embargo defaults from release O still in place.
    assert cfg.sample_weight_half_life_bars == 720
    assert cfg.embargo_bars == 1


def test_walk_forward_config_accepts_15m_long_horizon():
    cfg = WalkForwardConfig(bar_minutes=15, feature_set="long_horizon")
    assert cfg.bar_minutes == 15
    assert cfg.feature_set == "long_horizon"


# ─── _make_supervised_for_feature_set ───


def _seconds_frame_for_target(
    n_minutes: int = 200, drift_per_min_bps: float = 1.0,
    symbol: str = "BTCUSDT", seed: int = 0,
) -> pd.DataFrame:
    """Synthetic 1-second frame — same shape as production input.

    Each minute has a RANDOM-SIGN drift (not monotonic) so the binary
    target after build_target has both classes — needed for CatBoost
    to fit at all in tests."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2026-04-29 00:00:00", tz="UTC")
    base = 77_000.0
    for m in range(n_minutes):
        # Random-walk style: each minute's drift is ±drift_per_min_bps
        # times a Gaussian shock — produces real sign variation in
        # next-minute returns, so binary y has both 0 and 1.
        shock_bps = rng.normal(0.0, drift_per_min_bps)
        base = base * (1.0 + shock_bps / 1e4)
        for s in range(60):
            ts = t0 + pd.Timedelta(minutes=m, seconds=s)
            rows.append({
                "ts": ts,
                "symbol": symbol,
                "ofi": float(rng.normal(0.0, 0.5)),
                "microprice": float(base + rng.normal(0.0, 1.0)),
                "depth_imb": float(rng.uniform(-0.2, 0.2)),
                "spread_bps": float(rng.uniform(0.5, 1.5)),
                "trade_imb": float(rng.normal(0.0, 0.001)),
                "n_updates": 10,
            })
    return pd.DataFrame(rows)


def test_make_supervised_microstructure_path_returns_microstructure_columns():
    df = _seconds_frame_for_target(n_minutes=10)
    X, y, meta = _make_supervised_for_feature_set(
        df, feature_set="microstructure", bar_minutes=1,
    )
    from app.highfreq.feature_pipeline import FEATURE_COLUMNS
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(meta)
    # y is binary int8 after target+neutral-band drop.
    assert y.dtype == np.int8


def test_make_supervised_long_horizon_path_returns_long_horizon_columns():
    """Long-horizon pipeline must produce LONG_HORIZON_FEATURE_COLUMNS,
    NOT microstructure. Pin to defend against silent fallback to the
    wrong pipeline if feature_set='long_horizon' branch breaks."""
    df = _seconds_frame_for_target(n_minutes=200, drift_per_min_bps=2.0)
    X, y, meta = _make_supervised_for_feature_set(
        df, feature_set="long_horizon", bar_minutes=5,
    )
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS,
    )
    assert list(X.columns) == LONG_HORIZON_FEATURE_COLUMNS
    assert len(X) == len(y) == len(meta)
    assert y.dtype == np.int8


def test_make_supervised_auto_routes_by_bar_minutes():
    """auto + bar_minutes=1 → microstructure; auto + bar_minutes=15
    → long_horizon. Pin the routing so a refactor of _resolve_feature_set
    can't silently send 15m runs through the microstructure pipeline."""
    from app.highfreq.feature_pipeline import FEATURE_COLUMNS as MICRO
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS as LH,
    )
    df = _seconds_frame_for_target(n_minutes=200)
    X1, _, _ = _make_supervised_for_feature_set(
        df, feature_set="auto", bar_minutes=1,
    )
    X15, _, _ = _make_supervised_for_feature_set(
        df, feature_set="auto", bar_minutes=15,
    )
    assert list(X1.columns) == MICRO
    assert list(X15.columns) == LH


def test_make_supervised_long_horizon_empty_input_returns_well_typed_empty():
    """Same defensive contract as make_supervised — empty DB query
    must return correctly-shaped empty triple, never KeyError."""
    empty = pd.DataFrame(columns=[
        "ts", "symbol", "ofi", "microprice", "depth_imb",
        "spread_bps", "trade_imb", "n_updates",
    ])
    X, y, meta = _make_supervised_for_feature_set(
        empty, feature_set="long_horizon", bar_minutes=15,
    )
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS,
    )
    assert X.empty
    assert list(X.columns) == LONG_HORIZON_FEATURE_COLUMNS
    assert y.empty
    assert meta.empty


# ─── walk_forward_evaluate fold-geometry scaling ───


def test_walk_forward_scales_initial_train_by_bar_minutes():
    """At bar_minutes=15 with initial_train_minutes=60 (1 hour of data
    = 4 bars at 15m), the loop should be willing to start training
    after just 4 bars of warmup — NOT after 60 bars (which would mean
    15 hours of warmup that the loop never accumulates on a 200-minute
    synthetic frame).  Pin so a refactor that forgets to divide by
    bar_minutes doesn't silently degrade to 0 folds.
    """
    from app.highfreq.feature_pipeline import aggregate_to_minute, build_target
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS, build_long_horizon_features,
    )
    from app.highfreq.trainer import walk_forward_evaluate

    catboost = pytest.importorskip("catboost")
    del catboost

    df = _seconds_frame_for_target(n_minutes=400, drift_per_min_bps=1.0, seed=7)
    minute_df = aggregate_to_minute(df, bar_minutes=15)
    targeted = build_target(minute_df, horizon=1, neutral_band_bps=0.0)
    keep = (targeted["y"] != -1)
    targeted = targeted.loc[keep].reset_index(drop=True)
    X = build_long_horizon_features(targeted)[LONG_HORIZON_FEATURE_COLUMNS]
    y = targeted["y"].astype(np.int8)
    meta = targeted[["symbol", "minute", "microprice_close", "return_bps"]]

    # 400 minutes @ 15m = ~26 bars. With initial_train_minutes=60 (=4 bars at 15m),
    # test_fold_minutes=15 (=1 bar), step_minutes=15 (=1 bar) we should produce
    # multiple folds.
    cfg = WalkForwardConfig(
        bar_minutes=15,
        initial_train_minutes=60,    # 4 bars at 15m
        test_fold_minutes=15,        # 1 bar
        step_minutes=15,             # 1 bar
        min_train_samples=2,         # accept tiny train folds for the test
        catboost_iterations=20,      # fast
        sample_weight_half_life_bars=0,
        embargo_bars=0,
    )
    folds, preds = walk_forward_evaluate(X, y, meta, config=cfg)

    # Should produce SOME folds — scaling broken would yield zero.
    assert len(folds) > 0, (
        "walk_forward_evaluate produced 0 folds on 26-bar 15m data; "
        "fold geometry probably did NOT scale by bar_minutes"
    )
    # Each fold's test slice is 1 bar at 15m (test_fold_bars=1 after scaling).
    for f in folds:
        assert f.n_test == 1, (
            f"fold n_test={f.n_test} but expected 1 (test_fold_minutes=15 / "
            f"bar_minutes=15 = 1 bar)"
        )


# ─── predictor: feature_set + bar_minutes accessors ───


def test_predictor_feature_set_defaults_to_microstructure(tmp_path):
    """Legacy metrics.json with no ``feature_set`` field → predictor
    falls back to microstructure. Pin so a refactor that changes the
    default doesn't silently switch the live inference path."""
    from app.highfreq.predictor import LivePredictor

    # Empty metrics file (legacy).
    metrics_path = tmp_path / "x_1m_metrics.json"
    metrics_path.write_text("{}")
    weights_path = tmp_path / "x_1m.cbm"  # never read by feature_set()
    p = LivePredictor(weights_path=weights_path, metrics_path=metrics_path)
    assert p.feature_set() == "microstructure"
    assert p.bar_minutes() == 1


def test_predictor_feature_set_reads_from_metrics(tmp_path):
    """When metrics.json has ``feature_set='long_horizon'`` and
    ``bar_minutes=15`` the predictor surfaces those values so the
    runner can dispatch to the right inference helper."""
    import json
    from app.highfreq.predictor import LivePredictor

    metrics_path = tmp_path / "x_15m_metrics.json"
    metrics_path.write_text(json.dumps({
        "symbol": "BTCUSDT",
        "feature_set": "long_horizon",
        "bar_minutes": 15,
        "dir_acc_mean": 0.66,
        "dir_acc_ci_low": 0.52,
        "dir_acc_ci_high": 0.80,
    }))
    weights_path = tmp_path / "x_15m.cbm"
    p = LivePredictor(weights_path=weights_path, metrics_path=metrics_path)
    assert p.feature_set() == "long_horizon"
    assert p.bar_minutes() == 15


def test_predictor_expected_columns_dispatches_by_feature_set(tmp_path):
    """Pin the train-vs-serve invariant: a long_horizon model gets
    24-column input shape, microstructure gets 18.  Defends against
    the 'Feature 18 is present in model but not in pool' CatBoost
    error that hit the first 15m paper-trader spawn."""
    import json
    from app.highfreq.feature_pipeline import FEATURE_COLUMNS
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS,
    )
    from app.highfreq.predictor import LivePredictor

    weights_path = tmp_path / "x.cbm"

    # microstructure path
    micro_metrics = tmp_path / "micro_metrics.json"
    micro_metrics.write_text(json.dumps({"feature_set": "microstructure"}))
    p_micro = LivePredictor(weights_path=weights_path, metrics_path=micro_metrics)
    assert p_micro._expected_feature_columns() == FEATURE_COLUMNS

    # long_horizon path
    lh_metrics = tmp_path / "lh_metrics.json"
    lh_metrics.write_text(json.dumps({"feature_set": "long_horizon"}))
    p_lh = LivePredictor(weights_path=weights_path, metrics_path=lh_metrics)
    assert p_lh._expected_feature_columns() == LONG_HORIZON_FEATURE_COLUMNS


# ─── build_latest_inference_bar_long_horizon ───


def test_build_latest_inference_bar_long_horizon_returns_none_on_empty():
    """Empty seconds frame → cold-start → None.  Same contract as the
    1m helper."""
    from app.highfreq.feature_pipeline_long_horizon import (
        build_latest_inference_bar_long_horizon,
    )
    out = build_latest_inference_bar_long_horizon(
        pd.DataFrame(), bar_minutes=15,
    )
    assert out is None


def test_build_latest_inference_bar_long_horizon_needs_20_bars():
    """The long-horizon pipeline uses EMA(20) / Bollinger(20) / RSI(14).
    Returning a feature row from <20 bars produces zero-valued
    indicators the model wasn't trained on. Pin: helper returns None
    rather than emit out-of-distribution features."""
    from app.highfreq.feature_pipeline_long_horizon import (
        build_latest_inference_bar_long_horizon,
    )
    # 60 minutes of seconds = 4 complete 15m bars (since the 1st bar
    # may need >450s coverage; we'll have 4 fully covered bars).
    df = _seconds_frame_for_target(n_minutes=60, drift_per_min_bps=2.0, seed=0)
    out = build_latest_inference_bar_long_horizon(df, bar_minutes=15)
    assert out is None, (
        "expected None when fewer than 20 long-horizon bars are available"
    )


def test_build_latest_inference_bar_long_horizon_returns_feature_row_at_15m():
    """With ≥20 complete 15m bars, helper returns (features, close)
    where features is indexed by LONG_HORIZON_FEATURE_COLUMNS.

    21 × 15 = 315 minutes of source seconds → 21 complete 15m bars
    after dropping the in-flight one — enough to bootstrap EMA(20)."""
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS,
        build_latest_inference_bar_long_horizon,
    )
    df = _seconds_frame_for_target(
        n_minutes=22 * 15, drift_per_min_bps=2.0, seed=3,
    )
    out = build_latest_inference_bar_long_horizon(df, bar_minutes=15)
    assert out is not None
    feats, close_mp = out
    assert list(feats.index) == LONG_HORIZON_FEATURE_COLUMNS
    assert isinstance(close_mp, float)
    # Sanity: synthetic price stays in BTC's range.
    assert close_mp > 50_000


def test_build_latest_inference_bar_long_horizon_drops_in_flight_bar():
    """The trainer fits on whole-bar aggregates. Helper must drop the
    in-flight bar (matches the 1m contract). Pin via a sentinel: if
    the in-flight bar leaks through, the close microprice would be
    the most-recent second's price; we want the prior complete bar's
    close instead."""
    from app.highfreq.feature_pipeline_long_horizon import (
        build_latest_inference_bar_long_horizon,
    )
    # 22 minutes × 15 = 330 mins of complete bars + a 5-min in-flight
    # tail with a deliberately-different price level.
    full = _seconds_frame_for_target(n_minutes=22 * 15, seed=4)
    # In-flight bar: prices ~10% higher to make leaks obvious.
    rng = np.random.default_rng(99)
    rows = []
    t0 = pd.Timestamp("2026-04-29 00:00:00", tz="UTC") + pd.Timedelta(
        minutes=22 * 15,
    )
    for s in range(5 * 60):  # 5 minutes of in-flight
        rows.append({
            "ts": t0 + pd.Timedelta(seconds=s),
            "symbol": "BTCUSDT",
            "ofi": float(rng.normal(0.0, 0.5)),
            "microprice": 100_000.0,  # sentinel high
            "depth_imb": float(rng.uniform(-0.2, 0.2)),
            "spread_bps": 1.0,
            "trade_imb": float(rng.normal(0.0, 0.001)),
            "n_updates": 10,
        })
    df = pd.concat([full, pd.DataFrame(rows)], ignore_index=True)
    out = build_latest_inference_bar_long_horizon(df, bar_minutes=15)
    assert out is not None
    _, close_mp = out
    # If the in-flight bar leaked, close would be ~100_000 (sentinel).
    assert close_mp < 90_000, (
        f"close_microprice={close_mp} suggests the in-flight bar leaked"
    )
