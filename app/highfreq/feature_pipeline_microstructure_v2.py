"""Microstructure v2: base 18 features + 4 trade-flow rolling features.

Why
===

The current ``microstructure`` pipeline (18 cols) summarises trade
flow with two scalars per bar: ``trade_imb_sum`` (signed sum of
trade-volume imbalance over the bar) and ``trade_imb_abs_sum``.
That collapses a 60-second time-series into a number — losing the
shape of how flow built up.

This v2 pipeline adds **four trade-flow rolling features** that
expose multi-bar dynamics:

1. ``trade_buy_ratio`` — within current bar:
   ``trade_imb_sum / max(trade_imb_abs_sum, ε)``. Signed in [-1, 1];
   1.0 = pure aggressive buying, -1.0 = pure aggressive selling.
2. ``trade_buy_ratio_lag1`` — same shifted 1 bar (memory of last min).
3. ``trade_buy_ratio_3bar_avg`` — 3-bar rolling mean of the ratio
   (smoothed flow direction over the last 3 minutes).
4. ``trade_imb_acceleration`` — current ``trade_imb_sum`` minus
   lag1 ``trade_imb_sum``. Positive = flow becoming MORE buy-
   aggressive; negative = flow becoming MORE sell-aggressive.

Total: **22 columns** (18 base + 4 trade-flow).

Defence narrative
-----------------

> "We extended the trade-flow representation from two scalars per
> bar to a 4-component multi-scale feature: instantaneous,
> 1-bar-lagged, 3-bar-averaged, and acceleration. The hypothesis
> was that flow-shape information at sub-minute resolution carries
> directional signal beyond the bar's net imbalance. Empirical
> A/B (T.18.c) measures the lift, if any."

Backward compatibility
----------------------

The base 18 columns are computed by the unchanged
:func:`app.highfreq.feature_pipeline.build_features`. v2 calls it
and appends the 4 trade-flow columns. So v1 models continue to
load + serve cleanly; only models trained with feature_set='microstructure_v2'
expect the 22-col input.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.highfreq.feature_pipeline import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    build_features as build_base_features,
)

logger = logging.getLogger(__name__)


#: Trade-flow rolling features added on top of the base 18 columns.
TRADE_FLOW_FEATURE_COLUMNS: list[str] = [
    "trade_buy_ratio",
    "trade_buy_ratio_lag1",
    "trade_buy_ratio_3bar_avg",
    "trade_imb_acceleration",
]


#: Full v2 column list = base 18 + trade-flow 4 = 22 columns.
MICROSTRUCTURE_V2_FEATURE_COLUMNS: list[str] = (
    list(BASE_FEATURE_COLUMNS) + list(TRADE_FLOW_FEATURE_COLUMNS)
)


def build_microstructure_v2_features(df_minutes: pd.DataFrame) -> pd.DataFrame:
    """Build the 22-col v2 feature matrix.

    Inputs columns expected match what
    :func:`feature_pipeline.aggregate_to_minute` produces:
    ``symbol``, ``minute``, ``trade_imb_sum``, ``trade_imb_abs_sum``,
    plus everything ``build_base_features`` consumes.

    Lags are computed PER-SYMBOL (groupby) so a multi-symbol frame
    doesn't bleed BTC's lag1 into ETH's row.
    """
    if df_minutes.empty:
        return pd.DataFrame(columns=MICROSTRUCTURE_V2_FEATURE_COLUMNS)

    base = build_base_features(df_minutes)
    eps = 1e-12

    # Sort+groupby per-symbol for the lags.
    df = df_minutes.copy().sort_values(["symbol", "minute"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)

    # 1) Within-bar buy ratio. Signed in [-1, 1].
    trade_buy_ratio = (
        df["trade_imb_sum"].astype(float)
        / df["trade_imb_abs_sum"].astype(float).clip(lower=eps)
    ).clip(-1.0, 1.0)

    # 2) Lag-1 of the ratio.
    df["_buy_ratio"] = trade_buy_ratio
    trade_buy_ratio_lag1 = (
        g["_buy_ratio"].shift(1).fillna(0.0).astype(float)
    )

    # 3) 3-bar rolling mean (current + last 2 bars).
    trade_buy_ratio_3bar = (
        g["_buy_ratio"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .astype(float)
    )

    # 4) Acceleration: current - lag1 of trade_imb_sum.
    trade_imb_lag1 = (
        g["trade_imb_sum"].shift(1).fillna(0.0).astype(float)
    )
    trade_imb_acceleration = (
        df["trade_imb_sum"].astype(float) - trade_imb_lag1
    )

    df = df.drop(columns=["_buy_ratio"])

    # Index-align to base. Both base + df share row order from the
    # input minute_df; sort_values above is stable + reset_index
    # keeps positional alignment.
    base = base.reset_index(drop=True)
    base["trade_buy_ratio"] = trade_buy_ratio.reset_index(drop=True).values
    base["trade_buy_ratio_lag1"] = trade_buy_ratio_lag1.reset_index(drop=True).values
    base["trade_buy_ratio_3bar_avg"] = trade_buy_ratio_3bar.reset_index(drop=True).values
    base["trade_imb_acceleration"] = trade_imb_acceleration.reset_index(drop=True).values

    # Reorder to canonical column list — defends against silent column
    # drift when the trainer / predictor compare schemas.
    base = base[MICROSTRUCTURE_V2_FEATURE_COLUMNS]
    return base.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_latest_inference_bar_microstructure_v2(
    df_seconds: pd.DataFrame,
) -> tuple[pd.Series, float] | None:
    """Live-inference helper paralleling
    :func:`feature_pipeline.build_latest_inference_bar` but emitting
    the 22-col v2 vector.

    Aggregates seconds → 1-min bars (drops in-flight current bar),
    requires ≥ 4 complete bars so lag1 + 3-bar rolling are defined
    on the latest row, returns ``(features, close_microprice)`` for
    the most recent COMPLETE bar.
    """
    from app.highfreq.feature_pipeline import (
        aggregate_to_minute, MIN_SECONDS_PER_MINUTE,
    )

    if df_seconds.empty:
        return None
    df = df_seconds.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if df["ts"].empty:
        return None

    # Drop the in-flight bar (mirrors v1 helper).
    now_floor = df["ts"].max().floor("1min")
    df = df.loc[df["ts"] < now_floor].copy()
    if df.empty:
        return None

    minute_df = aggregate_to_minute(df, bar_minutes=1)
    # Need ≥ 4 bars so the 3-bar rolling + lag1 are well-defined for
    # the latest row.
    if len(minute_df) < 4:
        return None

    feats = build_microstructure_v2_features(minute_df)
    if feats.empty:
        return None
    last_close = float(minute_df.iloc[-1]["microprice_close"])
    return feats.iloc[-1], last_close
