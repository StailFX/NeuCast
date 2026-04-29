"""Joint multi-symbol feature pipeline — ONE model trained on
BTC + ETH + BNB simultaneously with symbol identity as a feature.

Why this exists
===============

Today we have THREE separate CatBoost models, one per symbol. Each
sees only its own data — ~1500-3000 OOS bars after neutral-band drop.

Joint training gives ONE model trained on the **pooled** data
(~6000-9000 bars). Hypothesis:

* **Regularisation through shared parameters** — the model learns a
  single decision boundary that has to work for all 3 symbols, which
  prevents overfitting to per-symbol idiosyncrasies.
* **Larger training set** — 3× more data per fit.
* **Symbol identity as a feature** lets the model encode per-symbol
  shifts (e.g. "depth_imb mean is +0.05 lower on BNB vs BTC") without
  losing the shared signal.

Honest risks
------------

* **May UNDERPERFORM per-symbol models** if the symbols' dynamics are
  too different. Joint training assumes shared underlying signal.
* **Class balance shifts**: BTC's base rate ≈ 0.50, but if BNB's is
  0.51 in the training window the joint model picks up that bias.

Empirically tested via multi_horizon_eval --feature-sets joint.
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


# Symbol-id features. CatBoost handles these as numeric (treats them
# like an ordinal lookup table); for the small symbol set we have,
# pure one-hot is equivalently expressive and simpler.
JOINT_SYMBOL_FEATURE_COLUMNS: list[str] = [
    "is_btc",  # 1 if symbol == BTCUSDT, else 0
    "is_eth",
    "is_bnb",
]

JOINT_FEATURE_COLUMNS: list[str] = list(BASE_FEATURE_COLUMNS) + list(JOINT_SYMBOL_FEATURE_COLUMNS)


# Joint + classical-TA features — pooled multi-symbol training with
# OHLC/RSI/Bollinger features (better suited for longer horizons).
# Same symbol-id one-hots appended for per-symbol identity.
def joint_long_horizon_columns() -> list[str]:
    """Build the canonical column order for joint+long_horizon mode.

    Centralised so trainer + predictor + tests can't drift."""
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS,
    )
    return list(LONG_HORIZON_FEATURE_COLUMNS) + list(JOINT_SYMBOL_FEATURE_COLUMNS)


def build_joint_long_horizon_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Long-horizon TA features + symbol-id one-hots for joint training.

    Used at 5m / 15m / 60m horizons where microstructure decays but
    OHLC + EMA + RSI + Bollinger carry signal.
    """
    from app.highfreq.feature_pipeline_long_horizon import (
        build_long_horizon_features,
    )
    cols = joint_long_horizon_columns()
    if minute_df.empty:
        return pd.DataFrame(columns=cols)

    feats = build_long_horizon_features(minute_df)
    sym = minute_df["symbol"].astype(str).str.upper()
    feats["is_btc"] = (sym == "BTCUSDT").astype(float).values
    feats["is_eth"] = (sym == "ETHUSDT").astype(float).values
    feats["is_bnb"] = (sym == "BNBUSDT").astype(float).values
    return feats[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_joint_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Compute base features + one-hot symbol identity.

    Input: per-bar frame from ``aggregate_to_minute``. The frame may
    contain MULTIPLE symbols — that's the point of joint training.

    Output: feature matrix with the pinned column order in
    :data:`JOINT_FEATURE_COLUMNS`. Same row order as input.
    """
    if minute_df.empty:
        return pd.DataFrame(columns=JOINT_FEATURE_COLUMNS)

    base = build_base_features(minute_df)

    sym = minute_df["symbol"].astype(str).str.upper()
    base["is_btc"] = (sym == "BTCUSDT").astype(float).values
    base["is_eth"] = (sym == "ETHUSDT").astype(float).values
    base["is_bnb"] = (sym == "BNBUSDT").astype(float).values

    return base[JOINT_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0.0)


def make_joint_supervised(
    df_secs_by_symbol: dict[str, pd.DataFrame],
    *,
    horizon: int = 1,
    neutral_band_bps: float = 1.0,
    bar_minutes: int = 1,
    use_long_horizon_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Pool seconds frames across symbols and build a unified
    (X, y, meta) supervised dataset.

    The trickiest invariant: ``build_target`` shifts microprice within
    each symbol's group, NOT across symbol boundaries. We rely on
    ``aggregate_to_minute``'s groupby preserving per-symbol ordering
    + ``build_target``'s ``groupby("symbol")`` shift.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix in JOINT_FEATURE_COLUMNS order, multi-symbol pooled.
    y : pd.Series
        Binary direction target.
    meta : pd.DataFrame
        ``symbol``, ``minute``, ``microprice_close``, ``return_bps`` —
        for join-back to predictions.
    """
    from app.highfreq.feature_pipeline import (
        aggregate_to_minute,
        build_target,
    )

    if not df_secs_by_symbol:
        return (
            pd.DataFrame(columns=JOINT_FEATURE_COLUMNS),
            pd.Series(dtype=np.int8),
            pd.DataFrame(columns=["symbol", "minute", "microprice_close", "return_bps"]),
        )

    pooled_secs = pd.concat(
        df_secs_by_symbol.values(), ignore_index=True, sort=False,
    )
    minute_df = aggregate_to_minute(pooled_secs, bar_minutes=bar_minutes)
    if minute_df.empty:
        return (
            pd.DataFrame(columns=JOINT_FEATURE_COLUMNS),
            pd.Series(dtype=np.int8),
            pd.DataFrame(columns=["symbol", "minute", "microprice_close", "return_bps"]),
        )
    targeted = build_target(
        minute_df, horizon=horizon, neutral_band_bps=neutral_band_bps,
    )
    if targeted.empty:
        return (
            pd.DataFrame(columns=JOINT_FEATURE_COLUMNS),
            pd.Series(dtype=np.int8),
            pd.DataFrame(columns=["symbol", "minute", "microprice_close", "return_bps"]),
        )

    keep = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
    targeted = targeted.loc[keep].reset_index(drop=True)
    if use_long_horizon_features:
        X = build_joint_long_horizon_features(targeted)
    else:
        X = build_joint_features(targeted)
    # CRITICAL: walk-forward CV needs chronological ordering. Joint
    # data has multiple symbols at the same minute — within-minute
    # order doesn't matter for training, but we sort by `minute` so
    # the train/test split happens on a clean time boundary.
    order = targeted["minute"].argsort(kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = targeted["y"].astype(np.int8).iloc[order].reset_index(drop=True)
    meta = targeted[["symbol", "minute", "microprice_close", "return_bps"]].iloc[order].reset_index(drop=True)
    return X, y, meta
