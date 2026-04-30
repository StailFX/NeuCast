"""Cross-asset + lagged feature pipeline — BTC features as inputs to
ETH/BNB models, plus lagged microstructure on the target symbol itself.

Why this exists
===============

Until now each symbol's model is trained in isolation: BTC sees only
BTC's order flow, ETH only ETH's. But empirically on Binance Spot at
minute timescale:

* ETH/BTC return correlation ≈ 0.85+
* BNB/BTC return correlation ≈ 0.7-0.85 (regime-dependent)
* BTC tends to LEAD by 5-30 seconds — ETH/BNB price discovery often
  follows after a BTC move propagates through cross-asset arbitrage.

If true, BTC's microstructure features at time t carry directional
information about ETH/BNB at t+1 that ETH/BNB's own features miss.
The model has access to the future of the leader, not just the
current state of the follower.

Lagged microstructure (separate gain)
-------------------------------------

CatBoost sees each bar independently. Even on a single symbol, the
last 2-3 minutes of OFI carry momentum information that the
1-minute aggregate hides. We add ``ofi_sum_lag1``, ``ofi_sum_lag2``,
``ofi_sum_lag3`` and ``microprice_return_bps_lag1`` so the model can
learn "OFI was positive for last 3 minutes → momentum regime".

Output schema
-------------

When ``reference_symbol`` is provided (e.g. ``"BTCUSDT"`` for ETH model):
    base 18 features (from feature_pipeline.FEATURE_COLUMNS)
  + 4 lagged features
  + 5 cross-asset features
  = 27 features total

When ``reference_symbol`` is None (BTC's own model):
    base 18
  + 4 lagged
  = 22 features total
  (cross-asset would point at itself; identity features are useless
   so we just don't add them)

Pure pipeline contract — same as the other feature pipelines:
takes already-aggregated minute bars + optional reference bars,
returns feature matrix in canonical column order.
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


# Lagged versions of the most predictive raw features.
LAGGED_FEATURE_COLUMNS: list[str] = [
    "ofi_sum_lag1",
    "ofi_sum_lag2",
    "ofi_sum_lag3",
    "microprice_return_bps_lag1",
]

# Cross-asset features — only added when a reference symbol is given.
CROSS_ASSET_FEATURE_COLUMNS: list[str] = [
    "ref_microprice_return_bps",      # current bar's reference return
    "ref_microprice_return_bps_lag1", # one-bar-ago reference return
    "ref_ofi_sum",                    # reference OFI right now (lead/lag)
    "ref_microprice_high_low_bps",    # reference vol regime proxy
    "ref_depth_imb_now",              # reference book imbalance
]


def feature_columns_for(reference_symbol: str | None) -> list[str]:
    """Build the canonical feature-column order for a model that uses
    cross-asset + lagged features. Centralised so trainer + predictor
    + tests cannot drift."""
    cols = list(BASE_FEATURE_COLUMNS) + list(LAGGED_FEATURE_COLUMNS)
    if reference_symbol:
        cols += list(CROSS_ASSET_FEATURE_COLUMNS)
    return cols


def _add_lagged_features(
    base_feats: pd.DataFrame,
    minute_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add lagged microstructure to the base feature matrix.

    Lags are computed PER-SYMBOL within the minute_df (groupby) — a
    multi-symbol DataFrame must NOT have BTC's OFI bleed into ETH's
    lag1 just because they're adjacent rows.
    """
    df = minute_df.sort_values(["symbol", "minute"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)

    ofi_lag1 = g["ofi_sum"].shift(1).fillna(0.0).astype(float)
    ofi_lag2 = g["ofi_sum"].shift(2).fillna(0.0).astype(float)
    ofi_lag3 = g["ofi_sum"].shift(3).fillna(0.0).astype(float)

    # microprice return lag1: need to recompute return per-bar then shift
    mp_close = df["microprice_close"].astype(float)
    mp_open = df["microprice_open"].astype(float).replace(0, np.nan)
    bar_ret_bps = ((mp_close - mp_open) / mp_open * 1e4).fillna(0.0)
    df["_bar_ret_bps"] = bar_ret_bps
    bar_ret_lag1 = df.groupby("symbol", sort=False)["_bar_ret_bps"].shift(1).fillna(0.0).astype(float)
    df = df.drop(columns=["_bar_ret_bps"])

    # Index alignment: base_feats was built off the same minute_df
    # rows in the same order, so positional concat is safe.
    out = base_feats.copy()
    out["ofi_sum_lag1"] = ofi_lag1.values
    out["ofi_sum_lag2"] = ofi_lag2.values
    out["ofi_sum_lag3"] = ofi_lag3.values
    out["microprice_return_bps_lag1"] = bar_ret_lag1.values
    return out


def _add_cross_asset_features(
    feats: pd.DataFrame,
    minute_df: pd.DataFrame,
    reference_minutes: pd.DataFrame,
    reference_symbol: str,
) -> pd.DataFrame:
    """Join reference-symbol bars onto minute_df by ``minute`` ts and
    expose them as features.

    The reference frame must have been produced by the same
    ``aggregate_to_minute`` call (aligned bar-floor timestamps). We
    inner-join on ``minute`` so target rows that have no aligned
    reference bar are dropped — keeps the feature matrix contiguous
    and well-defined.

    Returns a (possibly filtered) feature matrix; if rows were dropped,
    the caller's downstream code (y, meta) must align via the returned
    index.
    """
    ref = reference_minutes[reference_minutes["symbol"] == reference_symbol].copy()
    if ref.empty:
        logger.warning(
            "cross-asset: no reference rows for %s — emitting NaN features",
            reference_symbol,
        )
        # Fill with zeros so the model still trains; honest because the
        # feature carries no signal in this case.
        for col in CROSS_ASSET_FEATURE_COLUMNS:
            feats[col] = 0.0
        return feats

    ref = ref.sort_values("minute").reset_index(drop=True)
    # Compute reference-side derived quantities.
    ref_open = ref["microprice_open"].astype(float).replace(0, np.nan)
    ref_close = ref["microprice_close"].astype(float)
    ref_high = ref["microprice_high"].astype(float)
    ref_low = ref["microprice_low"].astype(float)

    ref["_ref_microprice_return_bps"] = ((ref_close - ref_open) / ref_open * 1e4).fillna(0.0)
    ref["_ref_microprice_return_bps_lag1"] = ref["_ref_microprice_return_bps"].shift(1).fillna(0.0)
    ref["_ref_ofi_sum"] = ref["ofi_sum"].astype(float)
    ref["_ref_microprice_high_low_bps"] = (
        (ref_high - ref_low) / ref_open * 1e4
    ).fillna(0.0)
    ref["_ref_depth_imb_now"] = ref["depth_imb_last"].astype(float)

    ref_keys = ref[[
        "minute",
        "_ref_microprice_return_bps",
        "_ref_microprice_return_bps_lag1",
        "_ref_ofi_sum",
        "_ref_microprice_high_low_bps",
        "_ref_depth_imb_now",
    ]].rename(columns={
        "_ref_microprice_return_bps": "ref_microprice_return_bps",
        "_ref_microprice_return_bps_lag1": "ref_microprice_return_bps_lag1",
        "_ref_ofi_sum": "ref_ofi_sum",
        "_ref_microprice_high_low_bps": "ref_microprice_high_low_bps",
        "_ref_depth_imb_now": "ref_depth_imb_now",
    })

    # Merge on `minute` — left join on the target frame so we keep ALL
    # target rows. NaN rows (target bar present but reference missing
    # at that minute) get zeros via fillna.
    target_minute = minute_df.sort_values(["symbol", "minute"]).reset_index(drop=True)["minute"]
    target_minute = target_minute.reset_index(drop=True)
    target_df = pd.DataFrame({"minute": target_minute})

    joined = target_df.merge(ref_keys, on="minute", how="left")
    for col in CROSS_ASSET_FEATURE_COLUMNS:
        feats[col] = joined[col].fillna(0.0).astype(float).values
    return feats


def build_cross_asset_features(
    minute_df: pd.DataFrame,
    *,
    reference_minutes: pd.DataFrame | None = None,
    reference_symbol: str | None = None,
) -> pd.DataFrame:
    """Build the enriched feature matrix.

    Parameters
    ----------
    minute_df
        Per-bar frame from ``aggregate_to_minute(...)``. The TARGET
        symbol's bars (the model being trained / served).
    reference_minutes
        Optional second per-bar frame containing the REFERENCE symbol's
        bars. Same column schema. If ``reference_symbol`` is None this
        is ignored.
    reference_symbol
        Symbol whose features to pipe in (typically ``"BTCUSDT"`` for
        ETH/BNB models). None ⇒ no cross-asset features added (used
        for BTC's own model — pointing at itself is identity).

    Returns
    -------
    pd.DataFrame
        Feature matrix in the order returned by ``feature_columns_for(
        reference_symbol)``.
    """
    expected_cols = feature_columns_for(reference_symbol)
    if minute_df.empty:
        return pd.DataFrame(columns=expected_cols)

    # Base 18 features — reuse existing pipeline.
    base = build_base_features(minute_df)

    # Lagged 4 features (always added).
    enriched = _add_lagged_features(base, minute_df)

    # Cross-asset 5 features (only when reference given).
    if reference_symbol and reference_minutes is not None:
        enriched = _add_cross_asset_features(
            enriched, minute_df, reference_minutes, reference_symbol,
        )

    # Reorder to canonical — defends against silent column reordering.
    enriched = enriched[expected_cols]
    return enriched.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_latest_inference_bar_cross_asset(
    df_seconds: pd.DataFrame,
    *,
    bar_minutes: int,
    reference_df_seconds: pd.DataFrame | None,
    reference_symbol: str | None,
) -> tuple[pd.Series, float] | None:
    """Cross-asset counterpart to
    :func:`feature_pipeline.build_latest_inference_bar`.

    Aggregates the target seconds frame at ``bar_minutes``, drops the
    in-flight current bar, then computes base + lagged + (optional)
    cross-asset features using the same ``build_cross_asset_features``
    pipeline the trainer uses. Reference seconds (typically BTC for
    ETH/BNB) are aggregated at the same bar size and joined by
    ``minute`` timestamps inside the pipeline.

    Returns ``(features, close_microprice)`` for the most recent
    COMPLETE bar, or ``None`` at cold-start / insufficient data.

    The lagged features need at least 4 complete bars (lag1/2/3 + the
    current bar). Cross-asset features need at least 1 reference bar
    aligned with the target bar. If reference data is missing for a
    given target bar, those columns get filled with 0.0 inside
    ``build_cross_asset_features`` — the model still produces a
    prediction, just with reduced signal on that bar.
    """
    if df_seconds.empty or bar_minutes <= 0:
        return None

    df = df_seconds.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if df["ts"].empty:
        return None

    # Drop the in-flight bar (mirrors the other helpers).
    bar_freq = f"{int(bar_minutes)}min"
    now_floor = df["ts"].max().floor(bar_freq)
    df = df.loc[df["ts"] < now_floor].copy()
    if df.empty:
        return None

    from app.highfreq.feature_pipeline import aggregate_to_minute

    minute_df = aggregate_to_minute(df, bar_minutes=bar_minutes)
    # Need ≥ 4 bars so lag3 is defined for the LAST row.
    if len(minute_df) < 4:
        return None

    ref_minutes: pd.DataFrame | None = None
    if reference_symbol and reference_df_seconds is not None and not reference_df_seconds.empty:
        ref = reference_df_seconds.copy()
        ref["ts"] = pd.to_datetime(ref["ts"], utc=True)
        # Align reference to the same wall-clock cutoff so we don't
        # join in-flight bars from BTC into a complete ETH bar.
        ref = ref.loc[ref["ts"] < now_floor].copy()
        if not ref.empty:
            ref_minutes = aggregate_to_minute(ref, bar_minutes=bar_minutes)

    feats_full = build_cross_asset_features(
        minute_df,
        reference_minutes=ref_minutes,
        reference_symbol=reference_symbol,
    )
    if feats_full.empty:
        return None

    last_close = float(minute_df.iloc[-1]["microprice_close"])
    return feats_full.iloc[-1], last_close
