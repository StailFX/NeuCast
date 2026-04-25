"""Shared feature-construction pipeline — single source of truth for both
train-time (``app.highfreq.trainer``) and inference-time
(``app.highfreq.predictor``).

Why a separate module
=====================

The most insidious bug in production ML is **feature drift between train
and serve** — when the trainer aggregates `microprice_close` as the last
value in a 1-min window but the live predictor uses the mean (because
someone reimplemented it slightly differently 6 months later), every
forecast is silently wrong and the only signal is degraded P&L over weeks.

Putting `build_features`, `build_target`, `aggregate_to_minute` and the
column list (``FEATURE_COLUMNS``) here means:

* the trainer fits a CatBoost on output of ``build_features``;
* the live predictor scores rows of ``build_features``;
* refactors that change feature semantics MUST go through this file —
  reviewers see one diff that affects both sides.

Constants live here too (``HORIZON_MIN``, ``NEUTRAL_BAND_BPS``,
``MIN_SECONDS_PER_MINUTE``) — same reasoning. Trainer's neutral band ≠
predictor's neutral band would mean the predictor serves on inputs the
trainer would have dropped.

Pure functions only
-------------------

This module has **zero side effects** — no logging side-effects beyond
INFO messages, no DB, no model loading, no I/O. Easy to unit-test, easy
to reason about. Ingest is in ``aggregator``, training-loop logic is in
``trainer``, model-serving is in ``predictor``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Configuration constants (shared across train/serve)
# ────────────────────────────────────────────────────────────────────────────

#: Forecast horizon in minutes. We predict ``sign(microprice_close[t+H] -
#: microprice_close[t])``. Sub-minute on Binance Spot is dominated by
#: tick noise; OFI signal-to-noise improves with aggregation
#: (Cont-Kukanov-Stoikov 2014, §5). See ADR-003.
HORIZON_MIN: int = 1

#: Bars where ``|return_bps| < NEUTRAL_BAND_BPS`` are dropped from the
#: training set and from the predictor's "actionable" output. 1 bp ≈
#: $7.7 on BTC at $77k — about the typical Binance Spot spread; below
#: this any "direction" is hard to monetise after fees regardless of
#: model quality. ADR-004.
NEUTRAL_BAND_BPS: float = 1.0

#: A 1-minute bar is kept only if at least this many of its 60 seconds
#: were observed. Coverage gaps usually mean a WS reconnect mid-minute
#: and the partial bar would inject noise (the OFI sum is not a good
#: estimator on a partial window).
MIN_SECONDS_PER_MINUTE: int = 30

#: Feature columns in the order the trainer fits and the predictor
#: serves. Changing this list IS a model-incompatibility break — bump
#: the model version and retrain.
FEATURE_COLUMNS: list[str] = [
    # 60-second window stats (the bar itself).
    "ofi_sum", "ofi_mean", "ofi_std",
    "depth_imb_mean", "depth_imb_std",
    "spread_bps_mean", "spread_bps_max",
    "trade_imb_sum", "trade_imb_abs_sum",
    "microprice_return_bps", "microprice_high_low_bps",
    "n_updates_sum",
    # Cross-scale / state features (current vs windowed).
    "spread_bps_now_to_mean",
    "depth_imb_now",
]


# ────────────────────────────────────────────────────────────────────────────
# Pure data transforms
# ────────────────────────────────────────────────────────────────────────────


def aggregate_to_minute(df_seconds: pd.DataFrame) -> pd.DataFrame:
    """Roll up 1-second rows into 1-minute bars.

    Parameters
    ----------
    df_seconds
        DataFrame with columns matching ``highfreq_ofi_1s``: ``ts``
        (datetime64[ns, UTC]), ``symbol``, ``ofi``, ``microprice``,
        ``depth_imb``, ``spread_bps``, ``trade_imb``, ``n_updates``.

    Returns
    -------
    pd.DataFrame
        Indexed by minute-floor ``ts`` with columns sufficient for
        :func:`build_features` and :func:`build_target`. Bars with
        fewer than :data:`MIN_SECONDS_PER_MINUTE` covered seconds are
        dropped (logged at INFO).
    """
    if df_seconds.empty:
        return df_seconds.copy()

    df = df_seconds.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["minute"] = df["ts"].dt.floor("1min")

    # Aggregate. ``microprice_open`` and ``microprice_close`` are picked
    # by event-time order — sort_values enforces that within each group.
    df = df.sort_values("ts")
    grouped = df.groupby(["symbol", "minute"], sort=True)

    out = grouped.agg(
        seconds_observed=("ts", "count"),
        ofi_sum=("ofi", "sum"),
        ofi_mean=("ofi", "mean"),
        ofi_std=("ofi", "std"),
        microprice_open=("microprice", "first"),
        microprice_close=("microprice", "last"),
        microprice_high=("microprice", "max"),
        microprice_low=("microprice", "min"),
        depth_imb_mean=("depth_imb", "mean"),
        depth_imb_std=("depth_imb", "std"),
        depth_imb_last=("depth_imb", "last"),
        spread_bps_mean=("spread_bps", "mean"),
        spread_bps_last=("spread_bps", "last"),
        spread_bps_max=("spread_bps", "max"),
        trade_imb_sum=("trade_imb", "sum"),
        trade_imb_abs_sum=("trade_imb", lambda s: float(np.sum(np.abs(s)))),
        n_updates_sum=("n_updates", "sum"),
    ).reset_index()

    # Discard bars with poor coverage (likely WS drop or restart).
    keep = out["seconds_observed"] >= MIN_SECONDS_PER_MINUTE
    dropped = (~keep).sum()
    if dropped > 0:
        logger.info(
            "aggregate_to_minute: dropped %d bars with <%d seconds coverage",
            dropped, MIN_SECONDS_PER_MINUTE,
        )
    return out.loc[keep].reset_index(drop=True)


def build_target(
    df_minutes: pd.DataFrame,
    *,
    horizon: int = HORIZON_MIN,
    neutral_band_bps: float = NEUTRAL_BAND_BPS,
) -> pd.DataFrame:
    """Append ``return_bps`` and binary ``y`` columns to a per-minute frame.

    The target for bar :math:`t` is

    .. math::

        y_t = \\mathbb{1}\\!\\left[
            \\frac{P^{\\text{micro}}_{t+H} - P^{\\text{micro}}_{t}}{P^{\\text{micro}}_{t}}
            \\cdot 10^4 > 0
        \\right]

    where :math:`P^{\\text{micro}}` is the **closing** microprice of the bar.
    Bars with absolute return below ``neutral_band_bps`` are flagged
    ``in_neutral_band = True`` so the caller can drop them.

    Returns a copy of ``df_minutes`` with these added columns:

    * ``return_bps`` (float) — forward return in basis points
    * ``y`` (int8) — 1 if return > 0, 0 otherwise; sentinel ``-1`` if
      the future bar is missing (last ``horizon`` rows of the frame)
    * ``in_neutral_band`` (bool)
    """
    if df_minutes.empty:
        out = df_minutes.copy()
        out["return_bps"] = pd.Series(dtype=float)
        out["y"] = pd.Series(dtype="int8")
        out["in_neutral_band"] = pd.Series(dtype=bool)
        return out

    out = df_minutes.copy().sort_values(["symbol", "minute"]).reset_index(drop=True)
    # Forward microprice within the same symbol; -horizon shift.
    out["microprice_future"] = (
        out.groupby("symbol")["microprice_close"].shift(-horizon)
    )
    # bps return.
    out["return_bps"] = (
        (out["microprice_future"] - out["microprice_close"])
        / out["microprice_close"]
        * 1e4
    )
    out["in_neutral_band"] = out["return_bps"].abs() < neutral_band_bps
    out["y"] = np.where(
        out["return_bps"].isna(),
        -1,
        (out["return_bps"] > 0).astype(np.int8),
    ).astype(np.int8)
    return out


def build_features(df_minutes: pd.DataFrame) -> pd.DataFrame:
    """Project the per-minute frame into the model's feature matrix.

    Returns a DataFrame with one row per input bar and exactly the
    columns listed in :data:`FEATURE_COLUMNS`. Indexed positionally —
    pair with the input frame's index for joining back to (symbol, minute).
    """
    if df_minutes.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = df_minutes.copy()
    # microprice return inside the bar (open → close), in bps.
    mp_ret_bps = (
        (df["microprice_close"] - df["microprice_open"])
        / df["microprice_open"]
        * 1e4
    )
    mp_hl_bps = (
        (df["microprice_high"] - df["microprice_low"])
        / df["microprice_open"]
        * 1e4
    )
    # Spread stress: how does the most recent spread compare to its
    # 60-s mean within this bar? >1 means liquidity is thinner *now*.
    spread_now_to_mean = np.where(
        df["spread_bps_mean"] > 0,
        df["spread_bps_last"] / df["spread_bps_mean"],
        1.0,
    )

    feats = pd.DataFrame({
        "ofi_sum": df["ofi_sum"].astype(float),
        "ofi_mean": df["ofi_mean"].astype(float),
        "ofi_std": df["ofi_std"].fillna(0.0).astype(float),
        "depth_imb_mean": df["depth_imb_mean"].astype(float),
        "depth_imb_std": df["depth_imb_std"].fillna(0.0).astype(float),
        "spread_bps_mean": df["spread_bps_mean"].astype(float),
        "spread_bps_max": df["spread_bps_max"].astype(float),
        "trade_imb_sum": df["trade_imb_sum"].astype(float),
        "trade_imb_abs_sum": df["trade_imb_abs_sum"].astype(float),
        "microprice_return_bps": mp_ret_bps.astype(float),
        "microprice_high_low_bps": mp_hl_bps.astype(float),
        "n_updates_sum": df["n_updates_sum"].astype(float),
        "spread_bps_now_to_mean": pd.Series(spread_now_to_mean, index=df.index, dtype=float),
        "depth_imb_now": df["depth_imb_last"].astype(float),
    }, index=df.index)

    # Replace any residual NaN/inf — CatBoost handles missing but we
    # prefer to be explicit so feature distributions are interpretable.
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_supervised(
    df_seconds: pd.DataFrame,
    *,
    horizon: int = HORIZON_MIN,
    neutral_band_bps: float = NEUTRAL_BAND_BPS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convenience: 1-second rows → ``(X, y, meta)`` ready for training.

    ``meta`` carries ``symbol``, ``minute``, ``microprice_close``,
    ``return_bps`` so the caller can join predictions back for
    sim-backtest P&L computation in :mod:`app.highfreq.backtest`.

    Bars with missing future returns (the last ``horizon`` rows) and
    bars in the neutral band are dropped — the trainer does not learn
    on noise.
    """
    minute_df = aggregate_to_minute(df_seconds)
    targeted = build_target(
        minute_df, horizon=horizon, neutral_band_bps=neutral_band_bps
    )
    # Drop unobservable / noise bars.
    keep = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
    targeted = targeted.loc[keep].reset_index(drop=True)

    X = build_features(targeted)
    y = targeted["y"].astype(np.int8)
    meta = targeted[["symbol", "minute", "microprice_close", "return_bps"]].copy()
    return X, y, meta


# ────────────────────────────────────────────────────────────────────────────
# Live-inference helper (Phase B)
# ────────────────────────────────────────────────────────────────────────────


def build_latest_feature_row(df_seconds: pd.DataFrame) -> pd.Series | None:
    """Build a single feature row for **live inference** at time of call.

    Use case: predictor wants to score "the market right now". Call this
    with the most recent ~120 s of 1-second rows from
    ``highfreq_ofi_1s``; it aggregates to minute, drops any partial
    current minute, and returns the most recent COMPLETE minute's
    feature vector.

    Returns ``None`` when there isn't a single complete bar (typical at
    cold-start or right after a reconnect).

    Why "complete minute" semantics
    -------------------------------

    The trainer fits on whole-minute aggregates: ``microprice_close`` is
    the last sample of the minute, ``ofi_sum`` is summed across the full
    60 s window. Serving on a partial bar (say, only 23 seconds of the
    current minute observed so far) means the predictor sees a feature
    distribution it was never trained on — features look smaller, OFI
    sums under-represent flow, the model output is unreliable. We only
    serve on whole minutes; the forecast updates once per minute.
    """
    if df_seconds.empty:
        return None

    df = df_seconds.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if df["ts"].empty:
        return None

    # Drop any rows that fall in the *current* minute (incomplete by
    # definition — even if `now` happens to be exactly at :00 the bar
    # still hasn't seen its full 60s).
    now_minute = df["ts"].max().floor("1min")
    df = df.loc[df["ts"] < now_minute].copy()
    if df.empty:
        return None

    minute_df = aggregate_to_minute(df)
    if minute_df.empty:
        return None

    feats = build_features(minute_df)
    if feats.empty:
        return None

    # Most recent complete-minute feature row.
    return feats.iloc[-1]
