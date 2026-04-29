"""Long-horizon feature pipeline — OHLC + classical TA, NOT microstructure.

Why a separate pipeline
=======================

Release K's multi-horizon eval showed a clear empirical result: the
14-feature microstructure set (OFI / depth_imb / spread / trade_imb)
gives **dir_acc 0.56-0.58 at 1-minute horizon** but **drops to chance
(0.45-0.50) at 5/15/60-minute horizons**.

Reason: microstructure features capture imbalances at **second-to-minute
timescale**. By the time a 5-minute bar closes, the OFI signal from
its first minute is already 4 minutes stale and has nothing to do
with the next 5-minute return.

Long-horizon prediction needs a fundamentally different feature class:

* **OHLC bar shape** — open, high, low, close, range, body ratio,
  upper / lower wick. Encodes "what kind of bar was this?" (bullish
  engulfing, doji, hammer, etc.).
* **Trend / momentum indicators** — EMA crossovers, ROC, MACD signal.
  Capture multi-bar directional bias that 1-bar microstructure misses.
* **Volatility regime** — Bollinger band z-score, rolling ATR ratio.
  Conditions the model on whether we're in a calm vs trending vs
  volatile regime.
* **Mean-reversion proxies** — RSI(14), distance-from-EMA in std-units.
  Captures "stretched" conditions where price tends to mean-revert.
* **Cross-bar momentum** — return autocorrelation across last N bars.
  Distinguishes momentum regime (positive autocorr) from mean-reverting
  regime (negative autocorr).

Calendar features (release α) carry over identically.

This is the canonical "TA features" set that classical retail trading
strategies are built on. Honest empirical hypothesis: at 15m+ horizon,
TA-derived signals carry information the microstructure set misses.
The eval tool tests this hypothesis quantitatively.

Same contract as feature_pipeline.build_features
------------------------------------------------

Pure function — takes the per-bar DataFrame from
``feature_pipeline.aggregate_to_minute(..., bar_minutes=N)`` and
returns a feature matrix with the column order in
:data:`LONG_HORIZON_FEATURE_COLUMNS`.

Output shape: 1 row per input bar, columns exactly the constant.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Order matters — fits + serves both reorder via this list, so a
# mismatch anywhere becomes a model-version break (intentional).
LONG_HORIZON_FEATURE_COLUMNS: list[str] = [
    # === OHLC bar shape (8) ===
    "bar_return_bps",        # close / open - 1, in bps
    "bar_range_bps",          # (high - low) / open, in bps
    "bar_body_ratio",         # |close - open| / (high - low + eps); 0..1
    "bar_upper_wick_ratio",   # (high - max(open, close)) / range
    "bar_lower_wick_ratio",   # (min(open, close) - low) / range
    "bar_close_to_high",      # (high - close) / range; 0 = closed at high
    "bar_close_to_low",       # (close - low) / range; 0 = closed at low
    "bar_volume_proxy",       # n_updates_sum as activity proxy

    # === Multi-bar momentum / trend (6) ===
    "ema_5_dist_bps",         # close vs EMA(5), in bps
    "ema_20_dist_bps",        # close vs EMA(20), in bps
    "ema_5_minus_20_bps",     # EMA(5) − EMA(20), in bps  (MACD signal proxy)
    "roc_3_bps",              # 3-bar rate of change in bps
    "roc_10_bps",              # 10-bar ROC in bps
    "return_autocorr_5",      # corr of last-5 returns vs lag-1; momentum vs MR

    # === Volatility regime (4) ===
    "bb_z_20",                # z-score vs 20-bar Bollinger band
    "atr_ratio_5_20",         # ATR(5) / ATR(20); high = trending burst
    "vol_pct_rank_50",        # percentile rank of |return| over last 50 bars
    "spread_bps_mean",        # carry from microstructure — still useful

    # === Mean-reversion (2) ===
    "rsi_14",                 # classic RSI 14
    "rsi_extreme_flag",       # 1 if rsi>70 or <30, else 0

    # === Calendar (4, mirror release α) ===
    "hour_of_day",
    "minute_of_hour",
    "day_of_week",
    "hour_of_week",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average — standard formulation."""
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _atr(highs: pd.Series, lows: pd.Series, closes: pd.Series,
         period: int) -> pd.Series:
    """Average True Range — Wilder's smoothing.

    TR = max(high - low, |high - prev_close|, |low - prev_close|).
    ATR = exponential moving average of TR with span ``period``.
    """
    prev_close = closes.shift(1)
    tr = pd.concat([
        (highs - lows).abs(),
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI(14). Standard textbook formula."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)  # neutral when no data


def build_long_horizon_features(df_bars: pd.DataFrame) -> pd.DataFrame:
    """Project a per-bar frame (output of ``aggregate_to_minute(...,
    bar_minutes=N)``) into the long-horizon feature matrix.

    Inputs columns expected (mirror feature_pipeline.build_features):
        symbol, minute, microprice_open, microprice_close,
        microprice_high, microprice_low, n_updates_sum, spread_bps_mean.

    Calendar derived from `minute` column. Indicators computed
    per-symbol (groupby) so a multi-symbol DataFrame doesn't bleed
    BTC's price levels into ETH's RSI etc.
    """
    if df_bars.empty:
        return pd.DataFrame(columns=LONG_HORIZON_FEATURE_COLUMNS)

    eps = 1e-12
    df = df_bars.copy().sort_values(["symbol", "minute"]).reset_index(drop=True)

    o = df["microprice_open"].astype(float)
    h = df["microprice_high"].astype(float)
    l = df["microprice_low"].astype(float)
    c = df["microprice_close"].astype(float)
    rng = (h - l).clip(lower=eps)
    body = (c - o).abs()

    bar_ret_bps = ((c - o) / o.replace(0, np.nan) * 1e4).fillna(0.0)
    bar_range_bps = (rng / o.replace(0, np.nan) * 1e4).fillna(0.0)
    bar_body_ratio = (body / rng).clip(0.0, 1.0)
    bar_upper_wick_ratio = ((h - pd.concat([o, c], axis=1).max(axis=1)) / rng).clip(0.0, 1.0)
    bar_lower_wick_ratio = ((pd.concat([o, c], axis=1).min(axis=1) - l) / rng).clip(0.0, 1.0)
    bar_close_to_high = ((h - c) / rng).clip(0.0, 1.0)
    bar_close_to_low = ((c - l) / rng).clip(0.0, 1.0)

    # Per-symbol multi-bar indicators.
    out_rows = []
    for sym, g in df.groupby("symbol", sort=False):
        gc = g["microprice_close"].astype(float)
        gh = g["microprice_high"].astype(float)
        gl = g["microprice_low"].astype(float)
        go = g["microprice_open"].astype(float)

        ema5 = _ema(gc, 5)
        ema20 = _ema(gc, 20)
        atr5 = _atr(gh, gl, gc, 5)
        atr20 = _atr(gh, gl, gc, 20)

        ema5_dist_bps = ((gc - ema5) / gc.replace(0, np.nan) * 1e4).fillna(0.0)
        ema20_dist_bps = ((gc - ema20) / gc.replace(0, np.nan) * 1e4).fillna(0.0)
        ema5_minus_20_bps = ((ema5 - ema20) / gc.replace(0, np.nan) * 1e4).fillna(0.0)

        roc3 = ((gc / gc.shift(3) - 1) * 1e4).fillna(0.0)
        roc10 = ((gc / gc.shift(10) - 1) * 1e4).fillna(0.0)

        # Return autocorrelation: rolling correlation of returns vs
        # lagged returns. High positive = momentum regime, negative =
        # mean-reverting.
        bar_returns = gc.pct_change()
        autocorr5 = (
            bar_returns.rolling(5)
            .apply(lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 3 else 0.0, raw=False)
            .fillna(0.0)
        )

        # Bollinger z-score (20-bar mean ± 2σ).
        roll_mean = gc.rolling(20, min_periods=2).mean()
        roll_std = gc.rolling(20, min_periods=2).std().replace(0, np.nan)
        bb_z = ((gc - roll_mean) / roll_std).fillna(0.0)

        atr_ratio = (atr5 / atr20.replace(0, np.nan)).fillna(1.0)

        # Vol percentile rank of |return| over last 50 bars.
        abs_ret = bar_returns.abs()
        vol_rank = (
            abs_ret.rolling(50, min_periods=5)
            .apply(lambda x: (x.iloc[-1] > x).mean() if len(x) > 0 else 0.5, raw=False)
            .fillna(0.5)
        )

        rsi = _rsi(gc, 14)
        rsi_extreme = ((rsi > 70) | (rsi < 30)).astype(float)

        sym_feat = pd.DataFrame({
            "ema_5_dist_bps": ema5_dist_bps,
            "ema_20_dist_bps": ema20_dist_bps,
            "ema_5_minus_20_bps": ema5_minus_20_bps,
            "roc_3_bps": roc3,
            "roc_10_bps": roc10,
            "return_autocorr_5": autocorr5,
            "bb_z_20": bb_z,
            "atr_ratio_5_20": atr_ratio,
            "vol_pct_rank_50": vol_rank,
            "rsi_14": rsi,
            "rsi_extreme_flag": rsi_extreme,
        }, index=g.index)
        out_rows.append(sym_feat)
    multi_bar = pd.concat(out_rows).sort_index()

    # Calendar (mirror α).
    minute_ts = pd.to_datetime(df["minute"], utc=True)
    hod = minute_ts.dt.hour.astype(float).values
    mof = minute_ts.dt.minute.astype(float).values
    dow = minute_ts.dt.dayofweek.astype(float).values
    how = (minute_ts.dt.dayofweek * 24 + minute_ts.dt.hour).astype(float).values

    feats = pd.DataFrame({
        "bar_return_bps": bar_ret_bps.values,
        "bar_range_bps": bar_range_bps.values,
        "bar_body_ratio": bar_body_ratio.values,
        "bar_upper_wick_ratio": bar_upper_wick_ratio.values,
        "bar_lower_wick_ratio": bar_lower_wick_ratio.values,
        "bar_close_to_high": bar_close_to_high.values,
        "bar_close_to_low": bar_close_to_low.values,
        "bar_volume_proxy": df["n_updates_sum"].astype(float).values,
        "ema_5_dist_bps": multi_bar["ema_5_dist_bps"].values,
        "ema_20_dist_bps": multi_bar["ema_20_dist_bps"].values,
        "ema_5_minus_20_bps": multi_bar["ema_5_minus_20_bps"].values,
        "roc_3_bps": multi_bar["roc_3_bps"].values,
        "roc_10_bps": multi_bar["roc_10_bps"].values,
        "return_autocorr_5": multi_bar["return_autocorr_5"].values,
        "bb_z_20": multi_bar["bb_z_20"].values,
        "atr_ratio_5_20": multi_bar["atr_ratio_5_20"].values,
        "vol_pct_rank_50": multi_bar["vol_pct_rank_50"].values,
        "spread_bps_mean": df["spread_bps_mean"].astype(float).values,
        "rsi_14": multi_bar["rsi_14"].values,
        "rsi_extreme_flag": multi_bar["rsi_extreme_flag"].values,
        "hour_of_day": hod,
        "minute_of_hour": mof,
        "day_of_week": dow,
        "hour_of_week": how,
    })

    # Final scrub for inf / NaN — CatBoost handles missing but explicit
    # is cleaner.
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ────────────────────────────────────────────────────────────────────────────
# Live-inference helper (Phase B+ multi-horizon)
# ────────────────────────────────────────────────────────────────────────────


def build_latest_inference_bar_long_horizon(
    df_seconds: pd.DataFrame, *, bar_minutes: int,
) -> tuple[pd.Series, float] | None:
    """Long-horizon counterpart to
    :func:`feature_pipeline.build_latest_inference_bar`.

    Aggregates the seconds frame at the requested ``bar_minutes`` (e.g.
    15 / 60), drops the in-flight current bar, then computes the
    long-horizon TA feature row + the closing microprice of that bar.

    Same "complete bar only" semantics as the 1-minute helper: the
    trainer fits on whole-bar aggregates, so serving on a partial bar
    means feeding out-of-distribution inputs.

    Returns ``(features, close_microprice)`` for the most recent
    COMPLETE bar of the requested size, or ``None`` at cold-start /
    insufficient data.

    Parameters
    ----------
    df_seconds
        Per-second OFI frame (same schema as the L2 ingest writes).
    bar_minutes
        Aggregation horizon. Must be ≥ 1.

    Notes
    -----
    The long-horizon pipeline needs MORE history than 1m to bootstrap:
    EMA(20) wants 20 bars of context, RSI(14) wants 14 bars of returns,
    Bollinger z-score wants 20 bars. Returning a feature row from a
    fresh aggregate with only one bar would produce zero-valued
    indicators that the model wasn't trained on. We let the caller
    pass enough seconds-history to produce ≥ 20 complete bars; if not
    enough, return ``None`` and the runner skips the tick.
    """
    if df_seconds.empty or bar_minutes <= 0:
        return None

    df = df_seconds.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if df["ts"].empty:
        return None

    # Drop the in-flight bar (mirrors the 1m helper). For a 15m bar at
    # wall-clock 12:37 UTC, the most recent complete bar floor is 12:30,
    # the in-flight one is also 12:30 (ends at 12:45) and gets dropped.
    bar_freq = f"{int(bar_minutes)}min"
    now_floor = df["ts"].max().floor(bar_freq)
    df = df.loc[df["ts"] < now_floor].copy()
    if df.empty:
        return None

    from app.highfreq.feature_pipeline import aggregate_to_minute

    minute_df = aggregate_to_minute(df, bar_minutes=bar_minutes)
    # Need enough bars to bootstrap the EMA(20) / Bollinger(20) /
    # RSI(14) indicators. Fewer than 20 → indicators ill-defined.
    if len(minute_df) < 20:
        return None

    feats_full = build_long_horizon_features(minute_df)
    if feats_full.empty:
        return None

    last_close = float(minute_df.iloc[-1]["microprice_close"])
    return feats_full.iloc[-1], last_close
