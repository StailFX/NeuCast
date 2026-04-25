"""Walk-forward CatBoost trainer for the 1-minute directional model.

Pipeline (each step is a pure function except :func:`load_seconds`)::

    load_seconds()      ─┐
                         ▼
    aggregate_to_minute() ──► build_target() + build_features()
                         ▼
            walk_forward_evaluate()  ──► (per-fold predictions)
                         ▼
              evaluate()  ──► dir_acc, bootstrap CI, F1, log_loss
                         ▼
              fit_final_model()  ──► weights/highfreq/<symbol>_1m.cbm

Design choices
--------------

The decisions below are driven by the constraints in
``docs/highfreq/architecture.md`` and the ADRs cited inline.

* **1-minute horizon, not 1-second.** Sub-minute crypto returns on
  Binance Spot are dominated by tick noise; OFI signal-to-noise improves
  with aggregation (Cont-Kukanov-Stoikov 2014, §5). See ADR-003.
* **Binary classification with neutral-band drop.** Bars where
  ``|return_1m| < neutral_band_bps`` are removed before training and
  evaluation — they're noise and a 50/50 classifier on noise is
  uninformative for paper P&L. See ADR-004.
* **Expanding-window walk-forward, not random k-fold.** Time-series
  leakage is the single most common silent killer in financial ML.
  We never let the model see a future bar. Initial train window is
  configurable (default 24 h); we step forward in 1 h test folds.
* **Bootstrap CI on dir_acc.** A 52 % point estimate on 200 samples is
  meaningless; we report the 95 % bootstrap CI and stamp a
  ``low_directional_skill`` flag if the lower bound ≤ 50 %. Same pattern
  used in ``app.prediction.run_prediction`` for the daily model.
* **CatBoost, not LightGBM/XGBoost.** Already in requirements, handles
  tabular features without preprocessing, robust to outliers, fast on
  CPU. ``thread_count=2`` per ADR-006 (coexist with main app on shared
  VPS).

Run from the CLI::

    python -m app.highfreq.trainer --symbol BTCUSDT --since-hours 48 \\
        --out weights/highfreq/btcusdt_1m.cbm

Run as a Celery beat task at 04:00 UTC daily — see Phase A.5.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

#: Forecast horizon. We predict the sign of the microprice change between
#: bar t and bar t+HORIZON_MIN. 1 minute is the sweet spot (ADR-003).
HORIZON_MIN: int = 1

#: Bars where ``|return_1m| < NEUTRAL_BAND_BPS`` are dropped before
#: training and evaluation. 1 bp ≈ $7.7 on BTC at $77k — about the
#: typical Binance Spot spread, i.e. below this threshold any "direction"
#: is hard to monetise after fees regardless of model quality. ADR-004.
NEUTRAL_BAND_BPS: float = 1.0

#: Minimum number of seconds inside a 1-minute bucket for it to be
#: considered observed. With @100ms cadence we expect ~600 frames/min,
#: so requiring ≥ 30 seconds of coverage discards bars where the
#: WebSocket dropped or the consumer was reconnecting.
MIN_SECONDS_PER_MINUTE: int = 30

#: Feature columns produced by :func:`build_features`. Exposed as a
#: module constant so :mod:`app.highfreq.predictor` can validate the
#: feature schema at inference time.
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
# Pure data transforms (testable)
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
        dropped.
    """
    if df_seconds.empty:
        return df_seconds.copy()

    df = df_seconds.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["minute"] = df["ts"].dt.floor("1min")

    grouped = df.groupby(["symbol", "minute"], sort=True)

    # Aggregate. ``microprice_open`` and ``microprice_close`` are picked
    # by event-time order (df is assumed sorted by ts within each group;
    # we enforce that via the sort_values below if not).
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
    * ``y`` (int8) — 1 if return > 0, 0 otherwise; NaN-equivalent
      ``-1`` if the future bar is missing
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
    """Convenience: 1-second rows → (X, y, meta) ready for training.

    ``meta`` carries ``symbol``, ``minute``, ``microprice_close``,
    ``return_bps`` so the caller can join predictions back for
    sim-backtest P&L computation in :mod:`app.highfreq.backtest`.

    Bars with missing future returns (the last ``horizon`` rows) and
    bars in the neutral band are dropped.
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
# Walk-forward evaluation
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WalkForwardConfig:
    """Tunable knobs for the walk-forward CV loop.

    Defaults assume ~7 days of accumulated data. With less, the loop
    will produce fewer folds — :func:`walk_forward_evaluate` skips
    cleanly if there isn't enough data to satisfy ``initial_train_minutes``.
    """

    initial_train_minutes: int = 24 * 60     # 1 day warm-up
    test_fold_minutes: int = 60              # 1 hour per test fold
    step_minutes: int = 60                   # advance by 1 hour
    min_train_samples: int = 200             # don't fit on toy folds
    catboost_iterations: int = 400
    catboost_depth: int = 5
    catboost_learning_rate: float = 0.05
    catboost_thread_count: int = 2           # ADR-006
    random_seed: int = 42


@dataclass
class FoldReport:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    dir_acc: float
    log_loss: float
    n_pos: int                 # n positives in test fold
    base_rate: float           # max(P(y=1), P(y=0)) — beat this to be useful


def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    config: WalkForwardConfig | None = None,
) -> tuple[list[FoldReport], pd.DataFrame]:
    """Expanding-window walk-forward CV.

    Returns
    -------
    folds
        Per-fold metrics — useful for plotting `dir_acc` over time.
    predictions
        DataFrame with columns ``minute, y_true, proba, y_pred, fold_idx``
        — one row per out-of-sample test sample, in chronological order.
        Pair with ``meta`` (joined on minute) for paper P&L computation.
    """
    cfg = config or WalkForwardConfig()
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "catboost is required for walk_forward_evaluate; "
            "pip install catboost>=1.2"
        ) from exc

    if len(X) < cfg.initial_train_minutes + cfg.test_fold_minutes:
        logger.warning(
            "walk_forward_evaluate: only %d minutes available, need ≥%d for one fold",
            len(X), cfg.initial_train_minutes + cfg.test_fold_minutes,
        )
        return [], pd.DataFrame(columns=["minute", "y_true", "proba", "y_pred", "fold_idx"])

    # Index is positional; meta provides the timestamp.
    minutes = pd.to_datetime(meta["minute"], utc=True).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    folds: list[FoldReport] = []
    pred_rows: list[dict[str, Any]] = []

    train_end = cfg.initial_train_minutes
    fold_idx = 0
    while train_end + cfg.test_fold_minutes <= len(X):
        test_end = train_end + cfg.test_fold_minutes

        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if len(X_tr) < cfg.min_train_samples:
            train_end += cfg.step_minutes
            continue

        clf = CatBoostClassifier(
            iterations=cfg.catboost_iterations,
            depth=cfg.catboost_depth,
            learning_rate=cfg.catboost_learning_rate,
            loss_function="Logloss",
            thread_count=cfg.catboost_thread_count,
            random_seed=cfg.random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        clf.fit(X_tr.values, y_tr.values)
        proba = clf.predict_proba(X_te.values)[:, 1]
        y_hat = (proba >= 0.5).astype(np.int8)
        dir_acc = float((y_hat == y_te.values).mean())
        ll = _binary_logloss(y_te.values, proba)
        base = max(float(y_te.mean()), 1.0 - float(y_te.mean()))

        folds.append(FoldReport(
            fold_idx=fold_idx,
            train_start=minutes.iloc[0],
            train_end=minutes.iloc[train_end - 1],
            test_start=minutes.iloc[train_end],
            test_end=minutes.iloc[test_end - 1],
            n_train=int(len(X_tr)),
            n_test=int(len(X_te)),
            dir_acc=dir_acc,
            log_loss=ll,
            n_pos=int(y_te.sum()),
            base_rate=base,
        ))
        for i in range(len(X_te)):
            pred_rows.append({
                "minute": minutes.iloc[train_end + i],
                "y_true": int(y_te.iloc[i]),
                "proba": float(proba[i]),
                "y_pred": int(y_hat[i]),
                "fold_idx": fold_idx,
            })
        train_end += cfg.step_minutes
        fold_idx += 1

    preds_df = pd.DataFrame(pred_rows)
    return folds, preds_df


def _binary_logloss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(proba, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap CI on directional accuracy
# ────────────────────────────────────────────────────────────────────────────

def bootstrap_dir_acc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """95 % bootstrap CI for directional accuracy.

    Same pattern as ``app.prediction.run_prediction`` for the daily
    model. Returns ``(point_estimate, ci_low, ci_high)``.
    """
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float((y_pred == y_true).mean())
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = (y_pred[idx] == y_true[idx]).mean()
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    return point, lo, hi


# ────────────────────────────────────────────────────────────────────────────
# Final-model fit + persistence
# ────────────────────────────────────────────────────────────────────────────

def fit_final_model(
    X: pd.DataFrame, y: pd.Series, *, config: WalkForwardConfig | None = None,
):
    """Fit a model on the full dataset for production inference.

    Returns the fitted ``CatBoostClassifier`` instance. Caller is
    responsible for calling ``.save_model(path, format="cbm")``.
    """
    cfg = config or WalkForwardConfig()
    from catboost import CatBoostClassifier

    clf = CatBoostClassifier(
        iterations=cfg.catboost_iterations,
        depth=cfg.catboost_depth,
        learning_rate=cfg.catboost_learning_rate,
        loss_function="Logloss",
        thread_count=cfg.catboost_thread_count,
        random_seed=cfg.random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    clf.fit(X.values, y.values)
    return clf


# ────────────────────────────────────────────────────────────────────────────
# DB layer (SQLAlchemy, synchronous — async only matters for the WS consumer)
# ────────────────────────────────────────────────────────────────────────────

def load_seconds(
    database_url: str, *, symbol: str, since_hours: float
) -> pd.DataFrame:
    """Read the last ``since_hours`` of 1-second rows for ``symbol``.

    Returns a DataFrame with the same columns as ``highfreq_ofi_1s``.
    """
    from sqlalchemy import create_engine, text  # local import keeps CLI lightweight

    eng = create_engine(database_url, future=True)
    query = text("""
        SELECT ts, symbol, ofi, microprice, depth_imb, spread_bps,
               trade_imb, vpin, n_updates, local_recv_ms
        FROM highfreq_ofi_1s
        WHERE symbol = :symbol
          AND ts >= now() - (:hours * interval '1 hour')
        ORDER BY ts ASC
    """)
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol, "hours": since_hours})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Top-level training run
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingReport:
    """JSON-serialisable summary of one training run."""

    symbol: str
    horizon_min: int
    neutral_band_bps: float
    n_seconds_loaded: int
    n_minutes_after_aggregation: int
    n_minutes_after_neutral_drop: int
    base_rate: float
    n_folds: int
    dir_acc_mean: float
    dir_acc_ci_low: float
    dir_acc_ci_high: float
    log_loss_mean: float
    folds: list[dict[str, Any]] = field(default_factory=list)
    low_directional_skill: bool = True
    weights_path: str | None = None
    elapsed_seconds: float = 0.0

    def to_json(self) -> str:
        """JSON-serialise the report. NaN/Inf are emitted as ``null`` so
        the output is RFC-7159 compliant and downstream UI / dashboards
        can parse it without falling back to a permissive parser."""
        def _scrub(o: Any) -> Any:
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: _scrub(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_scrub(v) for v in o]
            return o
        return json.dumps(_scrub(asdict(self)), indent=2, default=str)


def run_training(
    database_url: str,
    *,
    symbol: str,
    since_hours: float,
    out_path: Path | None,
    config: WalkForwardConfig | None = None,
) -> TrainingReport:
    """Full training pipeline. Returns a ``TrainingReport`` for logging."""
    cfg = config or WalkForwardConfig()
    started = time.monotonic()

    df_secs = load_seconds(database_url, symbol=symbol, since_hours=since_hours)
    logger.info("loaded %d seconds of data for %s", len(df_secs), symbol)

    minute_df = aggregate_to_minute(df_secs)
    n_min = len(minute_df)
    logger.info("aggregated to %d minute bars", n_min)

    X, y, meta = make_supervised(df_secs)
    n_min_kept = len(X)
    logger.info(
        "after target+neutral-band drop: %d bars (%.1f%% kept)",
        n_min_kept, 100.0 * n_min_kept / max(1, n_min),
    )
    base_rate = float(max(y.mean(), 1.0 - y.mean())) if len(y) else float("nan")

    folds, preds = walk_forward_evaluate(X, y, meta, config=cfg)
    if folds:
        dir_acc_arr = np.array([f.dir_acc for f in folds])
        ll_arr = np.array([f.log_loss for f in folds])
        # CI is built from per-prediction outcomes (not per-fold means)
        # so it reflects sample size, not fold count.
        _, ci_lo, ci_hi = bootstrap_dir_acc_ci(
            preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
            seed=cfg.random_seed,
        )
        dir_acc_mean = float(dir_acc_arr.mean())
        ll_mean = float(ll_arr.mean())
    else:
        dir_acc_mean = ll_mean = ci_lo = ci_hi = float("nan")

    weights_path: str | None = None
    if out_path is not None and len(X) >= cfg.min_train_samples:
        clf = fit_final_model(X, y, config=cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save_model(str(out_path), format="cbm")
        weights_path = str(out_path)
        logger.info("saved final model to %s", out_path)

    report = TrainingReport(
        symbol=symbol,
        horizon_min=HORIZON_MIN,
        neutral_band_bps=NEUTRAL_BAND_BPS,
        n_seconds_loaded=len(df_secs),
        n_minutes_after_aggregation=n_min,
        n_minutes_after_neutral_drop=n_min_kept,
        base_rate=base_rate,
        n_folds=len(folds),
        dir_acc_mean=dir_acc_mean,
        dir_acc_ci_low=ci_lo,
        dir_acc_ci_high=ci_hi,
        log_loss_mean=ll_mean,
        folds=[asdict(f) for f in folds],
        # Cautious default: claim "we have skill" only when the 95 % CI
        # lower bound is strictly above chance. NaN (no data) keeps the
        # flag set so the UI doesn't show a misleading green badge.
        low_directional_skill=(math.isnan(ci_lo) or ci_lo <= 0.5),
        weights_path=weights_path,
        elapsed_seconds=time.monotonic() - started,
    )
    return report


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m app.highfreq.trainer",
        description="Walk-forward CatBoost trainer for the 1-min directional model.",
    )
    p.add_argument("--symbol", default="BTCUSDT", help="trading pair (default: BTCUSDT)")
    p.add_argument(
        "--since-hours", type=float, default=72.0,
        help="how many hours of 1-second history to load (default: 72)",
    )
    p.add_argument(
        "--out", default="weights/highfreq/btcusdt_1m.cbm",
        help="path to write the final fitted CatBoost model "
             "(set to '' to skip saving)",
    )
    p.add_argument(
        "--report", default="weights/highfreq/btcusdt_1m_metrics.json",
        help="path to write the JSON metrics report",
    )
    p.add_argument(
        "--initial-train-minutes", type=int, default=24 * 60,
        help="initial train window for walk-forward (default: 1440 = 1 day)",
    )
    p.add_argument(
        "--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is required")
        return 2

    cfg = WalkForwardConfig(initial_train_minutes=args.initial_train_minutes)
    out = Path(args.out) if args.out else None

    report = run_training(
        dsn, symbol=args.symbol, since_hours=args.since_hours,
        out_path=out, config=cfg,
    )

    # Console-friendly summary.
    logger.info(
        "TRAINING DONE | symbol=%s | bars=%d | folds=%d | "
        "dir_acc=%.4f [%.4f, %.4f] | logloss=%.4f | base_rate=%.4f | "
        "low_skill=%s | elapsed=%.1fs",
        report.symbol, report.n_minutes_after_neutral_drop, report.n_folds,
        report.dir_acc_mean, report.dir_acc_ci_low, report.dir_acc_ci_high,
        report.log_loss_mean, report.base_rate,
        report.low_directional_skill, report.elapsed_seconds,
    )

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report.to_json())
        logger.info("metrics report written to %s", rp)
    else:
        print(report.to_json())  # noqa: T201

    return 0 if report.n_folds > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
