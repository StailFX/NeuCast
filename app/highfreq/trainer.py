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
# Feature-pipeline imports
# ────────────────────────────────────────────────────────────────────────────
#
# Constants and feature transforms are deliberately defined in
# :mod:`app.highfreq.feature_pipeline` and re-exported here. This avoids
# train-vs-serve drift: the live predictor (Phase B) imports the SAME
# functions, guaranteeing CatBoost sees the exact distribution at score
# time that it saw at fit time. Keep the canonical home in
# ``feature_pipeline``; do not redefine these names here.

from app.highfreq.feature_pipeline import (  # noqa: E402  (re-export)
    FEATURE_COLUMNS,
    HORIZON_MIN,
    MIN_SECONDS_PER_MINUTE,
    NEUTRAL_BAND_BPS,
    aggregate_to_minute,
    build_features,
    build_target,
    make_supervised,
)


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
    #: Days of the most-recent data the trainer is FORBIDDEN from
    #: looking at — neither walk-forward CV folds nor the final-model
    #: fit see these bars. Reserved for ``tools/eval_frozen_holdout.py``
    #: to evaluate the deployed model on data it has truly never seen.
    #:
    #: Why this is different from walk-forward CV: walk-forward IS
    #: out-of-sample for individual folds, but hyperparameters
    #: (CatBoost depth, learning rate, neutral-band threshold) were
    #: tuned by inspecting CV outputs, so there's a mild leak. A
    #: frozen holdout is the canonical way to claim "I have no leak"
    #: under academic scrutiny — the trainer literally cannot tune
    #: against data it never loads.
    #:
    #: 0 disables the holdout (use during early bootstrapping when
    #: every bar of data matters); 7 is the default once the system
    #: is warm — gives ``eval_frozen_holdout.py`` a sample of
    #: ~7 × 24 × 60 ≈ 10k bars after neutral-band drop, plenty for
    #: a tight Wilson CI.
    frozen_holdout_days: int = 7


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


def binom_test_p_greater_half(n_correct: int, n_total: int) -> float:
    """One-sided exact binomial test for "model is better than random".

    Returns ``P(X >= n_correct | X ~ Binomial(n_total, 0.5))`` — the
    probability of seeing at least this many correct calls if the model
    had no directional skill.

    Why a separate metric from the bootstrap CI
    -------------------------------------------

    Bootstrap CI answers "what's the plausible range of the accuracy?"
    The p-value answers "could this number have come from random
    guessing?" The two are complementary: a CI of ``[0.49, 0.55]``
    around 0.52 means we **don't know** whether we have skill, and a
    well-defined p-value (e.g. ``0.18``) puts a precise number on that
    uncertainty. Defence-grade reviewers ask for both.

    Implementation: scipy is already a transitive dependency through
    scikit-learn (used by the trainer for ``GroupKFold`` etc.), so
    importing it here doesn't grow the deployment footprint.

    NaN return is the safe default when the run produced no folds yet
    — the UI can render "—" instead of a misleading "p < ..." string.
    """
    if n_total <= 0:
        return float("nan")
    from scipy.stats import binomtest  # transitive dep via sklearn
    return float(binomtest(k=n_correct, n=n_total, p=0.5, alternative="greater").pvalue)


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
    #: One-sided binomial p-value testing H₀ "no directional skill"
    #: (proportion correct = 0.5) against H_a "model is better than
    #: random". Computed on the pooled walk-forward predictions
    #: (not per-fold) so it reflects sample size honestly. ``NaN``
    #: when no folds were produced.
    dir_acc_p_value: float
    log_loss_mean: float
    folds: list[dict[str, Any]] = field(default_factory=list)
    low_directional_skill: bool = True
    weights_path: str | None = None
    elapsed_seconds: float = 0.0
    #: Days of recent data the trainer was forbidden to look at — kept
    #: as a separate frozen holdout for ``tools/eval_frozen_holdout.py``.
    #: 0 means the holdout was disabled (early bootstrap mode).
    frozen_holdout_days: int = 0
    #: Bars excluded from training/CV by the frozen holdout. ``None``
    #: when the holdout is disabled.
    n_minutes_in_holdout: int | None = None
    #: Cutoff time: bars with ``minute >= holdout_cutoff_iso`` were
    #: held out. ``None`` when disabled or no data was available.
    holdout_cutoff_iso: str | None = None

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

    # Frozen holdout split. Bars with ``minute >= cutoff`` are
    # reserved for ``tools/eval_frozen_holdout.py`` — neither
    # walk-forward CV nor the final-model fit see them. This is the
    # academic-grade "I have no leak" answer.
    n_in_holdout: int | None = None
    holdout_cutoff_iso: str | None = None
    if cfg.frozen_holdout_days > 0 and not meta.empty:
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        cutoff = _dt.now(tz=_tz.utc) - _td(days=int(cfg.frozen_holdout_days))
        # Compare timezone-aware (meta["minute"] is tz-UTC by upstream contract).
        cutoff_pd = pd.Timestamp(cutoff)
        train_mask = meta["minute"] < cutoff_pd
        n_in_holdout = int((~train_mask).sum())
        holdout_cutoff_iso = cutoff_pd.isoformat()
        if n_in_holdout > 0:
            logger.info(
                "frozen-holdout: excluding %d bars at minute >= %s "
                "(holdout_days=%d)",
                n_in_holdout, holdout_cutoff_iso, cfg.frozen_holdout_days,
            )
            X = X.loc[train_mask].reset_index(drop=True)
            y = y.loc[train_mask].reset_index(drop=True)
            meta = meta.loc[train_mask].reset_index(drop=True)

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
        # One-sided p-value on the pooled predictions (same sample as
        # the CI). Conservative: tests p > 0.5 strictly, so a model
        # that happens to be biased toward the majority class doesn't
        # get a free "significant" badge — it has to actually beat
        # random by enough to overcome ``n_total`` worth of variance.
        n_correct_total = int(
            (preds["y_pred"].to_numpy() == preds["y_true"].to_numpy()).sum()
        )
        n_total = int(len(preds))
        dir_acc_p_value = binom_test_p_greater_half(n_correct_total, n_total)
    else:
        dir_acc_mean = ll_mean = ci_lo = ci_hi = dir_acc_p_value = float("nan")

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
        dir_acc_p_value=dir_acc_p_value,
        log_loss_mean=ll_mean,
        frozen_holdout_days=int(cfg.frozen_holdout_days),
        n_minutes_in_holdout=n_in_holdout,
        holdout_cutoff_iso=holdout_cutoff_iso,
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
        "--frozen-holdout-days", type=int, default=7,
        help="reserve last N days from training/CV — used by "
             "eval_frozen_holdout.py for true OOS evaluation. "
             "Set 0 during early bootstrap when every bar matters.",
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

    cfg = WalkForwardConfig(
        initial_train_minutes=args.initial_train_minutes,
        frozen_holdout_days=args.frozen_holdout_days,
    )
    out = Path(args.out) if args.out else None

    # Normalise symbol to uppercase here (templated systemd units pass
    # lowercase via %I; DB stores uppercase). Single canonical form
    # keeps DB queries hitting the right rows.
    report = run_training(
        dsn, symbol=args.symbol.upper(), since_hours=args.since_hours,
        out_path=out, config=cfg,
    )

    # Console-friendly summary.
    logger.info(
        "TRAINING DONE | symbol=%s | bars=%d | folds=%d | "
        "dir_acc=%.4f [%.4f, %.4f] | p=%.4f | logloss=%.4f | base_rate=%.4f | "
        "low_skill=%s | elapsed=%.1fs",
        report.symbol, report.n_minutes_after_neutral_drop, report.n_folds,
        report.dir_acc_mean, report.dir_acc_ci_low, report.dir_acc_ci_high,
        report.dir_acc_p_value,
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

    # Heartbeat: per-symbol "trainer last success" gauge for the
    # textfile_collector. We emit on data-loaded runs even when
    # ``n_folds == 0`` (cold-start, ETH/BNB still accumulating
    # bars) — the alert we care about is "trainer didn't run AT
    # ALL for >25h", not "trainer ran but didn't yet have enough
    # data for a fold". The fold-count signal already lives in
    # the report payload and the UI surfaces it as 0/1500 progress.
    from app.highfreq.cron_metrics import write_cron_success
    sym = report.symbol  # already uppercased
    write_cron_success(
        "neucast_hf_trainer_last_success_timestamp_seconds",
        file_stem=f"neucast_hf_trainer_{sym.lower()}",
        labels={"symbol": sym},
    )

    return 0 if report.n_folds > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
