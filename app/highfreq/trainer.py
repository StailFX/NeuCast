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


def _resolve_feature_set(feature_set: str, bar_minutes: int) -> str:
    """Resolve ``feature_set='auto'`` based on ``bar_minutes``.

    1-minute bars → microstructure (production-tested for OFI signal).
    5+ minute bars → long_horizon TA (microstructure decays into noise
    at that scale; OHLC/EMA/RSI capture multi-bar momentum + regime).
    """
    valid = (
        "microstructure", "long_horizon", "cross_asset",
        "microstructure_v2", "microstructure_v3",
    )
    if feature_set != "auto":
        if feature_set not in valid:
            raise ValueError(
                f"feature_set must be 'auto' or one of {valid}, "
                f"got {feature_set!r}"
            )
        return feature_set
    return "microstructure" if bar_minutes <= 1 else "long_horizon"


def _make_supervised_for_feature_set(
    df_secs: pd.DataFrame,
    *,
    feature_set: str,
    bar_minutes: int,
    horizon: int = HORIZON_MIN,
    neutral_band_bps: float = NEUTRAL_BAND_BPS,
    reference_df_secs: pd.DataFrame | None = None,
    target_symbol: str | None = None,
    futures_df_secs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Pipeline-aware (X, y, meta) builder.

    For ``microstructure`` calls the canonical :func:`make_supervised`.
    For ``long_horizon`` inlines the same shape: aggregate → target →
    drop unobservable + neutral-band → build long-horizon features.
    For ``cross_asset`` builds base + lagged + (optionally) BTC-reference
    features — needs ``reference_df_secs`` (BTC's seconds) when the
    target symbol is ETH/BNB.

    Returns the same ``(X, y, meta)`` triple regardless of source so
    walk-forward CV / fit_final_model don't care about the pipeline.
    """
    resolved = _resolve_feature_set(feature_set, bar_minutes)
    if resolved == "microstructure":
        return make_supervised(
            df_secs,
            horizon=horizon,
            neutral_band_bps=neutral_band_bps,
            bar_minutes=bar_minutes,
        )

    if resolved == "long_horizon":
        # long_horizon path. We replicate make_supervised's logic but swap
        # build_features for build_long_horizon_features.
        from app.highfreq.feature_pipeline_long_horizon import (
            LONG_HORIZON_FEATURE_COLUMNS,
            build_long_horizon_features,
        )
        minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
        targeted = build_target(
            minute_df, horizon=horizon, neutral_band_bps=neutral_band_bps,
        )
        if targeted.empty:
            empty_X = pd.DataFrame(columns=LONG_HORIZON_FEATURE_COLUMNS)
            empty_y = pd.Series(dtype=np.int8, name="y")
            empty_meta = pd.DataFrame(columns=[
                "symbol", "minute", "microprice_close", "return_bps",
            ])
            return empty_X, empty_y, empty_meta
        keep = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
        targeted = targeted.loc[keep].reset_index(drop=True)
        X = build_long_horizon_features(targeted)[LONG_HORIZON_FEATURE_COLUMNS]
        y = targeted["y"].astype(np.int8)
        meta = targeted[
            ["symbol", "minute", "microprice_close", "return_bps"]
        ].copy()
        return X, y, meta

    if resolved == "microstructure_v3":
        # T.23 (2026-05-04): base 18 microstructure cols + 5
        # futures-basis cols (basis_bps, basis_change, ofi_diff,
        # funding_bps, mark_premium_bps). T.24 A/B at production
        # geometry measured +20pp dir_acc lift. The futures seconds
        # come in via the new ``futures_df_secs`` param.
        from app.highfreq.feature_pipeline_microstructure_v3 import (
            build_microstructure_v3_features,
            microstructure_v3_feature_columns,
        )
        cols_v3 = microstructure_v3_feature_columns()
        minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
        targeted = build_target(
            minute_df, horizon=horizon, neutral_band_bps=neutral_band_bps,
        )
        if targeted.empty:
            empty_X = pd.DataFrame(columns=cols_v3)
            empty_y = pd.Series(dtype=np.int8, name="y")
            empty_meta = pd.DataFrame(columns=[
                "symbol", "minute", "microprice_close", "return_bps",
            ])
            return empty_X, empty_y, empty_meta
        keep = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
        targeted = targeted.loc[keep].reset_index(drop=True)
        X = build_microstructure_v3_features(
            targeted, futures_seconds_df=futures_df_secs,
            bar_minutes=bar_minutes,
        )
        X = X[cols_v3]
        y = targeted["y"].astype(np.int8)
        meta = targeted[
            ["symbol", "minute", "microprice_close", "return_bps"]
        ].copy()
        return X, y, meta

    if resolved == "microstructure_v2":
        # T.18.c (2026-05-03): base 18 microstructure cols +
        # 4 trade-flow rolling features. Same supervised contract
        # as base — just a wider feature matrix.
        from app.highfreq.feature_pipeline_microstructure_v2 import (
            MICROSTRUCTURE_V2_FEATURE_COLUMNS,
            build_microstructure_v2_features,
        )
        minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
        targeted = build_target(
            minute_df, horizon=horizon, neutral_band_bps=neutral_band_bps,
        )
        if targeted.empty:
            empty_X = pd.DataFrame(columns=MICROSTRUCTURE_V2_FEATURE_COLUMNS)
            empty_y = pd.Series(dtype=np.int8, name="y")
            empty_meta = pd.DataFrame(columns=[
                "symbol", "minute", "microprice_close", "return_bps",
            ])
            return empty_X, empty_y, empty_meta
        keep = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
        targeted = targeted.loc[keep].reset_index(drop=True)
        X = build_microstructure_v2_features(targeted)
        X = X[MICROSTRUCTURE_V2_FEATURE_COLUMNS]
        y = targeted["y"].astype(np.int8)
        meta = targeted[
            ["symbol", "minute", "microprice_close", "return_bps"]
        ].copy()
        return X, y, meta

    # cross_asset path (release T 2026-04-29). For BTC the reference is
    # itself, so we just don't add cross-asset features (only base+lagged).
    # For ETH/BNB we point at BTC and need reference seconds.
    from app.highfreq.feature_pipeline_cross_asset import (
        build_cross_asset_features,
        feature_columns_for,
    )
    sym_upper = (target_symbol or "").upper()
    reference_symbol: str | None = (
        None if sym_upper == "BTCUSDT" else "BTCUSDT"
    )
    cols = feature_columns_for(reference_symbol)

    minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
    ref_minutes: pd.DataFrame | None = None
    if reference_symbol is not None and reference_df_secs is not None:
        ref_minutes = aggregate_to_minute(
            reference_df_secs, bar_minutes=bar_minutes,
        )
    elif reference_symbol is not None:
        logger.warning(
            "cross_asset requested for %s but no reference_df_secs "
            "provided — falling back to base + lagged only",
            target_symbol,
        )

    targeted = build_target(
        minute_df, horizon=horizon, neutral_band_bps=neutral_band_bps,
    )
    if targeted.empty:
        empty_X = pd.DataFrame(columns=cols)
        empty_y = pd.Series(dtype=np.int8, name="y")
        empty_meta = pd.DataFrame(columns=[
            "symbol", "minute", "microprice_close", "return_bps",
        ])
        return empty_X, empty_y, empty_meta
    keep = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
    targeted = targeted.loc[keep].reset_index(drop=True)
    X = build_cross_asset_features(
        targeted,
        reference_minutes=ref_minutes,
        reference_symbol=reference_symbol,
    )
    X = X[cols]
    y = targeted["y"].astype(np.int8)
    meta = targeted[
        ["symbol", "minute", "microprice_close", "return_bps"]
    ].copy()
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
    #: Bar size in minutes for aggregation (release R, multi-horizon
    #: support). 1 = original 1-minute contract; 5/15/60 = longer
    #: horizons that catch where E[|move|] grows enough to clear the
    #: fee burden (see ADR-017 trade-off table). Multi-horizon eval
    #: empirically validated this lift on 15m + 60m horizons.
    bar_minutes: int = 1
    #: Feature pipeline. ``"auto"`` picks microstructure for 1-minute
    #: bars (the production-tested pipeline) and long-horizon TA
    #: (OHLC + EMA + RSI + Bollinger) for ≥5-minute bars where
    #: microstructure decays into noise. ``"microstructure"`` /
    #: ``"long_horizon"`` force the pipeline regardless of bar size.
    feature_set: str = "auto"
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
    #: Half-life (in BARS) for exponential sample weighting. The
    #: trainer assigns each training bar a weight ``2 ** (-age / hl)``
    #: where ``age`` is bars-from-most-recent. This gives recent bars
    #: more influence than old ones — hedges against concept drift
    #: in market regime. Set to 0 to disable (uniform weighting,
    #: original behaviour). Default 720 bars (= ~12 hours at 1-min
    #: granularity, or proportionally less at longer horizons): the
    #: most recent 12 h of data dominates the fit, but bars going
    #: back 24-48 h still contribute meaningfully.
    sample_weight_half_life_bars: int = 720
    #: Embargo (López de Prado): drop the last ``embargo_bars`` rows
    #: from the train fold before fitting. Prevents subtle leakage
    #: where the train fold's last bar's target overlaps with the test
    #: fold (since target is forward-shifted by ``horizon``). With
    #: horizon=1 the leak is at most 1 bar of theoretical overlap, but
    #: best academic practice is to embargo it explicitly.
    embargo_bars: int = 1
    #: Purge (López de Prado): also drop bars near the train→test
    #: boundary on the TRAIN side that could have target-leakage with
    #: the test fold's first ``horizon`` bars. With horizon=1 +
    #: embargo=1 the purge is 0 (already covered by embargo). Reserve
    #: for future multi-horizon targets.
    purge_bars: int = 0


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

    # Scale fold geometry by bar_minutes (release R, multi-horizon).
    # Config fields are semantically "minutes of WALL-CLOCK data", but
    # X has ONE ROW PER BAR, so for 5/15/60-minute bars the positional
    # indices need scaling. ``initial_train_minutes=1440`` (1 day) maps
    # to 1440 bars on 1m, 96 bars on 15m, 24 bars on 60m.
    bm = max(1, int(cfg.bar_minutes))
    initial_train_bars = max(cfg.min_train_samples, cfg.initial_train_minutes // bm)
    test_fold_bars = max(1, cfg.test_fold_minutes // bm)
    step_bars = max(1, cfg.step_minutes // bm)

    if len(X) < initial_train_bars + test_fold_bars:
        logger.warning(
            "walk_forward_evaluate: only %d bars available, need ≥%d for one fold "
            "(bar_minutes=%d, initial_train_bars=%d, test_fold_bars=%d)",
            len(X), initial_train_bars + test_fold_bars,
            bm, initial_train_bars, test_fold_bars,
        )
        return [], pd.DataFrame(columns=["minute", "y_true", "proba", "y_pred", "fold_idx"])

    # Index is positional; meta provides the timestamp.
    minutes = pd.to_datetime(meta["minute"], utc=True).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    folds: list[FoldReport] = []
    pred_rows: list[dict[str, Any]] = []

    # Code-review Perf-medium #21 (2026-05-04): warm-start the next
    # fold's CatBoost from the previous fold's fit via ``init_model=``.
    # The expanding-window training set grows by ``step_bars`` per
    # fold; the model from the previous fold has already digested
    # ~99% of the new fold's training data, so the new fit only needs
    # to "absorb" the recently-arrived bars. Empirically saves
    # 30-50% wall-time on production geometry (33-39 folds × 3 symbols).
    #
    # Opt-out via ``HF_DISABLE_WARM_START=1`` (correctness fallback
    # if a future regression shows fold-to-fold contamination — the
    # cold-start path is the gold reference).
    _warm_start_enabled = os.getenv("HF_DISABLE_WARM_START", "").strip() not in ("1", "true", "yes", "on")
    prev_clf: "CatBoostClassifier | None" = None

    train_end = initial_train_bars
    fold_idx = 0
    while train_end + test_fold_bars <= len(X):
        test_end = train_end + test_fold_bars

        # Apply embargo + purge (López de Prado). Embargo: drop the
        # last ``embargo_bars`` of the train fold so the model can't
        # peek at info that overlaps with test fold via forward target.
        # Purge (target-leak guard): drop additional train bars within
        # ``purge_bars`` of the boundary.
        train_eff_end = max(0, train_end - cfg.embargo_bars - cfg.purge_bars)
        X_tr, y_tr = X.iloc[:train_eff_end], y.iloc[:train_eff_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if len(X_tr) < cfg.min_train_samples:
            # Skip-fold path: advance by the SAME bar-scaled step the
            # successful path uses. Pre-release-R this branch used the
            # raw ``cfg.step_minutes`` which was correct only when
            # bar_minutes=1; on 15m bars that meant skipping 60 bars
            # at a time (15 hours of data) and producing 0 folds even
            # when only the first fold needed to be skipped.
            train_end += step_bars
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
        # Sample weighting (release O 2026-04-29): exponential decay
        # so most-recent bars influence the fit more than old ones.
        # Hedges against concept drift in market regime. Disabled
        # (uniform weights) when half_life <= 0.
        sample_weights = _exponential_sample_weights(
            n=len(X_tr),
            half_life=cfg.sample_weight_half_life_bars,
        )
        # Warm-start path: pass the previous fold's fitted classifier
        # as ``init_model``. CatBoost continues from that ensemble
        # rather than starting from scratch.
        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weights}
        if _warm_start_enabled and prev_clf is not None:
            try:
                fit_kwargs["init_model"] = prev_clf
            except Exception:
                # Defensive — older CatBoost builds may not accept
                # init_model in fit(); fall through to cold start.
                pass
        try:
            clf.fit(X_tr.values, y_tr.values, **fit_kwargs)
        except TypeError:
            # CatBoost build doesn't accept init_model — cold-start
            # fallback so the trainer never hard-fails on a perf knob.
            fit_kwargs.pop("init_model", None)
            clf.fit(X_tr.values, y_tr.values, **fit_kwargs)
        proba = clf.predict_proba(X_te.values)[:, 1]
        y_hat = (proba >= 0.5).astype(np.int8)
        dir_acc = float((y_hat == y_te.values).mean())
        ll = _binary_logloss(y_te.values, proba)
        base = max(float(y_te.mean()), 1.0 - float(y_te.mean()))

        folds.append(FoldReport(
            fold_idx=fold_idx,
            train_start=minutes.iloc[0],
            # Use the effective train_end (after embargo+purge) so the
            # reported timestamp reflects what the model actually saw.
            train_end=minutes.iloc[max(0, train_eff_end - 1)],
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
        # Stash the fitted clf for the next fold's warm-start (#21).
        if _warm_start_enabled:
            prev_clf = clf
        train_end += step_bars
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
    # Code-review Perf-medium (2026-05-04): vectorised bootstrap.
    # Was a Python ``for i in range(n_resamples)`` loop — for n_resamples
    # 1000 × n=2500 the loop spent most of its time in interpreter
    # overhead. The vectorised form runs ~15-20× faster on the
    # walk-forward CV inner loop, which fires once per fold (33-39 folds
    # at production geometry × 3 symbols).
    correct = (y_pred == y_true).astype(np.float32)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = correct[idx].mean(axis=1)
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


def bayesian_dir_acc_ci(
    n_correct: int,
    n_total: int,
    *,
    alpha: float = 0.05,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[float, float, float]:
    """Bayesian credible interval for ``dir_acc`` via Beta-Binomial posterior.

    The Beta distribution is the conjugate prior of the Binomial
    likelihood, so the posterior given ``n_correct`` successes out of
    ``n_total`` is::

        posterior = Beta(prior_alpha + n_correct, prior_beta + n_total - n_correct)

    The credible interval is the (α/2, 1−α/2) quantile range of that
    posterior — by default a 95 % equal-tailed interval.

    Returns ``(point, lo, hi)`` where ``point`` is the posterior MEAN
    (NOT the MLE ``k/n``).  Mean-of-Beta is::

        E[θ] = (prior_alpha + k) / (prior_alpha + prior_beta + n)

    With the default Beta(1, 1) uniform prior this differs from k/n by a
    Laplace smoothing of one success and one failure.

    Why a separate metric from the bootstrap CI
    -------------------------------------------

    The bootstrap CI is a *frequentist* coverage interval: "if I rerun
    the experiment many times, the true value is inside this range
    95 % of the time".  A Bayesian credible interval is a *posterior*
    statement: "given my prior + this data, there's a 95 % posterior
    probability that the true value is in this range".  The latter is
    what most people *think* a CI means; reporting both gives reviewers
    the choice and demonstrates statistical literacy.

    For dir_acc on a dichotomous outcome (correct / incorrect) the
    Beta-Binomial posterior is the canonical, exact answer — no
    bootstrap resampling needed.  The two intervals tend to coincide
    for large n and a uniform prior; they diverge meaningfully on small
    samples (where the bootstrap is too narrow because it doesn't
    propagate the right uncertainty about extreme proportions).

    Returns
    -------
    (point, lo, hi)
        Posterior mean and (1 − alpha) credible interval bounds.
        ``(NaN, NaN, NaN)`` when no data was observed yet.
    """
    if n_total <= 0:
        return float("nan"), float("nan"), float("nan")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError(
            f"prior parameters must be positive, "
            f"got alpha={prior_alpha} beta={prior_beta}"
        )
    if not (0 <= n_correct <= n_total):
        raise ValueError(
            f"n_correct={n_correct} must be in [0, n_total={n_total}]"
        )

    a = prior_alpha + float(n_correct)
    b = prior_beta + float(n_total - n_correct)
    point = a / (a + b)
    from scipy.stats import beta  # transitive dep via sklearn
    lo = float(beta.ppf(alpha / 2.0, a, b))
    hi = float(beta.ppf(1.0 - alpha / 2.0, a, b))
    return float(point), lo, hi


# ────────────────────────────────────────────────────────────────────────────
# Final-model fit + persistence
# ────────────────────────────────────────────────────────────────────────────

def _exponential_sample_weights(
    *, n: int, half_life: int,
) -> np.ndarray | None:
    """Build a numpy array of length ``n`` with exponential decay
    weights so the last sample (most recent) has weight 1 and a
    sample ``half_life`` bars older has weight 0.5.

    Returns ``None`` when ``half_life <= 0`` (signals "use uniform
    weights" to CatBoost).

    Pure helper — no side effects. Tested separately.
    """
    if n <= 0 or half_life is None or half_life <= 0:
        return None
    # Position 0 is the OLDEST bar, n-1 is the MOST RECENT.
    # age[i] = (n-1) - i  → age=0 for newest, age=n-1 for oldest.
    age = np.arange(n - 1, -1, -1, dtype=float)
    return np.power(2.0, -age / float(half_life))


def fit_final_model(
    X: pd.DataFrame, y: pd.Series, *, config: WalkForwardConfig | None = None,
    init_from: str | Path | None = None,
):
    """Fit a model on the full dataset for production inference.

    Returns the fitted ``CatBoostClassifier`` instance. Caller is
    responsible for calling ``.save_model(path, format="cbm")``.

    ``init_from`` (release T.15, 2026-05-01): path to a pre-trained
    ``.cbm`` checkpoint. When given, CatBoost continues fitting from
    that checkpoint via the ``init_model=`` parameter — incremental
    (transfer) learning. The pre-trained model must have been fit
    with the SAME feature schema (column count + order). Typical
    workflow: pretrain on years of historical Klines via
    ``app.highfreq.pretrain``, then fine-tune on recent live OFI
    data here.
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
    sample_weights = _exponential_sample_weights(
        n=len(X), half_life=cfg.sample_weight_half_life_bars,
    )
    init_model = None
    if init_from is not None:
        init_path = Path(init_from)
        if not init_path.exists():
            raise FileNotFoundError(
                f"--init-from points to {init_path} but the file is missing"
            )
        init_model = CatBoostClassifier()
        init_model.load_model(str(init_path))
        logger.info(
            "fine-tuning: loaded pre-trained checkpoint from %s "
            "(tree_count=%d)",
            init_path, init_model.tree_count_,
        )
    clf.fit(X.values, y.values, sample_weight=sample_weights, init_model=init_model)
    return clf


# ────────────────────────────────────────────────────────────────────────────
# DB layer (SQLAlchemy, synchronous — async only matters for the WS consumer)
# ────────────────────────────────────────────────────────────────────────────

#: Allowed values for the trainer ``--venue`` flag (release S phase 3).
VENUE_TABLES: dict[str, str] = {
    "spot": "highfreq_ofi_1s",
    "futures": "highfreq_futures_ofi_1s",
}


# Code-review M-1 (2026-05-04): pre-baked SQL queries per venue,
# instead of f-string interpolation of a whitelisted table name +
# conditional ``extra_cols``. The previous shape was *safe today*
# (table name post-whitelist, extra_cols post-whitelist), but the
# pattern itself ("trust f-string into raw SQL") is the seed of the
# next SQL-injection regression — a future maintainer adding a
# ``--filter`` flag will inevitably stick it into the same f-string.
#
# Dispatching a fully-formed ``text(...)`` per venue eliminates that
# precedent: there is no live string-formatting against any input,
# even validated input. New venues are added by appending an entry to
# this dict, NOT by editing a format string.
_LOAD_SECONDS_QUERIES: dict[str, "Any"] = {}


def _build_load_seconds_queries() -> dict[str, "Any"]:
    """Lazy: SQLAlchemy is a fairly heavy import; keep the CLI snappy
    when ``--help`` is the only thing the user wants."""
    from sqlalchemy import text
    return {
        "spot": text(
            "SELECT ts, symbol, ofi, microprice, depth_imb, spread_bps, "
            "trade_imb, vpin, n_updates, local_recv_ms "
            "FROM highfreq_ofi_1s "
            "WHERE symbol = :symbol "
            "AND ts >= now() - (:hours * interval '1 hour') "
            "ORDER BY ts ASC"
        ),
        "futures": text(
            "SELECT ts, symbol, ofi, microprice, depth_imb, spread_bps, "
            "trade_imb, vpin, n_updates, local_recv_ms, "
            "mark_price, funding_rate "
            "FROM highfreq_futures_ofi_1s "
            "WHERE symbol = :symbol "
            "AND ts >= now() - (:hours * interval '1 hour') "
            "ORDER BY ts ASC"
        ),
    }


def load_seconds(
    database_url: str,
    *,
    symbol: str,
    since_hours: float,
    venue: str = "spot",
) -> pd.DataFrame:
    """Read the last ``since_hours`` of 1-second rows for ``symbol``.

    ``venue`` selects the source table:
    * ``"spot"`` (default, backwards compatible) → ``highfreq_ofi_1s``
    * ``"futures"`` → ``highfreq_futures_ofi_1s``

    Returns a DataFrame with the columns required by the
    feature-pipeline aggregator. Futures rows additionally carry
    ``mark_price`` / ``funding_rate`` — these are SELECTED for
    futures venue only (release T.23, 2026-05-04: needed by the
    ``microstructure_v3`` feature pipeline which computes basis
    and mark-premium features against the spot side).
    """
    if venue not in VENUE_TABLES:
        raise ValueError(
            f"venue must be one of {sorted(VENUE_TABLES)}, got {venue!r}"
        )

    from sqlalchemy import create_engine

    # Code-review M-1 (2026-05-04): dispatch dict of pre-baked queries —
    # no f-string interpolation against any input, even validated.
    global _LOAD_SECONDS_QUERIES
    if not _LOAD_SECONDS_QUERIES:
        _LOAD_SECONDS_QUERIES = _build_load_seconds_queries()
    query = _LOAD_SECONDS_QUERIES[venue]

    eng = create_engine(database_url, future=True)
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
    #: Calibration diagnostics (release M.δ). ``None`` when no folds
    #: produced — there are no OOS probabilities to fit a calibrator
    #: on. Lower is better for both metrics.
    calibrator_brier: float | None = None
    calibrator_ece: float | None = None
    #: Bayesian 95 % credible interval (Beta-Binomial posterior, uniform
    #: prior). Reported alongside the bootstrap CI for two reasons:
    #: 1) the posterior interpretation ("there's a 95 % posterior
    #: probability the true dir_acc is in this range") matches what
    #: most reviewers intuit when they read "95 % CI"; 2) for a
    #: dichotomous outcome the Beta-Binomial posterior is the canonical,
    #: exact answer that doesn't depend on resampling. ``None`` when
    #: no folds have been produced yet. Release Q (2026-04-29).
    dir_acc_bayesian_ci_low: float | None = None
    dir_acc_bayesian_ci_high: float | None = None
    #: Bar size in minutes the model was trained at (release R). The
    #: predictor reads this so it can aggregate live seconds → bars
    #: at the right granularity at inference time.
    bar_minutes: int = 1
    #: Feature pipeline used at training time (release R). The
    #: predictor reads this so the live inference path uses the SAME
    #: pipeline that produced the .cbm — avoids the Feature 18 vs 24
    #: column-count drift between train (long_horizon, 24 cols) and
    #: serve (microstructure, 18 cols) that broke the first 15m
    #: paper-trader spawn.
    feature_set: str = "microstructure"
    #: Split-conformal nonconformity quantile q at α = 0.10 (T.17.b).
    #: Defines a 90 %-coverage prediction interval around the live
    #: ``prob_up``: ``[max(0, prob - q), min(1, prob + q)]``.
    #:
    #: Conformal scores are |proba_oos - y_oos| over the pooled
    #: walk-forward predictions; ``q`` is the
    #: ⌈(n+1)(1-α)⌉ / n quantile of those scores. The coverage
    #: guarantee — ``P(true_outcome ∈ interval) ≥ 1 - α`` — holds
    #: under exchangeability of calibration vs test data, which the
    #: walk-forward CV approximately satisfies (rolling-origin
    #: contemporaneous folds). Modern academic mainstream:
    #: Vovk-Gammerman-Shafer 2005, Angelopoulos-Bates 2023.
    #:
    #: ``None`` when no folds produced (cold start). The predictor
    #: reads from metrics.json and emits the interval on the
    #: forecast endpoint when present.
    conformal_q_alpha_0_10: float | None = None
    conformal_q_alpha_0_05: float | None = None
    conformal_n_calibration: int | None = None

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
    venue: str = "spot",
    init_from: str | Path | None = None,
) -> TrainingReport:
    """Full training pipeline. Returns a ``TrainingReport`` for logging.

    ``venue`` selects which OFI table the trainer reads from
    (``"spot"`` → ``highfreq_ofi_1s``; ``"futures"`` →
    ``highfreq_futures_ofi_1s``). Default ``"spot"`` preserves the
    original contract — existing systemd timer / CLI calls continue
    to read the spot table without changes (release S phase 3).

    ``init_from`` (release T.15, 2026-05-01): path to a pre-trained
    ``.cbm`` checkpoint produced by ``app.highfreq.pretrain``. When
    given, the final-fit step uses CatBoost's incremental learning
    (``init_model=``) so the new tree ensemble continues from the
    pretrained one. Feature schema must match — caller is responsible
    for setting ``--feature-set`` to the same pipeline used at pretrain.
    """
    cfg = config or WalkForwardConfig()
    started = time.monotonic()

    df_secs = load_seconds(
        database_url, symbol=symbol, since_hours=since_hours, venue=venue,
    )
    logger.info(
        "loaded %d seconds of data for %s (venue=%s)",
        len(df_secs), symbol, venue,
    )

    bm = max(1, int(cfg.bar_minutes))
    fs = _resolve_feature_set(cfg.feature_set, bm)
    logger.info(
        "bar_minutes=%d feature_set=%s (resolved from %s)",
        bm, fs, cfg.feature_set,
    )

    # cross_asset for ETH/BNB needs BTC's seconds at the SAME window so
    # the bar timestamps align after aggregation. BTC's own model has no
    # reference (would point at itself). Spot-only — futures cross-asset
    # is not yet wired (would need separate venue split).
    reference_df_secs: pd.DataFrame | None = None
    if fs == "cross_asset" and symbol.upper() != "BTCUSDT":
        try:
            reference_df_secs = load_seconds(
                database_url,
                symbol="BTCUSDT",
                since_hours=since_hours,
                venue="spot",
            )
            logger.info(
                "loaded %d reference seconds (BTCUSDT spot) for cross_asset features",
                len(reference_df_secs),
            )
        except Exception:
            logger.warning(
                "failed to load BTCUSDT reference for cross_asset; "
                "falling back to base + lagged features",
                exc_info=True,
            )
            reference_df_secs = None

    # microstructure_v3 (T.23, 2026-05-04): also load the same window
    # of FUTURES seconds (perpetual). Used to compute basis_bps,
    # mark_premium, ofi_diff, funding_bps over each spot bar. Cold-
    # start safe — the v3 pipeline zero-fills when futures rows are
    # missing, so a futures-ingest gap doesn't block training.
    futures_df_secs: pd.DataFrame | None = None
    if fs == "microstructure_v3":
        try:
            futures_df_secs = load_seconds(
                database_url,
                symbol=symbol,
                since_hours=since_hours,
                venue="futures",
            )
            logger.info(
                "loaded %d futures seconds for %s (microstructure_v3)",
                len(futures_df_secs), symbol,
            )
        except Exception:
            logger.warning(
                "failed to load %s futures seconds for microstructure_v3; "
                "v3 will zero-fill the 5 futures cols",
                symbol, exc_info=True,
            )
            futures_df_secs = None

    minute_df = aggregate_to_minute(df_secs, bar_minutes=bm)
    n_min = len(minute_df)
    logger.info("aggregated to %d %d-minute bars", n_min, bm)

    X, y, meta = _make_supervised_for_feature_set(
        df_secs,
        feature_set=cfg.feature_set,
        bar_minutes=bm,
        reference_df_secs=reference_df_secs,
        target_symbol=symbol,
        futures_df_secs=futures_df_secs,
    )
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
    # Split-conformal nonconformity quantiles. Computed on the pooled
    # walk-forward OOS predictions (which already approximate
    # exchangeable calibration vs test data via rolling-origin
    # contemporaneous folds). For each row the score is
    # |proba_oos - y_true|; the (n+1)(1-α)/n quantile of those scores
    # gives a 1-α-coverage prediction-interval halfwidth. Released as
    # T.17.b. ``None`` when no folds (cold start).
    conformal_q_alpha_0_10: float | None = None
    conformal_q_alpha_0_05: float | None = None
    conformal_n_calibration: int | None = None
    if folds and len(preds) > 0:
        proba_arr = preds["proba"].to_numpy()
        y_arr = preds["y_true"].to_numpy()
        scores = np.abs(proba_arr - y_arr)
        n_cal = int(len(scores))
        conformal_n_calibration = n_cal
        # Inflated quantile per Angelopoulos & Bates 2023 — accounts
        # for finite-sample exchangeability so coverage holds at
        # the nominal 1-α level rather than slightly under.
        for alpha, target in (
            (0.10, "conformal_q_alpha_0_10"),
            (0.05, "conformal_q_alpha_0_05"),
        ):
            q_idx = math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
            q_idx_clipped = min(1.0, max(0.0, q_idx))
            q_value = float(np.quantile(scores, q_idx_clipped))
            if target == "conformal_q_alpha_0_10":
                conformal_q_alpha_0_10 = q_value
            else:
                conformal_q_alpha_0_05 = q_value
        logger.info(
            "conformal: q@α=0.10 = %.4f, q@α=0.05 = %.4f (n_cal=%d)",
            conformal_q_alpha_0_10, conformal_q_alpha_0_05, n_cal,
        )

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
        # Bayesian credible interval — defence-grade companion to the
        # bootstrap CI. Uniform Beta(1,1) prior so the posterior is
        # essentially the data with Laplace smoothing.
        try:
            _, bayes_lo, bayes_hi = bayesian_dir_acc_ci(
                n_correct=n_correct_total, n_total=n_total,
            )
        except Exception:
            bayes_lo = bayes_hi = float("nan")
    else:
        dir_acc_mean = ll_mean = ci_lo = ci_hi = dir_acc_p_value = float("nan")
        bayes_lo = bayes_hi = float("nan")

    weights_path: str | None = None
    calibrator_brier: float | None = None
    calibrator_ece: float | None = None
    if out_path is not None and len(X) >= cfg.min_train_samples:
        clf = fit_final_model(X, y, config=cfg, init_from=init_from)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Archive the existing weights (if any) BEFORE we overwrite —
        # T.17.d production safety. If today's training produces a
        # regressing model we can rollback in 1 command.
        try:
            from app.highfreq.model_archive import archive_existing
            archive_existing(out_path, keep_last_n=7)
        except Exception:  # noqa: BLE001
            logger.warning(
                "model archive failed (non-fatal); proceeding with overwrite",
                exc_info=True,
            )
        clf.save_model(str(out_path), format="cbm")
        weights_path = str(out_path)
        logger.info("saved final model to %s", out_path)

        # Probability calibration (release M.δ, 2026-04-29). Fit Platt
        # scaler on the pooled OOS predictions from walk-forward CV
        # — this is the most honest calibration data we have, since
        # those probabilities were produced on data the model never
        # saw at fit time. Save alongside the .cbm; predictor loads
        # both. Computed reliability metrics surface in the report
        # for the defence-grade reliability diagram.
        if folds and len(preds) > 0:
            try:
                from app.highfreq.calibration import (
                    calibrator_path_for,
                    compute_reliability_curve,
                    fit_platt_calibrator,
                    fit_isotonic_calibrator,
                    save_calibrator,
                )
                # T.18.a (2026-05-03): default to isotonic regression
                # for n_oos ≥ 1000 (Niculescu-Mizil & Caruana 2005
                # crossover point). Fall back to Platt if env opts in.
                cal_type = os.getenv("HF_CALIBRATOR_TYPE", "auto").strip().lower()
                if cal_type == "platt":
                    cal_fit = fit_platt_calibrator
                elif cal_type == "isotonic":
                    cal_fit = fit_isotonic_calibrator
                else:  # auto
                    cal_fit = (
                        fit_isotonic_calibrator
                        if len(preds) >= 1000
                        else fit_platt_calibrator
                    )
                logger.info(
                    "calibrator: %s (HF_CALIBRATOR_TYPE=%s, n_oos=%d)",
                    cal_fit.__name__, cal_type, len(preds),
                )
                cal = cal_fit(
                    preds["proba"].to_numpy(),
                    preds["y_true"].to_numpy(),
                )
                cal_path = calibrator_path_for(out_path)
                save_calibrator(cal, cal_path)
                logger.info("saved calibrator to %s", cal_path)
                # Pre-calibration reliability for reporting.
                rc = compute_reliability_curve(
                    preds["proba"].to_numpy(),
                    preds["y_true"].to_numpy(),
                    n_bins=10,
                )
                calibrator_brier = float(rc.brier_score)
                calibrator_ece = float(rc.ece)
                logger.info(
                    "raw model reliability: brier=%.4f ece=%.4f (lower is better)",
                    calibrator_brier, calibrator_ece,
                )
            except Exception:
                logger.warning(
                    "calibration fit/save failed (model still saved); "
                    "predictor will fall back to raw probabilities",
                    exc_info=True,
                )

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
        calibrator_brier=calibrator_brier,
        calibrator_ece=calibrator_ece,
        dir_acc_bayesian_ci_low=bayes_lo,
        dir_acc_bayesian_ci_high=bayes_hi,
        bar_minutes=bm,
        feature_set=fs,
        conformal_q_alpha_0_10=conformal_q_alpha_0_10,
        conformal_q_alpha_0_05=conformal_q_alpha_0_05,
        conformal_n_calibration=conformal_n_calibration,
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
        "--test-fold-minutes", type=int, default=60,
        help="test-fold size in minutes (default 60 = 1 hour). At "
             "small bar sizes scale-down: e.g. for 30m bars, "
             "--test-fold-minutes 30 = 1 bar per fold.",
    )
    p.add_argument(
        "--step-minutes", type=int, default=60,
        help="walk-forward step (default 60). Like --test-fold-minutes "
             "this is in wall-clock minutes; trainer divides by "
             "bar_minutes to get bars-per-step.",
    )
    p.add_argument(
        "--min-train-samples", type=int, default=200,
        help="minimum bars required to attempt a CV fold fit (default "
             "200). Lower this for low-density bar sizes (30m, 1h, ...) "
             "where total bar count is naturally small. Trade-off: "
             "smaller min → more folds but each fit is on fewer rows, "
             "noisier per-fold dir_acc estimate.",
    )
    p.add_argument(
        "--frozen-holdout-days", type=int, default=7,
        help="reserve last N days from training/CV — used by "
             "eval_frozen_holdout.py for true OOS evaluation. "
             "Set 0 during early bootstrap when every bar matters.",
    )
    p.add_argument(
        "--bar-minutes", type=int, default=1,
        help="bar size in minutes. 1 = production 1-minute model "
             "(microstructure features); 5/15/60 = longer horizons "
             "with long-horizon TA features (auto-selected). Output "
             "weights path should reflect the horizon, e.g. "
             "btcusdt_15m.cbm. Release R / 2026-04-29.",
    )
    p.add_argument(
        "--venue", default="spot", choices=("spot", "futures"),
        help="OFI table to read from. 'spot' = highfreq_ofi_1s "
             "(production default); 'futures' = highfreq_futures_ofi_1s "
             "(release S USDM Perpetual Futures). Caller is responsible "
             "for routing --out to the right directory "
             "(weights/highfreq/<symbol>_*.cbm vs "
             "weights/highfreq/futures/<symbol>_*.cbm). Release S phase 3.",
    )
    p.add_argument(
        "--feature-set", default="auto",
        choices=("auto", "microstructure", "long_horizon", "cross_asset",
                 "microstructure_v2", "microstructure_v3"),
        help="feature pipeline. 'auto' (default) picks microstructure "
             "for 1m bars and long_horizon TA for ≥5m. 'cross_asset' "
             "adds BTC reference features to ETH/BNB models — empirically "
             "lifts dir_acc by ~1pp on ETH/BNB. Force a specific "
             "pipeline only for ablation studies.",
    )
    p.add_argument(
        "--sample-weight-half-life-bars",
        type=int,
        default=int(os.getenv("HF_SAMPLE_WEIGHT_HALF_LIFE", "720")),
        help="exponential decay half-life (in bars) for sample weighting. "
             "Default 720 (release O). Set to 0 to disable — empirically "
             "disabled weighting lifts dir_acc by 1-5pp on stable market "
             "regimes (the deweighting of older training bars without "
             "drift compensation is structurally a headwind on stable "
             "windows). Override via HF_SAMPLE_WEIGHT_HALF_LIFE env.",
    )
    p.add_argument(
        "--init-from", default=os.getenv("HF_INIT_FROM"),
        help="path to a pre-trained .cbm checkpoint (release T.15). When "
             "set, CatBoost continues fitting from that checkpoint via "
             "init_model= — this is incremental / transfer learning. "
             "Typical workflow: pretrain on years of historical Klines "
             "via `python -m app.highfreq.pretrain`, then fine-tune on "
             "recent live OFI data here. Feature schema MUST match — "
             "set --feature-set to the same pipeline used at pretrain "
             "(default 'long_horizon' for Klines pretrains).",
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
        test_fold_minutes=args.test_fold_minutes,
        step_minutes=args.step_minutes,
        min_train_samples=args.min_train_samples,
        frozen_holdout_days=args.frozen_holdout_days,
        bar_minutes=args.bar_minutes,
        feature_set=args.feature_set,
        sample_weight_half_life_bars=args.sample_weight_half_life_bars,
    )
    out = Path(args.out) if args.out else None

    # Normalise symbol to uppercase here (templated systemd units pass
    # lowercase via %I; DB stores uppercase). Single canonical form
    # keeps DB queries hitting the right rows.
    from datetime import datetime as _dt, timezone as _tz
    run_started_at = _dt.now(tz=_tz.utc)
    report = run_training(
        dsn, symbol=args.symbol.upper(), since_hours=args.since_hours,
        out_path=out, config=cfg, venue=args.venue,
        init_from=args.init_from,
    )

    # Append-only training history. Fail-soft — a successful run that
    # couldn't log itself is still a successful run (the .cbm + JSON
    # are already on disk).
    try:
        from app.highfreq.training_history import persist_run_sync
        persist_run_sync(dsn, report, run_started_at=run_started_at)
    except Exception:
        logger.warning("training_history persist failed", exc_info=True)

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
