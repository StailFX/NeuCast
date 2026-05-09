"""Joint multi-symbol trainer (production version).

Reads OFI seconds for BTC + ETH + BNB, builds a joint supervised set
with symbol-id one-hots (see ``app.highfreq.feature_pipeline_joint``),
runs walk-forward CV, fits a final pooled model on ALL bars, fits a
calibrator on the OOS predictions, and writes everything to disk in
the same shape per-symbol trainers use.

Why a separate tool
===================

``tools/multi_horizon_eval.py`` evaluates joint training but throws
the per-fold models away. Production needs the **persisted** weights
+ a calibrator file the predictor can load. This tool reuses the
joint feature pipeline / supervised builder, but wires in the same
"save .cbm + calibrator + metrics + heartbeat + history-row" tail as
``app.highfreq.trainer.run`` does for solo models.

Output contract
---------------

* ``--out`` (default ``weights/highfreq/joint_1m.cbm``) — final
  CatBoost classifier fit on the entire pooled dataset.
* ``<out>_calibrator.pkl`` — Platt or isotonic calibrator fit on
  OOS predictions.
* ``--report`` (default ``weights/highfreq/joint_1m_metrics.json``) —
  same schema as ``app.highfreq.trainer.TrainingReport`` so the
  dashboard / training_history / parsing layers don't care that this
  is a joint model. ``symbol`` is the literal string ``"JOINT"`` so
  downstream UIs can distinguish.

Defence-grade: dir_acc + Wilson 95 % CI + binomial p-value + Bayesian
beta-binomial CI + reliability (brier / ECE) + per-fold breakdown.
The CI/p story is on the **pooled** walk-forward predictions (n_folds
× test_fold_bars), not per-fold means — that's the honest sample size.

Run from Tokyo
--------------
::

    set -a; source /etc/neucast/env; set +a
    cd /opt/neucast
    sudo -u stailfx --preserve-env=DATABASE_URL \\
        /opt/neucast/venv/bin/python -m tools.train_joint_1m \\
        --since-hours 240 \\
        --out /opt/neucast/weights/highfreq/joint_1m.cbm \\
        --report /opt/neucast/weights/highfreq/joint_1m_metrics.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_joint_1m")


# Matching the eval tool's CV hyperparameters so spike → production
# numbers are directly comparable. These are also the values that
# produced the 0.5547 / p=7e-33 result on 09.05 fresh data.
_DEFAULT_INITIAL_TRAIN_BARS_PER_HOUR = 60   # 60 bars/h × 24h = 1440 (1m) → scaled
_DEFAULT_TEST_FOLD_BARS_PER_HOUR = 60       # 60 bars/h × 1h
_CATBOOST_ITERATIONS = 200
_CATBOOST_DEPTH = 5
_CATBOOST_LEARNING_RATE = 0.05
_CATBOOST_RANDOM_SEED = 42

# Production solo trainer defaults — keep parity for fair comparisons.
_SAMPLE_WEIGHT_HALF_LIFE_BARS = 720   # 12h at 1m
_EMBARGO_BARS = 1


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — same impl as multi_horizon_eval."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _binom_p_value_greater_half(k: int, n: int) -> float:
    """One-sided p-value for H₀: p=0.5 vs H_a: p>0.5. Same as eval tool."""
    if n <= 0:
        return float("nan")
    from scipy.stats import binomtest
    res = binomtest(k=k, n=n, p=0.5, alternative="greater")
    return float(res.pvalue)


def _bayesian_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95 % Beta-Binomial posterior credible interval (uniform prior)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    from scipy.stats import beta
    a = k + 1
    b = (n - k) + 1
    return (float(beta.ppf(alpha / 2, a, b)),
            float(beta.ppf(1 - alpha / 2, a, b)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+",
                   default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                   help="symbols to pool. Default: all 3 production symbols.")
    p.add_argument("--since-hours", type=float, default=240.0,
                   help="hours of OFI history to load per symbol (default 240 "
                        "= 10 days; matches eval-tool default once we have "
                        "enough data accumulated).")
    p.add_argument("--bar-minutes", type=int, default=1,
                   help="bar size; 1 = 1m microstructure (production default). "
                        "5/15/60 use long-horizon TA features automatically.")
    p.add_argument("--neutral-band-bps", type=float, default=None,
                   help="override default. Auto-scales as sqrt(bar_minutes) "
                        "× 1bp if not given — preserves z-score across horizons.")
    p.add_argument("--out", default="weights/highfreq/joint_1m.cbm",
                   help="path to save the final fitted CatBoost model.")
    p.add_argument("--report", default="weights/highfreq/joint_1m_metrics.json",
                   help="path to save the metrics report JSON.")
    p.add_argument("--use-long-horizon-features", action="store_true",
                   help="use TA features (OHLC/EMA/RSI) instead of "
                        "microstructure. Recommended for bar-minutes >= 5.")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is required")
        return 2

    started_at = datetime.now(tz=timezone.utc)
    t0 = time.monotonic()

    # ── 1. Load OFI seconds for all symbols ───────────────────────────
    from app.highfreq.trainer import load_seconds
    df_secs_by_symbol = {}
    for sym in args.symbols:
        df = load_seconds(dsn, symbol=sym, since_hours=args.since_hours)
        df_secs_by_symbol[sym] = df
        logger.info("%s: %d seconds loaded", sym, len(df))

    n_seconds_loaded = sum(len(df) for df in df_secs_by_symbol.values())

    # ── 2. Build joint supervised dataset ─────────────────────────────
    from app.highfreq.feature_pipeline_joint import (
        make_joint_supervised,
        JOINT_FEATURE_COLUMNS,
        joint_long_horizon_columns,
    )
    from app.highfreq.feature_pipeline import aggregate_to_minute
    import pandas as pd

    bm = int(args.bar_minutes)
    neutral_bps = (
        args.neutral_band_bps if args.neutral_band_bps is not None
        else 1.0 * math.sqrt(bm)
    )
    use_lh = bool(args.use_long_horizon_features) or bm >= 5
    feature_set_label = "joint_long_horizon" if use_lh else "joint"
    feature_columns = (
        joint_long_horizon_columns() if use_lh else JOINT_FEATURE_COLUMNS
    )
    logger.info(
        "config: bar_minutes=%d feature_set=%s neutral_band_bps=%.3f n_features=%d",
        bm, feature_set_label, neutral_bps, len(feature_columns),
    )

    # Diagnostic: post-aggregation bar count.
    pooled_secs = pd.concat(
        df_secs_by_symbol.values(), ignore_index=True, sort=False,
    )
    minute_df = aggregate_to_minute(pooled_secs, bar_minutes=bm)
    n_bars_agg = int(len(minute_df))

    X, y, _meta = make_joint_supervised(
        df_secs_by_symbol,
        neutral_band_bps=neutral_bps,
        bar_minutes=bm,
        use_long_horizon_features=use_lh,
    )
    n_bars_kept = int(len(X))
    logger.info(
        "supervised: %d bars after aggregation, %d after neutral-band drop "
        "(%.1f %% kept)",
        n_bars_agg, n_bars_kept,
        100.0 * n_bars_kept / max(1, n_bars_agg),
    )

    # ── 3. Walk-forward CV ────────────────────────────────────────────
    initial_train_bars = max(60, int(_DEFAULT_INITIAL_TRAIN_BARS_PER_HOUR * 24 / bm))
    test_fold_bars = max(5, int(_DEFAULT_TEST_FOLD_BARS_PER_HOUR / bm))
    step_bars = test_fold_bars

    if n_bars_kept < initial_train_bars + test_fold_bars:
        logger.error(
            "not enough data: need >= %d bars (initial=%d + test=%d), have %d",
            initial_train_bars + test_fold_bars,
            initial_train_bars, test_fold_bars, n_bars_kept,
        )
        return 1

    from catboost import CatBoostClassifier

    folds_meta: list[dict] = []
    pooled_y_true: list[int] = []
    pooled_y_pred: list[int] = []
    pooled_proba: list[float] = []  # for calibration fitting later
    pooled_log_loss: list[float] = []
    train_end = initial_train_bars
    fold_idx = 0

    while train_end + test_fold_bars <= n_bars_kept:
        train_eff_end = max(0, train_end - max(0, _EMBARGO_BARS))
        X_tr = X.iloc[:train_eff_end].to_numpy()
        y_tr = y.iloc[:train_eff_end].to_numpy()
        X_te = X.iloc[train_end:train_end + test_fold_bars].to_numpy()
        y_te = y.iloc[train_end:train_end + test_fold_bars].to_numpy()
        if len(X_tr) < 100 or len(set(y_tr.tolist())) < 2:
            train_end += step_bars
            continue

        clf = CatBoostClassifier(
            iterations=_CATBOOST_ITERATIONS,
            depth=_CATBOOST_DEPTH,
            learning_rate=_CATBOOST_LEARNING_RATE,
            loss_function="Logloss",
            thread_count=int(os.getenv("OMP_NUM_THREADS", "2")),
            random_seed=_CATBOOST_RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
        if _SAMPLE_WEIGHT_HALF_LIFE_BARS > 0:
            n_tr = len(X_tr)
            age = np.arange(n_tr - 1, -1, -1, dtype=float)
            sw = np.power(2.0, -age / float(_SAMPLE_WEIGHT_HALF_LIFE_BARS))
            clf.fit(X_tr, y_tr, sample_weight=sw)
        else:
            clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_te)[:, 1]
        y_hat = (proba > 0.5).astype(int)

        # Per-fold dir_acc + log-loss for the report breakdown.
        n_correct = int((y_hat == y_te).sum())
        n_total = int(len(y_te))
        fold_dir_acc = n_correct / max(1, n_total)
        # numerically safe log-loss
        eps = 1e-15
        ll = -float(
            (y_te * np.log(np.clip(proba, eps, 1 - eps))
             + (1 - y_te) * np.log(np.clip(1 - proba, eps, 1 - eps)))
            .mean()
        )

        folds_meta.append({
            "fold_idx": fold_idx,
            "n_train": int(len(X_tr)),
            "n_test": n_total,
            "dir_acc": fold_dir_acc,
            "log_loss": ll,
        })
        pooled_y_true.extend(y_te.tolist())
        pooled_y_pred.extend(y_hat.tolist())
        pooled_proba.extend(proba.tolist())
        pooled_log_loss.append(ll)

        fold_idx += 1
        train_end += step_bars

    n_folds = len(folds_meta)
    if n_folds == 0:
        logger.error("no folds produced; aborting")
        return 1

    yt = np.array(pooled_y_true)
    yp = np.array(pooled_y_pred)
    n_correct = int((yt == yp).sum())
    n_total = int(len(yt))
    dir_acc = n_correct / n_total
    ci_lo, ci_hi = _wilson_ci(n_correct, n_total)
    bayes_lo, bayes_hi = _bayesian_ci(n_correct, n_total)
    p_value = _binom_p_value_greater_half(n_correct, n_total)
    base_rate = float(max(yt.mean(), 1 - yt.mean()))
    log_loss_mean = float(np.mean(pooled_log_loss))
    low_skill = bool(ci_lo <= 0.5)

    logger.info(
        "WALK-FORWARD CV | folds=%d | dir_acc=%.4f [%.4f, %.4f] | p=%.2e | "
        "log_loss=%.4f | base_rate=%.4f | low_skill=%s",
        n_folds, dir_acc, ci_lo, ci_hi, p_value,
        log_loss_mean, base_rate, low_skill,
    )

    # ── 4. Final model on FULL pooled dataset ─────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Archive existing weights — same safety pattern as solo trainer.
    try:
        from app.highfreq.model_archive import archive_existing
        archive_existing(out_path, keep_last_n=7)
    except Exception:  # noqa: BLE001
        logger.warning(
            "model archive failed (non-fatal); proceeding with overwrite",
            exc_info=True,
        )

    final = CatBoostClassifier(
        iterations=_CATBOOST_ITERATIONS,
        depth=_CATBOOST_DEPTH,
        learning_rate=_CATBOOST_LEARNING_RATE,
        loss_function="Logloss",
        thread_count=int(os.getenv("OMP_NUM_THREADS", "2")),
        random_seed=_CATBOOST_RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )
    if _SAMPLE_WEIGHT_HALF_LIFE_BARS > 0:
        n_full = len(X)
        age = np.arange(n_full - 1, -1, -1, dtype=float)
        sw = np.power(2.0, -age / float(_SAMPLE_WEIGHT_HALF_LIFE_BARS))
        final.fit(X.to_numpy(), y.to_numpy(), sample_weight=sw)
    else:
        final.fit(X.to_numpy(), y.to_numpy())
    final.save_model(str(out_path), format="cbm")
    logger.info("saved final model to %s", out_path)

    # ── 5. Calibrator on OOS pooled predictions ───────────────────────
    calibrator_brier: float | None = None
    calibrator_ece: float | None = None
    try:
        from app.highfreq.calibration import (
            calibrator_path_for,
            compute_reliability_curve,
            fit_isotonic_calibrator,
            fit_platt_calibrator,
            save_calibrator,
        )
        cal_type = os.getenv("HF_CALIBRATOR_TYPE", "auto").strip().lower()
        n_oos = len(pooled_proba)
        if cal_type == "platt":
            cal_fit = fit_platt_calibrator
        elif cal_type == "isotonic":
            cal_fit = fit_isotonic_calibrator
        else:  # auto
            cal_fit = (
                fit_isotonic_calibrator if n_oos >= 1000
                else fit_platt_calibrator
            )
        logger.info(
            "calibrator: %s (HF_CALIBRATOR_TYPE=%s, n_oos=%d)",
            cal_fit.__name__, cal_type, n_oos,
        )
        cal = cal_fit(np.array(pooled_proba), yt)
        cal_path = calibrator_path_for(out_path)
        save_calibrator(cal, cal_path)
        logger.info("saved calibrator to %s", cal_path)
        rc = compute_reliability_curve(np.array(pooled_proba), yt, n_bins=10)
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

    # ── 6. Build report (matches solo trainer's TrainingReport schema) ──
    elapsed = time.monotonic() - t0
    report = {
        "symbol": "JOINT",
        "horizon_min": bm,    # solo trainer's field is bm; honour the convention
        "neutral_band_bps": neutral_bps,
        "n_seconds_loaded": n_seconds_loaded,
        "n_minutes_after_aggregation": n_bars_agg,
        "n_minutes_after_neutral_drop": n_bars_kept,
        "base_rate": base_rate,
        "n_folds": n_folds,
        "dir_acc_mean": dir_acc,
        "dir_acc_ci_low": ci_lo,
        "dir_acc_ci_high": ci_hi,
        "dir_acc_p_value": p_value,
        "dir_acc_bayesian_ci_low": bayes_lo,
        "dir_acc_bayesian_ci_high": bayes_hi,
        "log_loss_mean": log_loss_mean,
        "folds": folds_meta,
        "low_directional_skill": low_skill,
        "weights_path": str(out_path),
        "elapsed_seconds": elapsed,
        "frozen_holdout_days": 0,
        "n_minutes_in_holdout": None,
        "holdout_cutoff_iso": None,
        "calibrator_brier": calibrator_brier,
        "calibrator_ece": calibrator_ece,
        "bar_minutes": bm,
        "feature_set": feature_set_label,
        # Joint-specific extras for the dashboard / thesis.
        "joint_symbols": list(args.symbols),
        "joint_n_features": len(feature_columns),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("metrics report written to %s", report_path)

    # ── 7. Heartbeat for the cron-stale Grafana rule ─────────────────
    try:
        from app.highfreq.cron_metrics import write_cron_success
        write_cron_success(
            metric_name="neucast_hf_trainer_joint_last_success_timestamp_seconds",
            file_stem=f"neucast_hf_trainer_joint_{bm}m",
        )
    except Exception:
        logger.warning("heartbeat write failed (non-fatal)", exc_info=True)

    # ── 8. training_history row (best-effort) ─────────────────────────
    try:
        from app.highfreq.training_history import persist_run_sync
        # Build a TrainingReport-compatible dataclass on the fly so we
        # can reuse the existing INSERT helper. Simpler than refactoring
        # persist_run_sync to take a dict.
        from dataclasses import make_dataclass
        FauxReport = make_dataclass("FauxReport", [(k, type(v)) for k, v in report.items()])
        persist_run_sync(
            dsn,
            FauxReport(**report),  # type: ignore[arg-type]
            run_started_at=started_at,
        )
    except Exception:
        logger.warning("training_history persist failed (non-fatal)", exc_info=True)

    logger.info(
        "TRAINING DONE | symbol=JOINT | bm=%dm | n_folds=%d | "
        "dir_acc=%.4f [%.4f, %.4f] | p=%.2e | base=%.4f | "
        "low_skill=%s | elapsed=%.1fs",
        bm, n_folds, dir_acc, ci_lo, ci_hi, p_value,
        base_rate, low_skill, elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
