"""Hyperparameter sweep — systematic CatBoost tuning via walk-forward CV.

Why this exists
===============

Until now the CatBoost hyperparameters in ``WalkForwardConfig`` are
fixed defaults (depth=5, iterations=400, lr=0.05). They were chosen
once on the daily-side experience and never re-tuned for the 1-min
HFT pipeline. This tool runs an explicit grid search across the most
impactful knobs and reports best per (symbol, horizon, feature_set).

Defence-grade: a reviewer asking "did you tune the model?" gets
"yes, here's the sweep matrix and sensitivity report" rather than
"defaults from CatBoost docs".

Honest disclaimer
-----------------

Walk-forward CV with hyperparameter selection has a **known leak**:
hyperparameters are picked to maximise CV performance, so the CV
score is biased upward. The frozen-holdout (release A) is the
intended counter-measure — once we have ≥14 days of data, the
holdout never saw the chosen params at all.

For now (bootstrap mode, no holdout active), this sweep's "best CV"
is a hint, not the truth. We surface ALL combos' results so the
defence narrative can show sensitivity rather than cherry-picked
peak.

Default grid
------------

Designed to span the meaningful CatBoost dynamic range:

    depth          ∈ {3, 5, 7}
    iterations     ∈ {200, 400, 800}
    learning_rate  ∈ {0.03, 0.05, 0.1}
    l2_leaf_reg    ∈ {1, 3, 9}

3 × 3 × 3 × 3 = 81 combos per (symbol, horizon, feature_set). At ~10 s
per combo for 1m / BTCUSDT / microstructure on Tokyo, that's
~14 minutes per sweep. Acceptable for a one-shot tuning run.

Run from Tokyo
--------------

::

    sudo -u stailfx --preserve-env=DATABASE_URL \\
      /opt/neucast/venv/bin/python -m tools.hyperparam_sweep \\
        --symbols BTCUSDT \\
        --horizons 1 \\
        --feature-sets microstructure \\
        --since-hours 96
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default grid — meaningful spans, not exhaustive. 81 combos.
DEFAULT_DEPTHS = (3, 5, 7)
DEFAULT_ITERATIONS = (200, 400, 800)
DEFAULT_LRS = (0.03, 0.05, 0.1)
DEFAULT_L2 = (1.0, 3.0, 9.0)


@dataclass
class SweepResult:
    symbol: str
    bar_minutes: int
    feature_set: str
    depth: int
    iterations: int
    learning_rate: float
    l2_leaf_reg: float
    n_folds: int
    n_bars_after_neutral_drop: int
    dir_acc: float | None
    dir_acc_ci_low: float | None
    dir_acc_ci_high: float | None
    dir_acc_p_value: float | None
    log_loss: float | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wilson(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _binom_p(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    from scipy.stats import binomtest
    return float(binomtest(k=k, n=n, p=0.5, alternative="greater").pvalue)


def evaluate_one_combo(
    df_secs: pd.DataFrame,
    *,
    symbol: str,
    bar_minutes: int,
    feature_set: str,
    reference_df_secs: pd.DataFrame | None,
    depth: int,
    iterations: int,
    learning_rate: float,
    l2_leaf_reg: float,
    initial_train_bars: int,
    test_fold_bars: int,
    step_bars: int,
    neutral_band_bps: float,
    random_seed: int = 42,
) -> SweepResult:
    """One combo: aggregate, build features, walk-forward CV, score."""
    from app.highfreq.feature_pipeline import (
        aggregate_to_minute,
        build_target,
    )
    if feature_set == "long_horizon":
        from app.highfreq.feature_pipeline_long_horizon import (
            LONG_HORIZON_FEATURE_COLUMNS as ACTIVE_COLS,
            build_long_horizon_features as _build,
        )
        ref_minutes = None
    elif feature_set == "cross_asset":
        from app.highfreq.feature_pipeline_cross_asset import (
            build_cross_asset_features,
            feature_columns_for,
        )
        reference_symbol = None if symbol.upper() == "BTCUSDT" else "BTCUSDT"
        ACTIVE_COLS = feature_columns_for(reference_symbol)
        ref_minutes = (
            aggregate_to_minute(reference_df_secs, bar_minutes=bar_minutes)
            if reference_df_secs is not None and reference_symbol is not None
            else None
        )
        def _build(d):
            return build_cross_asset_features(
                d, reference_minutes=ref_minutes, reference_symbol=reference_symbol,
            )
    else:
        from app.highfreq.feature_pipeline import (
            FEATURE_COLUMNS as ACTIVE_COLS,
            build_features as _build,
        )

    started = time.monotonic()
    minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
    targeted = build_target(
        minute_df, horizon=1, neutral_band_bps=neutral_band_bps,
    )
    if targeted.empty:
        return SweepResult(
            symbol=symbol, bar_minutes=bar_minutes, feature_set=feature_set,
            depth=depth, iterations=iterations, learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg, n_folds=0, n_bars_after_neutral_drop=0,
            dir_acc=None, dir_acc_ci_low=None, dir_acc_ci_high=None,
            dir_acc_p_value=None, log_loss=None,
            elapsed_seconds=time.monotonic() - started,
        )
    keep_mask = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
    targeted = targeted.loc[keep_mask].reset_index(drop=True)
    X = _build(targeted)[ACTIVE_COLS]
    y = targeted["y"].astype(np.int8)
    n_kept = len(X)

    if n_kept < initial_train_bars + test_fold_bars:
        return SweepResult(
            symbol=symbol, bar_minutes=bar_minutes, feature_set=feature_set,
            depth=depth, iterations=iterations, learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg, n_folds=0,
            n_bars_after_neutral_drop=n_kept,
            dir_acc=None, dir_acc_ci_low=None, dir_acc_ci_high=None,
            dir_acc_p_value=None, log_loss=None,
            elapsed_seconds=time.monotonic() - started,
        )

    from catboost import CatBoostClassifier
    yt_pool: list[int] = []
    yp_pool: list[int] = []
    proba_pool: list[float] = []
    n_folds = 0
    train_end = initial_train_bars
    while train_end + test_fold_bars <= n_kept:
        X_tr = X.iloc[:train_end].to_numpy()
        y_tr = y.iloc[:train_end].to_numpy()
        X_te = X.iloc[train_end:train_end + test_fold_bars].to_numpy()
        y_te = y.iloc[train_end:train_end + test_fold_bars].to_numpy()
        if len(set(y_tr.tolist())) < 2:
            train_end += step_bars
            continue
        clf = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="Logloss",
            thread_count=2,
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        y_hat = (proba > 0.5).astype(int)
        yt_pool.extend(y_te.tolist())
        yp_pool.extend(y_hat.tolist())
        proba_pool.extend(proba.tolist())
        n_folds += 1
        train_end += step_bars

    if not yt_pool:
        return SweepResult(
            symbol=symbol, bar_minutes=bar_minutes, feature_set=feature_set,
            depth=depth, iterations=iterations, learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg, n_folds=0,
            n_bars_after_neutral_drop=n_kept,
            dir_acc=None, dir_acc_ci_low=None, dir_acc_ci_high=None,
            dir_acc_p_value=None, log_loss=None,
            elapsed_seconds=time.monotonic() - started,
        )

    yt = np.array(yt_pool)
    yp = np.array(yp_pool)
    p_arr = np.array(proba_pool)
    n_correct = int((yt == yp).sum())
    n_total = int(len(yt))
    dir_acc = n_correct / n_total
    ci_lo, ci_hi = _wilson(n_correct, n_total)
    p_value = _binom_p(n_correct, n_total)
    eps = 1e-7
    p_clip = np.clip(p_arr, eps, 1 - eps)
    log_loss = float(-(yt * np.log(p_clip) + (1 - yt) * np.log(1 - p_clip)).mean())

    return SweepResult(
        symbol=symbol, bar_minutes=bar_minutes, feature_set=feature_set,
        depth=depth, iterations=iterations, learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg, n_folds=n_folds,
        n_bars_after_neutral_drop=n_kept,
        dir_acc=dir_acc, dir_acc_ci_low=ci_lo, dir_acc_ci_high=ci_hi,
        dir_acc_p_value=p_value, log_loss=log_loss,
        elapsed_seconds=time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    p.add_argument("--horizons", type=int, nargs="+", default=[1])
    p.add_argument("--feature-sets", nargs="+",
                   default=["microstructure"],
                   choices=["microstructure", "long_horizon", "cross_asset"])
    p.add_argument("--since-hours", type=float, default=96.0)
    p.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    p.add_argument("--iterations", type=int, nargs="+", default=list(DEFAULT_ITERATIONS))
    p.add_argument("--lrs", type=float, nargs="+", default=list(DEFAULT_LRS))
    p.add_argument("--l2", type=float, nargs="+", default=list(DEFAULT_L2))
    p.add_argument("--neutral-band-bps", type=float, default=None)
    p.add_argument("--out", default="weights/highfreq/hyperparam_sweep.json")
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

    from app.highfreq.trainer import load_seconds

    btc_df = None
    if "cross_asset" in args.feature_sets and any(
        s.upper() != "BTCUSDT" for s in args.symbols
    ):
        logger.info("pre-loading BTCUSDT for cross-asset reference")
        btc_df = load_seconds(dsn, symbol="BTCUSDT", since_hours=args.since_hours)

    grid = list(itertools.product(args.depths, args.iterations, args.lrs, args.l2))
    total = len(grid) * len(args.symbols) * len(args.horizons) * len(args.feature_sets)
    logger.info("hyperparameter sweep: %d combinations total", total)

    results: list[SweepResult] = []
    sweep_started = time.monotonic()
    done = 0
    for symbol in args.symbols:
        df_secs = load_seconds(dsn, symbol=symbol, since_hours=args.since_hours)
        for h in args.horizons:
            initial_train = max(60, int(24 * 60 / h))
            test_bars = max(5, int(60 / h))
            step = test_bars
            neutral = args.neutral_band_bps if args.neutral_band_bps is not None \
                else 1.0 * math.sqrt(h)
            for fset in args.feature_sets:
                ref = btc_df if fset == "cross_asset" and symbol.upper() != "BTCUSDT" \
                    else None
                for (depth, iters, lr, l2) in grid:
                    done += 1
                    res = evaluate_one_combo(
                        df_secs, symbol=symbol, bar_minutes=h, feature_set=fset,
                        reference_df_secs=ref,
                        depth=depth, iterations=iters,
                        learning_rate=lr, l2_leaf_reg=l2,
                        initial_train_bars=initial_train,
                        test_fold_bars=test_bars, step_bars=step,
                        neutral_band_bps=neutral,
                    )
                    results.append(res)
                    if res.dir_acc is not None:
                        logger.info(
                            "[%d/%d] %s %dm %s d%d i%d lr%.3f l2%.1f → dir_acc %.4f p=%.2e (%.1fs)",
                            done, total, symbol, h, fset[:4],
                            depth, iters, lr, l2,
                            res.dir_acc, res.dir_acc_p_value, res.elapsed_seconds,
                        )
                    else:
                        logger.info(
                            "[%d/%d] %s %dm %s d%d i%d lr%.3f l2%.1f → INSUFFICIENT",
                            done, total, symbol, h, fset[:4], depth, iters, lr, l2,
                        )

    total_elapsed = time.monotonic() - sweep_started
    logger.info("sweep complete in %.1fs", total_elapsed)

    # Best per (symbol, horizon, feature_set) by dir_acc.
    print()
    print("# Best params per (symbol, horizon, feature_set)")
    print()
    print("| symbol | bar | features | best params | dir_acc | CI | p |")
    print("|---|---:|---|---|---:|---|---:|")
    by_key: dict[tuple, list[SweepResult]] = {}
    for r in results:
        if r.dir_acc is None:
            continue
        by_key.setdefault((r.symbol, r.bar_minutes, r.feature_set), []).append(r)
    for key, rs in sorted(by_key.items()):
        best = max(rs, key=lambda r: r.dir_acc or -1)
        params = (
            f"d={best.depth} iter={best.iterations} "
            f"lr={best.learning_rate:.3f} l2={best.l2_leaf_reg:.1f}"
        )
        ci = f"[{best.dir_acc_ci_low:.3f}, {best.dir_acc_ci_high:.3f}]"
        p_str = (
            f"{best.dir_acc_p_value:.2e}" if best.dir_acc_p_value < 0.01
            else f"{best.dir_acc_p_value:.3f}"
        )
        print(
            f"| **{best.symbol}** | **{best.bar_minutes}m** | {best.feature_set[:4]} | "
            f"{params} | {best.dir_acc:.4f} | {ci} | {p_str} |"
        )

    # Sensitivity: spread of dir_acc across the grid for each (symbol, horizon, feature_set).
    print()
    print("# Sensitivity (dir_acc range across grid)")
    print()
    print("| symbol | bar | features | n_combos | min | mean | max | range |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for key, rs in sorted(by_key.items()):
        accs = [r.dir_acc for r in rs if r.dir_acc is not None]
        sym, bar, fset = key
        if not accs:
            continue
        print(
            f"| {sym} | {bar}m | {fset[:4]} | {len(accs)} | "
            f"{min(accs):.4f} | {sum(accs)/len(accs):.4f} | "
            f"{max(accs):.4f} | {max(accs)-min(accs):.4f} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": total_elapsed,
        "n_combinations": total,
        "results": [r.to_dict() for r in results],
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
