"""A/B eval: does adding futures-basis features lift dir_acc?

Purpose
=======

Yesterday's "futures funding is NULL" finding was wrong — the
``poll_funding_rate`` cron job back-fills funding_rate + mark_price
on the seconds-level table at 99%+ coverage (40h of data on disk
right now, growing).

This script builds candidate "futures basis" features on top of the
existing microstructure (or cross_asset) pipeline and measures the
walk-forward dir_acc lift, WITHOUT touching any production code path:

* Loads spot OFI seconds (for the model's own pipeline).
* Loads futures OFI seconds at the same wall-clock window.
* Joins by ``(ts, symbol)`` and computes per-bar:
    1. ``basis_bps``: (fut_microprice - spot_microprice) / spot * 1e4
    2. ``basis_change_bps``: 1-bar diff of basis_bps (lead/lag dynamics)
    3. ``ofi_diff``: fut_ofi_sum - spot_ofi_sum  (cross-venue OFI divergence)
    4. ``funding_bps_avg``: per-minute average of funding_rate × 1e4
    5. ``mark_premium_bps``: (mark_price - spot_microprice) / spot * 1e4
* Trains baseline (microstructure-only) AND augmented (microstructure +
  5 futures-basis features) CatBoost models with identical walk-forward
  geometry. Reports the dir_acc delta + permutation p-value on the
  delta itself.

Run from Tokyo
--------------

::

    python -m tools.futures_basis_eval --symbol BTCUSDT --since-hours 36
    python -m tools.futures_basis_eval --symbols BTCUSDT ETHUSDT BNBUSDT --since-hours 36

Note ``--since-hours`` is capped by the futures-table depth (~40h as
of 2026-05-01); going beyond that gets you fewer rows in the join.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


FUTURES_BASIS_FEATURE_COLUMNS: list[str] = [
    "basis_bps_close",
    "basis_change_bps",
    "ofi_diff_sum",
    "funding_bps_mean",
    "mark_premium_bps_close",
]


@dataclass
class ABResult:
    symbol: str
    n_seconds_spot: int
    n_seconds_futures: int
    n_bars_aligned: int
    n_folds_baseline: int
    n_folds_augmented: int
    dir_acc_baseline: float
    dir_acc_augmented: float
    dir_acc_delta: float
    ci_baseline_low: float
    ci_baseline_high: float
    ci_augmented_low: float
    ci_augmented_high: float
    perm_p_delta: float
    feature_importances: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def _load_seconds(database_url: str, *, table: str, symbol: str,
                  since_hours: float) -> pd.DataFrame:
    """Load seconds-level rows from spot or futures OFI table."""
    from sqlalchemy import create_engine, text
    eng = create_engine(database_url, future=True)
    extra_cols = ""
    if table == "highfreq_futures_ofi_1s":
        extra_cols = ", mark_price, funding_rate"
    query = text(f"""
        SELECT ts, symbol, ofi, microprice, depth_imb, spread_bps,
               trade_imb, vpin, n_updates, local_recv_ms{extra_cols}
        FROM {table}
        WHERE symbol = :symbol
          AND ts >= now() - (:hours * interval '1 hour')
        ORDER BY ts ASC
    """)
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol, "hours": since_hours})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def build_futures_basis_features(
    spot_minute_df: pd.DataFrame,
    futures_seconds_df: pd.DataFrame,
    *,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Aggregate futures seconds → bars → join onto spot bars by minute.

    Returns a DataFrame indexed positionally on ``spot_minute_df``'s
    rows, with the 5 futures-basis features. Rows where futures data
    is missing get zero-filled (model still trains).
    """
    from app.highfreq.feature_pipeline import aggregate_to_minute

    if futures_seconds_df.empty:
        return pd.DataFrame(
            0.0,
            index=range(len(spot_minute_df)),
            columns=FUTURES_BASIS_FEATURE_COLUMNS,
        )

    # Aggregate futures seconds → bars (same minute floor as spot).
    fut_min = aggregate_to_minute(futures_seconds_df, bar_minutes=bar_minutes)
    if fut_min.empty:
        return pd.DataFrame(
            0.0,
            index=range(len(spot_minute_df)),
            columns=FUTURES_BASIS_FEATURE_COLUMNS,
        )

    # Compute per-minute funding & mark_price (mean across the bar's seconds).
    fut_extra = futures_seconds_df.copy()
    fut_extra["minute"] = fut_extra["ts"].dt.floor(f"{bar_minutes}min")
    extra_agg = fut_extra.groupby(["symbol", "minute"]).agg(
        funding_rate_mean=("funding_rate", "mean"),
        mark_price_mean=("mark_price", "mean"),
    ).reset_index()
    fut_min = fut_min.merge(extra_agg, on=["symbol", "minute"], how="left")

    # Spot bars: use microprice_close as the spot reference price.
    spot = spot_minute_df[["minute", "symbol", "microprice_close"]].rename(
        columns={"microprice_close": "spot_microprice_close"},
    )
    fut = fut_min[["minute", "symbol", "microprice_close", "ofi_sum",
                   "funding_rate_mean", "mark_price_mean"]].rename(
        columns={"microprice_close": "fut_microprice_close",
                 "ofi_sum": "fut_ofi_sum"},
    )
    joined = spot.merge(fut, on=["minute", "symbol"], how="left")

    # Spot OFI sum is the raw column we need to compute ofi_diff. Pull it.
    spot_ofi = spot_minute_df[["minute", "symbol", "ofi_sum"]].rename(
        columns={"ofi_sum": "spot_ofi_sum"},
    )
    joined = joined.merge(spot_ofi, on=["minute", "symbol"], how="left")

    # Compute the 5 features.
    spot_close = joined["spot_microprice_close"].astype(float).replace(0, np.nan)
    fut_close = joined["fut_microprice_close"].astype(float)
    mark = joined["mark_price_mean"].astype(float)

    basis_bps = ((fut_close - spot_close) / spot_close * 1e4).fillna(0.0)
    basis_change = basis_bps.diff().fillna(0.0)
    ofi_diff = (joined["fut_ofi_sum"].astype(float).fillna(0.0)
                - joined["spot_ofi_sum"].astype(float).fillna(0.0))
    funding_bps = (joined["funding_rate_mean"].astype(float) * 1e4).fillna(0.0)
    mark_premium_bps = ((mark - spot_close) / spot_close * 1e4).fillna(0.0)

    out = pd.DataFrame({
        "basis_bps_close": basis_bps.astype(float).values,
        "basis_change_bps": basis_change.astype(float).values,
        "ofi_diff_sum": ofi_diff.astype(float).values,
        "funding_bps_mean": funding_bps.astype(float).values,
        "mark_premium_bps_close": mark_premium_bps.astype(float).values,
    })
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out.index = range(len(spot_minute_df))
    return out


def run_ab(database_url: str, *, symbol: str, since_hours: float,
           seed: int = 42, initial_train_hours: int = 10) -> ABResult:
    """Train baseline + augmented models at identical fold geometry.

    ``initial_train_hours``
        Walk-forward warm-up window. Default 10h (chosen for the
        first A/B when the futures table only had ~36h of data).
        For the **production-geometry replication**, use 24 (matches
        ``WalkForwardConfig`` defaults). This shrinks the fold count
        but the larger train window is what production actually
        uses, so the lift seen here transfers more cleanly.
    """
    from app.highfreq.trainer import (
        WalkForwardConfig, _make_supervised_for_feature_set,
        bootstrap_dir_acc_ci, walk_forward_evaluate,
    )
    from app.highfreq.feature_pipeline import aggregate_to_minute
    from catboost import CatBoostClassifier

    started = time.monotonic()

    # 1) Load spot + futures seconds.
    spot_secs = _load_seconds(database_url, table="highfreq_ofi_1s",
                              symbol=symbol, since_hours=since_hours)
    fut_secs = _load_seconds(database_url, table="highfreq_futures_ofi_1s",
                             symbol=symbol, since_hours=since_hours)
    logger.info("loaded %d spot, %d futures seconds for %s",
                len(spot_secs), len(fut_secs), symbol)

    # 2) Build baseline microstructure (X_base, y, meta).
    cfg = WalkForwardConfig(
        initial_train_minutes=initial_train_hours * 60,
        test_fold_minutes=60,
        step_minutes=60,
        frozen_holdout_days=0,
        bar_minutes=1,
        feature_set="microstructure",
        sample_weight_half_life_bars=0,
    )
    X_base, y, meta = _make_supervised_for_feature_set(
        spot_secs, feature_set=cfg.feature_set, bar_minutes=cfg.bar_minutes,
    )
    if X_base.empty:
        raise RuntimeError(f"{symbol}: no spot bars after pipeline")

    # 3) Build matching minute-frame for spot so we can join futures features.
    spot_min_df = aggregate_to_minute(spot_secs, bar_minutes=cfg.bar_minutes)
    # The supervised pipeline drops in-flight bars + neutral-band bars; we
    # need a matching mask to align futures features with X_base.
    # `meta` carries `minute` for every X_base row, so re-join futures
    # features by minute on `meta`.
    fut_feats = build_futures_basis_features(spot_min_df, fut_secs)
    # fut_feats is keyed by spot_min_df row index; align via merge on minute.
    fut_feats["_minute"] = pd.to_datetime(spot_min_df["minute"], utc=True).reset_index(drop=True)
    fut_aligned = (
        meta[["minute"]]
        .reset_index(drop=True)
        .merge(fut_feats, left_on="minute", right_on="_minute", how="left")
        .drop(columns=["_minute"])
        .fillna(0.0)
    )
    # Drop the `minute` col before concat — X_base has only feature cols.
    fut_aligned = fut_aligned.drop(columns=["minute"])
    X_aug = pd.concat(
        [X_base.reset_index(drop=True), fut_aligned.reset_index(drop=True)],
        axis=1,
    )

    # 4) Train + walk-forward on each.
    folds_b, preds_b = walk_forward_evaluate(X_base, y, meta, config=cfg)
    folds_a, preds_a = walk_forward_evaluate(X_aug, y, meta, config=cfg)

    if not folds_b or not folds_a:
        raise RuntimeError(f"{symbol}: insufficient folds (baseline={len(folds_b)}, augmented={len(folds_a)})")

    acc_b = float((preds_b["y_pred"].to_numpy() == preds_b["y_true"].to_numpy()).mean())
    acc_a = float((preds_a["y_pred"].to_numpy() == preds_a["y_true"].to_numpy()).mean())
    delta = acc_a - acc_b

    _, ci_b_lo, ci_b_hi = bootstrap_dir_acc_ci(
        preds_b["y_true"].to_numpy(), preds_b["y_pred"].to_numpy(), seed=seed,
    )
    _, ci_a_lo, ci_a_hi = bootstrap_dir_acc_ci(
        preds_a["y_true"].to_numpy(), preds_a["y_pred"].to_numpy(), seed=seed,
    )

    # Permutation p-value on the DELTA: pool all predictions, shuffle the
    # "augmented vs baseline" assignment 1000× and check how often the
    # shuffled delta ≥ observed.
    rng = np.random.default_rng(seed)
    n = len(preds_b)
    if len(preds_a) != n:
        # In rare cases the augmented model produces a different fold count
        # because of NaN handling; fall back to point comparison only.
        perm_p = float("nan")
    else:
        c_b = (preds_b["y_pred"].to_numpy() == preds_b["y_true"].to_numpy()).astype(int)
        c_a = (preds_a["y_pred"].to_numpy() == preds_a["y_true"].to_numpy()).astype(int)
        # diff[i] = augmented_correct[i] - baseline_correct[i] ∈ {-1, 0, 1}
        diff = c_a - c_b
        observed = float(diff.mean())
        # Permutation: each row's diff sign is randomly flipped (±diff).
        n_perm = 1000
        sample = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            signs = rng.choice([-1, 1], size=n)
            sample[i] = float((signs * diff).mean())
        perm_p = float((sample >= observed).sum() + 1) / (n_perm + 1)

    # Feature importance on the augmented model (re-fit on all data).
    final = CatBoostClassifier(
        iterations=cfg.catboost_iterations,
        depth=cfg.catboost_depth,
        learning_rate=cfg.catboost_learning_rate,
        loss_function="Logloss",
        thread_count=cfg.catboost_thread_count,
        random_seed=cfg.random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    final.fit(X_aug.values, y.values)
    importances = final.get_feature_importance()
    feature_importances = dict(zip(X_aug.columns.tolist(), importances.tolist()))
    # Restrict to futures features for legibility.
    futures_importances = {
        k: feature_importances[k]
        for k in FUTURES_BASIS_FEATURE_COLUMNS
        if k in feature_importances
    }

    return ABResult(
        symbol=symbol,
        n_seconds_spot=len(spot_secs),
        n_seconds_futures=len(fut_secs),
        n_bars_aligned=len(X_aug),
        n_folds_baseline=len(folds_b),
        n_folds_augmented=len(folds_a),
        dir_acc_baseline=acc_b,
        dir_acc_augmented=acc_a,
        dir_acc_delta=delta,
        ci_baseline_low=ci_b_lo,
        ci_baseline_high=ci_b_hi,
        ci_augmented_low=ci_a_lo,
        ci_augmented_high=ci_a_hi,
        perm_p_delta=perm_p,
        feature_importances=futures_importances,
        elapsed_seconds=time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", action="append", default=None,
                   help="symbol to evaluate; pass multiple times")
    p.add_argument("--since-hours", type=float, default=36.0,
                   help="window of history (default 36h, capped by futures depth)")
    p.add_argument("--initial-train-hours", type=int, default=10,
                   help="walk-forward warm-up window (default 10h; "
                        "use 24 for production-geometry replication)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="weights/highfreq/futures_basis_eval.json")
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

    symbols = args.symbol or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    results: list[ABResult] = []
    for sym in symbols:
        sym = sym.upper()
        logger.info("=" * 60)
        logger.info("A/B for %s (since_hours=%s)", sym, args.since_hours)
        try:
            r = run_ab(dsn, symbol=sym, since_hours=args.since_hours,
                       seed=args.seed,
                       initial_train_hours=args.initial_train_hours)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", sym, exc, exc_info=True)
            continue
        results.append(r)
        logger.info(
            "%s: baseline=%.4f → augmented=%.4f (Δ=%+.4f, perm_p=%.5g) "
            "n_bars=%d folds=%d/%d elapsed=%.1fs",
            sym, r.dir_acc_baseline, r.dir_acc_augmented, r.dir_acc_delta,
            r.perm_p_delta, r.n_bars_aligned, r.n_folds_baseline, r.n_folds_augmented,
            r.elapsed_seconds,
        )
        logger.info("  futures feature importances: %s",
                    {k: round(v, 2) for k, v in r.feature_importances.items()})

    if results:
        from datetime import datetime, timezone
        from pathlib import Path
        out = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "since_hours": args.since_hours,
            "results": [asdict(r) for r in results],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        logger.info("wrote %s", args.out)

        # Markdown summary printed to stdout for human consumption.
        print()  # noqa: T201
        print("| symbol | bars | folds | baseline | augmented | Δ      | perm p   |")  # noqa: T201
        print("|--------|------|-------|----------|-----------|--------|----------|")  # noqa: T201
        for r in results:
            print(  # noqa: T201
                f"| {r.symbol} | {r.n_bars_aligned} | "
                f"{r.n_folds_baseline}/{r.n_folds_augmented} | "
                f"{r.dir_acc_baseline:.4f} | {r.dir_acc_augmented:.4f} | "
                f"{r.dir_acc_delta:+.4f} | {r.perm_p_delta:.4g} |"
            )

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
