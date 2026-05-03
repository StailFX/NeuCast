"""A/B comparison: Triple-Barrier Labels vs fixed-horizon direction
target on the same live data (T.17.a).

The trainer's default target is binary direction:
``y = sign(microprice_close[t+1] - microprice_close[t])``. That's
the simplest baseline. López de Prado's triple-barrier labels are
trade-aligned: ``y = 1`` if a long entry would have hit a +tp_bps
take-profit before a -sl_bps stop-loss within ``time_stop_bars``.

This script trains BOTH on the same live OFI window and reports
whether TBL gives a meaningfully different / better classifier.

Outputs
=======

Markdown table to stdout + ``weights/highfreq/tbl_vs_direction.json``::

    | symbol | n_dir | dir_acc_dir | n_tbl | dir_acc_tbl | Δ |
    |--------|-------|-------------|-------|-------------|---|
    | BTCUSDT | 6641 | 0.5519 | 4128 | 0.5712 | +1.93pp |

Run
---

::

    python -m tools.tbl_vs_direction_eval \\
        --symbols BTCUSDT ETHUSDT BNBUSDT --since-hours 165
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ABResult:
    symbol: str
    bar_minutes: int
    feature_set: str
    # Fixed-horizon direction target.
    n_direction: int
    dir_acc_direction: float
    ci_low_direction: float
    ci_high_direction: float
    # Triple-barrier binary target.
    tp_bps: float
    sl_bps: float
    time_stop_bars: int
    n_tbl_total: int            # bars after lookahead
    n_tbl_tp: int                # tbl_y == 1
    n_tbl_sl: int                # tbl_y == 0
    n_tbl_timestop: int          # tbl_y == 2
    n_tbl_train: int             # bars used for tbl fit (TP + SL only)
    dir_acc_tbl: float
    ci_low_tbl: float
    ci_high_tbl: float
    delta: float                 # dir_acc_tbl - dir_acc_direction
    elapsed_seconds: float = 0.0


def _build_supervised_with_tbl(
    df_secs: pd.DataFrame,
    *,
    target_symbol: str,
    feature_set: str,
    bar_minutes: int,
    tp_bps: float,
    sl_bps: float,
    time_stop_bars: int,
    reference_df_secs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build (X, y_tbl, meta) where y_tbl ∈ {0, 1} from the TBL
    pipeline. Time-stop bars (tbl_y == 2) and insufficient-lookahead
    bars (tbl_y == -1) are dropped — the binary classifier learns
    to discriminate "long would have hit TP" vs "long would have
    hit SL", which is what's actionable for a paper-trader."""
    from app.highfreq.feature_pipeline import (
        aggregate_to_minute,
        build_triple_barrier_labels,
        build_features,
        FEATURE_COLUMNS,
    )

    minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
    if minute_df.empty:
        return pd.DataFrame(), pd.Series(dtype=np.int8), pd.DataFrame()

    labelled = build_triple_barrier_labels(
        minute_df,
        tp_bps=tp_bps, sl_bps=sl_bps, time_stop_bars=time_stop_bars,
    )
    keep = labelled["tbl_y"].isin([0, 1])  # drop -1 + 2
    labelled = labelled.loc[keep].reset_index(drop=True)
    if labelled.empty:
        return pd.DataFrame(), pd.Series(dtype=np.int8), pd.DataFrame()

    if feature_set == "long_horizon":
        from app.highfreq.feature_pipeline_long_horizon import (
            LONG_HORIZON_FEATURE_COLUMNS,
            build_long_horizon_features,
        )
        X = build_long_horizon_features(labelled)[LONG_HORIZON_FEATURE_COLUMNS]
    elif feature_set == "cross_asset":
        from app.highfreq.feature_pipeline_cross_asset import (
            build_cross_asset_features, feature_columns_for,
        )
        sym_upper = (target_symbol or "").upper()
        ref_sym = None if sym_upper == "BTCUSDT" else "BTCUSDT"
        ref_minutes: pd.DataFrame | None = None
        if ref_sym is not None and reference_df_secs is not None:
            ref_minutes = aggregate_to_minute(
                reference_df_secs, bar_minutes=bar_minutes,
            )
        X = build_cross_asset_features(
            labelled, reference_minutes=ref_minutes, reference_symbol=ref_sym,
        )
        X = X[feature_columns_for(ref_sym)]
    else:
        X = build_features(labelled)[FEATURE_COLUMNS]

    y_tbl = labelled["tbl_y"].astype(np.int8)
    meta = labelled[
        ["symbol", "minute", "microprice_close",
         "tbl_first_hit", "tbl_first_hit_bars"]
    ].copy()
    return X, y_tbl, meta


def run_ab(
    *,
    database_url: str,
    symbol: str,
    since_hours: float,
    bar_minutes: int,
    feature_set: str,
    tp_bps: float,
    sl_bps: float,
    time_stop_bars: int,
    seed: int = 42,
) -> ABResult:
    """Train CatBoost on direction target + TBL target, walk-forward
    each, return both metrics + delta."""
    from app.highfreq.trainer import (
        WalkForwardConfig,
        _make_supervised_for_feature_set,
        bootstrap_dir_acc_ci,
        load_seconds,
        walk_forward_evaluate,
    )

    started = time.monotonic()

    # Load live OFI seconds.
    df_secs = load_seconds(database_url, symbol=symbol, since_hours=since_hours)
    ref_secs = None
    if feature_set == "cross_asset" and symbol.upper() != "BTCUSDT":
        ref_secs = load_seconds(
            database_url, symbol="BTCUSDT", since_hours=since_hours,
        )

    # ── Direction target (existing pipeline) ────────────────────────
    X_dir, y_dir, meta_dir = _make_supervised_for_feature_set(
        df_secs,
        feature_set=feature_set,
        bar_minutes=bar_minutes,
        reference_df_secs=ref_secs,
        target_symbol=symbol,
    )
    cfg = WalkForwardConfig(
        initial_train_minutes=24 * 60,
        bar_minutes=bar_minutes,
        feature_set=feature_set,
        sample_weight_half_life_bars=0,
        frozen_holdout_days=0,
    )
    folds_dir, preds_dir = walk_forward_evaluate(X_dir, y_dir, meta_dir, config=cfg)
    if not folds_dir:
        raise RuntimeError(f"{symbol}: direction baseline produced no CV folds")
    dir_acc_dir = float(
        (preds_dir["y_pred"].to_numpy() == preds_dir["y_true"].to_numpy()).mean()
    )
    _, ci_dir_lo, ci_dir_hi = bootstrap_dir_acc_ci(
        preds_dir["y_true"].to_numpy(), preds_dir["y_pred"].to_numpy(),
        seed=seed,
    )

    # ── TBL target ─────────────────────────────────────────────────
    X_tbl, y_tbl_full, meta_tbl_full = _build_supervised_with_tbl(
        df_secs,
        target_symbol=symbol,
        feature_set=feature_set,
        bar_minutes=bar_minutes,
        tp_bps=tp_bps, sl_bps=sl_bps, time_stop_bars=time_stop_bars,
        reference_df_secs=ref_secs,
    )
    # Count distribution: aggregate again to compute time_stop / TP / SL.
    from app.highfreq.feature_pipeline import (
        aggregate_to_minute,
        build_triple_barrier_labels,
    )
    minute_df_full = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
    labelled_full = build_triple_barrier_labels(
        minute_df_full,
        tp_bps=tp_bps, sl_bps=sl_bps, time_stop_bars=time_stop_bars,
    )
    eligible = labelled_full[labelled_full["tbl_y"] != -1]
    n_total = int(len(eligible))
    n_tp = int((eligible["tbl_y"] == 1).sum())
    n_sl = int((eligible["tbl_y"] == 0).sum())
    n_ts = int((eligible["tbl_y"] == 2).sum())

    if X_tbl.empty:
        raise RuntimeError(
            f"{symbol}: TBL pipeline produced no labelled bars; "
            f"thresholds may be too tight or data too sparse"
        )

    folds_tbl, preds_tbl = walk_forward_evaluate(X_tbl, y_tbl_full, meta_tbl_full, config=cfg)
    if not folds_tbl:
        raise RuntimeError(
            f"{symbol}: TBL fit produced no CV folds; n_train={len(X_tbl)}"
        )
    dir_acc_tbl = float(
        (preds_tbl["y_pred"].to_numpy() == preds_tbl["y_true"].to_numpy()).mean()
    )
    _, ci_tbl_lo, ci_tbl_hi = bootstrap_dir_acc_ci(
        preds_tbl["y_true"].to_numpy(), preds_tbl["y_pred"].to_numpy(),
        seed=seed,
    )

    return ABResult(
        symbol=symbol,
        bar_minutes=bar_minutes,
        feature_set=feature_set,
        n_direction=int(len(preds_dir)),
        dir_acc_direction=dir_acc_dir,
        ci_low_direction=ci_dir_lo,
        ci_high_direction=ci_dir_hi,
        tp_bps=tp_bps, sl_bps=sl_bps, time_stop_bars=time_stop_bars,
        n_tbl_total=n_total,
        n_tbl_tp=n_tp, n_tbl_sl=n_sl, n_tbl_timestop=n_ts,
        n_tbl_train=int(len(preds_tbl)),
        dir_acc_tbl=dir_acc_tbl,
        ci_low_tbl=ci_tbl_lo,
        ci_high_tbl=ci_tbl_hi,
        delta=dir_acc_tbl - dir_acc_dir,
        elapsed_seconds=time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", action="append", default=None,
                   help="symbols to evaluate; default BTC/ETH/BNB")
    p.add_argument("--since-hours", type=float, default=165.0)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--tp-bps", type=float, default=5.0,
                   help="take-profit threshold in bps (default 5)")
    p.add_argument("--sl-bps", type=float, default=5.0,
                   help="stop-loss threshold in bps (default 5)")
    p.add_argument("--time-stop-bars", type=int, default=10,
                   help="time-stop horizon in bars (default 10)")
    p.add_argument("--out", default="weights/highfreq/tbl_vs_direction.json")
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

    symbols = [s.upper() for s in (args.symbol or ["BTCUSDT", "ETHUSDT", "BNBUSDT"])]

    # Per-symbol feature_set: BTC microstructure, ETH/BNB cross_asset
    # (matches production T.13).
    fs_for = lambda s: "microstructure" if s == "BTCUSDT" else "cross_asset"

    results: list[ABResult] = []
    for sym in symbols:
        logger.info("=" * 60)
        logger.info("A/B for %s (since=%.0fh tp=%.1fbp sl=%.1fbp ts=%d)",
                    sym, args.since_hours, args.tp_bps, args.sl_bps,
                    args.time_stop_bars)
        try:
            r = run_ab(
                database_url=dsn,
                symbol=sym,
                since_hours=args.since_hours,
                bar_minutes=args.bar_minutes,
                feature_set=fs_for(sym),
                tp_bps=args.tp_bps,
                sl_bps=args.sl_bps,
                time_stop_bars=args.time_stop_bars,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", sym, exc, exc_info=True)
            continue
        results.append(r)
        logger.info(
            "%s | dir(n=%d acc=%.4f CI=[%.4f, %.4f]) | "
            "tbl(n=%d tp=%d sl=%d ts=%d acc=%.4f CI=[%.4f, %.4f]) | "
            "Δ=%+.4f | %.1fs",
            sym, r.n_direction, r.dir_acc_direction,
            r.ci_low_direction, r.ci_high_direction,
            r.n_tbl_train, r.n_tbl_tp, r.n_tbl_sl, r.n_tbl_timestop,
            r.dir_acc_tbl, r.ci_low_tbl, r.ci_high_tbl,
            r.delta, r.elapsed_seconds,
        )

    if not results:
        return 1

    # Markdown summary.
    print()  # noqa: T201
    print("| symbol | n_dir | dir_acc_dir | n_tbl_train | TP/SL/TS | dir_acc_tbl | Δ |")  # noqa: T201
    print("|--------|-------|-------------|-------------|----------|-------------|---|")  # noqa: T201
    for r in results:
        sign = "+" if r.delta >= 0 else ""
        print(  # noqa: T201
            f"| {r.symbol} | {r.n_direction} | {r.dir_acc_direction:.4f} "
            f"| {r.n_tbl_train} | {r.n_tbl_tp}/{r.n_tbl_sl}/{r.n_tbl_timestop} "
            f"| {r.dir_acc_tbl:.4f} | {sign}{r.delta * 100:.2f}pp |"
        )

    # JSON dump.
    from datetime import datetime, timezone
    out = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "params": {
            "since_hours": args.since_hours,
            "bar_minutes": args.bar_minutes,
            "tp_bps": args.tp_bps,
            "sl_bps": args.sl_bps,
            "time_stop_bars": args.time_stop_bars,
        },
        "results": [asdict(r) for r in results],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
