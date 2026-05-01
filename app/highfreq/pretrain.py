"""Pre-train CatBoost on multi-year historical klines.

Pipeline
========

1. Load a parquet of minute-bars produced by
   ``tools.binance_klines_download``.
2. Build target (``sign(microprice_close[t+H] - microprice_close[t])``)
   + neutral-band drop, identical to the live trainer's pipeline.
3. Build features via ``feature_pipeline_long_horizon`` — OHLC + EMA +
   RSI + Bollinger + calendar features. These don't need OFI / depth /
   spread (Klines doesn't carry those), so the pipeline is happy with
   the zero-filled neutral columns from the downloader.
4. Walk-forward CV (1-day folds across multi-year history → ~700+ folds).
5. Fit final CatBoost model on the FULL multi-year history.
6. Save .cbm + a small metrics.json.

The output ``.cbm`` is the **pre-trained checkpoint** that the live
trainer (``app.highfreq.trainer``) loads via ``--init-from`` and
fine-tunes on the recent live OFI data via CatBoost's ``init_model=``
parameter.

Why long_horizon (not microstructure)
-------------------------------------

Microstructure features (OFI, depth_imb, spread_bps, vpin) come from
L2 book updates which Binance Klines doesn't carry. We could fill
zeros and pretend, but the model would learn "OFI is always 0", which
HURTS at fine-tune time when real OFI shows up. Better: use the
already-implemented long_horizon pipeline (24 cols, OHLC-derived only)
that doesn't touch microstructure columns.

Trade-off: long_horizon is empirically -0.5pp vs microstructure on
1-min bars on small samples. The hypothesis is that years of pretrain
overcomes that gap by giving the model regime knowledge that 5 days
of training never could.

Run from any host
-----------------

::

    python -m app.highfreq.pretrain \\
        --data data/historical/btcusdt_1m_klines.parquet \\
        --symbol BTCUSDT \\
        --out weights/highfreq/btcusdt_1m_pretrained.cbm

The output .cbm is then consumed by the live trainer:

    python -m app.highfreq.trainer \\
        --symbol BTCUSDT --since-hours 93 --feature-set long_horizon \\
        --init-from weights/highfreq/btcusdt_1m_pretrained.cbm \\
        --out weights/highfreq/btcusdt_1m.cbm
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PretrainReport:
    """JSON-serialisable summary written next to the .cbm."""
    symbol: str
    pretrained_at: str
    data_path: str
    n_bars_loaded: int
    n_bars_after_neutral_drop: int
    n_folds: int
    dir_acc_mean: float | None
    dir_acc_ci_low: float | None
    dir_acc_ci_high: float | None
    log_loss_mean: float | None
    base_rate: float | None
    bar_minutes: int
    feature_set: str = "long_horizon"
    weights_path: str | None = None
    elapsed_seconds: float = 0.0

    def to_json(self) -> str:
        def _scrub(o: Any) -> Any:
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: _scrub(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_scrub(v) for v in o]
            return o
        return json.dumps(_scrub(asdict(self)), indent=2, default=str)


def _build_supervised(
    minute_df: pd.DataFrame,
    *,
    horizon: int,
    neutral_band_bps: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Same shape as trainer._make_supervised_for_feature_set, but
    operates on a minute-bar DataFrame already aggregated (klines, not
    seconds). Avoids re-running ``aggregate_to_minute`` since klines
    are pre-aggregated."""
    from app.highfreq.feature_pipeline import build_target
    from app.highfreq.feature_pipeline_long_horizon import (
        LONG_HORIZON_FEATURE_COLUMNS,
        build_long_horizon_features,
    )

    targeted = build_target(
        minute_df, horizon=horizon, neutral_band_bps=neutral_band_bps,
    )
    if targeted.empty:
        empty_X = pd.DataFrame(columns=LONG_HORIZON_FEATURE_COLUMNS)
        empty_y = pd.Series(dtype=np.int8, name="y")
        empty_meta = pd.DataFrame(
            columns=["symbol", "minute", "microprice_close", "return_bps"],
        )
        return empty_X, empty_y, empty_meta
    keep = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
    targeted = targeted.loc[keep].reset_index(drop=True)
    X = build_long_horizon_features(targeted)[LONG_HORIZON_FEATURE_COLUMNS]
    y = targeted["y"].astype(np.int8)
    meta = targeted[["symbol", "minute", "microprice_close", "return_bps"]].copy()
    return X, y, meta


def run_pretrain(
    *,
    data_path: Path,
    symbol: str,
    out_path: Path | None,
    bar_minutes: int = 1,
    horizon: int = 1,
    neutral_band_bps: float = 1.0,
    catboost_iterations: int = 800,
    catboost_depth: int = 5,
    catboost_learning_rate: float = 0.05,
    random_seed: int = 42,
    walk_forward: bool = True,
) -> PretrainReport:
    """Load parquet → build supervised → walk-forward CV → fit final."""
    from app.highfreq.trainer import (
        WalkForwardConfig, walk_forward_evaluate,
        bootstrap_dir_acc_ci,
    )
    from datetime import datetime, timezone

    started = time.monotonic()
    logger.info("loading parquet from %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("loaded %d minute-bars for %s", len(df), symbol)
    if df.empty:
        return PretrainReport(
            symbol=symbol,
            pretrained_at=datetime.now(tz=timezone.utc).isoformat(),
            data_path=str(data_path),
            n_bars_loaded=0,
            n_bars_after_neutral_drop=0,
            n_folds=0,
            dir_acc_mean=None,
            dir_acc_ci_low=None,
            dir_acc_ci_high=None,
            log_loss_mean=None,
            base_rate=None,
            bar_minutes=bar_minutes,
            elapsed_seconds=time.monotonic() - started,
        )

    # Filter to the right symbol if the parquet has multiple.
    df = df[df["symbol"].str.upper() == symbol.upper()].copy()
    df = df.sort_values("minute").reset_index(drop=True)

    X, y, meta = _build_supervised(
        df, horizon=horizon, neutral_band_bps=neutral_band_bps,
    )
    logger.info(
        "after target+neutral-band drop: %d bars (%.1f%% kept)",
        len(X), 100.0 * len(X) / max(1, len(df)),
    )
    if len(X) < 200:
        raise RuntimeError(
            f"only {len(X)} bars after target/neutral-band drop — "
            f"need ≥200 for any meaningful walk-forward"
        )

    base_rate = float(max(y.mean(), 1.0 - y.mean()))

    # Walk-forward CV (optional — can skip when pretraining on huge
    # data and we only care about the final fit).
    n_folds = 0
    dir_acc_mean = None
    ci_lo = ci_hi = None
    log_loss_mean = None
    if walk_forward:
        cfg = WalkForwardConfig(
            initial_train_minutes=24 * 60,
            test_fold_minutes=60,
            step_minutes=60,
            min_train_samples=200,
            catboost_iterations=catboost_iterations,
            catboost_depth=catboost_depth,
            catboost_learning_rate=catboost_learning_rate,
            bar_minutes=bar_minutes,
            feature_set="long_horizon",
            sample_weight_half_life_bars=0,  # uniform on multi-year history
            frozen_holdout_days=0,
        )
        # Reduce iterations during CV to avoid hours of compute on
        # millions of bars. The final fit uses full iterations.
        cfg_cv = WalkForwardConfig(**{**asdict(cfg), "catboost_iterations": 200})
        logger.info("walk-forward CV…")
        folds, preds = walk_forward_evaluate(X, y, meta, config=cfg_cv)
        n_folds = len(folds)
        if folds:
            dir_acc_mean = float(np.array([f.dir_acc for f in folds]).mean())
            log_loss_mean = float(np.array([f.log_loss for f in folds]).mean())
            _, ci_lo, ci_hi = bootstrap_dir_acc_ci(
                preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                seed=random_seed,
            )
            logger.info(
                "walk-forward: %d folds, dir_acc=%.4f CI=[%.4f, %.4f] logloss=%.4f",
                n_folds, dir_acc_mean, ci_lo, ci_hi, log_loss_mean,
            )

    # Final fit on FULL data — this is the pre-trained checkpoint.
    weights_path: str | None = None
    if out_path is not None:
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=catboost_iterations,
            depth=catboost_depth,
            learning_rate=catboost_learning_rate,
            loss_function="Logloss",
            thread_count=2,
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        logger.info("fitting final CatBoost on %d bars (this is the pretrain)…", len(X))
        clf.fit(X.values, y.values)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save_model(str(out_path), format="cbm")
        weights_path = str(out_path)
        logger.info("saved pretrained checkpoint to %s", out_path)

    return PretrainReport(
        symbol=symbol,
        pretrained_at=datetime.now(tz=timezone.utc).isoformat(),
        data_path=str(data_path),
        n_bars_loaded=len(df),
        n_bars_after_neutral_drop=len(X),
        n_folds=n_folds,
        dir_acc_mean=dir_acc_mean,
        dir_acc_ci_low=ci_lo,
        dir_acc_ci_high=ci_hi,
        log_loss_mean=log_loss_mean,
        base_rate=base_rate,
        bar_minutes=bar_minutes,
        weights_path=weights_path,
        elapsed_seconds=time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True,
                   help="parquet path (output of tools.binance_klines_download)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--out", default=None,
                   help="output .cbm path; default <data_dir>/<symbol>_pretrained.cbm")
    p.add_argument("--report", default=None,
                   help="output JSON path; default same dir as --out, .json")
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--neutral-band-bps", type=float, default=1.0)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--no-walk-forward", action="store_true",
                   help="skip walk-forward CV (faster — useful when "
                        "you just want the .cbm checkpoint)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("data file not found: %s", data_path)
        return 2

    out_path = Path(args.out) if args.out else (
        data_path.parent / f"{args.symbol.lower()}_{args.bar_minutes}m_pretrained.cbm"
    )
    report_path = Path(args.report) if args.report else out_path.with_suffix(".json")

    report = run_pretrain(
        data_path=data_path,
        symbol=args.symbol,
        out_path=out_path,
        bar_minutes=args.bar_minutes,
        horizon=args.horizon,
        neutral_band_bps=args.neutral_band_bps,
        catboost_iterations=args.iterations,
        catboost_depth=args.depth,
        catboost_learning_rate=args.learning_rate,
        random_seed=args.seed,
        walk_forward=not args.no_walk_forward,
    )
    report_path.write_text(report.to_json())
    logger.info(
        "PRETRAIN DONE | %s | bars=%d folds=%d dir_acc=%.4f → %s",
        report.symbol, report.n_bars_after_neutral_drop, report.n_folds,
        report.dir_acc_mean if report.dir_acc_mean is not None else float("nan"),
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
