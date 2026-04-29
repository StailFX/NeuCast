"""Magnitude-regression evaluation (release P / roadmap ε).

Why this exists
===============

Production ships a *classification* model — predicts ``P(price up at t+1)``
for a directional bet. That works for "should I be long or short next
minute?" but throws away two pieces of information the operator wants:

1. **Confidence (size)** — a 60% probability bet on a 1-bp move is worth
   far less than a 60% probability bet on a 4-bp move. Kelly sizing needs
   ``E[return]``, not ``P(positive)``.
2. **Fee-aware filtering** — at retail fees (~7.5 bp/side, 15 bp
   round-trip), a sub-2-bp expected move is unprofitable regardless of
   directional skill. The classifier emits one signal per bar; the
   regressor lets us *skip* low-magnitude bars cleanly.

This tool is **offline-only**. It never writes a ``.cbm``, never touches
production weights, and the existing classification path keeps running
unchanged. The output is a markdown report comparing:

* MAE / RMSE / R² / IC  — does the regressor capture magnitude?
* Sign accuracy — when sign(predicted return) is taken as a signal,
  does it match the classifier's dir_acc? (Apples-to-apples.)
* Threshold curves — at "trade only when |E[r]| > θ" filter levels
  (θ ∈ {0, 1, 2, 4, 8} bp), what's the hit rate, mean realized
  return, and after-fee P&L per fee tier?

Empirical question
------------------

If the regressor's sign accuracy is materially below the classifier's,
the classifier is genuinely a better directional model and the
regressor's only edge is fee-aware filtering. If sign accuracy is
*comparable*, the regressor strictly dominates (same direction +
free magnitude info → free Kelly sizing).

We don't know the answer until we run it on real data. That's the
point of this tool.

Usage
-----

::

    python -m tools.regression_eval \\
        --symbols BTCUSDT ETHUSDT BNBUSDT \\
        --horizons 1 5 15 \\
        --since-hours 168

Output:
    weights/highfreq/regression_eval.json   (machine-readable)
    docs/highfreq/regression_eval_report.md  (defence-grade markdown)
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


# Fee tiers, mirror app.highfreq.fee_tiers (one bar of code; keeps this
# script standalone-runnable without that import).
FEE_TIERS_BPS: dict[str, float] = {
    "retail":    7.5,
    "vip5":      1.0,
    "vip9":      0.0,
    "mm_rebate": -0.4,
}

#: Threshold curve points (bps) — at each |E[r]| threshold we report
#: how many trades passed the filter, what fraction had matching sign,
#: and the realized P&L per fee tier.
DEFAULT_THRESHOLDS_BPS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)


@dataclass
class ThresholdRow:
    """Slice of OOS predictions where |predicted_bps| > threshold."""
    threshold_bps: float
    n_trades: int
    fraction_kept: float                # n_trades / n_total_predictions
    sign_accuracy: float | None         # share where sign(pred)==sign(actual)
    mean_realized_bps: float | None     # E[realized return] over the slice
    mean_abs_realized_bps: float | None
    # Conditional on taking the trade (signed by sign(pred)), what's the
    # mean realized return AFTER subtracting fees per tier?
    realized_pnl_bps: dict[str, float | None] = field(default_factory=dict)


@dataclass
class RegressionEvalRow:
    """One row of the comparison — per (symbol, bar_minutes)."""
    symbol: str
    bar_minutes: int
    n_seconds_loaded: int
    n_bars_after_aggregation: int
    n_bars_kept: int                  # bars used for training (no NaN)
    n_folds: int
    n_predictions: int                # pooled OOS sample size
    # Regression goodness-of-fit on pooled OOS predictions.
    mae_bps: float | None             # mean absolute error
    rmse_bps: float | None            # root-mean-square error
    r2: float | None                  # coefficient of determination
    ic_pearson: float | None          # Pearson IC (correlation)
    ic_spearman: float | None         # rank IC (more robust to outliers)
    # Sign-based bridge to classifier — directly comparable to dir_acc.
    sign_accuracy: float | None
    # Threshold curves (Kelly / fee-aware sizing analysis).
    thresholds: list[ThresholdRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────── helpers ─────────────────────────

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation that returns None when undefined (constant
    arrays or NaN). Defensive against the degenerate case where the
    regressor predicts a constant — don't blow up, return None."""
    if len(a) < 2 or len(b) < 2:
        return None
    sa = a.std()
    sb = b.std()
    if sa == 0 or sb == 0 or not np.isfinite(sa) or not np.isfinite(sb):
        return None
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else None


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation (Spearman) — more robust than Pearson when one
    side has fat tails, common in short-horizon return distributions."""
    if len(a) < 2 or len(b) < 2:
        return None
    # scipy is a transitive dep through sklearn; safe to import.
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, b, nan_policy="omit")
        return float(rho) if np.isfinite(rho) else None
    except ImportError:
        return None


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Coefficient of determination. Returns None when the variance of
    ``y_true`` is zero (no signal to explain)."""
    if len(y_true) == 0:
        return None
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _threshold_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold_bps: float,
) -> ThresholdRow:
    """Compute per-threshold diagnostics.

    For trades where ``|y_pred| > threshold_bps`` we measure:
    * how many trades passed,
    * fraction kept (vs total),
    * sign accuracy (fraction where sign(pred) == sign(actual)),
    * mean realized return,
    * realized P&L per fee tier (signed by sign(pred)).
    """
    mask = np.abs(y_pred) > threshold_bps
    n_total = len(y_pred)
    n_trades = int(mask.sum())
    fraction_kept = float(n_trades / n_total) if n_total else 0.0

    if n_trades == 0:
        return ThresholdRow(
            threshold_bps=float(threshold_bps),
            n_trades=0,
            fraction_kept=fraction_kept,
            sign_accuracy=None,
            mean_realized_bps=None,
            mean_abs_realized_bps=None,
            realized_pnl_bps={tier: None for tier in FEE_TIERS_BPS},
        )

    yt = y_true[mask]
    yp = y_pred[mask]
    sign_match = (np.sign(yp) == np.sign(yt)) & (yp != 0)
    sign_acc = float(sign_match.mean()) if n_trades else None
    mean_realized = float(yt.mean())
    mean_abs_realized = float(np.abs(yt).mean())

    # Realized P&L per tier: take long when pred>0, short when pred<0,
    # capture sign(pred)*y_true minus round-trip fees.
    signed_realized = np.sign(yp) * yt
    pnl_per_tier: dict[str, float | None] = {}
    for tier, fee_bps_per_side in FEE_TIERS_BPS.items():
        roundtrip = 2.0 * fee_bps_per_side
        pnl_per_tier[tier] = float(signed_realized.mean() - roundtrip)

    return ThresholdRow(
        threshold_bps=float(threshold_bps),
        n_trades=n_trades,
        fraction_kept=fraction_kept,
        sign_accuracy=sign_acc,
        mean_realized_bps=mean_realized,
        mean_abs_realized_bps=mean_abs_realized,
        realized_pnl_bps=pnl_per_tier,
    )


# ───────────────────────── walk-forward CV ─────────────────────────

def evaluate_one_regression(
    df_secs: pd.DataFrame,
    *,
    symbol: str,
    bar_minutes: int,
    initial_train_bars: int,
    test_fold_bars: int,
    step_bars: int,
    neutral_band_bps: float = 1.0,  # mirror classifier's drop for apples-to-apples
    catboost_iterations: int = 400,
    catboost_depth: int = 5,
    catboost_learning_rate: float = 0.05,
    random_seed: int = 42,
    sample_weight_half_life: int = 720,
    embargo_bars: int = 1,
    thresholds_bps: tuple[float, ...] = DEFAULT_THRESHOLDS_BPS,
) -> RegressionEvalRow:
    """Walk-forward CV with CatBoostRegressor on continuous return_bps.

    Mirrors ``tools.multi_horizon_eval.evaluate_one_horizon`` for the
    classifier path so results are directly comparable: same data, same
    bar size, same neutral-band drop, same fold geometry.
    """
    from app.highfreq.feature_pipeline import (
        FEATURE_COLUMNS,
        aggregate_to_minute,
        build_features,
        build_target,
    )

    n_seconds_loaded = int(len(df_secs))

    minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
    n_bars_agg = int(len(minute_df))

    targeted = build_target(minute_df, horizon=1, neutral_band_bps=neutral_band_bps)
    if targeted.empty:
        return RegressionEvalRow(
            symbol=symbol, bar_minutes=bar_minutes,
            n_seconds_loaded=n_seconds_loaded,
            n_bars_after_aggregation=n_bars_agg,
            n_bars_kept=0, n_folds=0, n_predictions=0,
            mae_bps=None, rmse_bps=None, r2=None,
            ic_pearson=None, ic_spearman=None,
            sign_accuracy=None,
            thresholds=[],
        )

    # Drop unobservable bars + neutral-band bars (matches classifier).
    keep_mask = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
    targeted = targeted.loc[keep_mask].reset_index(drop=True)
    X = build_features(targeted)[FEATURE_COLUMNS]
    # Continuous target — return_bps in basis points.
    y = targeted["return_bps"].astype(float)
    n_bars_kept = int(len(X))

    if n_bars_kept < initial_train_bars + test_fold_bars:
        return RegressionEvalRow(
            symbol=symbol, bar_minutes=bar_minutes,
            n_seconds_loaded=n_seconds_loaded,
            n_bars_after_aggregation=n_bars_agg,
            n_bars_kept=n_bars_kept, n_folds=0, n_predictions=0,
            mae_bps=None, rmse_bps=None, r2=None,
            ic_pearson=None, ic_spearman=None,
            sign_accuracy=None,
            thresholds=[],
        )

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise RuntimeError("catboost required: pip install catboost")

    pooled_y_true: list[float] = []
    pooled_y_pred: list[float] = []
    n_folds = 0
    train_end = initial_train_bars
    while train_end + test_fold_bars <= n_bars_kept:
        train_eff_end = max(0, train_end - max(0, embargo_bars))
        X_tr = X.iloc[:train_eff_end].to_numpy()
        y_tr = y.iloc[:train_eff_end].to_numpy()
        X_te = X.iloc[train_end:train_end + test_fold_bars].to_numpy()
        y_te = y.iloc[train_end:train_end + test_fold_bars].to_numpy()
        if len(X_tr) < 100:
            train_end += step_bars
            continue
        # Skip degenerate folds where all y_tr are identical (CatBoost
        # would produce a constant, contaminating IC with NaN).
        if y_tr.std() == 0:
            train_end += step_bars
            continue
        reg = CatBoostRegressor(
            iterations=catboost_iterations,
            depth=catboost_depth,
            learning_rate=catboost_learning_rate,
            loss_function="RMSE",
            thread_count=2,
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        if sample_weight_half_life > 0:
            n_tr = len(X_tr)
            age = np.arange(n_tr - 1, -1, -1, dtype=float)
            sw = np.power(2.0, -age / float(sample_weight_half_life))
            reg.fit(X_tr, y_tr, sample_weight=sw)
        else:
            reg.fit(X_tr, y_tr)
        y_hat = reg.predict(X_te)
        pooled_y_true.extend(y_te.tolist())
        pooled_y_pred.extend(y_hat.tolist())
        n_folds += 1
        train_end += step_bars

    if not pooled_y_true:
        return RegressionEvalRow(
            symbol=symbol, bar_minutes=bar_minutes,
            n_seconds_loaded=n_seconds_loaded,
            n_bars_after_aggregation=n_bars_agg,
            n_bars_kept=n_bars_kept, n_folds=0, n_predictions=0,
            mae_bps=None, rmse_bps=None, r2=None,
            ic_pearson=None, ic_spearman=None,
            sign_accuracy=None,
            thresholds=[],
        )

    yt = np.asarray(pooled_y_true, dtype=float)
    yp = np.asarray(pooled_y_pred, dtype=float)

    mae = float(np.abs(yt - yp).mean())
    rmse = float(math.sqrt(((yt - yp) ** 2).mean()))
    r2 = _r2(yt, yp)
    ic_p = _safe_corr(yt, yp)
    ic_s = _spearman_corr(yt, yp)

    # Sign accuracy on full pooled OOS sample (no threshold filter yet).
    sign_match_all = (np.sign(yp) == np.sign(yt)) & (yp != 0)
    sign_acc = float(sign_match_all.mean()) if len(yp) else None

    threshold_rows = [
        _threshold_slice(yt, yp, threshold_bps=t) for t in thresholds_bps
    ]

    return RegressionEvalRow(
        symbol=symbol, bar_minutes=bar_minutes,
        n_seconds_loaded=n_seconds_loaded,
        n_bars_after_aggregation=n_bars_agg,
        n_bars_kept=n_bars_kept, n_folds=n_folds,
        n_predictions=int(len(yp)),
        mae_bps=mae, rmse_bps=rmse, r2=r2,
        ic_pearson=ic_p, ic_spearman=ic_s,
        sign_accuracy=sign_acc,
        thresholds=threshold_rows,
    )


# ───────────────────────── markdown rendering ─────────────────────────

def _fmt_or_dash(x: float | None, fmt: str = "{:.4f}") -> str:
    """Format ``x`` or render '—' for None / NaN — defence-grade tables
    must never carry a 'nan' string into a slide."""
    if x is None:
        return "—"
    try:
        if not math.isfinite(float(x)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return fmt.format(x)


def render_markdown_report(rows: list[RegressionEvalRow]) -> str:
    """Render a markdown report from the eval rows.

    Two tables:
    1. Summary — one row per (symbol, bar_minutes) with regression
       goodness-of-fit + sign accuracy.
    2. Threshold detail — for each row, the per-threshold curve.
    """
    lines: list[str] = []
    lines.append("# Magnitude regression — defence-grade evaluation\n")
    lines.append(
        "Walk-forward CV with CatBoostRegressor on continuous "
        "``return_bps`` target.  All metrics are pooled OOS — never "
        "the trainer's view of itself.\n"
    )

    # Summary table.
    lines.append("## Summary\n")
    lines.append(
        "| symbol | bar_min | n_pred | n_folds | MAE (bp) | RMSE (bp) "
        "| R² | IC_p | IC_s | sign_acc |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        lines.append(
            f"| {r.symbol} | {r.bar_minutes} | {r.n_predictions} | "
            f"{r.n_folds} | {_fmt_or_dash(r.mae_bps, '{:.3f}')} | "
            f"{_fmt_or_dash(r.rmse_bps, '{:.3f}')} | "
            f"{_fmt_or_dash(r.r2, '{:.4f}')} | "
            f"{_fmt_or_dash(r.ic_pearson, '{:.4f}')} | "
            f"{_fmt_or_dash(r.ic_spearman, '{:.4f}')} | "
            f"{_fmt_or_dash(r.sign_accuracy, '{:.4f}')} |"
        )
    lines.append("")

    # Per-symbol threshold detail.
    for r in rows:
        if not r.thresholds:
            continue
        lines.append(
            f"## {r.symbol} — bar_minutes={r.bar_minutes} "
            f"(n_predictions={r.n_predictions})\n"
        )
        lines.append(
            "| θ (bp) | n_trades | %kept | sign_acc | E[r] (bp) "
            "| E[|r|] (bp) | retail | vip5 | vip9 | mm_rebate |"
        )
        lines.append(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for t in r.thresholds:
            tier_pnl = t.realized_pnl_bps
            lines.append(
                f"| {t.threshold_bps:.1f} | {t.n_trades} | "
                f"{t.fraction_kept * 100:.1f}% | "
                f"{_fmt_or_dash(t.sign_accuracy, '{:.4f}')} | "
                f"{_fmt_or_dash(t.mean_realized_bps, '{:.3f}')} | "
                f"{_fmt_or_dash(t.mean_abs_realized_bps, '{:.3f}')} | "
                f"{_fmt_or_dash(tier_pnl.get('retail'), '{:+.3f}')} | "
                f"{_fmt_or_dash(tier_pnl.get('vip5'), '{:+.3f}')} | "
                f"{_fmt_or_dash(tier_pnl.get('vip9'), '{:+.3f}')} | "
                f"{_fmt_or_dash(tier_pnl.get('mm_rebate'), '{:+.3f}')} |"
            )
        lines.append("")

    lines.append(
        "Notes: ``θ`` is the |E[r]| filter in bps (trade only when the "
        "regressor's expected return exceeds the threshold).  ``sign_acc`` "
        "is the share of trades where ``sign(pred) == sign(actual)`` — "
        "directly comparable to the classifier's ``dir_acc``.  Tier "
        "columns are mean realized P&L per trade after subtracting "
        "round-trip fees.\n"
    )
    return "\n".join(lines)


# ───────────────────────── CLI ─────────────────────────

def _load_seconds(database_url: str, *, symbol: str, since_hours: float) -> pd.DataFrame:
    """Re-implement the trainer's load_seconds locally to avoid a
    cross-module dependency in this offline tool. Same query."""
    from sqlalchemy import create_engine, text
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.regression_eval",
        description=__doc__,
    )
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 15],
                   help="bar size in minutes for each row")
    p.add_argument("--since-hours", type=float, default=168.0,
                   help="hours of 1-second history to load")
    p.add_argument("--initial-train-bars", type=int, default=1440,
                   help="walk-forward initial train (default 1440 = 1 day at 1m)")
    p.add_argument("--test-fold-bars", type=int, default=60)
    p.add_argument("--step-bars", type=int, default=60)
    p.add_argument("--sample-weight-half-life", type=int, default=720)
    p.add_argument("--embargo-bars", type=int, default=1)
    p.add_argument("--neutral-band-bps", type=float, default=1.0)
    p.add_argument("--out-json",
                   default="weights/highfreq/regression_eval.json")
    p.add_argument("--out-md",
                   default="docs/highfreq/regression_eval_report.md")
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

    started = time.monotonic()
    rows: list[RegressionEvalRow] = []

    # Load each symbol once, reuse across horizons.
    by_symbol: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        sym_u = sym.upper()
        logger.info("loading %s seconds (since_hours=%s)", sym_u, args.since_hours)
        by_symbol[sym_u] = _load_seconds(
            dsn, symbol=sym_u, since_hours=args.since_hours,
        )
        logger.info("  loaded %d rows for %s", len(by_symbol[sym_u]), sym_u)

    for sym_u, df_secs in by_symbol.items():
        for bar_minutes in args.horizons:
            logger.info(
                "evaluating %s @ bar_minutes=%d", sym_u, bar_minutes
            )
            row = evaluate_one_regression(
                df_secs,
                symbol=sym_u,
                bar_minutes=bar_minutes,
                initial_train_bars=args.initial_train_bars,
                test_fold_bars=args.test_fold_bars,
                step_bars=args.step_bars,
                neutral_band_bps=args.neutral_band_bps,
                sample_weight_half_life=args.sample_weight_half_life,
                embargo_bars=args.embargo_bars,
            )
            rows.append(row)
            logger.info(
                "  done: n_pred=%d MAE=%s sign_acc=%s",
                row.n_predictions,
                _fmt_or_dash(row.mae_bps, "{:.3f}"),
                _fmt_or_dash(row.sign_accuracy, "{:.4f}"),
            )

    # Persist artefacts.
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "config": {
            "symbols": args.symbols,
            "horizons": args.horizons,
            "since_hours": args.since_hours,
            "initial_train_bars": args.initial_train_bars,
            "sample_weight_half_life": args.sample_weight_half_life,
            "embargo_bars": args.embargo_bars,
            "neutral_band_bps": args.neutral_band_bps,
        },
        "rows": [r.to_dict() for r in rows],
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", out_json)

    md = render_markdown_report(rows)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    logger.info("wrote %s", out_md)

    print(md)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
