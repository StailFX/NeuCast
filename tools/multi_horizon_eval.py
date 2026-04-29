"""Multi-horizon evaluation: train + walk-forward CV across multiple
prediction horizons on the same OFI 1-second history.

Why this exists
===============

The 1-minute model on retail-tier fees is **mathematically unprofitable**
even at proven directional skill (dir_acc 0.548, p < 1e-5):

    E[P&L per trade] = (2*p − 1) × E[|move|] − fees
                     = 0.10 × 4 bp − 15 bp = −14.6 bp/trade

The fee burden eats the edge. But the SAME model architecture / SAME
features applied to 5 / 15 / 60-minute bars sees E[|move|] grow
roughly with sqrt(horizon) — at 60-minute horizon, E[|move|] ≈ 30-40
bp, so even a 5-percentage-point edge clears retail fees easily:

    1h horizon: 0.10 × 35 bp − 15 bp = +2.0 bp/trade ✅

This tool sweeps `[1, 5, 15, 60]` minute horizons on the same history
and produces a comparison table — the canonical defence-grade artefact
for "where does the edge become commercially monetisable".

Output
------

Prints a markdown table to stdout AND writes
``weights/highfreq/multi_horizon_eval.json`` for downstream rendering
(slides, UI block, Telegram bot).

Each row reports:
    * symbol + bar_minutes
    * n_seconds_loaded / n_bars_after_aggregation / n_bars_after_neutral_drop
    * n_folds achieved by walk-forward CV
    * dir_acc point estimate + Wilson 95% CI + binomial p-value
    * mean |return| (bps) — proxy for E[|move|]
    * estimated E[P&L per trade] at retail / vip5 / vip9 / mm_rebate fee tiers
    * verdict per fee tier: profitable / breakeven / unprofitable

The fee-tier estimate is the bridge from "we have edge" to "where
does it pay" — explicitly defence-grade.

Run from Tokyo
--------------

::

    sudo -u stailfx --preserve-env=DATABASE_URL \
      /opt/neucast/venv/bin/python -m tools.multi_horizon_eval \
        --symbols BTCUSDT ETHUSDT BNBUSDT \
        --horizons 1 5 15 60 \
        --since-hours 96
"""
from __future__ import annotations

import argparse
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


# Fee tiers, mirror app.highfreq.fee_tiers (one bar of code; not
# imported to keep this script standalone-runnable).
FEE_TIERS_BPS: dict[str, float] = {
    "retail":    7.5,
    "vip5":      1.0,
    "vip9":      0.0,
    "mm_rebate": -0.4,
}


@dataclass
class HorizonEvalRow:
    """One row of the comparison table — per (symbol, bar_minutes, feature_set)."""
    symbol: str
    bar_minutes: int
    n_seconds_loaded: int
    n_bars_after_aggregation: int
    n_bars_after_neutral_drop: int
    n_folds: int
    dir_acc: float | None
    dir_acc_ci_low: float | None
    dir_acc_ci_high: float | None
    dir_acc_p_value: float | None
    base_rate: float | None
    mean_abs_return_bps: float | None
    # E[P&L per trade] at each fee tier, given dir_acc and mean_abs_return:
    # E[P&L] = (2*dir_acc − 1) * mean_abs_return − fees_roundtrip_bps
    pnl_per_trade_bps: dict[str, float | None]
    # Defaults at end to preserve dataclass argument-order rules.
    feature_set: str = "microstructure"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _binom_p_value_greater_half(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    from scipy.stats import binomtest
    return float(binomtest(k=k, n=n, p=0.5, alternative="greater").pvalue)


def _expected_pnl_bps(
    *,
    dir_acc: float | None,
    mean_abs_return_bps: float | None,
    fee_bps_per_side: float,
) -> float | None:
    """Bridge from "model skill" to "trader P&L" assuming:
    * Trader takes EVERY signal (worst case for fee burden).
    * Realized return per trade = ±mean_abs_return_bps with prob
      (dir_acc, 1−dir_acc) per side.

    Returns None when inputs are insufficient.
    """
    if dir_acc is None or mean_abs_return_bps is None:
        return None
    if not (0.0 <= dir_acc <= 1.0):
        return None
    edge = 2.0 * dir_acc - 1.0          # signed edge (positive = skill)
    fee_roundtrip = 2.0 * fee_bps_per_side
    return float(edge * mean_abs_return_bps - fee_roundtrip)


def evaluate_one_horizon(
    df_secs: pd.DataFrame,
    *,
    symbol: str,
    bar_minutes: int,
    initial_train_bars: int,
    test_fold_bars: int,
    step_bars: int,
    neutral_band_bps: float,
    feature_set: str = "microstructure",  # 'microstructure' | 'long_horizon' | 'cross_asset'
    reference_df_secs: pd.DataFrame | None = None,  # for cross_asset only
    catboost_iterations: int = 200,
    catboost_depth: int = 5,
    catboost_learning_rate: float = 0.05,
    random_seed: int = 42,
    sample_weight_half_life: int = 0,  # 0 = disabled (release O default for eval)
    embargo_bars: int = 0,              # 0 = no embargo
) -> HorizonEvalRow:
    """Aggregate, build supervised, walk-forward CV. Returns one row.

    ``feature_set`` selects which feature pipeline to use:
    * 'microstructure' (default) — original 14+4 calendar features.
      Optimised for 1-min horizon.
    * 'long_horizon' — OHLC + classical TA features
      (LONG_HORIZON_FEATURE_COLUMNS). Hypothesis: works better at 5m+
      where microstructure decays.
    """
    from app.highfreq.feature_pipeline import (
        aggregate_to_minute,
        build_features,
        build_target,
    )
    if feature_set == "long_horizon":
        from app.highfreq.feature_pipeline_long_horizon import (
            LONG_HORIZON_FEATURE_COLUMNS as ACTIVE_FEATURE_COLUMNS,
            build_long_horizon_features as _build_features_active,
        )
    elif feature_set == "cross_asset":
        from app.highfreq.feature_pipeline_cross_asset import (
            build_cross_asset_features,
            feature_columns_for,
        )
        # Cross-asset features need a reference symbol — for ETH/BNB
        # we point at BTC; for BTC itself we have no reference and
        # fall back to base + lagged.
        reference_symbol = (
            None if symbol.upper() == "BTCUSDT" else "BTCUSDT"
        )
        ACTIVE_FEATURE_COLUMNS = feature_columns_for(reference_symbol)
        # Curry into the same single-arg signature the other branches use.
        _ref_minutes_holder: dict = {"ref": None}
        def _build_features_active(targeted_df: pd.DataFrame) -> pd.DataFrame:
            return build_cross_asset_features(
                targeted_df,
                reference_minutes=_ref_minutes_holder["ref"],
                reference_symbol=reference_symbol,
            )
    else:
        from app.highfreq.feature_pipeline import (
            FEATURE_COLUMNS as ACTIVE_FEATURE_COLUMNS,
            build_features as _build_features_active,
        )

    n_seconds_loaded = int(len(df_secs))

    # 1) Aggregate to bars + count after coverage drop.
    minute_df = aggregate_to_minute(df_secs, bar_minutes=bar_minutes)
    n_bars_agg = int(len(minute_df))

    # 1b) For cross_asset mode: also aggregate the reference symbol's
    #     seconds at the SAME horizon, then stash for the closure.
    if feature_set == "cross_asset" and reference_df_secs is not None:
        ref_minutes = aggregate_to_minute(reference_df_secs, bar_minutes=bar_minutes)
        _ref_minutes_holder["ref"] = ref_minutes

    # 2) Build target + features using the SELECTED pipeline.
    #    (Cannot reuse make_supervised because it hardcodes the
    #    microstructure pipeline; we inline its logic here so we can
    #    swap _build_features_active.)
    targeted = build_target(minute_df, horizon=1, neutral_band_bps=neutral_band_bps)
    if targeted.empty:
        X = pd.DataFrame(columns=ACTIVE_FEATURE_COLUMNS)
        y = pd.Series(dtype=np.int8)
    else:
        keep_mask = (targeted["y"] != -1) & (~targeted["in_neutral_band"])
        targeted = targeted.loc[keep_mask].reset_index(drop=True)
        X = _build_features_active(targeted)
        # Reorder to canonical column order — defence against silent drift.
        X = X[ACTIVE_FEATURE_COLUMNS]
        y = targeted["y"].astype(np.int8)
    n_bars_kept = int(len(X))

    # mean |return| in bps — across all post-aggregation bars BEFORE
    # neutral-band drop, so it reflects raw market vol at this horizon.
    if not minute_df.empty:
        # Compute per-bar return from microprice_close shift(-1).
        sym_df = minute_df.sort_values(["symbol", "minute"]).copy()
        sym_df["mp_close_next"] = sym_df.groupby("symbol")["microprice_close"].shift(-1)
        sym_df["return_bps"] = (
            (sym_df["mp_close_next"] - sym_df["microprice_close"])
            / sym_df["microprice_close"] * 1e4
        )
        mean_abs_return = float(sym_df["return_bps"].abs().mean()) if (
            sym_df["return_bps"].notna().any()
        ) else None
    else:
        mean_abs_return = None

    # 3) Walk-forward CV (manual, fold-step in BARS not minutes — the
    #    aggregator already collapsed time). Skip if not enough data.
    if n_bars_kept < initial_train_bars + test_fold_bars:
        # Insufficient — return early with what we have.
        return HorizonEvalRow(
            symbol=symbol, bar_minutes=bar_minutes, feature_set=feature_set,
            n_seconds_loaded=n_seconds_loaded,
            n_bars_after_aggregation=n_bars_agg,
            n_bars_after_neutral_drop=n_bars_kept,
            n_folds=0,
            dir_acc=None, dir_acc_ci_low=None, dir_acc_ci_high=None,
            dir_acc_p_value=None,
            base_rate=float(max(y.mean(), 1 - y.mean())) if len(y) else None,
            mean_abs_return_bps=mean_abs_return,
            pnl_per_trade_bps={k: None for k in FEE_TIERS_BPS},
        )

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise RuntimeError("catboost required: pip install catboost")

    folds_y_true: list[int] = []
    folds_y_pred: list[int] = []
    n_folds = 0
    train_end = initial_train_bars
    while train_end + test_fold_bars <= n_bars_kept:
        # Embargo: drop the last ``embargo_bars`` from train fold to
        # avoid target-leakage with test fold (release O.7C).
        train_eff_end = max(0, train_end - max(0, embargo_bars))
        X_tr = X.iloc[:train_eff_end].to_numpy()
        y_tr = y.iloc[:train_eff_end].to_numpy()
        X_te = X.iloc[train_end:train_end + test_fold_bars].to_numpy()
        y_te = y.iloc[train_end:train_end + test_fold_bars].to_numpy()
        # Skip degenerate folds where one class isn't present in train.
        if len(X_tr) < 100 or len(set(y_tr.tolist())) < 2:
            train_end += step_bars
            continue
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
        # Sample weighting: recent bars matter more (release O.3D).
        if sample_weight_half_life > 0:
            n_tr = len(X_tr)
            age = np.arange(n_tr - 1, -1, -1, dtype=float)
            sw = np.power(2.0, -age / float(sample_weight_half_life))
            clf.fit(X_tr, y_tr, sample_weight=sw)
        else:
            clf.fit(X_tr, y_tr)
        y_hat = (clf.predict_proba(X_te)[:, 1] > 0.5).astype(int)
        folds_y_true.extend(y_te.tolist())
        folds_y_pred.extend(y_hat.tolist())
        n_folds += 1
        train_end += step_bars

    if not folds_y_true:
        return HorizonEvalRow(
            symbol=symbol, bar_minutes=bar_minutes, feature_set=feature_set,
            n_seconds_loaded=n_seconds_loaded,
            n_bars_after_aggregation=n_bars_agg,
            n_bars_after_neutral_drop=n_bars_kept,
            n_folds=0,
            dir_acc=None, dir_acc_ci_low=None, dir_acc_ci_high=None,
            dir_acc_p_value=None,
            base_rate=float(max(y.mean(), 1 - y.mean())) if len(y) else None,
            mean_abs_return_bps=mean_abs_return,
            pnl_per_trade_bps={k: None for k in FEE_TIERS_BPS},
        )

    yt = np.array(folds_y_true)
    yp = np.array(folds_y_pred)
    n_correct = int((yt == yp).sum())
    n_total = int(len(yt))
    dir_acc = n_correct / n_total
    ci_lo, ci_hi = _wilson_ci(n_correct, n_total)
    p_value = _binom_p_value_greater_half(n_correct, n_total)
    base_rate = float(max(yt.mean(), 1 - yt.mean()))

    pnl_tiers = {
        tier: _expected_pnl_bps(
            dir_acc=dir_acc, mean_abs_return_bps=mean_abs_return,
            fee_bps_per_side=fee_bps,
        )
        for tier, fee_bps in FEE_TIERS_BPS.items()
    }

    return HorizonEvalRow(
        symbol=symbol, bar_minutes=bar_minutes, feature_set=feature_set,
        n_seconds_loaded=n_seconds_loaded,
        n_bars_after_aggregation=n_bars_agg,
        n_bars_after_neutral_drop=n_bars_kept,
        n_folds=n_folds,
        dir_acc=dir_acc,
        dir_acc_ci_low=ci_lo,
        dir_acc_ci_high=ci_hi,
        dir_acc_p_value=p_value,
        base_rate=base_rate,
        mean_abs_return_bps=mean_abs_return,
        pnl_per_trade_bps=pnl_tiers,
    )


def evaluate_joint_horizon(
    df_secs_by_symbol: dict[str, pd.DataFrame],
    *,
    bar_minutes: int,
    initial_train_bars: int,
    test_fold_bars: int,
    step_bars: int,
    neutral_band_bps: float,
    use_long_horizon_features: bool = False,
    catboost_iterations: int = 200,
    catboost_depth: int = 5,
    catboost_learning_rate: float = 0.05,
    random_seed: int = 42,
    sample_weight_half_life: int = 0,
    embargo_bars: int = 0,
) -> HorizonEvalRow:
    """Train ONE model on pooled BTC+ETH+BNB data with symbol-id features.

    Walk-forward CV operates on the chronologically-sorted pooled
    dataset. Each fold trains on past pooled bars and tests on the
    next slice — the test set may contain multiple symbols.

    When ``use_long_horizon_features`` is True, uses TA features
    (OHLC/EMA/RSI/Bollinger) instead of microstructure — appropriate
    for 5m+ horizons where microstructure decays.
    """
    from app.highfreq.feature_pipeline_joint import (
        make_joint_supervised,
    )
    from app.highfreq.feature_pipeline import aggregate_to_minute

    n_seconds_loaded = sum(len(df) for df in df_secs_by_symbol.values())

    # Aggregate pooled for the diagnostic n_bars_after_aggregation.
    pooled_secs = pd.concat(df_secs_by_symbol.values(), ignore_index=True, sort=False)
    minute_df = aggregate_to_minute(pooled_secs, bar_minutes=bar_minutes)
    n_bars_agg = int(len(minute_df))

    X, y, meta = make_joint_supervised(
        df_secs_by_symbol,
        neutral_band_bps=neutral_band_bps,
        bar_minutes=bar_minutes,
        use_long_horizon_features=use_long_horizon_features,
    )
    n_bars_kept = int(len(X))

    # mean |return| in bps over post-aggregation bars (across all symbols).
    if not minute_df.empty:
        sym_df = minute_df.sort_values(["symbol", "minute"]).copy()
        sym_df["mp_close_next"] = sym_df.groupby("symbol")["microprice_close"].shift(-1)
        sym_df["return_bps"] = (
            (sym_df["mp_close_next"] - sym_df["microprice_close"])
            / sym_df["microprice_close"] * 1e4
        )
        mean_abs_return = (
            float(sym_df["return_bps"].abs().mean())
            if sym_df["return_bps"].notna().any() else None
        )
    else:
        mean_abs_return = None

    if n_bars_kept < initial_train_bars + test_fold_bars:
        return HorizonEvalRow(
            symbol="JOINT", bar_minutes=bar_minutes, feature_set="joint",
            n_seconds_loaded=n_seconds_loaded,
            n_bars_after_aggregation=n_bars_agg,
            n_bars_after_neutral_drop=n_bars_kept,
            n_folds=0,
            dir_acc=None, dir_acc_ci_low=None, dir_acc_ci_high=None,
            dir_acc_p_value=None,
            base_rate=float(max(y.mean(), 1 - y.mean())) if len(y) else None,
            mean_abs_return_bps=mean_abs_return,
            pnl_per_trade_bps={k: None for k in FEE_TIERS_BPS},
        )

    from catboost import CatBoostClassifier

    folds_y_true: list[int] = []
    folds_y_pred: list[int] = []
    n_folds = 0
    train_end = initial_train_bars
    while train_end + test_fold_bars <= n_bars_kept:
        train_eff_end = max(0, train_end - max(0, embargo_bars))
        X_tr = X.iloc[:train_eff_end].to_numpy()
        y_tr = y.iloc[:train_eff_end].to_numpy()
        X_te = X.iloc[train_end:train_end + test_fold_bars].to_numpy()
        y_te = y.iloc[train_end:train_end + test_fold_bars].to_numpy()
        if len(X_tr) < 100 or len(set(y_tr.tolist())) < 2:
            train_end += step_bars
            continue
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
        if sample_weight_half_life > 0:
            n_tr = len(X_tr)
            age = np.arange(n_tr - 1, -1, -1, dtype=float)
            sw = np.power(2.0, -age / float(sample_weight_half_life))
            clf.fit(X_tr, y_tr, sample_weight=sw)
        else:
            clf.fit(X_tr, y_tr)
        y_hat = (clf.predict_proba(X_te)[:, 1] > 0.5).astype(int)
        folds_y_true.extend(y_te.tolist())
        folds_y_pred.extend(y_hat.tolist())
        n_folds += 1
        train_end += step_bars

    if not folds_y_true:
        return HorizonEvalRow(
            symbol="JOINT", bar_minutes=bar_minutes, feature_set="joint",
            n_seconds_loaded=n_seconds_loaded,
            n_bars_after_aggregation=n_bars_agg,
            n_bars_after_neutral_drop=n_bars_kept,
            n_folds=0,
            dir_acc=None, dir_acc_ci_low=None, dir_acc_ci_high=None,
            dir_acc_p_value=None,
            base_rate=float(max(y.mean(), 1 - y.mean())) if len(y) else None,
            mean_abs_return_bps=mean_abs_return,
            pnl_per_trade_bps={k: None for k in FEE_TIERS_BPS},
        )

    yt = np.array(folds_y_true)
    yp = np.array(folds_y_pred)
    n_correct = int((yt == yp).sum())
    n_total = int(len(yt))
    dir_acc = n_correct / n_total
    ci_lo, ci_hi = _wilson_ci(n_correct, n_total)
    p_value = _binom_p_value_greater_half(n_correct, n_total)
    base_rate = float(max(yt.mean(), 1 - yt.mean()))

    pnl_tiers = {
        tier: _expected_pnl_bps(
            dir_acc=dir_acc, mean_abs_return_bps=mean_abs_return,
            fee_bps_per_side=fee_bps,
        )
        for tier, fee_bps in FEE_TIERS_BPS.items()
    }

    return HorizonEvalRow(
        symbol="JOINT", bar_minutes=bar_minutes, feature_set="joint",
        n_seconds_loaded=n_seconds_loaded,
        n_bars_after_aggregation=n_bars_agg,
        n_bars_after_neutral_drop=n_bars_kept,
        n_folds=n_folds,
        dir_acc=dir_acc,
        dir_acc_ci_low=ci_lo,
        dir_acc_ci_high=ci_hi,
        dir_acc_p_value=p_value,
        base_rate=base_rate,
        mean_abs_return_bps=mean_abs_return,
        pnl_per_trade_bps=pnl_tiers,
    )


def _verdict(pnl_bps: float | None) -> str:
    if pnl_bps is None:
        return "—"
    if pnl_bps > 1.0:
        return "✅ profitable"
    if pnl_bps > -1.0:
        return "≈ breakeven"
    return "❌ loss"


def _markdown_table(rows: list[HorizonEvalRow]) -> str:
    lines = []
    lines.append("# Multi-horizon evaluation\n")
    lines.append("Same OFI 1-second history → same model architecture → "
                  "different aggregation horizon. Defence-grade artefact "
                  "showing where directional edge becomes monetisable.\n")
    lines.append("| symbol | bar | features | n_kept | folds | dir_acc | CI | p | E[\\|move\\|] | retail | vip9 | mm_rebate |")
    lines.append("|---|---:|---|---:|---:|---:|---|---:|---:|---|---|---|")
    for r in rows:
        fset = r.feature_set[:4]  # 'micr' or 'long'
        if r.dir_acc is None:
            lines.append(
                f"| {r.symbol} | {r.bar_minutes}m | {fset} | "
                f"{r.n_bars_after_neutral_drop} | "
                f"{r.n_folds} | — | — | — | — | — | — | — |"
            )
            continue
        ci = f"[{r.dir_acc_ci_low:.3f}, {r.dir_acc_ci_high:.3f}]"
        p_str = f"{r.dir_acc_p_value:.2e}" if r.dir_acc_p_value < 0.01 else f"{r.dir_acc_p_value:.3f}"
        mar = f"{r.mean_abs_return_bps:.1f}bp" if r.mean_abs_return_bps is not None else "—"
        retail = _verdict(r.pnl_per_trade_bps.get("retail"))
        vip9 = _verdict(r.pnl_per_trade_bps.get("vip9"))
        mmr = _verdict(r.pnl_per_trade_bps.get("mm_rebate"))
        lines.append(
            f"| **{r.symbol}** | **{r.bar_minutes}m** | {fset} | "
            f"{r.n_bars_after_neutral_drop} | {r.n_folds} | "
            f"{r.dir_acc:.4f} | {ci} | {p_str} | {mar} | "
            f"{retail} | {vip9} | {mmr} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 15, 60],
                   help="bar sizes in minutes to compare")
    p.add_argument("--feature-sets", nargs="+",
                   default=["microstructure"],
                   choices=["microstructure", "long_horizon", "cross_asset",
                            "joint", "joint_long_horizon"],
                   help="which feature pipelines to evaluate. Pass multiple "
                        "to compare side-by-side. cross_asset uses BTC as "
                        "reference for ETH/BNB; joint pools 3 symbols with "
                        "symbol-id features; joint_long_horizon combines "
                        "joint pooling with classical TA features (OHLC, "
                        "EMA, RSI, BB) — better suited for 5m/15m+.")
    p.add_argument("--since-hours", type=float, default=96.0)
    p.add_argument("--sample-weight-half-life", type=int, default=0,
                   help="Exponential decay half-life in bars for sample "
                        "weighting (recent > old). 0 disables. "
                        "Default 0; production trainer uses 720 (= 12h at 1m).")
    p.add_argument("--embargo-bars", type=int, default=0,
                   help="Drop this many bars at the train→test boundary "
                        "to prevent target-leakage. 0 disables. Production "
                        "trainer uses 1.")
    p.add_argument("--neutral-band-bps", type=float, default=None,
                   help="override default. Auto-scales as sqrt(bar_minutes) × 1bp "
                        "if not given — preserves the same z-score across horizons.")
    p.add_argument("--out", default="weights/highfreq/multi_horizon_eval.json")
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

    rows: list[HorizonEvalRow] = []
    started = time.monotonic()

    # Pre-load BTC seconds once if cross_asset mode requested for any
    # non-BTC symbol (avoids redundant DB queries).
    btc_df_secs = None
    if "cross_asset" in args.feature_sets and any(
        s.upper() != "BTCUSDT" for s in args.symbols
    ):
        logger.info("=" * 60)
        logger.info("pre-loading BTCUSDT seconds for cross-asset reference")
        btc_df_secs = load_seconds(dsn, symbol="BTCUSDT", since_hours=args.since_hours)
        logger.info("loaded %d BTC rows for cross-asset reference", len(btc_df_secs))

    # If any joint variant is requested, special-case: load all symbols'
    # seconds ONCE, train a single pooled model, and short-circuit the
    # per-symbol loop for those feature_sets.
    joint_variants = [
        f for f in args.feature_sets if f in ("joint", "joint_long_horizon")
    ]
    if joint_variants:
        logger.info("=" * 60)
        logger.info("loading all symbols for joint training")
        joint_secs = {
            sym: load_seconds(dsn, symbol=sym, since_hours=args.since_hours)
            for sym in args.symbols
        }
        for sym, df in joint_secs.items():
            logger.info("  %s: %d seconds", sym, len(df))
        for h in args.horizons:
            initial_train_bars = max(60, int(24 * 60 / h))
            test_fold_bars = max(5, int(60 / h))
            step_bars = test_fold_bars
            neutral = args.neutral_band_bps if args.neutral_band_bps is not None \
                else 1.0 * math.sqrt(h)
            for fset in joint_variants:
                use_lh = (fset == "joint_long_horizon")
                logger.info(
                    "%s @ %dm neutral=%.2fbp init_train=%d test=%d",
                    fset, h, neutral, initial_train_bars, test_fold_bars,
                )
                try:
                    row = evaluate_joint_horizon(
                        joint_secs, bar_minutes=h,
                        initial_train_bars=initial_train_bars,
                        test_fold_bars=test_fold_bars,
                        step_bars=step_bars,
                        neutral_band_bps=neutral,
                        use_long_horizon_features=use_lh,
                    )
                    # Tag with which joint variant for the table.
                    row = HorizonEvalRow(**{**asdict(row), "feature_set": fset})
                    rows.append(row)
                    if row.dir_acc is None:
                        logger.info(
                            "%s @ %dm: insufficient data (n=%d)",
                            fset, h, row.n_bars_after_neutral_drop,
                        )
                    else:
                        logger.info(
                            "%s @ %dm: dir_acc=%.4f [%.3f, %.3f] p=%.2e folds=%d",
                            fset, h, row.dir_acc, row.dir_acc_ci_low,
                            row.dir_acc_ci_high, row.dir_acc_p_value, row.n_folds,
                        )
                except Exception:
                    logger.exception("%s eval failed at %dm", fset, h)

    for symbol in args.symbols:
        logger.info("=" * 60)
        logger.info("loading %s seconds (since_hours=%g)", symbol, args.since_hours)
        df_secs = load_seconds(dsn, symbol=symbol, since_hours=args.since_hours)
        logger.info("loaded %d rows for %s", len(df_secs), symbol)
        for h in args.horizons:
            # Walk-forward params scale with horizon: short bars need
            # bigger train/test windows in BARS, long bars need fewer.
            # Heuristic: aim for ~24 hours of train + 1 hour of test.
            initial_train_bars = max(60, int(24 * 60 / h))
            test_fold_bars = max(5, int(60 / h))
            step_bars = test_fold_bars
            neutral = args.neutral_band_bps
            if neutral is None:
                # Scale neutral band as sqrt(horizon) — preserves the same
                # z-score threshold across horizons. At 1m it's the
                # original 1bp; at 60m it's 1*sqrt(60) ≈ 7.7bp.
                neutral = 1.0 * math.sqrt(h)
            for fset in args.feature_sets:
                if fset in ("joint", "joint_long_horizon"):
                    # Joint variants are symbol-agnostic; handled above.
                    # Skip here so we don't double-evaluate.
                    continue
                logger.info(
                    "  horizon=%dm fset=%s neutral=%.2fbp init_train=%d test=%d",
                    h, fset, neutral, initial_train_bars, test_fold_bars,
                )
                # Cross-asset for non-BTC: pass BTC as reference. For
                # BTC: ref is None (handled internally).
                ref_secs = None
                if fset == "cross_asset" and symbol.upper() != "BTCUSDT":
                    ref_secs = btc_df_secs
                try:
                    row = evaluate_one_horizon(
                        df_secs, symbol=symbol, bar_minutes=h,
                        feature_set=fset,
                        reference_df_secs=ref_secs,
                        initial_train_bars=initial_train_bars,
                        test_fold_bars=test_fold_bars,
                        step_bars=step_bars,
                        neutral_band_bps=neutral,
                        sample_weight_half_life=args.sample_weight_half_life,
                        embargo_bars=args.embargo_bars,
                    )
                except Exception:
                    logger.exception(
                        "eval failed for symbol=%s horizon=%d fset=%s",
                        symbol, h, fset,
                    )
                    continue
                rows.append(row)
                if row.dir_acc is None:
                    logger.info(
                        "    %s @ %dm fset=%s: insufficient data (n=%d, need %d)",
                        symbol, h, fset, row.n_bars_after_neutral_drop,
                        initial_train_bars + test_fold_bars,
                    )
                else:
                    logger.info(
                        "    %s @ %dm fset=%s: dir_acc=%.4f [%.3f, %.3f] "
                        "p=%.2e folds=%d mean|move|=%.1fbp",
                        symbol, h, fset, row.dir_acc, row.dir_acc_ci_low,
                        row.dir_acc_ci_high, row.dir_acc_p_value, row.n_folds,
                        row.mean_abs_return_bps if row.mean_abs_return_bps else 0.0,
                    )

    elapsed = time.monotonic() - started
    logger.info("done in %.1fs", elapsed)

    table_md = _markdown_table(rows)
    print()
    print(table_md)
    print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "symbols": args.symbols,
        "horizons": args.horizons,
        "since_hours": args.since_hours,
        "elapsed_seconds": elapsed,
        "rows": [r.to_dict() for r in rows],
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
