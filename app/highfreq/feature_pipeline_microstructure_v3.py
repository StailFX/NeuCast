"""Microstructure v3: base 18 features + 5 futures-basis features (T.23).

Why
===

T.24 A/B (96h, 45-53 walk-forward folds, perm p < 0.001) found that
adding 5 futures-basis features to the base microstructure pipeline
lifts dir_acc by **+19 to +22 percentage points** consistently across
all 3 symbols. The lift survives at production geometry (24h initial
train, walk-forward over 4 days). The dominant feature is
``mark_premium_bps_close`` (importance ~30, vs ~3-5 for the others)
— the futures **mark price** leads spot microprice in crypto by
seconds-to-minutes, and at 1m-bar aggregation that lead is large
enough to dominate the prediction.

Five futures-basis features
---------------------------

1. ``basis_bps_close`` — (fut_microprice - spot_microprice) / spot * 1e4.
   The classic basis, signed: positive = futures premium.
2. ``basis_change_bps`` — 1-bar diff of basis. Captures whether the
   premium is widening or tightening (lead-lag dynamics).
3. ``ofi_diff_sum`` — futures OFI sum minus spot OFI sum at the same
   bar. Cross-venue order-flow divergence.
4. ``funding_bps_mean`` — per-minute average of funding_rate × 1e4.
   Encodes the cost-of-carry between perpetual and spot.
5. ``mark_premium_bps_close`` — (mark_price - spot_microprice) / spot * 1e4.
   The dominant feature; mark embeds futures-mid + funding-adjustment
   and at 1-second sampling carries the futures lead.

Total: **23 columns** (18 base + 5 futures-basis).

Backward compatibility
----------------------

The base 18 columns are computed by the unchanged
:func:`app.highfreq.feature_pipeline.build_features`. v3 calls it,
adds the 5 futures columns aligned by minute, and returns the wider
matrix. Models trained on ``microstructure`` continue to load + serve
cleanly; only models trained with ``feature_set='microstructure_v3'``
expect the 23-col input.

Cold-start / missing-data handling
----------------------------------

If the futures table has no rows aligned with the spot bar (early
ingest gap, ingest downtime, or per-bar futures coverage hole), the
5 futures columns zero-fill for that bar. CatBoost handles the
zero-input gracefully — it just becomes a non-informative split
candidate at that row. This means the v3 pipeline NEVER blocks
training, even when futures ingest is broken.

Defence narrative
-----------------

> "We added 5 features encoding spot-vs-futures basis dynamics. The
> hypothesis was that the perpetual mark price leads spot microprice
> on the seconds-to-minutes scale, and that this lead survives 1-min
> bar aggregation enough to be a usable predictor. Empirical A/B at
> production walk-forward geometry (T.24.c) measured a +19 to +22pp
> lift on all 3 symbols, with feature importance dominated by
> mark_premium (importance ~30 vs ~3-5 for the other 4 components).
> We deployed v3 as a canary on BTC first, watched live paper-trader
> dir_acc for 24h, then expanded."
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Five futures-basis feature columns appended to the base 18.
# Order matters — the trainer's saved feature-order check pins this.
FUTURES_BASIS_FEATURE_COLUMNS: list[str] = [
    "basis_bps_close",
    "basis_change_bps",
    "ofi_diff_sum",
    "funding_bps_mean",
    "mark_premium_bps_close",
]


def _zero_filled_futures_block(n_rows: int) -> pd.DataFrame:
    """Used when futures data is missing — model still trains."""
    return pd.DataFrame(
        0.0,
        index=range(n_rows),
        columns=FUTURES_BASIS_FEATURE_COLUMNS,
    )


def build_futures_basis_block(
    spot_minute_df: pd.DataFrame,
    futures_seconds_df: pd.DataFrame | None,
    *,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Build the 5 futures-basis features aligned to ``spot_minute_df``.

    Returns a DataFrame indexed positionally on ``spot_minute_df``'s
    rows. Missing rows zero-fill so the trainer never crashes on
    futures ingest gaps. All values are guaranteed finite (no
    NaN/Inf leak into CatBoost).
    """
    from app.highfreq.feature_pipeline import aggregate_to_minute

    n = len(spot_minute_df)
    if n == 0:
        return _zero_filled_futures_block(0)

    if futures_seconds_df is None or futures_seconds_df.empty:
        logger.info(
            "v3: futures_seconds_df missing/empty (n_spot_bars=%d) — "
            "zero-filling 5 futures features",
            n,
        )
        return _zero_filled_futures_block(n)

    # Aggregate futures seconds → bars at same minute floor as spot.
    fut_min = aggregate_to_minute(futures_seconds_df, bar_minutes=bar_minutes)
    if fut_min.empty:
        return _zero_filled_futures_block(n)

    # Per-minute funding & mark_price (mean across the bar's seconds).
    fut_extra = futures_seconds_df.copy()
    fut_extra["minute"] = fut_extra["ts"].dt.floor(f"{bar_minutes}min")
    extra_agg = fut_extra.groupby(["symbol", "minute"]).agg(
        funding_rate_mean=("funding_rate", "mean"),
        mark_price_mean=("mark_price", "mean"),
    ).reset_index()
    fut_min = fut_min.merge(extra_agg, on=["symbol", "minute"], how="left")

    # Spot side: microprice_close + ofi_sum.
    spot = spot_minute_df[["minute", "symbol", "microprice_close",
                           "ofi_sum"]].rename(columns={
        "microprice_close": "spot_microprice_close",
        "ofi_sum": "spot_ofi_sum",
    })
    fut = fut_min[["minute", "symbol", "microprice_close", "ofi_sum",
                   "funding_rate_mean", "mark_price_mean"]].rename(
        columns={
            "microprice_close": "fut_microprice_close",
            "ofi_sum": "fut_ofi_sum",
        },
    )
    joined = spot.merge(fut, on=["minute", "symbol"], how="left")

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
    out.index = range(n)

    n_populated = int((out != 0.0).any(axis=1).sum())
    logger.info(
        "v3: built futures-basis block (%d rows, %d non-zero)",
        n, n_populated,
    )
    return out


def microstructure_v3_feature_columns() -> list[str]:
    """Return the full 23-column list. Matches what
    :func:`build_microstructure_v3_features` produces."""
    from app.highfreq.feature_pipeline import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS) + list(FUTURES_BASIS_FEATURE_COLUMNS)


def build_microstructure_v3_features(
    targeted_minute_df: pd.DataFrame,
    *,
    futures_seconds_df: pd.DataFrame | None,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Full v3 feature matrix: base 18 cols + 5 futures-basis cols.

    Parameters
    ----------
    targeted_minute_df
        Spot minute-frame after ``build_target`` (expected to carry
        ``minute``, ``symbol``, plus the columns ``build_features``
        consumes).
    futures_seconds_df
        Raw seconds rows from ``highfreq_futures_ofi_1s`` joined by
        symbol. ``None`` or empty → zero-fill.
    bar_minutes
        Aggregation period in minutes (production: 1).

    Returns
    -------
    DataFrame with columns ``microstructure_v3_feature_columns()``.
    """
    from app.highfreq.feature_pipeline import build_features

    base = build_features(targeted_minute_df).reset_index(drop=True)
    futures_block = build_futures_basis_block(
        targeted_minute_df, futures_seconds_df, bar_minutes=bar_minutes,
    ).reset_index(drop=True)

    out = pd.concat([base, futures_block], axis=1)
    expected = microstructure_v3_feature_columns()
    missing = set(expected) - set(out.columns)
    if missing:
        raise RuntimeError(f"v3: missing expected columns {sorted(missing)}")
    # Lock column order — saved-model dispatch depends on it.
    return out[expected]
