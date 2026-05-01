"""Robustness checks for the directional model — answers "is the
result statistically real or did we just catch a regime?".

The trainer's walk-forward CV emits an i.i.d. bootstrap CI + a
binomial p-value testing H₀=0.5. Both are useful but neither closes
the two big honest concerns from a sceptical defender:

1. **Time-series autocorrelation underestimates uncertainty.** Adjacent
   1-minute bars are correlated; an i.i.d. bootstrap acts as if every
   bar is an independent draw, narrowing the CI artificially. A
   *block* bootstrap that resamples 60-minute contiguous blocks
   preserves autocorrelation and produces honest coverage.

2. **A trend-fitting model fakes skill.** If the entire training
   window was one regime (e.g. uptrend), a model that's just biased
   toward "up" gets ~55% dir_acc — but on a different regime it'd
   collapse to 45%. The right counter-test is a *permutation test*:
   shuffle the true labels 1000 times, compute the null distribution
   of dir_acc, see where the actual point sits in that distribution.
   p-value < 0.01 means "this number couldn't plausibly have come
   from a labels-permuted run".

Plus three slicing checks for stability:

3. **Per-day stability** — bucket realized predictions by UTC date,
   compute dir_acc per day with Wilson CI. Wide spread → regime-
   sensitive; tight spread → robust.

4. **Per-hour-of-day** — heatmap by UTC hour. Reveals "model only
   works on Asian session" patterns.

5. **Regime-conditional** — bucket by trailing 60-min return sign
   (uptrend / downtrend / sideways). dir_acc per regime tells us
   whether the signal generalises across market modes.

Source of truth: ``predictions_log`` — the actual live predictions
the production predictor made, joined with realized_microprice_1m
backfilled by the realized-accuracy job. This is more honest than
re-running walk-forward CV because it captures in-flight bar
boundaries, late ticks, and any quirks of the production path.

Run from Tokyo
--------------

::

    python -m tools.robustness_suite --symbol BTCUSDT
    python -m tools.robustness_suite --symbol BTCUSDT --symbol ETHUSDT --symbol BNBUSDT

Writes ``weights/highfreq/<symbol>_1m_robustness.json``. The web
endpoint ``/api/highfreq/robustness?symbol=...`` serves the latest
report.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobustnessReport:
    """JSON-serialisable robustness suite output."""
    symbol: str
    generated_at: str
    n_predictions: int
    n_bootstrap: int
    n_permutations: int
    block_size_minutes: int

    # Point estimate (matches the simple realized accuracy)
    dir_acc: float
    n_correct: int
    n_total: int

    # Block bootstrap CI (autocorrelation-aware)
    block_bootstrap_ci_low: float
    block_bootstrap_ci_high: float

    # Permutation test
    permutation_p_value: float
    permutation_null_mean: float
    permutation_null_std: float
    # Z-score: how many null-distribution stdevs above the null mean
    # the observed dir_acc sits. > 3 = strong evidence.
    permutation_z_score: float

    # Per-day stability
    per_day: list[dict[str, Any]] = field(default_factory=list)
    per_day_min: float | None = None
    per_day_max: float | None = None
    per_day_std: float | None = None
    per_day_all_above_chance: bool = False

    # Per-hour-of-day
    per_hour: list[dict[str, Any]] = field(default_factory=list)

    # Regime-conditional
    per_regime: list[dict[str, Any]] = field(default_factory=list)

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


# ──────────────────────────────────────────────────────────────────────────
# Pure stat helpers — testable without DB
# ──────────────────────────────────────────────────────────────────────────


def block_bootstrap_dir_acc(
    correct: np.ndarray,
    *,
    block_size: int,
    n_resamples: int,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Block bootstrap CI for dir_acc.

    Resamples WITH REPLACEMENT contiguous blocks of length
    ``block_size`` (in original-row units, NOT minutes if rows are
    e.g. predictions; the caller passes block_size matching the row
    semantics — for 1-min predictions, block_size=60 = 1-hour blocks).

    Why blocks: on autocorrelated time series, an i.i.d. bootstrap
    acts as if every observation is independent and gives narrower
    CIs than the truth. A block bootstrap preserves within-block
    autocorrelation, giving the canonical Politis-Romano (1994)
    "moving block bootstrap" CI.

    ``correct`` is a 1-D array of 0/1 ints — was the prediction right
    on this row?

    Returns ``(ci_low, ci_high)`` for the (1-α) CI.
    """
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    block_size = min(block_size, n)
    rng = np.random.default_rng(seed)
    # Number of blocks needed to cover n rows.
    n_blocks = int(math.ceil(n / block_size))
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        # Pick random starting points; each block is [start, start+block_size).
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Build the resample by concatenating blocks; trim to n rows so the
        # estimate lives on the same sample size as the original.
        chunks = [correct[s:s + block_size] for s in starts]
        resampled = np.concatenate(chunks)[:n]
        samples[i] = float(resampled.mean())
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def permutation_test_dir_acc(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    n_permutations: int,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Permutation test against H₀ "model has no directional signal".

    Procedure: shuffle ``y_true`` ``n_permutations`` times. For each
    shuffled labels vector, compute ``(y_pred == shuffled).mean()``.
    The fraction of those shuffled accuracies ≥ the observed accuracy
    is the one-sided permutation p-value.

    This is STRICTER than the binomial p-value: binomial tests
    against "fair coin" (50/50 random labels with NO structure),
    while permutation tests against "labels are independent of
    predictions" — the right H₀ for "are we just lucky on a regime".

    Returns
    -------
    p_value, null_mean, null_std, z_score
        p_value: P(null dir_acc ≥ observed) under the permutation null.
        z_score: (observed - null_mean) / null_std — how many stdevs.
    """
    n = len(y_pred)
    if n == 0 or n != len(y_true):
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    observed = float((y_pred == y_true).mean())
    samples = np.empty(n_permutations, dtype=float)
    y_true_copy = y_true.copy()
    for i in range(n_permutations):
        rng.shuffle(y_true_copy)
        samples[i] = float((y_pred == y_true_copy).mean())
    p = float((samples >= observed).sum() + 1) / float(n_permutations + 1)
    null_mean = float(samples.mean())
    null_std = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
    z = (observed - null_mean) / null_std if null_std > 0 else float("inf")
    return p, null_mean, null_std, float(z)


def _wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def per_day_stability(
    timestamps_utc: np.ndarray,
    correct: np.ndarray,
) -> list[dict[str, Any]]:
    """Bucket by UTC date, compute dir_acc per bucket with Wilson CI.

    ``timestamps_utc`` is a 1-D array of pandas-style datetime64[ns]
    or numpy datetime64. Returns a list of dicts, one per day,
    chronologically.
    """
    import pandas as pd  # local import keeps the helper testable
    if len(timestamps_utc) == 0:
        return []
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps_utc, utc=True),
                       "correct": correct})
    df["date"] = df["ts"].dt.date
    out: list[dict[str, Any]] = []
    for d, sub in df.groupby("date"):
        n = int(len(sub))
        hits = int(sub["correct"].sum())
        acc = hits / n if n > 0 else float("nan")
        lo, hi = _wilson_ci(hits, n)
        out.append({
            "date": str(d),
            "n": n,
            "hits": hits,
            "dir_acc": acc,
            "ci_low": lo,
            "ci_high": hi,
        })
    return out


def per_hour_breakdown(
    timestamps_utc: np.ndarray,
    correct: np.ndarray,
) -> list[dict[str, Any]]:
    """Bucket by UTC hour-of-day (0-23). Returns 24 rows always (zero-fill)."""
    import pandas as pd
    out = [{"hour_utc": h, "n": 0, "hits": 0, "dir_acc": None,
            "ci_low": None, "ci_high": None} for h in range(24)]
    if len(timestamps_utc) == 0:
        return out
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps_utc, utc=True),
                       "correct": correct})
    df["hour"] = df["ts"].dt.hour
    for h, sub in df.groupby("hour"):
        n = int(len(sub))
        hits = int(sub["correct"].sum())
        acc = hits / n if n > 0 else None
        lo, hi = _wilson_ci(hits, n) if n > 0 else (None, None)
        out[int(h)] = {
            "hour_utc": int(h),
            "n": n,
            "hits": hits,
            "dir_acc": acc,
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


def regime_conditional_accuracy(
    timestamps_utc: np.ndarray,
    correct: np.ndarray,
    microprices: np.ndarray,
    *,
    lookback_minutes: int = 60,
    sideways_threshold_bps: float = 10.0,
) -> list[dict[str, Any]]:
    """Bucket by trailing-trend regime: uptrend / sideways / downtrend.

    Regime for prediction at time t = sign of microprice change over
    the past ``lookback_minutes``. If |change| < ``sideways_threshold_bps``
    the regime is "sideways"; positive change = uptrend; negative = downtrend.

    Returns a list of 3 dicts (one per regime).
    """
    import pandas as pd
    if len(timestamps_utc) == 0:
        return []
    df = pd.DataFrame({
        "ts": pd.to_datetime(timestamps_utc, utc=True),
        "correct": correct,
        "microprice": microprices,
    }).sort_values("ts").reset_index(drop=True)

    # Compute trailing return for each row by joining on ts - lookback.
    df["ts_lookback"] = df["ts"] - pd.Timedelta(minutes=lookback_minutes)
    # merge_asof needs sorted on the lookup column.
    lookback_df = df[["ts", "microprice"]].rename(
        columns={"ts": "ts_lookback", "microprice": "mp_lookback"},
    ).sort_values("ts_lookback").reset_index(drop=True)
    merged = pd.merge_asof(
        df, lookback_df, on="ts_lookback", direction="backward",
    )
    merged["return_bps"] = (
        (merged["microprice"] - merged["mp_lookback"]) / merged["mp_lookback"] * 1e4
    )
    # Drop rows where we couldn't compute a trailing return (early rows).
    merged = merged.dropna(subset=["return_bps"]).reset_index(drop=True)

    def _bucket(r: float) -> str:
        if r > sideways_threshold_bps:
            return "uptrend"
        if r < -sideways_threshold_bps:
            return "downtrend"
        return "sideways"
    merged["regime"] = merged["return_bps"].apply(_bucket)

    out: list[dict[str, Any]] = []
    for regime in ("uptrend", "sideways", "downtrend"):
        sub = merged[merged["regime"] == regime]
        n = int(len(sub))
        hits = int(sub["correct"].sum()) if n > 0 else 0
        acc = hits / n if n > 0 else None
        lo, hi = _wilson_ci(hits, n) if n > 0 else (None, None)
        out.append({
            "regime": regime,
            "n": n,
            "hits": hits,
            "dir_acc": acc,
            "ci_low": lo,
            "ci_high": hi,
        })
    return out


# ──────────────────────────────────────────────────────────────────────────
# DB layer + orchestration
# ──────────────────────────────────────────────────────────────────────────


def load_predictions(database_url: str, *, symbol: str) -> "pd.DataFrame":  # noqa: F821
    """Load realized predictions from ``predictions_log``.

    Returns a DataFrame with columns ts (UTC), prob_up, signal,
    microprice, realized_correct, realized_microprice_1m. Drops
    rows where realized_correct IS NULL (still in-flight) or where
    signal is 'neutral' (no directional claim — can't be right/wrong).
    """
    import pandas as pd
    from sqlalchemy import create_engine, text
    eng = create_engine(database_url, future=True)
    sql = text("""
        SELECT ts, prob_up, signal, microprice,
               realized_correct, realized_microprice_1m
        FROM predictions_log
        WHERE symbol = :symbol
          AND realized_correct IS NOT NULL
          AND signal IN ('up', 'down')
        ORDER BY ts ASC
    """)
    with eng.connect() as conn:
        df = pd.read_sql(sql, conn, params={"symbol": symbol})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def run_robustness(
    database_url: str,
    *,
    symbol: str,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    block_size_minutes: int = 60,
    seed: int = 42,
) -> RobustnessReport:
    from datetime import datetime, timezone
    import pandas as pd

    df = load_predictions(database_url, symbol=symbol)
    n = len(df)
    logger.info("loaded %d realized predictions for %s", n, symbol)
    if n == 0:
        return RobustnessReport(
            symbol=symbol,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            n_predictions=0,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            block_size_minutes=block_size_minutes,
            dir_acc=float("nan"),
            n_correct=0, n_total=0,
            block_bootstrap_ci_low=float("nan"),
            block_bootstrap_ci_high=float("nan"),
            permutation_p_value=float("nan"),
            permutation_null_mean=float("nan"),
            permutation_null_std=float("nan"),
            permutation_z_score=float("nan"),
        )

    correct = df["realized_correct"].astype(int).to_numpy()
    n_correct = int(correct.sum())
    dir_acc = float(correct.mean())

    # signal == 'up' → y_pred = 1, signal == 'down' → y_pred = 0
    y_pred = (df["signal"] == "up").astype(int).to_numpy()
    # y_true derived: prediction was correct → y_true = y_pred,
    # else y_true = 1 - y_pred.
    y_true = np.where(correct == 1, y_pred, 1 - y_pred)

    # Block bootstrap CI.
    t0 = time.monotonic()
    ci_lo, ci_hi = block_bootstrap_dir_acc(
        correct, block_size=block_size_minutes,
        n_resamples=n_bootstrap, seed=seed,
    )
    logger.info(
        "block bootstrap (block=%d, n=%d): CI=[%.4f, %.4f] in %.1fs",
        block_size_minutes, n_bootstrap, ci_lo, ci_hi, time.monotonic() - t0,
    )

    # Permutation test.
    t0 = time.monotonic()
    perm_p, perm_mean, perm_std, perm_z = permutation_test_dir_acc(
        y_pred, y_true, n_permutations=n_permutations, seed=seed,
    )
    logger.info(
        "permutation test (n=%d): p=%.5g null_mean=%.4f z=%.2f in %.1fs",
        n_permutations, perm_p, perm_mean, perm_z, time.monotonic() - t0,
    )

    # Per-day stability.
    ts = df["ts"].to_numpy()
    per_day = per_day_stability(ts, correct)
    if per_day:
        accs = [d["dir_acc"] for d in per_day]
        per_day_min = float(min(accs))
        per_day_max = float(max(accs))
        per_day_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        per_day_all_above_chance = all(d["ci_low"] > 0.5 for d in per_day)
    else:
        per_day_min = per_day_max = per_day_std = None
        per_day_all_above_chance = False

    # Per-hour-of-day.
    per_hour = per_hour_breakdown(ts, correct)

    # Regime conditional.
    per_regime = regime_conditional_accuracy(
        ts, correct, df["microprice"].to_numpy(),
    )

    return RobustnessReport(
        symbol=symbol,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        n_predictions=n,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        block_size_minutes=block_size_minutes,
        dir_acc=dir_acc,
        n_correct=n_correct,
        n_total=n,
        block_bootstrap_ci_low=ci_lo,
        block_bootstrap_ci_high=ci_hi,
        permutation_p_value=perm_p,
        permutation_null_mean=perm_mean,
        permutation_null_std=perm_std,
        permutation_z_score=perm_z,
        per_day=per_day,
        per_day_min=per_day_min,
        per_day_max=per_day_max,
        per_day_std=per_day_std,
        per_day_all_above_chance=per_day_all_above_chance,
        per_hour=per_hour,
        per_regime=per_regime,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", action="append",
                   default=None,
                   help="symbol to evaluate. Pass multiple times for "
                        "multi-symbol run. Default: BTCUSDT, ETHUSDT, BNBUSDT.")
    p.add_argument("--out-dir", default="weights/highfreq",
                   help="directory to write <symbol>_1m_robustness.json")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--block-size-minutes", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for sym in symbols:
        sym = sym.upper()
        logger.info("=" * 60)
        logger.info("running robustness suite for %s", sym)
        try:
            report = run_robustness(
                dsn, symbol=sym,
                n_bootstrap=args.n_bootstrap,
                n_permutations=args.n_permutations,
                block_size_minutes=args.block_size_minutes,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", sym, exc, exc_info=True)
            rc = 1
            continue
        out_path = out_dir / f"{sym.lower()}_1m_robustness.json"
        out_path.write_text(report.to_json())
        logger.info(
            "%s: dir_acc=%.4f block_ci=[%.4f, %.4f] perm_p=%.5g z=%.2f n=%d "
            "→ %s",
            sym, report.dir_acc, report.block_bootstrap_ci_low,
            report.block_bootstrap_ci_high, report.permutation_p_value,
            report.permutation_z_score, report.n_predictions, out_path,
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
