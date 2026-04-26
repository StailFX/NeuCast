"""Grid-search the (entry_long, entry_short) thresholds against historical paper trades.

Usage (CLI)
-----------

    python -m app.highfreq.threshold_search \\
        --symbol BTCUSDT \\
        --since-hours 168 \\          # last 7 days
        --long-grid 0.50,0.52,0.55,0.58,0.60 \\
        --short-grid 0.40,0.42,0.45,0.48,0.50 \\
        --metric sharpe \\            # or: pnl_total, win_rate
        --top 5

Reads from ``paper_trades`` (the trader's actual closed-trade rows)
and **re-simulates** what would have happened under each threshold
pair *given the same prob_up signals*. Cheap because we don't re-run
the predictor — we only re-classify already-recorded prob_up values.

Why this exists
---------------

The defaults ``(0.55, 0.45)`` were chosen heuristically (symmetric
around 0.50, fee-aware). Once we have ≥ a few hundred paper trades
the grid-search lets us answer "would tightening to (0.58, 0.42)
have improved Sharpe at the cost of fewer trades?" with hindsight.

The output is ADVISORY — we don't auto-update the runner config.
Defense talking point: "the system surfaces optimisation candidates
but defers the actual change to a human-reviewed PR".

Math
----

For each (long_thr, short_thr) candidate:
  * **Re-classify** every paper trade by its ``entry_prob_up``:
      - prob_up >= long_thr   → would have opened LONG
      - prob_up <= short_thr  → would have opened SHORT
      - else                  → SKIPPED (no trade)
  * For trades whose original side matches the new classification,
    keep the recorded P&L. For SKIPPED trades, contribute 0 (we'd
    not have opened them).
  * Note this is approximate — the *actual* trades that would have
    opened differ in their TIMING (a tighter threshold means fewer
    trades, potentially missing horizon overlaps). For threshold
    *tuning* this is the standard simplification; full re-simulation
    requires re-walking the predictions DataFrame, which the trainer
    doesn't currently persist (Phase D enhancement).

Metrics:
  * ``pnl_total``    — sum of pnl_usd
  * ``pnl_bps_avg``  — mean(pnl_bps) per kept trade
  * ``win_rate``     — share of kept trades with pnl_usd >= 0
  * ``sharpe``       — mean(pnl_bps) / std(pnl_bps) on kept trades
  * ``n_trades``     — kept trade count (exposure proxy)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# psycopg2 is a CLI-only runtime dep — pure-function tests don't need it
# and shouldn't fail on local envs without the driver. Import is deferred
# to load_paper_trades().

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridCell:
    long_thr: float
    short_thr: float
    n_trades: int
    pnl_total_usd: float
    pnl_bps_avg: float
    pnl_bps_std: float
    win_rate: float
    sharpe: float

    def to_row(self) -> dict:
        return {
            "long_thr": round(self.long_thr, 4),
            "short_thr": round(self.short_thr, 4),
            "n_trades": self.n_trades,
            "pnl_total_usd": round(self.pnl_total_usd, 4),
            "pnl_bps_avg": round(self.pnl_bps_avg, 4),
            "pnl_bps_std": round(self.pnl_bps_std, 4),
            "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe, 4),
        }


# ──────────────────────────────────────────────────────────────────
# Pure simulation function (testable without DB)
# ──────────────────────────────────────────────────────────────────


def evaluate_threshold(
    trades: pd.DataFrame, *, long_thr: float, short_thr: float,
) -> GridCell:
    """Re-classify each row and roll up metrics. Pure function.

    Expected columns in ``trades``:
      * entry_prob_up : float
      * side          : "long" | "short"
      * pnl_usd       : float
      * pnl_bps       : float
    """
    if trades.empty:
        return GridCell(long_thr, short_thr, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    p = trades["entry_prob_up"].astype(float)
    # Decision under candidate thresholds.
    would_long = p >= long_thr
    would_short = p <= short_thr

    # Keep rows where the WOULD decision matches the actual recorded side.
    # (We can't re-realize a trade we wouldn't have opened.)
    keep_long = would_long & (trades["side"] == "long")
    keep_short = would_short & (trades["side"] == "short")
    kept = trades[keep_long | keep_short]

    if kept.empty:
        return GridCell(long_thr, short_thr, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnl_usd = kept["pnl_usd"].astype(float)
    pnl_bps = kept["pnl_bps"].astype(float)
    bps_std = float(pnl_bps.std() or 1e-9)
    return GridCell(
        long_thr=long_thr,
        short_thr=short_thr,
        n_trades=int(len(kept)),
        pnl_total_usd=float(pnl_usd.sum()),
        pnl_bps_avg=float(pnl_bps.mean()),
        pnl_bps_std=bps_std,
        win_rate=float((pnl_usd >= 0).mean()),
        sharpe=float(pnl_bps.mean() / bps_std),
    )


def grid_search(
    trades: pd.DataFrame,
    *,
    long_grid: Iterable[float],
    short_grid: Iterable[float],
) -> list[GridCell]:
    """Cartesian product of (long, short) thresholds. Skips invalid pairs
    where short_thr >= long_thr."""
    cells: list[GridCell] = []
    for lt in long_grid:
        for st in short_grid:
            if st >= lt:
                continue
            cells.append(evaluate_threshold(trades, long_thr=lt, short_thr=st))
    return cells


# ──────────────────────────────────────────────────────────────────
# DB loader (sync psycopg2 — matches the trainer's pattern)
# ──────────────────────────────────────────────────────────────────


def load_paper_trades(
    *, dsn: str, symbol: str, since_hours: float,
) -> pd.DataFrame:
    """Load closed paper trades for ``symbol`` from the last
    ``since_hours``. Returns empty frame if none."""
    import psycopg2  # noqa: WPS433 — CLI-only dep, lazy
    import psycopg2.extras
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT entry_ts, exit_ts, side, entry_prob_up, "
                "       pnl_usd, pnl_bps "
                "FROM paper_trades "
                "WHERE symbol = %s "
                "  AND exit_ts > now() - make_interval(hours => %s) "
                "ORDER BY exit_ts ASC",
                (symbol.upper(), float(since_hours)),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=[
            "entry_ts", "exit_ts", "side", "entry_prob_up", "pnl_usd", "pnl_bps",
        ])
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def _parse_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--since-hours", type=float, default=168.0,
                   help="lookback window (default 7 days)")
    p.add_argument(
        "--long-grid",
        default="0.50,0.52,0.55,0.58,0.60,0.62,0.65",
        help="comma-separated long thresholds",
    )
    p.add_argument(
        "--short-grid",
        default="0.35,0.38,0.40,0.42,0.45,0.48,0.50",
        help="comma-separated short thresholds",
    )
    p.add_argument(
        "--metric", default="sharpe",
        choices=("sharpe", "pnl_total_usd", "pnl_bps_avg", "win_rate", "n_trades"),
        help="rank cells by this metric (descending)",
    )
    p.add_argument("--top", type=int, default=10, help="show top-N cells")
    p.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    if not args.dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2

    df = load_paper_trades(dsn=args.dsn, symbol=args.symbol, since_hours=args.since_hours)
    if df.empty:
        print(f"no paper trades for {args.symbol} in last {args.since_hours}h — nothing to search")
        return 1

    long_grid = _parse_grid(args.long_grid)
    short_grid = _parse_grid(args.short_grid)

    cells = grid_search(df, long_grid=long_grid, short_grid=short_grid)
    if not cells:
        print("empty grid (all short_thr >= long_thr) — adjust --long-grid / --short-grid")
        return 1

    rows = [c.to_row() for c in cells]
    out_df = pd.DataFrame(rows).sort_values(args.metric, ascending=False)

    print(f"\nGrid-search results for {args.symbol} "
          f"({len(df)} trades over {args.since_hours}h, "
          f"{len(cells)} valid threshold pairs):\n")
    print(out_df.head(args.top).to_string(index=False))
    print()

    best = out_df.iloc[0]
    print(f"Best by {args.metric}: long_thr={best['long_thr']}  "
          f"short_thr={best['short_thr']}  "
          f"sharpe={best['sharpe']}  pnl=${best['pnl_total_usd']}  "
          f"n={int(best['n_trades'])}")
    print("\n→ Update PaperTraderConfig defaults via PR if this beats the "
          "current (0.55, 0.45) AND has enough sample size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
