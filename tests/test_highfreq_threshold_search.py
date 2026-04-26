"""Tests for ``app.highfreq.threshold_search`` — offline grid-search."""
from __future__ import annotations

import pandas as pd
import pytest

from app.highfreq.threshold_search import (
    GridCell,
    evaluate_threshold,
    grid_search,
)


def _trades(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ───────────────────────── evaluate_threshold ─────────────────────────


def test_evaluate_threshold_empty_df_returns_zeros():
    cell = evaluate_threshold(_trades([]), long_thr=0.55, short_thr=0.45)
    assert cell.n_trades == 0
    assert cell.pnl_total_usd == 0.0
    assert cell.sharpe == 0.0


def test_evaluate_threshold_keeps_long_trades_above_threshold():
    """Trades with prob >= long_thr that were ACTUALLY long → kept."""
    trades = _trades([
        {"entry_prob_up": 0.70, "side": "long", "pnl_usd": +0.5, "pnl_bps": 50},
        {"entry_prob_up": 0.60, "side": "long", "pnl_usd": +0.3, "pnl_bps": 30},
        {"entry_prob_up": 0.50, "side": "long", "pnl_usd": +0.1, "pnl_bps": 10},  # excluded
    ])
    cell = evaluate_threshold(trades, long_thr=0.55, short_thr=0.45)
    assert cell.n_trades == 2
    assert cell.pnl_total_usd == pytest.approx(0.8)


def test_evaluate_threshold_keeps_short_trades_below_threshold():
    trades = _trades([
        {"entry_prob_up": 0.20, "side": "short", "pnl_usd": +0.4, "pnl_bps": 40},
        {"entry_prob_up": 0.30, "side": "short", "pnl_usd": -0.2, "pnl_bps": -20},
        {"entry_prob_up": 0.50, "side": "short", "pnl_usd": +0.1, "pnl_bps": 10},  # excluded
    ])
    cell = evaluate_threshold(trades, long_thr=0.55, short_thr=0.45)
    assert cell.n_trades == 2
    assert cell.pnl_total_usd == pytest.approx(0.2)


def test_evaluate_threshold_excludes_side_mismatch():
    """Even if prob >= long_thr, if side='short' it's not a long-decision
    we can replay — exclude."""
    trades = _trades([
        {"entry_prob_up": 0.70, "side": "short", "pnl_usd": +1.0, "pnl_bps": 100},
    ])
    cell = evaluate_threshold(trades, long_thr=0.55, short_thr=0.45)
    assert cell.n_trades == 0


def test_evaluate_threshold_win_rate_and_sharpe():
    """3 wins + 2 losses → win_rate 0.6; Sharpe is mean/std of pnl_bps."""
    trades = _trades([
        {"entry_prob_up": 0.7, "side": "long", "pnl_usd": +1.0, "pnl_bps": +20.0},
        {"entry_prob_up": 0.7, "side": "long", "pnl_usd": +0.5, "pnl_bps": +10.0},
        {"entry_prob_up": 0.7, "side": "long", "pnl_usd": +0.5, "pnl_bps": +10.0},
        {"entry_prob_up": 0.7, "side": "long", "pnl_usd": -0.3, "pnl_bps": -5.0},
        {"entry_prob_up": 0.7, "side": "long", "pnl_usd": -0.5, "pnl_bps": -10.0},
    ])
    cell = evaluate_threshold(trades, long_thr=0.55, short_thr=0.45)
    assert cell.n_trades == 5
    assert cell.win_rate == pytest.approx(0.6)
    # mean_bps = 5.0; std_bps via pandas std (ddof=1) ≈ 12.04
    assert cell.pnl_bps_avg == pytest.approx(5.0)
    assert cell.sharpe == pytest.approx(5.0 / cell.pnl_bps_std)


# ───────────────────────── grid_search ─────────────────────────


def test_grid_search_skips_invalid_pairs():
    """short_thr >= long_thr is non-sensical (overlapping decision regions)."""
    cells = grid_search(
        _trades([{"entry_prob_up": 0.7, "side": "long",
                  "pnl_usd": 1.0, "pnl_bps": 20.0}]),
        long_grid=[0.50, 0.55, 0.60],
        short_grid=[0.40, 0.50, 0.55],
    )
    # Valid pairs only: (0.50, 0.40), (0.55, 0.40), (0.55, 0.50),
    # (0.60, 0.40), (0.60, 0.50), (0.60, 0.55) — 6 pairs out of 9.
    assert len(cells) == 6
    for c in cells:
        assert c.short_thr < c.long_thr


def test_grid_search_returns_one_cell_per_valid_pair():
    cells = grid_search(
        _trades([]),
        long_grid=[0.55, 0.60],
        short_grid=[0.40, 0.45],
    )
    # All 4 pairs valid (both shorts < both longs).
    assert len(cells) == 4


def test_grid_cell_to_row_round_trips_to_dict():
    c = GridCell(
        long_thr=0.55, short_thr=0.45,
        n_trades=10, pnl_total_usd=1.234567,
        pnl_bps_avg=5.0, pnl_bps_std=12.0,
        win_rate=0.6, sharpe=0.4167,
    )
    row = c.to_row()
    assert row["pnl_total_usd"] == 1.2346  # rounded to 4 dp
    assert row["n_trades"] == 10
    assert row["sharpe"] == 0.4167
