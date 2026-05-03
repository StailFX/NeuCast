"""Tests for ``build_triple_barrier_labels`` (T.17.a).

López de Prado triple-barrier labeling. For each bar we simulate
a long entry at ``microprice_close`` and check the next
``time_stop_bars`` bars' high/low trajectory:

* TP first → tbl_y = 1
* SL first → tbl_y = 0
* Neither → tbl_y = 2 (time_stop)
* Insufficient lookahead → tbl_y = -1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.highfreq.feature_pipeline import build_triple_barrier_labels


def _bar(symbol: str, minute_offset: int, *,
         o: float, c: float, h: float, l: float) -> dict:
    """Build a single minute-bar row in the schema build_triple_barrier_labels expects."""
    base_ts = pd.Timestamp("2026-05-03T10:00:00", tz="UTC")
    return {
        "symbol": symbol,
        "minute": base_ts + pd.Timedelta(minutes=minute_offset),
        "microprice_open": o,
        "microprice_close": c,
        "microprice_high": h,
        "microprice_low": l,
    }


def test_tbl_tp_hit_first_labels_1():
    """Entry at 1000. TP=+5bp (1000.5). Next bar high=1001 → TP hit."""
    bars = [
        _bar("BTC", 0, o=1000, c=1000, h=1000, l=999.9),
        _bar("BTC", 1, o=1000, c=1000.4, h=1001.0, l=999.9),
        _bar("BTC", 2, o=1000.4, c=1000.4, h=1000.4, l=1000.4),
    ]
    df = pd.DataFrame(bars)
    out = build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=2)
    # Bar 0: TP barrier at 1000.5; bar 1's high 1001 ≥ 1000.5 → label 1.
    assert out.iloc[0]["tbl_y"] == 1
    assert out.iloc[0]["tbl_first_hit"] == "tp"
    assert out.iloc[0]["tbl_first_hit_bars"] == 1


def test_tbl_sl_hit_first_labels_0():
    """Entry at 1000. SL=-5bp (999.5). Next bar low=999 → SL hit."""
    bars = [
        _bar("BTC", 0, o=1000, c=1000, h=1000.05, l=999.9),
        _bar("BTC", 1, o=1000, c=999.6, h=1000.05, l=999.0),
        _bar("BTC", 2, o=999.6, c=999.6, h=999.6, l=999.6),
    ]
    df = pd.DataFrame(bars)
    out = build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=2)
    assert out.iloc[0]["tbl_y"] == 0
    assert out.iloc[0]["tbl_first_hit"] == "sl"


def test_tbl_time_stop_when_neither_barrier_hit():
    """Entry at 1000. TP=+5bp, SL=-5bp. Price stays in [999.9, 1000.1].
    No barrier hit in 2 bars → label = 2 (time_stop)."""
    bars = [
        _bar("BTC", 0, o=1000, c=1000, h=1000, l=1000),
        _bar("BTC", 1, o=1000, c=1000.05, h=1000.1, l=999.9),
        _bar("BTC", 2, o=1000.05, c=1000.0, h=1000.1, l=999.95),
    ]
    df = pd.DataFrame(bars)
    out = build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=2)
    assert out.iloc[0]["tbl_y"] == 2
    assert out.iloc[0]["tbl_first_hit"] == "time_stop"
    assert out.iloc[0]["tbl_first_hit_bars"] == 2


def test_tbl_insufficient_lookahead_labels_minus_1():
    """The last ``time_stop_bars`` bars of each symbol cannot be
    labeled because we don't have enough future bars."""
    bars = [
        _bar("BTC", 0, o=1000, c=1000, h=1000, l=1000),
        _bar("BTC", 1, o=1000, c=1000, h=1000, l=1000),
    ]
    df = pd.DataFrame(bars)
    out = build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=3)
    # Both bars need 3 future bars; only have 1 (bar 1) and 0 (bar 2). Both insufficient.
    assert out.iloc[0]["tbl_y"] == -1
    assert out.iloc[1]["tbl_y"] == -1
    assert out.iloc[0]["tbl_first_hit"] == "insufficient"


def test_tbl_simultaneous_tp_sl_in_same_bar_picks_sl():
    """If a single future bar's range covers BOTH barriers
    (large-range bar), tie-break = SL wins (conservative for a
    long-side label, matches realistic execution where you can't
    know intra-bar order without tick data)."""
    bars = [
        _bar("BTC", 0, o=1000, c=1000, h=1000, l=1000),
        # Bar 1 sweeps both: low=999 hits SL, high=1001 hits TP.
        _bar("BTC", 1, o=1000, c=1000, h=1001, l=999),
    ]
    df = pd.DataFrame(bars)
    out = build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=1)
    # Conservative: SL wins.
    assert out.iloc[0]["tbl_y"] == 0
    assert out.iloc[0]["tbl_first_hit"] == "sl"


def test_tbl_per_symbol_isolation():
    """Two symbols in same DataFrame: BTC's lookahead must NOT include
    ETH's bars (cross-symbol bleed would be a silent bug)."""
    bars = [
        # BTC: entry 1000, bar 1 close at 1000 (no TP/SL hit), should
        # be time_stop.
        _bar("BTC", 0, o=1000, c=1000, h=1000.0, l=1000.0),
        _bar("BTC", 1, o=1000, c=1000, h=1000.05, l=999.95),
        # ETH: entry 2000, bar 1 hits TP at 2001.
        _bar("ETH", 0, o=2000, c=2000, h=2000, l=2000),
        _bar("ETH", 1, o=2000, c=2002, h=2002, l=2000),
    ]
    df = pd.DataFrame(bars)
    out = build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=1)
    # Sort by (symbol, minute) for deterministic indexing.
    out = out.sort_values(["symbol", "minute"]).reset_index(drop=True)
    btc_first = out[out["symbol"] == "BTC"].iloc[0]
    eth_first = out[out["symbol"] == "ETH"].iloc[0]
    assert btc_first["tbl_y"] == 2  # time_stop, no barriers hit
    assert eth_first["tbl_y"] == 1  # TP at 2001 ≥ 2001 (entry × 1.0005)


def test_tbl_empty_input_returns_empty_with_correct_dtype():
    df = pd.DataFrame(columns=[
        "symbol", "minute", "microprice_open",
        "microprice_close", "microprice_high", "microprice_low",
    ])
    out = build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=10)
    assert out.empty
    assert "tbl_y" in out.columns
    assert "tbl_first_hit" in out.columns
    assert "tbl_first_hit_bars" in out.columns


def test_tbl_rejects_invalid_thresholds():
    df = pd.DataFrame([_bar("BTC", 0, o=1000, c=1000, h=1000, l=1000)])
    with pytest.raises(ValueError, match="tp_bps and sl_bps"):
        build_triple_barrier_labels(df, tp_bps=0.0, sl_bps=5.0, time_stop_bars=10)
    with pytest.raises(ValueError, match="time_stop_bars"):
        build_triple_barrier_labels(df, tp_bps=5.0, sl_bps=5.0, time_stop_bars=0)


def test_tbl_asymmetric_tp_sl_thresholds():
    """tp_bps and sl_bps don't have to be equal — some traders run
    asymmetric brackets (e.g. TP=10bp, SL=3bp for low-frequency
    momentum)."""
    bars = [
        _bar("BTC", 0, o=1000, c=1000, h=1000, l=1000),
        # tp=10bp = 1001, sl=3bp = 999.7. Bar 1 hits low=999.5 → SL first.
        _bar("BTC", 1, o=1000, c=1000.5, h=1001.5, l=999.5),
    ]
    df = pd.DataFrame(bars)
    out = build_triple_barrier_labels(df, tp_bps=10.0, sl_bps=3.0, time_stop_bars=1)
    assert out.iloc[0]["tbl_y"] == 0
    assert out.iloc[0]["tbl_first_hit"] == "sl"
