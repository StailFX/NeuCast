"""Tests for ``tools.binance_klines_download``.

The HTTP fetch is mocked so tests don't depend on Binance's API
availability. The pure ``klines_to_minute_df`` adapter is tested
against the canonical klines payload shape so a Binance API change
(adding a column / reordering) breaks loudly here, not silently in
production.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tools.binance_klines_download import (
    KLINES_COLUMNS,
    fetch_klines_chunk,
    klines_to_minute_df,
)


# Real Binance kline payload format (12 columns):
# [open_time_ms, open, high, low, close, volume, close_time_ms,
#  quote_volume, n_trades, taker_buy_base, taker_buy_quote, "0"]
_SAMPLE_KLINES = [
    [1714542000000, "60170.41", "60176.00", "60156.81", "60156.81",
     "3.0987", 1714542059999, "186437.7", 337,
     "1.8797", "113091.84", "0"],
    [1714542060000, "60156.82", "60156.82", "60133.26", "60133.26",
     "3.4491", 1714542119999, "207457.8", 311,
     "0.7479", "44986.22", "0"],
    [1714542120000, "60133.27", "60133.27", "60030.00", "60033.99",
     "10.677", 1714542179999, "641418.9", 815,
     "2.5908", "155593.33", "0"],
]


def test_klines_columns_pin_canonical_binance_schema():
    """If Binance adds/reorders columns this fails; we want to know
    before our trainer silently mis-aligns."""
    assert KLINES_COLUMNS == [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "n_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "_ignore",
    ]


def test_klines_to_minute_df_happy_path_emits_minute_bar_schema():
    df = klines_to_minute_df([_SAMPLE_KLINES], symbol="BTCUSDT")
    assert len(df) == 3
    expected_cols = {
        "minute", "symbol",
        "microprice_open", "microprice_close", "microprice_high", "microprice_low",
        "n_updates_sum", "trade_imb_sum", "trade_imb_abs_sum",
        "ofi_sum", "ofi_mean", "ofi_std",
        "depth_imb_mean", "depth_imb_std", "depth_imb_last",
        "spread_bps_mean", "spread_bps_last", "spread_bps_max",
        "seconds_observed",
    }
    assert expected_cols.issubset(set(df.columns))
    # Symbol normalised uppercase.
    assert (df["symbol"] == "BTCUSDT").all()
    # Numeric conversion happened (strings → float).
    assert df["microprice_open"].dtype in (np.float64, float)
    assert df["microprice_close"].iloc[0] == 60156.81
    # n_trades → n_updates_sum, kept as int.
    assert df["n_updates_sum"].iloc[0] == 337


def test_klines_to_minute_df_computes_trade_imb_in_minus_1_to_plus_1():
    df = klines_to_minute_df([_SAMPLE_KLINES], symbol="BTCUSDT")
    # taker_buy_base 1.8797 / total 3.0987 = 0.6065 → trade_imb = 0.213
    # Formula: 2 * 1.8797 / 3.0987 - 1 ≈ 0.2131
    assert abs(df["trade_imb_sum"].iloc[0] - (2 * 1.8797 / 3.0987 - 1)) < 1e-3
    # Bounded.
    assert (df["trade_imb_sum"].abs() <= 1.0 + 1e-6).all()


def test_klines_to_minute_df_neutral_zeros_for_l2_columns():
    """OFI / depth / spread aren't in klines → must be zero-filled so
    long_horizon pipeline doesn't choke on missing columns."""
    df = klines_to_minute_df([_SAMPLE_KLINES], symbol="BTCUSDT")
    for col in ("ofi_sum", "ofi_mean", "ofi_std",
                "depth_imb_mean", "depth_imb_std", "depth_imb_last",
                "spread_bps_mean", "spread_bps_last", "spread_bps_max"):
        assert (df[col] == 0.0).all(), f"{col} should be zero-filled"


def test_klines_to_minute_df_empty_input_returns_empty_df():
    df = klines_to_minute_df([], symbol="BTCUSDT")
    assert df.empty
    df2 = klines_to_minute_df([[]], symbol="BTCUSDT")
    assert df2.empty


def test_klines_to_minute_df_handles_zero_volume_without_division_error():
    """If a bar has 0 volume (rare but possible on illiquid pairs),
    trade_imb should be 0.0 not NaN/Inf."""
    zero_vol_kline = list(_SAMPLE_KLINES[0])
    zero_vol_kline[5] = "0.0"  # volume
    zero_vol_kline[9] = "0.0"  # taker_buy_base_volume
    df = klines_to_minute_df([[zero_vol_kline]], symbol="ETHUSDT")
    assert df["trade_imb_sum"].iloc[0] == 0.0
    assert not np.isnan(df["trade_imb_sum"].iloc[0])
    assert not np.isinf(df["trade_imb_sum"].iloc[0])


def test_klines_to_minute_df_minute_column_is_utc_datetime():
    df = klines_to_minute_df([_SAMPLE_KLINES], symbol="BTCUSDT")
    assert df["minute"].dt.tz is not None
    # 1714542000000 ms = 2024-05-01T05:40:00 UTC.
    assert df["minute"].iloc[0] == pd.Timestamp("2024-05-01T05:40:00", tz="UTC")


def test_klines_to_minute_df_sorted_by_minute_ascending():
    """Even if Binance returns out-of-order chunks (rare), output must
    be sorted so the downstream walk-forward sees increasing time."""
    out_of_order = list(reversed(_SAMPLE_KLINES))
    df = klines_to_minute_df([out_of_order], symbol="BTCUSDT")
    assert (df["minute"].diff().dropna() > pd.Timedelta(0)).all()


def test_fetch_klines_chunk_passes_correct_params():
    """The fetch helper must hit Binance's documented endpoint with
    the right params; our pretrain pipeline depends on it."""
    fake_resp = type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: _SAMPLE_KLINES,
    })()

    captured: dict = {}

    class FakeSession:
        def get(self, url, params, timeout):
            captured["url"] = url
            captured["params"] = params
            captured["timeout"] = timeout
            return fake_resp

    sess = FakeSession()
    out = fetch_klines_chunk(
        "btcusdt", start_ms=1714000000000, end_ms=1714099999999,
        interval="1m", limit=500, session=sess,
    )
    assert out == _SAMPLE_KLINES
    # URL & params pinned so we catch silent regressions.
    assert "binance.com" in captured["url"]
    assert captured["params"]["symbol"] == "BTCUSDT"   # uppercased
    assert captured["params"]["interval"] == "1m"
    assert captured["params"]["limit"] == 500
    assert captured["params"]["startTime"] == 1714000000000
    assert captured["params"]["endTime"] == 1714099999999
