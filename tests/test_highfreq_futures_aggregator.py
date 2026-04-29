"""Tests for ``app.highfreq.futures_aggregator`` — the parallel-to-spot
aggregator that writes to ``highfreq_futures_ofi_1s`` (release S, ADR-019).

Pin contracts that differ from the spot aggregator:

* mark-price + funding-rate stamping (latest-observed carry-forward,
  not bucketed)
* NULL handling for symbols that never received a mark frame
* tolerance for the dispatch ordering in :func:`asyncio.run` —
  on_snapshot / on_trade / on_mark_price are async and may interleave

We don't open a real Postgres connection here; a stub pool collects
the would-be INSERT records so we can inspect them.
"""
from __future__ import annotations

import asyncio
import math

import pytest

from app.highfreq.futures_aggregator import (
    AggregatedRow,
    FuturesAggregator,
    _safe,
    _safe_or_none,
)
from app.highfreq.futures_l2_consumer import MarkPriceUpdate
from app.highfreq.l2_consumer import L2Snapshot, Trade


# ─── _safe / _safe_or_none ───


def test_safe_replaces_nan_with_zero():
    assert _safe(float("nan")) == 0.0
    assert _safe(float("inf")) == 0.0
    assert _safe(float("-inf")) == 0.0


def test_safe_passes_finite_through():
    assert _safe(0.0) == 0.0
    assert _safe(-1.5) == -1.5
    assert _safe(123.456) == 123.456


def test_safe_or_none_preserves_none():
    """Mark-price / funding-rate columns can be unobserved — that's
    semantically different from "overflowed". Pin: None stays None."""
    assert _safe_or_none(None) is None


def test_safe_or_none_replaces_nan_with_none():
    """An aberrant NaN gets translated to None (NULL in Postgres)
    rather than pretending we observed 0.0 — keeps the data honest."""
    assert _safe_or_none(float("nan")) is None
    assert _safe_or_none(float("inf")) is None


def test_safe_or_none_passes_finite_through():
    assert _safe_or_none(65430.10) == 65430.10
    assert _safe_or_none(-0.0001) == -0.0001


# ─── on_mark_price stateful tracking ───


def test_on_mark_price_updates_latest_state():
    """Mark-price frames go to a per-symbol stateful cache, not a bucket.
    Pin: the latest one wins, regardless of arrival ordering."""
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT"],
    )
    mp1 = MarkPriceUpdate(
        event_time_ms=1000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65000.0, index_price=64999.0,
        funding_rate=0.0001, next_funding_ms=1234,
    )
    mp2 = MarkPriceUpdate(
        event_time_ms=2000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65010.0, index_price=65009.0,
        funding_rate=0.00012, next_funding_ms=1234,
    )
    asyncio.run(agg.on_mark_price(mp1))
    asyncio.run(agg.on_mark_price(mp2))
    latest = agg._latest_mark["BTCUSDT"]
    assert latest.mark_price == 65010.0
    assert latest.funding_rate == pytest.approx(0.00012)
    assert agg.mark_frames_seen == 2


def test_on_mark_price_partitions_by_symbol():
    """BTC mark frames don't pollute ETH state. Pin: per-symbol cache."""
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT", "ETHUSDT"],
    )
    asyncio.run(agg.on_mark_price(MarkPriceUpdate(
        event_time_ms=1000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65000.0, index_price=0.0,
        funding_rate=0.0001, next_funding_ms=0,
    )))
    asyncio.run(agg.on_mark_price(MarkPriceUpdate(
        event_time_ms=1000, local_recv_ms=0, symbol="ETHUSDT",
        mark_price=3500.0, index_price=0.0,
        funding_rate=-0.0002, next_funding_ms=0,
    )))
    assert agg._latest_mark["BTCUSDT"].mark_price == 65000.0
    assert agg._latest_mark["ETHUSDT"].mark_price == 3500.0


# ─── _finalize: row stamping with mark / no-mark cases ───


def _make_bucket_with_one_frame(symbol: str, second_ms: int):
    """Synthetic single-frame bucket — bypass the on_snapshot path so
    the test focuses on _finalize."""
    from app.highfreq.futures_aggregator import _SecondBucket
    from app.highfreq.ofi_features import FrameFeatures

    bucket = _SecondBucket(symbol=symbol, second_ms=second_ms)
    feat = FrameFeatures(
        event_time_ms=second_ms,
        symbol=symbol,
        ofi=1.5, microprice=65000.0, depth_imb=0.1,
        spread_bps=1.2, mid=65000.0,
    )
    bucket.add_frame(feat)
    return bucket


def test_finalize_with_mark_observed_stamps_all_three_columns():
    """When a mark-price frame has been observed for the symbol, the
    emitted row carries mark_price / funding_rate / next_funding_ms."""
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT"],
    )
    asyncio.run(agg.on_mark_price(MarkPriceUpdate(
        event_time_ms=999_000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65430.10, index_price=65420.50,
        funding_rate=0.00012345, next_funding_ms=1711238400000,
    )))
    bucket = _make_bucket_with_one_frame("BTCUSDT", 1000)
    row = agg._finalize(bucket, jitter=(0, 5))
    assert isinstance(row, AggregatedRow)
    assert row.mark_price == pytest.approx(65430.10)
    assert row.funding_rate == pytest.approx(0.00012345)
    assert row.next_funding_ms == 1711238400000


def test_finalize_without_mark_emits_none_for_extra_columns():
    """No mark-price frame yet → mark/funding columns are None →
    Postgres stores NULL. Pin: don't emit 0.0 or NaN."""
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT"],
    )
    bucket = _make_bucket_with_one_frame("BTCUSDT", 1000)
    row = agg._finalize(bucket, jitter=(0, 0))
    assert row.mark_price is None
    assert row.funding_rate is None
    assert row.next_funding_ms is None
    # OFI / microprice fields ARE populated though (they came from the bucket).
    assert row.microprice == pytest.approx(65000.0)


def test_finalize_carries_forward_latest_mark_across_seconds():
    """Mark frame at t=1000ms carries forward to bars at 2000ms, 3000ms,
    etc. — until a newer mark replaces it. This is the stated contract
    in the module docstring (slow-moving funding rate, latest-sample
    semantics)."""
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT"],
    )
    asyncio.run(agg.on_mark_price(MarkPriceUpdate(
        event_time_ms=1000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65000.0, index_price=0.0,
        funding_rate=0.0001, next_funding_ms=0,
    )))
    row1 = agg._finalize(
        _make_bucket_with_one_frame("BTCUSDT", 1000),
        jitter=(0, 0),
    )
    row2 = agg._finalize(
        _make_bucket_with_one_frame("BTCUSDT", 2000),
        jitter=(0, 0),
    )
    row3 = agg._finalize(
        _make_bucket_with_one_frame("BTCUSDT", 3000),
        jitter=(0, 0),
    )
    assert row1.mark_price == row2.mark_price == row3.mark_price == 65000.0
    assert row1.funding_rate == row2.funding_rate == row3.funding_rate
    # Now a fresher frame at t=4000.
    asyncio.run(agg.on_mark_price(MarkPriceUpdate(
        event_time_ms=4000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65020.0, index_price=0.0,
        funding_rate=0.00015, next_funding_ms=0,
    )))
    row4 = agg._finalize(
        _make_bucket_with_one_frame("BTCUSDT", 4000),
        jitter=(0, 0),
    )
    assert row4.mark_price == 65020.0
    assert row4.funding_rate == pytest.approx(0.00015)


def test_finalize_independent_per_symbol():
    """BTC mark cache doesn't bleed into ETH rows."""
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT", "ETHUSDT"],
    )
    asyncio.run(agg.on_mark_price(MarkPriceUpdate(
        event_time_ms=1000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65000.0, index_price=0.0,
        funding_rate=0.0001, next_funding_ms=0,
    )))
    btc_row = agg._finalize(
        _make_bucket_with_one_frame("BTCUSDT", 2000), jitter=(0, 0),
    )
    eth_row = agg._finalize(
        _make_bucket_with_one_frame("ETHUSDT", 2000), jitter=(0, 0),
    )
    assert btc_row.mark_price == 65000.0
    assert eth_row.mark_price is None  # ETH never got a mark frame


# ─── Constructor + counters ───


def test_constructor_initialises_counters():
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT"],
    )
    assert agg.rows_emitted == 0
    assert agg.rows_written == 0
    assert agg.mark_frames_seen == 0


def test_constructor_uppercases_symbols():
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["btcusdt", "ETHUSDT"],
    )
    assert agg.symbols == ["BTCUSDT", "ETHUSDT"]


# ─── Negative-funding sanity ───


def test_negative_funding_rate_preserved_in_row():
    """When shorts are paying longs (bearish-cost regime) funding_rate
    is negative. Pin: row carries the signed value, no abs() applied."""
    agg = FuturesAggregator(
        database_url="postgres://stub", symbols=["BTCUSDT"],
    )
    asyncio.run(agg.on_mark_price(MarkPriceUpdate(
        event_time_ms=1000, local_recv_ms=0, symbol="BTCUSDT",
        mark_price=65000.0, index_price=0.0,
        funding_rate=-0.0005, next_funding_ms=0,
    )))
    row = agg._finalize(
        _make_bucket_with_one_frame("BTCUSDT", 2000), jitter=(0, 0),
    )
    assert row.funding_rate == pytest.approx(-0.0005)
