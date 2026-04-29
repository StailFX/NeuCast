"""Tests for ``app.highfreq.futures_l2_consumer`` — the futures-side
WebSocket consumer parallel to the spot ``L2Consumer``.

Runs entirely on synthetic frames; no real WebSocket is opened. We pin
the parser contracts (frame → typed dataclass) and the URL builder so
a regression here is caught at CI time.
"""
from __future__ import annotations

import asyncio

import pytest

from app.highfreq.futures_l2_consumer import (
    BINANCE_FUTURES_WSS_BASE,
    FuturesL2Consumer,
    MarkPriceUpdate,
)


# ─── URL builder ───


def test_url_builder_uses_fstream_host():
    """The futures host is fstream.binance.com, NOT stream.binance.com.
    Pin: a refactor that defaults to spot URL would silently subscribe
    to the wrong venue."""
    c = FuturesL2Consumer(["BTCUSDT"])
    url = c._build_url()
    assert url.startswith(BINANCE_FUTURES_WSS_BASE)
    assert "fstream.binance.com" in url
    assert "stream.binance.com:9443" not in url


def test_url_builder_subscribes_to_all_three_streams_per_symbol():
    """Per ADR-019: depth + trade + markPrice. Three streams per symbol.
    Spot consumer subscribes to two (no markPrice) — pin the third."""
    c = FuturesL2Consumer(["BTCUSDT"])
    url = c._build_url()
    assert "btcusdt@depth20@100ms" in url
    assert "btcusdt@trade" in url
    assert "btcusdt@markPrice@1s" in url


def test_url_builder_handles_multiple_symbols():
    c = FuturesL2Consumer(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    url = c._build_url()
    for sym in ("btcusdt", "ethusdt", "bnbusdt"):
        assert f"{sym}@depth20@100ms" in url
        assert f"{sym}@trade" in url
        assert f"{sym}@markPrice@1s" in url
    # 3 symbols × 3 streams = 9 stream segments.
    streams_part = url.split("?streams=")[1]
    assert len(streams_part.split("/")) == 9


def test_url_builder_lowercases_symbols():
    """Binance requires lowercase in stream names."""
    c = FuturesL2Consumer(["BTCUSDT"])
    url = c._build_url()
    assert "btcusdt" in url
    # Lowercase guarantees no UPPER form leaks into the URL.
    streams_part = url.split("?streams=")[1]
    assert "BTCUSDT" not in streams_part


def test_url_builder_respects_depth_and_speed():
    c = FuturesL2Consumer(["BTCUSDT"], depth_levels=10, update_speed_ms=1000)
    url = c._build_url()
    assert "btcusdt@depth10@1000ms" in url


# ─── _parse_mark_price ───


def _mp_frame(**overrides) -> dict:
    """Reasonable Binance @markPrice@1s payload."""
    base = {
        "e": "markPriceUpdate",
        "E": 1711234567890,
        "s": "BTCUSDT",
        "p": "65430.10",       # mark price
        "i": "65420.50",       # index price
        "P": "65425.00",       # estimated settle (mostly ignored)
        "r": "0.00012345",     # funding rate (1.23 bp / 8h)
        "T": 1711238400000,    # next funding ms
    }
    base.update(overrides)
    return base


def test_mark_price_parses_complete_frame():
    out = FuturesL2Consumer._parse_mark_price(_mp_frame(), local_recv_ms=999)
    assert isinstance(out, MarkPriceUpdate)
    assert out.symbol == "BTCUSDT"
    assert out.mark_price == pytest.approx(65430.10)
    assert out.index_price == pytest.approx(65420.50)
    assert out.funding_rate == pytest.approx(0.00012345)
    assert out.next_funding_ms == 1711238400000
    assert out.event_time_ms == 1711234567890
    assert out.local_recv_ms == 999


def test_mark_price_returns_none_on_wrong_event_type():
    """Dispatcher might pass a non-markPrice frame here by mistake.
    Pin: we return None rather than building a confused MarkPriceUpdate."""
    not_mark = _mp_frame(e="trade")
    out = FuturesL2Consumer._parse_mark_price(not_mark, local_recv_ms=0)
    assert out is None


def test_mark_price_returns_none_on_missing_required_fields():
    """``E``, ``s``, ``p`` are required. Missing any of them → None."""
    no_E = _mp_frame()
    no_E.pop("E")
    assert FuturesL2Consumer._parse_mark_price(no_E, local_recv_ms=0) is None

    no_s = _mp_frame()
    no_s.pop("s")
    assert FuturesL2Consumer._parse_mark_price(no_s, local_recv_ms=0) is None

    no_p = _mp_frame()
    no_p.pop("p")
    assert FuturesL2Consumer._parse_mark_price(no_p, local_recv_ms=0) is None


def test_mark_price_tolerates_missing_optional_fields():
    """Index price / next-funding-ms / funding-rate may be missing in
    edge frames. Defaults to 0.0 / 0 — caller decides if that's usable."""
    minimal = {
        "e": "markPriceUpdate",
        "E": 1711234567890,
        "s": "BTCUSDT",
        "p": "65430.10",
    }
    out = FuturesL2Consumer._parse_mark_price(minimal, local_recv_ms=0)
    assert out is not None
    assert out.mark_price == pytest.approx(65430.10)
    assert out.index_price == 0.0
    assert out.funding_rate == 0.0
    assert out.next_funding_ms == 0


def test_mark_price_negative_funding_rate_preserved():
    """Funding rate is signed: positive = longs pay shorts (bullish-cost),
    negative = shorts pay longs (bearish-cost). The sign matters for the
    paper-trader's funding-cost calc — pin it doesn't get clipped."""
    frame = _mp_frame(r="-0.00050000")
    out = FuturesL2Consumer._parse_mark_price(frame, local_recv_ms=0)
    assert out is not None
    assert out.funding_rate == pytest.approx(-0.0005)


def test_mark_price_uppercases_symbol():
    """Defensive: Binance sends upper, but if a bug ever sends lower
    we still want consistent uppercase symbols downstream so the SQL
    primary key (symbol, ts) doesn't fragment."""
    frame = _mp_frame(s="btcusdt")
    out = FuturesL2Consumer._parse_mark_price(frame, local_recv_ms=0)
    assert out is not None
    assert out.symbol == "BTCUSDT"


# ─── L2 + trade parsers (sanity — they're shared with spot) ───


def test_snapshot_parser_uses_stream_for_symbol():
    """The futures @depth20 frame doesn't have an "s" field for the
    partial-book stream — symbol comes from the stream name. Same as
    spot. Pin so a refactor that drops stream-arg breaks loudly."""
    data = {
        "lastUpdateId": 1234,
        "bids": [["65430.10", "1.5"], ["65430.00", "2.0"]],
        "asks": [["65430.20", "1.0"], ["65430.30", "0.5"]],
    }
    snap = FuturesL2Consumer._parse_snapshot(
        data, stream="btcusdt@depth20@100ms", local_recv_ms=999,
    )
    assert snap is not None
    assert snap.symbol == "BTCUSDT"
    assert snap.bids[0] == (65430.10, 1.5)
    assert snap.asks[-1] == (65430.30, 0.5)


def test_snapshot_parser_handles_futures_short_form_b_a():
    """USDM Futures @depth20 uses single-letter ``b``/``a`` keys.
    Spot uses ``bids``/``asks``. Pin: parser accepts the futures form,
    otherwise we silently drop ALL depth frames and only ingest trades.
    This was caught by the first live deploy on Tokyo (snaps=0
    despite frames=9000+)."""
    data = {
        "e": "depthUpdate",
        "E": 1711234567890,
        "T": 1711234567880,
        "s": "BTCUSDT",
        "U": 1234,
        "u": 1240,
        "pu": 1233,
        "b": [["65430.10", "1.5"], ["65430.00", "2.0"]],
        "a": [["65430.20", "1.0"], ["65430.30", "0.5"]],
    }
    snap = FuturesL2Consumer._parse_snapshot(
        data, stream="btcusdt@depth20@100ms", local_recv_ms=999,
    )
    assert snap is not None
    assert snap.bids[0] == (65430.10, 1.5)
    assert snap.asks[-1] == (65430.30, 0.5)


def test_snapshot_parser_returns_none_when_no_bids_or_asks():
    """Frame with neither ``bids``/``asks`` nor ``b``/``a`` returns
    None — defensive against malformed envelopes."""
    data = {"lastUpdateId": 1, "noise": "data"}
    snap = FuturesL2Consumer._parse_snapshot(
        data, stream="btcusdt@depth20@100ms", local_recv_ms=0,
    )
    assert snap is None


def test_trade_parser_basic():
    data = {
        "e": "trade",
        "E": 1711234567890,
        "s": "BTCUSDT",
        "p": "65430.10",
        "q": "0.001",
        "m": True,  # buyer is maker
    }
    out = FuturesL2Consumer._parse_trade(data, local_recv_ms=42)
    assert out is not None
    assert out.symbol == "BTCUSDT"
    assert out.is_buyer_maker is True


# ─── Dispatch routing ───


@pytest.mark.asyncio
async def test_dispatch_routes_mark_price_to_callback():
    """Mark-price frames must hit on_mark_price, NOT on_snapshot."""
    received_mp: list[MarkPriceUpdate] = []
    received_snap = []

    async def on_mp(m): received_mp.append(m)
    async def on_snap(s): received_snap.append(s)

    c = FuturesL2Consumer(
        ["BTCUSDT"], on_mark_price=on_mp, on_snapshot=on_snap,
    )
    raw = (
        '{"stream":"btcusdt@markPrice@1s","data":'
        '{"e":"markPriceUpdate","E":1711234567890,"s":"BTCUSDT",'
        '"p":"65430.10","i":"65420.50","P":"65425.00",'
        '"r":"0.0001","T":1711238400000}}'
    )
    await c._dispatch(raw)
    assert len(received_mp) == 1
    assert len(received_snap) == 0
    assert received_mp[0].funding_rate == pytest.approx(0.0001)


@pytest.mark.asyncio
async def test_dispatch_routes_depth_to_snapshot_callback():
    received_snap = []
    received_mp: list[MarkPriceUpdate] = []

    async def on_snap(s): received_snap.append(s)
    async def on_mp(m): received_mp.append(m)

    c = FuturesL2Consumer(
        ["BTCUSDT"], on_snapshot=on_snap, on_mark_price=on_mp,
    )
    raw = (
        '{"stream":"btcusdt@depth20@100ms","data":'
        '{"lastUpdateId":1234,"bids":[["65430","1"]],"asks":[["65431","2"]]}}'
    )
    await c._dispatch(raw)
    assert len(received_snap) == 1
    assert len(received_mp) == 0


@pytest.mark.asyncio
async def test_dispatch_handles_unknown_stream_silently():
    """Binance occasionally adds new envelope fields. Parser must skip
    rather than raise — pin: no exception, no callback fired."""
    received_anything = []

    async def cb(*args): received_anything.append(args)

    c = FuturesL2Consumer(
        ["BTCUSDT"], on_snapshot=cb, on_trade=cb, on_mark_price=cb,
    )
    raw = '{"stream":"btcusdt@futureFeature","data":{"foo":"bar"}}'
    # Should not raise.
    await c._dispatch(raw)
    assert received_anything == []


# ─── Constructor validation ───


def test_constructor_rejects_empty_symbols():
    with pytest.raises(ValueError):
        FuturesL2Consumer([])


def test_constructor_rejects_invalid_depth():
    with pytest.raises(ValueError):
        FuturesL2Consumer(["BTCUSDT"], depth_levels=15)


def test_constructor_rejects_invalid_speed():
    with pytest.raises(ValueError):
        FuturesL2Consumer(["BTCUSDT"], update_speed_ms=500)


def test_constructor_uppercases_symbols():
    c = FuturesL2Consumer(["btcusdt", "ETHUSDT", "bnbusdt"])
    assert c.symbols == ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


# ─── stop() behaviour ───


@pytest.mark.asyncio
async def test_stop_signals_run_loop_to_exit():
    """stop() sets the internal event so run_forever returns at the
    next iteration. Pin: idempotent — calling twice doesn't raise."""
    c = FuturesL2Consumer(["BTCUSDT"])
    c.stop()
    c.stop()  # idempotent
    assert c._stop.is_set()


# ─── Counter / metrics interface ───


def test_counters_initialised_to_zero():
    c = FuturesL2Consumer(["BTCUSDT"])
    assert c.frames_received == 0
    assert c.snapshots_dispatched == 0
    assert c.trades_dispatched == 0
    assert c.mark_price_dispatched == 0
    assert c.reconnect_count == 0


@pytest.mark.asyncio
async def test_dispatch_increments_mark_price_counter():
    """The counter must tick on each successful mark-price dispatch
    so health endpoints can detect "no funding updates in N seconds"
    drift conditions."""
    c = FuturesL2Consumer(["BTCUSDT"])
    raw = (
        '{"stream":"btcusdt@markPrice@1s","data":'
        '{"e":"markPriceUpdate","E":1711234567890,"s":"BTCUSDT",'
        '"p":"65430.10","r":"0.0001","T":1711238400000}}'
    )
    await c._dispatch(raw)
    await c._dispatch(raw)
    assert c.mark_price_dispatched == 2
