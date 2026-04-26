"""Tests for ``app.highfreq.order_executor`` — pluggable execution
interface.

Three things to pin:

1. **Protocol contract**: every implementation must satisfy
   ``isinstance(x, OrderExecutor)`` (runtime-checkable Protocol). Pin
   so a future "let's pass a dict" refactor breaks loudly.
2. **Sim implementation matches current behaviour**: ``SimulatedExecutor``
   produces the same fill price + fee as the inline code in
   ``PaperTrader._open_position`` — ensures swapping it in later doesn't
   change paper-trade outputs.
3. **Live executor refuses to construct without explicit confirmation
   env**: belt-and-suspenders against a typo turning sim into mainnet.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from app.highfreq.order_executor import (
    BinanceLiveExecutor,
    BinanceTestnetExecutor,
    ExecutedFill,
    OrderExecutor,
    SimulatedExecutor,
    make_executor,
)


# ───── Protocol ─────


def test_simulated_executor_satisfies_protocol():
    assert isinstance(SimulatedExecutor(), OrderExecutor)


def test_testnet_executor_satisfies_protocol():
    """Even though it's a stub, it must be structurally an executor —
    so the runner can hold a reference to it without type errors."""
    e = BinanceTestnetExecutor(api_key="x", api_secret="y")
    assert isinstance(e, OrderExecutor)


# ───── SimulatedExecutor: byte-identical to current trader math ─────


@pytest.mark.asyncio
async def test_simulated_open_long_returns_fill_at_microprice():
    """Sim long: fill price == bar microprice, fee from configured bps."""
    e = SimulatedExecutor(maker_fee_bps_per_side=7.5)
    ts = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    fill = await e.open_long(symbol="BTCUSDT", qty=0.001, ts=ts, microprice=77_000.0)

    assert isinstance(fill, ExecutedFill)
    assert fill.fill_price == 77_000.0
    assert fill.fill_ts == ts
    assert fill.is_sim is True
    assert fill.venue_order_id is None
    # Fee = 7.5 bp * notional (qty * price) = 7.5e-4 * 77 = 0.0578.
    assert fill.fee_usd == pytest.approx(77.0 * 7.5e-4, rel=1e-6)


@pytest.mark.asyncio
async def test_simulated_open_short_returns_fill_at_microprice():
    """Sim short uses the same microprice — reflects the existing
    sim contract (no bid/ask spread modelled)."""
    e = SimulatedExecutor()
    ts = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    fill = await e.open_short(symbol="BTCUSDT", qty=0.002, ts=ts, microprice=77_500.0)
    assert fill.fill_price == 77_500.0
    assert fill.is_sim is True


@pytest.mark.asyncio
async def test_simulated_close_uses_microprice_regardless_of_side():
    """Close at the bar's microprice — symmetric to open. The "side"
    param is reserved for live executors that need to know whether
    to send buy-back or sell-back."""
    e = SimulatedExecutor()
    ts = datetime(2026, 4, 27, 12, 1, tzinfo=timezone.utc)
    fill_long = await e.close_position(
        symbol="BTCUSDT", qty=0.001, side="long",
        ts=ts, microprice=77_100.0, venue_order_id=None,
    )
    fill_short = await e.close_position(
        symbol="BTCUSDT", qty=0.001, side="short",
        ts=ts, microprice=77_100.0, venue_order_id=None,
    )
    assert fill_long.fill_price == fill_short.fill_price == 77_100.0
    assert fill_long.is_sim and fill_short.is_sim


@pytest.mark.asyncio
async def test_simulated_fee_matches_paper_trader_fee_function():
    """Cross-check: the fee SimulatedExecutor charges for one side
    must equal ``app.highfreq.paper_trader.fee_per_side`` for the
    same inputs. If those drift, sim trades from the executor differ
    from sim trades the trader's inline path produces — silent break."""
    from app.highfreq.paper_trader import fee_per_side

    e = SimulatedExecutor(maker_fee_bps_per_side=7.5)
    ts = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
    fill = await e.open_long(symbol="BTCUSDT", qty=0.5, ts=ts, microprice=77_000.0)
    assert fill.fee_usd == pytest.approx(fee_per_side(0.5, 77_000.0, 7.5), rel=1e-9)


# ───── Stub executors raise NotImplementedError loudly ─────


@pytest.mark.asyncio
async def test_testnet_executor_raises_not_implemented():
    """Stub: pointing the runner at testnet before it's wired in must
    fail fast, not silently no-op."""
    e = BinanceTestnetExecutor(api_key="x", api_secret="y")
    ts = datetime(2026, 4, 27, tzinfo=timezone.utc)

    with pytest.raises(NotImplementedError):
        await e.open_long(symbol="BTCUSDT", qty=1.0, ts=ts, microprice=77_000.0)
    with pytest.raises(NotImplementedError):
        await e.close_position(
            symbol="BTCUSDT", qty=1.0, side="long",
            ts=ts, microprice=77_000.0, venue_order_id=None,
        )


# ───── Live executor: refuses to construct without env confirmation ─────


def test_live_executor_refuses_construction_without_env(monkeypatch):
    """Belt-and-suspenders. A typo in ``HF_EXECUTOR_KIND`` ("live"
    instead of "testnet") would otherwise be the most expensive bug
    imaginable — without this guard, the runner would happily
    construct the live executor and route real-money orders."""
    monkeypatch.delenv("HF_LIVE_TRADING_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="HF_LIVE_TRADING_CONFIRMED"):
        BinanceLiveExecutor(api_key="x", api_secret="y")


def test_live_executor_constructs_with_explicit_env(monkeypatch):
    """When the env IS set, construction succeeds — confirms the
    guard is not over-broad. Doesn't actually call any methods
    (those are stubs and would raise)."""
    monkeypatch.setenv("HF_LIVE_TRADING_CONFIRMED", "1")
    e = BinanceLiveExecutor(api_key="x", api_secret="y")
    assert isinstance(e, OrderExecutor)


# ───── Factory ─────


def test_make_executor_default_is_sim(monkeypatch):
    monkeypatch.delenv("HF_EXECUTOR_KIND", raising=False)
    e = make_executor()
    assert isinstance(e, SimulatedExecutor)


def test_make_executor_kind_sim(monkeypatch):
    monkeypatch.setenv("HF_EXECUTOR_KIND", "sim")
    assert isinstance(make_executor(), SimulatedExecutor)


def test_make_executor_unrecognised_kind_falls_back_to_sim(monkeypatch):
    """A typo in HF_EXECUTOR_KIND falls back to sim with a warning —
    NOT to live. If we ever flip this to "fail loudly on typo", that's
    fine, but the safer default is sim."""
    monkeypatch.setenv("HF_EXECUTOR_KIND", "definitely-not-a-kind")
    e = make_executor()
    assert isinstance(e, SimulatedExecutor)


def test_make_executor_testnet_requires_creds(monkeypatch):
    monkeypatch.setenv("HF_EXECUTOR_KIND", "testnet")
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="BINANCE_TESTNET_API_KEY"):
        make_executor()


def test_make_executor_live_requires_confirmation_and_creds(monkeypatch):
    monkeypatch.setenv("HF_EXECUTOR_KIND", "live")
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.delenv("HF_LIVE_TRADING_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="HF_LIVE_TRADING_CONFIRMED"):
        make_executor()


def test_make_executor_live_works_with_full_setup(monkeypatch):
    monkeypatch.setenv("HF_EXECUTOR_KIND", "live")
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HF_LIVE_TRADING_CONFIRMED", "1")
    e = make_executor()
    assert isinstance(e, BinanceLiveExecutor)
