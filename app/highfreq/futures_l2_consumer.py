"""Binance USDM Perpetual Futures Level-2 + mark-price WebSocket consumer.

Parallel to :mod:`app.highfreq.l2_consumer` for the spot venue (ADR-019).
Subscribes to three streams per symbol:

* ``<symbol>@depth20@100ms`` — partial book snapshot (top-20 levels, 100 ms cadence).
  Same shape as spot — we reuse :class:`L2Snapshot` from the spot consumer.
* ``<symbol>@trade`` — every executed trade. Same shape — reuses :class:`Trade`.
* ``<symbol>@markPrice@1s`` — mark price + funding rate update every 1 s.
  Futures-specific; produces :class:`MarkPriceUpdate` frames.

Why a separate consumer (not a venue flag on the spot one)
-----------------------------------------------------------

The spot consumer has been running in production for weeks feeding the
existing paper traders. A "venue-aware" refactor that flips a URL based
on a constructor flag is a high-risk change with no upside for the spot
side. The futures consumer is a true clone — identical resilience /
backoff / reconnect logic, different URL + one extra stream type.

If futures research dead-ends (insufficient liquidity, funding-rate
volatility eats edge, etc.) we just delete this file and the futures
table — the spot side never knew it existed. See ADR-019.

Mark price + funding rate
--------------------------

USDM contracts pay funding every 8h. The @markPrice@1s stream fires
once per second with::

    {
        "e": "markPriceUpdate",
        "E": 1577653440000,      # event time
        "s": "BTCUSDT",          # symbol
        "p": "11794.15",         # mark price
        "i": "11784.62",         # index price
        "P": "11784.25",         # estimated settle price (only useful 1h before settlement)
        "r": "0.00038167",       # funding rate (fraction-of-1, e.g. 0.0001 = 1bp/8h)
        "T": 1577656800000       # next funding settlement time (ms)
    }

We persist ``r`` as ``funding_rate``, ``T`` as ``next_funding_ms``,
and ``p`` as ``mark_price``. The downstream aggregator interpolates
the funding rate across 1-second bars when @markPrice@1s lags slightly
(updates aren't atomic with depth).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

import websockets
from websockets.exceptions import ConnectionClosed

# Reuse spot's L2Snapshot + Trade — the message shape is identical
# between spot and futures @depth + @trade streams. The MarkPriceUpdate
# dataclass below is futures-specific.
from app.highfreq.l2_consumer import L2Snapshot, Trade

logger = logging.getLogger(__name__)

# USDM Futures uses ``fstream.binance.com``, distinct from the spot
# host. Combined-stream endpoint takes the same ``?streams=`` query
# format as spot.
BINANCE_FUTURES_WSS_BASE = "wss://fstream.binance.com/stream"


@dataclass(frozen=True)
class MarkPriceUpdate:
    """One frame of the @markPrice@1s stream — mark + funding rate.

    Funding rate is stored as a fraction-of-1 (Binance native form):
    ``0.0001`` = 1 bp per 8 h. To convert to bps, multiply by 1e4.
    """

    event_time_ms: int
    local_recv_ms: int
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_ms: int


SnapshotCB = Callable[[L2Snapshot], Awaitable[None]]
TradeCB = Callable[[Trade], Awaitable[None]]
MarkPriceCB = Callable[[MarkPriceUpdate], Awaitable[None]]


class FuturesL2Consumer:
    """Asyncio multiplexed consumer for Binance USDM Futures.

    Same lifecycle / reconnect contract as the spot :class:`L2Consumer`:

    * ``run_forever()`` blocks until ``stop()`` (or task cancellation).
    * Reconnects with exponential backoff (1 s → 60 s, jittered) on any
      ``ConnectionClosed`` or transient error.
    * Frame counters surfaced for health endpoints.

    Parameters
    ----------
    symbols
        Iterable of upper-case Binance symbols, e.g. ``["BTCUSDT"]``.
        We use the spot symbol form (no ``.P`` suffix) since that's the
        column convention in :sql:`highfreq_futures_ofi_1s`.
    on_snapshot, on_trade, on_mark_price
        Async callbacks. All optional. Mark-price callback is the new
        one; the other two mirror :class:`L2Consumer`.
    depth_levels
        5 / 10 / 20 (Binance allowed values). Default 20.
    update_speed_ms
        100 / 1000 (Binance). Default 100.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        on_snapshot: SnapshotCB | None = None,
        on_trade: TradeCB | None = None,
        on_mark_price: MarkPriceCB | None = None,
        depth_levels: int = 20,
        update_speed_ms: int = 100,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        if depth_levels not in (5, 10, 20):
            raise ValueError("depth_levels must be 5, 10, or 20")
        if update_speed_ms not in (100, 1000):
            raise ValueError("update_speed_ms must be 100 or 1000")
        self.on_snapshot = on_snapshot
        self.on_trade = on_trade
        self.on_mark_price = on_mark_price
        self.depth_levels = depth_levels
        self.update_speed_ms = update_speed_ms
        self._stop = asyncio.Event()
        # Counters surfaced by health endpoints / Prometheus.
        self.frames_received = 0
        self.snapshots_dispatched = 0
        self.trades_dispatched = 0
        self.mark_price_dispatched = 0
        self.last_event_time_ms: dict[str, int] = {}
        self.reconnect_count = 0

    # ──────────────────────────────────────────────────────────────────
    # Public API (mirrors L2Consumer for substitutability in callers)
    # ──────────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        backoff_s = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_and_consume()
                backoff_s = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — top-level retry loop
                self.reconnect_count += 1
                jitter = random.uniform(0.5, 1.5)
                wait_s = min(60.0, backoff_s * jitter)
                logger.warning(
                    "FuturesL2Consumer disconnected (%s: %s); "
                    "reconnecting in %.1fs (attempt %d)",
                    type(e).__name__, e, wait_s, self.reconnect_count,
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=wait_s)
                except asyncio.TimeoutError:
                    pass
                backoff_s = min(60.0, backoff_s * 2.0)

    def stop(self) -> None:
        self._stop.set()

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _build_url(self) -> str:
        streams: list[str] = []
        for sym in self.symbols:
            lo = sym.lower()
            streams.append(f"{lo}@depth{self.depth_levels}@{self.update_speed_ms}ms")
            streams.append(f"{lo}@trade")
            streams.append(f"{lo}@markPrice@1s")
        return f"{BINANCE_FUTURES_WSS_BASE}?streams={'/'.join(streams)}"

    async def _connect_and_consume(self) -> None:
        url = self._build_url()
        logger.info("FuturesL2Consumer connecting: %s", url)
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=2 ** 20,
        ) as ws:
            logger.info(
                "FuturesL2Consumer connected, %d streams (3 per symbol × %d symbols)",
                len(self.symbols) * 3, len(self.symbols),
            )
            async for raw in ws:
                if self._stop.is_set():
                    break
                self.frames_received += 1
                try:
                    await self._dispatch(raw)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "FuturesL2Consumer dispatch failed for frame: %r",
                        raw[:200],
                    )

    async def _dispatch(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        stream = msg.get("stream", "")
        data = msg.get("data") or msg
        local_recv_ms = int(time.time() * 1000)

        # Mark-price first (cheapest discriminator: distinct event field).
        if "@markPrice" in stream or data.get("e") == "markPriceUpdate":
            mp = self._parse_mark_price(data, local_recv_ms)
            if mp is not None:
                self.mark_price_dispatched += 1
                self.last_event_time_ms[mp.symbol] = mp.event_time_ms
                if self.on_mark_price is not None:
                    await self.on_mark_price(mp)
            return

        if "@depth" in stream or ("asks" in data and "bids" in data):
            snap = self._parse_snapshot(data, stream, local_recv_ms)
            if snap is not None:
                self.snapshots_dispatched += 1
                self.last_event_time_ms[snap.symbol] = snap.event_time_ms
                if self.on_snapshot is not None:
                    await self.on_snapshot(snap)
            return

        if "@trade" in stream or data.get("e") == "trade":
            trade = self._parse_trade(data, local_recv_ms)
            if trade is not None:
                self.trades_dispatched += 1
                self.last_event_time_ms[trade.symbol] = trade.event_time_ms
                if self.on_trade is not None:
                    await self.on_trade(trade)
            return

        # Unknown stream type: ignore (Binance adds envelope fields
        # over time; ignoring is safer than crashing).

    # ──────────────────────────────────────────────────────────────────
    # Parsers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_snapshot(
        data: dict, stream: str, local_recv_ms: int,
    ) -> L2Snapshot | None:
        # Same shape as spot — futures @depth20 returns identical fields.
        event_time_ms = int(data.get("E", local_recv_ms))
        if "@" in stream:
            symbol = stream.split("@", 1)[0].upper()
        else:
            symbol = data.get("s", "").upper()
        if not symbol:
            return None
        try:
            bids = tuple((float(p), float(q)) for p, q in data.get("bids", []))
            asks = tuple((float(p), float(q)) for p, q in data.get("asks", []))
        except (TypeError, ValueError):
            return None
        if not bids or not asks:
            return None
        return L2Snapshot(
            event_time_ms=event_time_ms,
            local_recv_ms=local_recv_ms,
            symbol=symbol,
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def _parse_trade(data: dict, local_recv_ms: int) -> Trade | None:
        try:
            return Trade(
                event_time_ms=int(data["E"]),
                local_recv_ms=local_recv_ms,
                symbol=str(data["s"]).upper(),
                price=float(data["p"]),
                qty=float(data["q"]),
                is_buyer_maker=bool(data.get("m", False)),
            )
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def _parse_mark_price(
        data: dict, local_recv_ms: int,
    ) -> MarkPriceUpdate | None:
        """Parse the futures-specific @markPrice@1s frame.

        Robust to:
        * Missing optional fields (i / P / T): treated as None / 0.
        * String numerics (Binance always sends strings) — coerce to float.
        * Wrong event type: returns None.
        """
        if data.get("e") != "markPriceUpdate":
            return None
        try:
            return MarkPriceUpdate(
                event_time_ms=int(data["E"]),
                local_recv_ms=local_recv_ms,
                symbol=str(data["s"]).upper(),
                mark_price=float(data["p"]),
                index_price=float(data.get("i", 0.0)),
                funding_rate=float(data.get("r", 0.0)),
                next_funding_ms=int(data.get("T", 0)),
            )
        except (KeyError, ValueError, TypeError):
            return None
