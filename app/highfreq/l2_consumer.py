"""Binance Spot Level-2 WebSocket consumer.

Subscribes to ``<symbol>@depth20@100ms`` (top-20 partial-book snapshot
every 100 ms) and ``<symbol>@trade`` (every executed trade) for one or
more symbols. Each incoming frame is parsed into an :class:`L2Snapshot`
or :class:`Trade` dataclass and dispatched to the user-supplied callbacks.

Why partial-book stream and not the diff stream
-----------------------------------------------

Binance offers two depth feeds:

* ``@depth`` — diff stream (every order-book delta). Requires REST
  snapshot to seed and careful sequence-number tracking.
* ``@depth20@100ms`` — partial book; full top-20 levels every 100 ms.
  No seeding required, no sequence tracking; each frame replaces the
  previous top-of-book entirely.

We use the partial book. For 1-second OFI aggregation the 100 ms cadence
is more than enough resolution (~10 frames/second/symbol), and it
eliminates an entire class of seeding bugs. See ADR-001 in
``docs/highfreq/architecture.md``.

Why event time, not local time
------------------------------

Each frame includes ``E`` (Binance matching-engine event time, UTC ms).
We persist ``E`` as the canonical timestamp and only record
local-receive time as a diagnostic — see ADR-002.

Resilience
----------

The consumer reconnects on any ``ConnectionClosed`` with exponential
backoff (1 s → 60 s, jittered). Reconnects are logged at WARNING level;
sustained reconnect loops trigger a structured ERROR log every minute
so operators see them in journalctl/Loki.
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

logger = logging.getLogger(__name__)

BINANCE_WSS_BASE = "wss://stream.binance.com:9443/stream"


@dataclass(frozen=True)
class L2Snapshot:
    """Top-N order book at a single event time."""

    event_time_ms: int
    local_recv_ms: int
    symbol: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Trade:
    """Single executed trade (Binance @trade stream)."""

    event_time_ms: int
    local_recv_ms: int
    symbol: str
    price: float
    qty: float
    is_buyer_maker: bool


SnapshotCB = Callable[[L2Snapshot], Awaitable[None]]
TradeCB = Callable[[Trade], Awaitable[None]]


class L2Consumer:
    """Asyncio multiplexed WebSocket consumer for Binance Spot L2 + trades.

    Parameters
    ----------
    symbols
        Iterable of upper-case Binance symbols, e.g. ``["BTCUSDT"]``.
    on_snapshot
        Async callback invoked once per L2 frame.
    on_trade
        Async callback invoked once per trade frame.
    depth_levels
        Number of partial-book levels to subscribe to. Binance allows
        5, 10, or 20. Default 20.
    update_speed_ms
        Stream cadence in milliseconds. Binance allows 100 or 1000.
        Default 100.

    Notes
    -----
    Callbacks must be cheap (microseconds, not milliseconds) — they run
    on the same event loop. Heavy work belongs in :mod:`app.highfreq.aggregator`.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        on_snapshot: SnapshotCB | None = None,
        on_trade: TradeCB | None = None,
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
        self.depth_levels = depth_levels
        self.update_speed_ms = update_speed_ms
        self._stop = asyncio.Event()
        # Counters surfaced by health endpoints.
        self.frames_received = 0
        self.snapshots_dispatched = 0
        self.trades_dispatched = 0
        self.last_event_time_ms: dict[str, int] = {}
        self.reconnect_count = 0

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Run until ``stop()`` is called or the task is cancelled."""
        backoff_s = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_and_consume()
                # Clean exit (server closed) — reset backoff.
                backoff_s = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — top-level retry loop
                self.reconnect_count += 1
                jitter = random.uniform(0.5, 1.5)
                wait_s = min(60.0, backoff_s * jitter)
                logger.warning(
                    "L2Consumer disconnected (%s: %s); reconnecting in %.1fs (attempt %d)",
                    type(e).__name__, e, wait_s, self.reconnect_count,
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=wait_s)
                except asyncio.TimeoutError:
                    pass
                backoff_s = min(60.0, backoff_s * 2.0)

    def stop(self) -> None:
        """Signal the consumer to exit at the next safe point."""
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
        return f"{BINANCE_WSS_BASE}?streams={'/'.join(streams)}"

    async def _connect_and_consume(self) -> None:
        url = self._build_url()
        logger.info("L2Consumer connecting: %s", url)
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=2 ** 20,
        ) as ws:
            logger.info("L2Consumer connected, %d streams", len(self.symbols) * 2)
            async for raw in ws:
                if self._stop.is_set():
                    break
                self.frames_received += 1
                try:
                    await self._dispatch(raw)
                except Exception:  # noqa: BLE001 — never kill the stream loop
                    logger.exception("L2Consumer dispatch failed for frame: %r", raw[:200])

    async def _dispatch(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        # Combined-stream envelope: {"stream": "...", "data": {...}}
        stream = msg.get("stream", "")
        data = msg.get("data") or msg
        local_recv_ms = int(time.time() * 1000)

        if "@depth" in stream or "asks" in data and "bids" in data:
            snap = self._parse_snapshot(data, stream, local_recv_ms)
            if snap is not None:
                self.snapshots_dispatched += 1
                self.last_event_time_ms[snap.symbol] = snap.event_time_ms
                if self.on_snapshot is not None:
                    await self.on_snapshot(snap)
        elif "@trade" in stream or data.get("e") == "trade":
            trade = self._parse_trade(data, local_recv_ms)
            if trade is not None:
                self.trades_dispatched += 1
                self.last_event_time_ms[trade.symbol] = trade.event_time_ms
                if self.on_trade is not None:
                    await self.on_trade(trade)
        # Unknown stream types are silently ignored — Binance occasionally
        # adds new envelope fields, and we don't want to break on them.

    @staticmethod
    def _parse_snapshot(
        data: dict, stream: str, local_recv_ms: int
    ) -> L2Snapshot | None:
        # Partial-book stream uses "lastUpdateId" but no "E" field — use
        # local_recv_ms as a stand-in only if there's no event time.
        event_time_ms = int(data.get("E", local_recv_ms))
        # Symbol is in the stream name for partial-book frames.
        # Stream looks like "btcusdt@depth20@100ms".
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
