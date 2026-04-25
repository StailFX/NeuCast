"""Binance Spot Level-2 WebSocket consumer.

Subscribes to ``<symbol>@depth20@100ms`` (top-20 order book snapshot every
100 ms) and ``<symbol>@trade`` (every executed trade) for one or more
symbols, holds the latest book + a short rolling buffer of trades in memory,
and forwards aggregated 1-second OFI / microprice / depth-imbalance rows
to :mod:`app.highfreq.aggregator`.

Design notes
------------

* **Event time, not local time.** Each WebSocket frame includes an ``E``
  field set by the Binance matching engine. We persist that (see ADR-002 in
  the architecture doc). Local-receive time is recorded as a diagnostic
  column only, never fed to the model.
* **In-memory only.** Raw L2 snapshots are *not* written to Postgres — only
  the 1-second OFI aggregates are persisted. See ADR-001.
* **Resilient to disconnects.** Reconnect with exponential backoff up to
  60 s; on each reconnect we re-snapshot the order book via REST so OFI
  computation does not depend on a long-lived stream.

The consumer is intended to run as its own asyncio task inside a dedicated
Docker container with ``mem_limit: 1.5g``; see ADR-006 for resource budget.
"""
from __future__ import annotations

# NOTE: This is a Phase-A.0 skeleton. Implementation lands in Phase A.2 —
# see docs/highfreq/architecture.md §7 for the roadmap.

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class L2Snapshot:
    """Top-N order book at a single event time."""

    event_time_ms: int        # Binance ``E`` field
    local_recv_ms: int        # diagnostic only
    symbol: str
    bids: tuple[tuple[float, float], ...]   # (price, qty), sorted desc
    asks: tuple[tuple[float, float], ...]   # (price, qty), sorted asc


@dataclass(frozen=True)
class Trade:
    """Single executed trade."""

    event_time_ms: int
    local_recv_ms: int
    symbol: str
    price: float
    qty: float
    is_buyer_maker: bool      # True → aggressive sell; False → aggressive buy


class L2Consumer:
    """Asyncio WebSocket consumer.

    Usage::

        consumer = L2Consumer(symbols=["BTCUSDT"], on_snapshot=cb_l2, on_trade=cb_t)
        await consumer.run_forever()
    """

    def __init__(self, symbols: list[str], *, on_snapshot=None, on_trade=None):
        self.symbols = symbols
        self.on_snapshot = on_snapshot
        self.on_trade = on_trade

    async def run_forever(self) -> None:
        """Main loop — connects, listens, reconnects on failure.

        Implementation TODO (Phase A.2):

        * connect to ``wss://stream.binance.com:9443/stream?streams=…``
        * for each frame, dispatch to ``_handle_depth_update`` or
          ``_handle_trade``
        * exponential-backoff reconnect with jitter
        * Prometheus counter for ``frames_received_total``
        """
        raise NotImplementedError("Phase A.2 — see architecture doc §7")
