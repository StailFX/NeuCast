"""1-second and 1-minute feature aggregation, Postgres writer.

Receives :class:`~app.highfreq.l2_consumer.L2Snapshot` and
:class:`~app.highfreq.l2_consumer.Trade` events from the consumer,
maintains a rolling window of per-frame features, emits one
:class:`AggregatedRow` per second containing OFI mean / sum / std /
microprice / depth-imbalance / spread / trade imbalance.

Persistence is async via ``asyncpg`` — see ADR-001 for why we don't use
TimescaleDB. The aggregator owns its own database connection pool and
writes are batched (default 5 rows ≈ 5 s flush cadence) to limit
transaction overhead while keeping data freshness within an acceptable
operator-visibility window.

Schema reference: ``docs/highfreq/architecture.md`` §6.
"""
from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

import asyncpg

from app.highfreq.l2_consumer import L2Snapshot, Trade
from app.highfreq.ofi_features import FrameFeatures, features_from_snapshot

logger = logging.getLogger(__name__)


@dataclass
class _SecondBucket:
    """In-flight aggregation state for one (symbol, second) pair."""

    symbol: str
    second_ms: int                                 # floor(event_time_ms / 1000) * 1000
    ofi_values: list[float] = field(default_factory=list)
    microprice_values: list[float] = field(default_factory=list)
    depth_imb_values: list[float] = field(default_factory=list)
    spread_bps_values: list[float] = field(default_factory=list)
    trade_imb: float = 0.0                         # signed sum of trade volume
    n_updates: int = 0

    def add_frame(self, feat: FrameFeatures) -> None:
        self.ofi_values.append(feat.ofi)
        self.microprice_values.append(feat.microprice)
        self.depth_imb_values.append(feat.depth_imb)
        self.spread_bps_values.append(feat.spread_bps)
        self.n_updates += 1

    def add_trade(self, t: Trade) -> None:
        # is_buyer_maker = True → aggressive sell hit a passive bid → sign = -qty
        # is_buyer_maker = False → aggressive buy lifted a passive ask → sign = +qty
        sign = -1.0 if t.is_buyer_maker else 1.0
        self.trade_imb += sign * t.qty


@dataclass(frozen=True)
class AggregatedRow:
    """One 1-second aggregated row, ready for Postgres."""

    second_ms: int
    symbol: str
    ofi: float
    microprice: float
    depth_imb: float
    spread_bps: float
    trade_imb: float
    vpin: float
    n_updates: int
    local_recv_ms_jitter: int       # max - min of local_recv_ms in window


class Aggregator:
    """Owns per-symbol state, emits 1-s rows, writes to Postgres.

    Lifecycle::

        agg = Aggregator(database_url="postgresql://...", symbols=["BTCUSDT"])
        await agg.start()
        try:
            consumer = L2Consumer(...,
                on_snapshot=agg.on_snapshot, on_trade=agg.on_trade)
            await consumer.run_forever()
        finally:
            await agg.flush()
            await agg.close()
    """

    def __init__(
        self,
        database_url: str,
        symbols: Iterable[str],
        *,
        depth_levels: int = 10,
        flush_batch_size: int = 5,
    ) -> None:
        self.database_url = database_url
        self.symbols = [s.upper() for s in symbols]
        self.depth_levels = depth_levels
        self.flush_batch_size = flush_batch_size
        # State.
        self._pool: asyncpg.Pool | None = None
        self._prev_snap: dict[str, L2Snapshot | None] = {s: None for s in self.symbols}
        self._buckets: dict[tuple[str, int], _SecondBucket] = {}
        self._jitter: dict[tuple[str, int], tuple[int, int]] = {}  # (min_ms, max_ms)
        self._pending_writes: deque[AggregatedRow] = deque()
        self._lock = asyncio.Lock()
        # Counters.
        self.rows_emitted = 0
        self.rows_written = 0

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._pool = await asyncpg.create_pool(
            self.database_url, min_size=1, max_size=2, command_timeout=10,
        )
        logger.info("Aggregator connected to Postgres (pool 1-2)")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def flush(self) -> None:
        """Force-write any pending rows (called on shutdown)."""
        async with self._lock:
            await self._flush_locked()

    # ──────────────────────────────────────────────────────────────────
    # Callbacks (wired into L2Consumer)
    # ──────────────────────────────────────────────────────────────────

    async def on_snapshot(self, snap: L2Snapshot) -> None:
        prev = self._prev_snap.get(snap.symbol)
        feat = features_from_snapshot(snap, prev, depth_levels=self.depth_levels)
        self._prev_snap[snap.symbol] = snap
        await self._add_to_bucket(snap.symbol, snap.event_time_ms, snap.local_recv_ms, feat)

    async def on_trade(self, t: Trade) -> None:
        # Trades go into the same 1-s bucket as snapshots based on event time.
        second_ms = (t.event_time_ms // 1000) * 1000
        key = (t.symbol, second_ms)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _SecondBucket(symbol=t.symbol, second_ms=second_ms)
            self._buckets[key] = bucket
        bucket.add_trade(t)

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    async def _add_to_bucket(
        self, symbol: str, event_time_ms: int, local_recv_ms: int, feat: FrameFeatures
    ) -> None:
        second_ms = (event_time_ms // 1000) * 1000
        key = (symbol, second_ms)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _SecondBucket(symbol=symbol, second_ms=second_ms)
            self._buckets[key] = bucket
        bucket.add_frame(feat)
        # Track local-recv jitter for diagnostics.
        cur = self._jitter.get(key)
        if cur is None:
            self._jitter[key] = (local_recv_ms, local_recv_ms)
        else:
            self._jitter[key] = (min(cur[0], local_recv_ms), max(cur[1], local_recv_ms))

        # Emit any complete buckets (any older than this one).
        await self._emit_completed(symbol, second_ms)

    async def _emit_completed(self, symbol: str, current_second_ms: int) -> None:
        # Find buckets strictly older than current second — they're complete.
        completed_keys = [
            k for k in self._buckets
            if k[0] == symbol and k[1] < current_second_ms
        ]
        for key in completed_keys:
            bucket = self._buckets.pop(key)
            jitter = self._jitter.pop(key, (0, 0))
            row = self._finalize(bucket, jitter)
            self._pending_writes.append(row)
            self.rows_emitted += 1

        if len(self._pending_writes) >= self.flush_batch_size:
            async with self._lock:
                await self._flush_locked()

    @staticmethod
    def _finalize(bucket: _SecondBucket, jitter: tuple[int, int]) -> AggregatedRow:
        ofi = sum(bucket.ofi_values) if bucket.ofi_values else 0.0
        microprice = (
            statistics.fmean(bucket.microprice_values)
            if bucket.microprice_values else 0.0
        )
        depth_imb = (
            statistics.fmean(bucket.depth_imb_values)
            if bucket.depth_imb_values else 0.0
        )
        spread_bps = (
            statistics.fmean(bucket.spread_bps_values)
            if bucket.spread_bps_values else 0.0
        )
        # VPIN computation requires multi-bucket volume buckets — Phase B.
        vpin = 0.0
        return AggregatedRow(
            second_ms=bucket.second_ms,
            symbol=bucket.symbol,
            ofi=_safe(ofi),
            microprice=_safe(microprice),
            depth_imb=_safe(depth_imb),
            spread_bps=_safe(spread_bps),
            trade_imb=_safe(bucket.trade_imb),
            vpin=vpin,
            n_updates=bucket.n_updates,
            local_recv_ms_jitter=max(0, jitter[1] - jitter[0]),
        )

    async def _flush_locked(self) -> None:
        if not self._pending_writes or self._pool is None:
            return
        rows: list[AggregatedRow] = list(self._pending_writes)
        self._pending_writes.clear()

        records = [
            (
                # to_timestamp expects seconds
                row.second_ms / 1000.0,
                row.symbol,
                row.ofi,
                row.microprice,
                row.depth_imb,
                row.spread_bps,
                row.trade_imb,
                row.vpin,
                row.n_updates,
                row.local_recv_ms_jitter,
            )
            for row in rows
        ]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO highfreq_ofi_1s (
                        ts, symbol, ofi, microprice, depth_imb,
                        spread_bps, trade_imb, vpin, n_updates, local_recv_ms
                    ) VALUES (
                        to_timestamp($1), $2, $3, $4, $5,
                        $6, $7, $8, $9, $10
                    )
                    ON CONFLICT (ts, symbol) DO NOTHING
                    """,
                    records,
                )
            self.rows_written += len(rows)
        except Exception:  # noqa: BLE001
            logger.exception("Aggregator flush failed (%d rows dropped)", len(rows))


def _safe(x: float) -> float:
    """Replace NaN/inf with 0.0 — Postgres rejects them in DOUBLE PRECISION."""
    if x is None or math.isnan(x) or math.isinf(x):
        return 0.0
    return x
