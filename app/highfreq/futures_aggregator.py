"""USDM Futures 1-second aggregator + Postgres writer.

Parallel to :mod:`app.highfreq.aggregator` for the spot venue (ADR-019).

Receives :class:`L2Snapshot` / :class:`Trade` / :class:`MarkPriceUpdate`
events from :class:`FuturesL2Consumer`, aggregates per (symbol, second)
and writes one row per second to ``highfreq_futures_ofi_1s``.

Differences from the spot aggregator
------------------------------------

The OFI / microprice / depth / spread / trade-imbalance aggregation
is identical (futures uses the exact same depth+trade frame format,
so the math reuses :mod:`app.highfreq.ofi_features` unchanged). The
new behaviour is:

* **Mark price + funding rate** — the @markPrice@1s stream pushes one
  frame per second per symbol. We track the *latest* values per symbol
  (stateful, not bucketed) and stamp every emitted second-row with
  whichever mark/funding we most recently observed. This is correct
  because funding rate is a slow-moving variable (changes by basis
  points per minute, not per second) — sampling latest at the
  emission boundary is plenty.

* **Wider table schema** — the SQL INSERT carries three extra columns:
  ``mark_price``, ``funding_rate``, ``next_funding_ms``. NULL on
  rows where we haven't received any mark frame yet (e.g. first second
  after restart).

Why a parallel module (not a venue flag on the spot one)
--------------------------------------------------------

See ADR-019. The spot aggregator has been running in production for
weeks. A "venue-aware" refactor that flips a table name based on a
constructor flag would be high-risk on the spot side with no upside.
The duplication here is intentional and bounded — feature_pipeline
and ofi_features are venue-agnostic, only the data-source plumbing
diverges.
"""
from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import asyncpg

from app.highfreq.futures_l2_consumer import MarkPriceUpdate
from app.highfreq.l2_consumer import L2Snapshot, Trade
from app.highfreq.ofi_features import FrameFeatures, features_from_snapshot

logger = logging.getLogger(__name__)


@dataclass
class _SecondBucket:
    """In-flight aggregation state for one (symbol, second) pair.

    Identical to the spot aggregator's ``_SecondBucket`` — we keep a
    private clone here so the two modules can evolve independently
    if futures-specific aggregation behaviour is ever needed (e.g.
    open-interest tracking, basis features against the spot price).
    """

    symbol: str
    second_ms: int
    ofi_values: list[float] = field(default_factory=list)
    microprice_values: list[float] = field(default_factory=list)
    depth_imb_values: list[float] = field(default_factory=list)
    spread_bps_values: list[float] = field(default_factory=list)
    trade_imb: float = 0.0
    n_updates: int = 0

    def add_frame(self, feat: FrameFeatures) -> None:
        self.ofi_values.append(feat.ofi)
        self.microprice_values.append(feat.microprice)
        self.depth_imb_values.append(feat.depth_imb)
        self.spread_bps_values.append(feat.spread_bps)
        self.n_updates += 1

    def add_trade(self, t: Trade) -> None:
        sign = -1.0 if t.is_buyer_maker else 1.0
        self.trade_imb += sign * t.qty


@dataclass(frozen=True)
class AggregatedRow:
    """One 1-second aggregated futures row, ready for Postgres.

    ``mark_price`` / ``funding_rate`` / ``next_funding_ms`` may be
    ``None`` when no mark-price frame has been observed yet for this
    symbol — Postgres stores these as NULL.
    """

    second_ms: int
    symbol: str
    ofi: float
    microprice: float
    depth_imb: float
    spread_bps: float
    trade_imb: float
    vpin: float
    n_updates: int
    local_recv_ms_jitter: int
    mark_price: float | None
    funding_rate: float | None
    next_funding_ms: int | None


class FuturesAggregator:
    """Owns per-symbol state, emits 1-s rows, writes to ``highfreq_futures_ofi_1s``.

    Lifecycle::

        agg = FuturesAggregator(database_url="postgresql://...",
                                symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        await agg.start()
        try:
            consumer = FuturesL2Consumer(
                symbols, on_snapshot=agg.on_snapshot,
                on_trade=agg.on_trade, on_mark_price=agg.on_mark_price,
            )
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
        self._prev_snap: dict[str, L2Snapshot | None] = {
            s: None for s in self.symbols
        }
        self._buckets: dict[tuple[str, int], _SecondBucket] = {}
        self._jitter: dict[tuple[str, int], tuple[int, int]] = {}
        # Latest mark-price snapshot per symbol — stateful (not bucketed).
        # Funding rate changes slowly (bps/minute, not bps/second) so
        # sampling latest at row-emission time is fine.
        self._latest_mark: dict[str, MarkPriceUpdate] = {}
        self._pending_writes: deque[AggregatedRow] = deque()
        self._lock = asyncio.Lock()
        # Counters surfaced to Prometheus / health endpoints.
        self.rows_emitted = 0
        self.rows_written = 0
        self.mark_frames_seen = 0

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._pool = await asyncpg.create_pool(
            self.database_url, min_size=1, max_size=2, command_timeout=10,
        )
        logger.info("FuturesAggregator connected to Postgres (pool 1-2)")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    # ──────────────────────────────────────────────────────────────────
    # Callbacks (wired into FuturesL2Consumer)
    # ──────────────────────────────────────────────────────────────────

    async def on_snapshot(self, snap: L2Snapshot) -> None:
        prev = self._prev_snap.get(snap.symbol)
        feat = features_from_snapshot(
            snap, prev, depth_levels=self.depth_levels,
        )
        self._prev_snap[snap.symbol] = snap
        await self._add_to_bucket(
            snap.symbol, snap.event_time_ms, snap.local_recv_ms, feat,
        )

    async def on_trade(self, t: Trade) -> None:
        second_ms = (t.event_time_ms // 1000) * 1000
        key = (t.symbol, second_ms)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _SecondBucket(symbol=t.symbol, second_ms=second_ms)
            self._buckets[key] = bucket
        bucket.add_trade(t)

    async def on_mark_price(self, mp: MarkPriceUpdate) -> None:
        """Update the latest-mark cache for ``mp.symbol``.

        We don't bucket these — the row write samples the latest
        value at emit time. Funding rate changes ~bps/minute so a
        100ms-1s lag between a depth bar's second and the most
        recent mark is operationally invisible.
        """
        self._latest_mark[mp.symbol] = mp
        self.mark_frames_seen += 1

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    async def _add_to_bucket(
        self,
        symbol: str,
        event_time_ms: int,
        local_recv_ms: int,
        feat: FrameFeatures,
    ) -> None:
        second_ms = (event_time_ms // 1000) * 1000
        key = (symbol, second_ms)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _SecondBucket(symbol=symbol, second_ms=second_ms)
            self._buckets[key] = bucket
        bucket.add_frame(feat)
        cur = self._jitter.get(key)
        if cur is None:
            self._jitter[key] = (local_recv_ms, local_recv_ms)
        else:
            self._jitter[key] = (
                min(cur[0], local_recv_ms),
                max(cur[1], local_recv_ms),
            )
        await self._emit_completed(symbol, second_ms)

    async def _emit_completed(
        self, symbol: str, current_second_ms: int,
    ) -> None:
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

    def _finalize(
        self, bucket: _SecondBucket, jitter: tuple[int, int],
    ) -> AggregatedRow:
        """Same OFI/microprice math as spot, plus mark/funding stamping."""
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
        # VPIN deferred to Phase B (same as spot).
        vpin = 0.0

        # Stamp the latest mark-price observation (carry-forward).
        # ``None`` means we haven't seen any mark frame for this symbol
        # yet — Postgres stores NULL.
        mp = self._latest_mark.get(bucket.symbol)
        mark_price = mp.mark_price if mp is not None else None
        funding_rate = mp.funding_rate if mp is not None else None
        next_funding_ms = mp.next_funding_ms if mp is not None else None

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
            mark_price=_safe_or_none(mark_price),
            funding_rate=_safe_or_none(funding_rate),
            next_funding_ms=next_funding_ms,
        )

    async def _flush_locked(self) -> None:
        if not self._pending_writes or self._pool is None:
            return
        rows: list[AggregatedRow] = list(self._pending_writes)
        self._pending_writes.clear()

        records = [
            (
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
                row.mark_price,
                row.funding_rate,
                row.next_funding_ms,
            )
            for row in rows
        ]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO highfreq_futures_ofi_1s (
                        ts, symbol, ofi, microprice, depth_imb,
                        spread_bps, trade_imb, vpin, n_updates,
                        local_recv_ms,
                        mark_price, funding_rate, next_funding_ms
                    ) VALUES (
                        to_timestamp($1), $2, $3, $4, $5,
                        $6, $7, $8, $9, $10,
                        $11, $12, $13
                    )
                    ON CONFLICT (ts, symbol) DO NOTHING
                    """,
                    records,
                )
            self.rows_written += len(rows)
        except Exception:  # noqa: BLE001
            logger.exception(
                "FuturesAggregator flush failed (%d rows dropped)", len(rows),
            )


def _safe(x: float) -> float:
    """Replace NaN/inf with 0.0 — Postgres rejects them in DOUBLE PRECISION."""
    if x is None or math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def _safe_or_none(x: float | None) -> float | None:
    """Like _safe but preserves None (so Postgres stores NULL).

    Used for mark/funding columns which can legitimately be unobserved
    yet — that's different from a numeric value that overflowed."""
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x
