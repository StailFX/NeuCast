"""Sub-sampled writer for top-N L2 snapshots into ``highfreq_l2_snapshots``.

Hooks into the same WS dispatch as ``Aggregator.on_snapshot`` (registered
side-by-side via the runner's fan-out callback) but writes to a
*different* table at a *different* cadence:

* **Aggregator**: every snapshot → derived OFI/microprice features
  (1 row/sec into ``highfreq_ofi_1s``).
* **This writer**: every Nth snapshot (default N=10 → 1 Hz) → raw
  top-N price/qty arrays into ``highfreq_l2_snapshots`` for the
  heatmap UI + Yandex S3 archival.

Design notes
------------

* **Independent pool.** Owns its own ``asyncpg.Pool`` (size 1-2) so a
  slow disk on the heatmap path can NEVER back-pressure the
  aggregator's ingest path. The aggregator's row-write contract is
  what powers the predictor — it must stay hot.

* **Sub-sampling.** Counts dispatched snapshots per symbol; writes
  every Nth. Reset on restart (we don't persist the counter — at
  worst we drop the first N-1 snapshots after a reboot, irrelevant
  for a heatmap that auto-scrolls).

* **Top-N trimming.** Binance gives depth20 (20 levels each side);
  we typically only want top-10 for the heatmap (denser visualisation,
  smaller storage). Trim happens on the writer side so the aggregator
  still sees full depth20 for OFI computation.

* **Batch flush.** Like the aggregator, accumulates rows in memory and
  flushes via ``executemany`` every ``flush_batch_size`` rows. With
  default 1 Hz × 3 symbols × batch=10, that's a flush every ~3.3 s —
  comfortable for Postgres.

* **Disabled by default.** Toggle via env ``HIGHFREQ_STORE_L2_SNAPSHOTS=1``
  in the runner. Safe to leave off on dev / Phase A boxes that don't
  need the heatmap.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from app.highfreq.l2_consumer import L2Snapshot

logger = logging.getLogger(__name__)


# Default top-N levels to write per side. depth20 from the WS is always
# available (see L2Consumer.depth_levels); trimming to 10 is the
# heatmap's natural density target.
DEFAULT_TOP_N: int = 10

# Default sub-sampling rate. With Binance's @100ms cadence, every-10th
# = 1 Hz, which matches what a typical UI heatmap renders at without
# visual aliasing.
DEFAULT_SAMPLE_EVERY_N: int = 10

# Flush this many rows at a time. Tuned for ~3-second flush interval at
# default 1 Hz × 3 symbols.
DEFAULT_FLUSH_BATCH: int = 10


# INSERT statement reused on every flush. Postgres accepts Python lists
# directly as DOUBLE PRECISION[] columns via asyncpg.
_INSERT_SQL = """
    INSERT INTO highfreq_l2_snapshots (
        ts, symbol, bids_price, bids_qty, asks_price, asks_qty
    ) VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (ts, symbol) DO NOTHING
"""


class L2SnapshotWriter:
    """Per-process sub-sampling writer; coexists with Aggregator on the
    same WS stream via the runner's fan-out callback."""

    def __init__(
        self,
        *,
        database_url: str,
        sample_every_n: int = DEFAULT_SAMPLE_EVERY_N,
        top_n_levels: int = DEFAULT_TOP_N,
        flush_batch_size: int = DEFAULT_FLUSH_BATCH,
        pool_min: int = 1,
        pool_max: int = 2,
    ) -> None:
        if sample_every_n < 1:
            raise ValueError("sample_every_n must be >= 1")
        if top_n_levels < 1:
            raise ValueError("top_n_levels must be >= 1")
        if flush_batch_size < 1:
            raise ValueError("flush_batch_size must be >= 1")

        self.database_url = database_url
        self.sample_every_n = sample_every_n
        self.top_n_levels = top_n_levels
        self.flush_batch_size = flush_batch_size
        self.pool_min = pool_min
        self.pool_max = pool_max

        self._pool: asyncpg.Pool | None = None
        self._buffer: list[tuple] = []
        self._buf_lock = asyncio.Lock()
        # Per-symbol monotonic counter for sub-sampling. Counter %
        # sample_every_n == 0 → write this snapshot.
        self._counters: dict[str, int] = {}

        # Stats — exposed for the runner's health log.
        self.snapshots_sampled: int = 0  # how many we kept (post-sub-sampling)
        self.snapshots_written: int = 0  # how many made it to Postgres
        self.snapshots_skipped: int = 0  # sub-sampled out (not kept)

    async def start(self) -> None:
        """Open the asyncpg pool. Idempotent."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self.database_url,
            min_size=self.pool_min,
            max_size=self.pool_max,
            command_timeout=10.0,
        )
        logger.info(
            "L2SnapshotWriter: connected pool min=%d max=%d "
            "sample_every_n=%d top_n=%d batch=%d",
            self.pool_min, self.pool_max,
            self.sample_every_n, self.top_n_levels, self.flush_batch_size,
        )

    async def on_snapshot(self, snap: "L2Snapshot") -> None:
        """Snapshot dispatch callback — sub-samples + buffers + flushes."""
        if self._pool is None:
            # Writer was never started (env flag off). Silently drop.
            return

        # Sub-sample: keep every Nth snapshot per symbol.
        sym = snap.symbol
        n = self._counters.get(sym, 0)
        self._counters[sym] = n + 1
        if n % self.sample_every_n != 0:
            self.snapshots_skipped += 1
            return

        # Trim to top_n_levels per side.
        bids = snap.bids[: self.top_n_levels]
        asks = snap.asks[: self.top_n_levels]
        if not bids or not asks:
            return  # defensive: empty book → nothing to write

        bids_price = [float(p) for p, _ in bids]
        bids_qty = [float(q) for _, q in bids]
        asks_price = [float(p) for p, _ in asks]
        asks_qty = [float(q) for _, q in asks]

        # L2Snapshot carries event_time_ms (Binance event time, ms since
        # epoch). Postgres column is TIMESTAMPTZ — convert here so the
        # writer is the only place we cross that boundary.
        ts = datetime.fromtimestamp(snap.event_time_ms / 1000.0, tz=timezone.utc)
        row = (ts, sym, bids_price, bids_qty, asks_price, asks_qty)

        async with self._buf_lock:
            self._buffer.append(row)
            self.snapshots_sampled += 1
            should_flush = len(self._buffer) >= self.flush_batch_size

        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        """Drain the buffer into Postgres. Safe to call when buffer empty."""
        if self._pool is None:
            return
        async with self._buf_lock:
            if not self._buffer:
                return
            rows = self._buffer
            self._buffer = []
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(_INSERT_SQL, rows)
            self.snapshots_written += len(rows)
        except Exception:
            logger.exception(
                "L2SnapshotWriter: flush of %d rows failed — dropping batch",
                len(rows),
            )
            # Don't re-buffer — we're sub-sampled archival data, dropping
            # one batch is acceptable and prevents memory growth on
            # persistent DB issues.

    async def close(self) -> None:
        """Drain remaining + close pool."""
        await self.flush()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def stats(self) -> dict:
        """JSON-friendly counters for the runner's health log."""
        return {
            "snapshots_sampled": self.snapshots_sampled,
            "snapshots_written": self.snapshots_written,
            "snapshots_skipped": self.snapshots_skipped,
            "buffer_size": len(self._buffer),
        }
