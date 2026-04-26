"""Tests for ``app.highfreq.l2_snapshot_writer`` — sub-sampling writer.

We mock the asyncpg pool entirely — these tests verify the
sub-sampling logic, top-N trimming, batch buffering, and graceful
behaviour when started/stopped or pool fails. The actual SQL is
exercised by the live deploy.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.highfreq.l2_consumer import L2Snapshot
from app.highfreq.l2_snapshot_writer import (
    DEFAULT_FLUSH_BATCH,
    DEFAULT_SAMPLE_EVERY_N,
    DEFAULT_TOP_N,
    L2SnapshotWriter,
)


UTC = timezone.utc


# ───────────────────────── helpers ─────────────────────────


def _snap(symbol: str = "BTCUSDT", n_levels: int = 20,
          ts_offset_s: int = 0) -> L2Snapshot:
    """Build a synthetic L2Snapshot with N levels each side.

    ``ts_offset_s`` shifts the event time so consecutive snapshots have
    distinct (ts, symbol) primary keys when written.
    """
    base_price = 78_000.0 if symbol == "BTCUSDT" else 2_000.0
    bids = tuple(
        (base_price - i * 0.5, 1.0 + i * 0.01) for i in range(n_levels)
    )
    asks = tuple(
        (base_price + i * 0.5, 1.0 + i * 0.01) for i in range(n_levels)
    )
    base_ts_ms = int(datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)
    return L2Snapshot(
        event_time_ms=base_ts_ms + ts_offset_s * 1000,
        local_recv_ms=base_ts_ms + ts_offset_s * 1000 + 5,
        symbol=symbol,
        bids=bids,
        asks=asks,
    )


def _writer_with_mock_pool(
    *, sample_every_n: int = 1, top_n: int = 10, batch: int = 100,
) -> L2SnapshotWriter:
    """Pre-started writer whose pool is a MagicMock that captures
    executemany calls but does no I/O."""
    w = L2SnapshotWriter(
        database_url="postgresql://stub:stub@localhost/stub",
        sample_every_n=sample_every_n,
        top_n_levels=top_n,
        flush_batch_size=batch,
    )
    # Manually wire a fake pool (skip real start()).
    conn = MagicMock()
    conn.executemany = AsyncMock(return_value=None)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    pool.close = AsyncMock(return_value=None)  # for writer.close() teardown
    w._pool = pool
    return w


def _conn_from(writer: L2SnapshotWriter) -> MagicMock:
    """Pull the AsyncMock connection out of the writer's mock pool."""
    return writer._pool.acquire.return_value.__aenter__.return_value


# ───────────────────────── sub-sampling ─────────────────────────


def test_writer_skips_first_when_sample_every_n_eq_2():
    """sample_every_n=2 → write counter 0, skip 1, write 2, skip 3, ..."""
    w = _writer_with_mock_pool(sample_every_n=2, batch=100)
    asyncio.run(_drive(w, [
        _snap(ts_offset_s=0),
        _snap(ts_offset_s=1),
        _snap(ts_offset_s=2),
        _snap(ts_offset_s=3),
        _snap(ts_offset_s=4),
    ]))
    # Counter 0, 2, 4 kept (3 total); 1, 3 skipped.
    assert w.snapshots_sampled == 3
    assert w.snapshots_skipped == 2


def test_writer_keeps_all_when_sample_every_n_eq_1():
    w = _writer_with_mock_pool(sample_every_n=1, batch=100)
    asyncio.run(_drive(w, [_snap(ts_offset_s=i) for i in range(5)]))
    assert w.snapshots_sampled == 5
    assert w.snapshots_skipped == 0


def test_writer_per_symbol_counter_is_independent():
    """Counters reset per symbol — BTCUSDT and ETHUSDT each get their
    own modulo cycle."""
    w = _writer_with_mock_pool(sample_every_n=2, batch=100)
    asyncio.run(_drive(w, [
        _snap(symbol="BTCUSDT", ts_offset_s=0),  # BTC counter=0 → keep
        _snap(symbol="ETHUSDT", ts_offset_s=0),  # ETH counter=0 → keep
        _snap(symbol="BTCUSDT", ts_offset_s=1),  # BTC counter=1 → skip
        _snap(symbol="ETHUSDT", ts_offset_s=1),  # ETH counter=1 → skip
        _snap(symbol="BTCUSDT", ts_offset_s=2),  # BTC counter=2 → keep
    ]))
    assert w.snapshots_sampled == 3  # BTC×2 + ETH×1


# ───────────────────────── top-N trimming ─────────────────────────


def test_writer_trims_to_top_n_levels():
    """Snapshot has 20 levels; writer with top_n=5 keeps only top 5
    on each side."""
    w = _writer_with_mock_pool(sample_every_n=1, top_n=5, batch=1)
    asyncio.run(_drive(w, [_snap(n_levels=20)]))

    # Inspect the row passed to executemany.
    conn = _conn_from(w)
    rows = conn.executemany.await_args.args[1]
    assert len(rows) == 1
    _ts, _sym, bids_price, bids_qty, asks_price, asks_qty = rows[0]
    assert len(bids_price) == 5
    assert len(asks_price) == 5
    assert len(bids_qty) == 5
    assert len(asks_qty) == 5


def test_writer_skips_empty_book_rows():
    """Snapshot with empty bids/asks → no row written (defensive)."""
    w = _writer_with_mock_pool(sample_every_n=1, batch=1)
    empty_snap = L2Snapshot(
        event_time_ms=int(datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC).timestamp() * 1000),
        local_recv_ms=int(datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC).timestamp() * 1000),
        symbol="BTCUSDT", bids=(), asks=(),
    )
    asyncio.run(_drive(w, [empty_snap]))
    assert w.snapshots_sampled == 0
    conn = _conn_from(w)
    conn.executemany.assert_not_awaited()


# ───────────────────────── batch flush ─────────────────────────


def test_writer_flushes_when_batch_full():
    """batch_size=3 → flush triggers on the 3rd kept snapshot."""
    w = _writer_with_mock_pool(sample_every_n=1, batch=3)
    asyncio.run(_drive(w, [
        _snap(ts_offset_s=0),  # buffered (1/3)
        _snap(ts_offset_s=1),  # buffered (2/3)
        _snap(ts_offset_s=2),  # FLUSH (3/3)
    ]))
    conn = _conn_from(w)
    assert conn.executemany.await_count == 1
    assert w.snapshots_written == 3


def test_writer_does_not_flush_below_batch_threshold():
    w = _writer_with_mock_pool(sample_every_n=1, batch=10)
    asyncio.run(_drive(w, [_snap(ts_offset_s=i) for i in range(5)]))
    conn = _conn_from(w)
    # Below batch size — never flushed.
    conn.executemany.assert_not_awaited()
    assert w.snapshots_written == 0
    # But all 5 sampled (in buffer).
    assert w.snapshots_sampled == 5


def test_writer_close_drains_buffer():
    """close() must call flush() before tearing down the pool."""
    w = _writer_with_mock_pool(sample_every_n=1, batch=10)
    # Capture the conn ref BEFORE close — close() sets _pool = None.
    conn = _conn_from(w)
    asyncio.run(_drive_then_close(w, [_snap(ts_offset_s=i) for i in range(3)]))
    # Even though batch wasn't full, close() forced a flush.
    assert conn.executemany.await_count == 1
    assert w.snapshots_written == 3


# ───────────────────────── pool not started ─────────────────────────


def test_on_snapshot_silently_drops_when_pool_not_started():
    """If env flag is off, writer is constructed but never start()-ed.
    on_snapshot must drop silently — no exception."""
    w = L2SnapshotWriter(
        database_url="postgresql://stub:stub@localhost/stub",
        sample_every_n=1, batch_size=1, flush_batch_size=1,
    ) if False else L2SnapshotWriter(
        database_url="postgresql://stub:stub@localhost/stub",
        sample_every_n=1, flush_batch_size=1,
    )
    # _pool is None — no start() called.
    asyncio.run(_drive(w, [_snap()]))
    assert w.snapshots_sampled == 0
    assert w.snapshots_written == 0


# ───────────────────────── flush failure resilience ─────────────────────────


def test_flush_failure_drops_batch_and_does_not_raise():
    """A DB error during executemany must NOT crash on_snapshot —
    we drop the batch and the next snapshot starts clean."""
    w = _writer_with_mock_pool(sample_every_n=1, batch=2)
    conn = _conn_from(w)
    conn.executemany.side_effect = RuntimeError("DB down")

    # Drive 2 snapshots → triggers flush → exception caught.
    asyncio.run(_drive(w, [_snap(ts_offset_s=0), _snap(ts_offset_s=1)]))
    assert w.snapshots_written == 0  # flush failed
    assert w.snapshots_sampled == 2  # but counter advanced

    # Now succeed: drive 2 more with a fixed mock.
    conn.executemany.side_effect = None
    asyncio.run(_drive(w, [_snap(ts_offset_s=2), _snap(ts_offset_s=3)]))
    assert w.snapshots_written == 2  # second flush worked


# ───────────────────────── ctor validation ─────────────────────────


def test_ctor_rejects_zero_sample_every_n():
    with pytest.raises(ValueError, match="sample_every_n"):
        L2SnapshotWriter(database_url="x", sample_every_n=0)


def test_ctor_rejects_zero_top_n():
    with pytest.raises(ValueError, match="top_n_levels"):
        L2SnapshotWriter(database_url="x", top_n_levels=0)


def test_ctor_rejects_zero_batch():
    with pytest.raises(ValueError, match="flush_batch_size"):
        L2SnapshotWriter(database_url="x", flush_batch_size=0)


# ───────────────────────── stats ─────────────────────────


def test_stats_returns_json_friendly_dict():
    w = _writer_with_mock_pool(sample_every_n=2, batch=100)
    asyncio.run(_drive(w, [_snap(ts_offset_s=i) for i in range(5)]))
    s = w.stats()
    assert set(s.keys()) == {
        "snapshots_sampled", "snapshots_written",
        "snapshots_skipped", "buffer_size",
    }
    assert s["snapshots_sampled"] == 3
    assert s["snapshots_skipped"] == 2
    assert s["buffer_size"] == 3


# ───────────────────────── async drivers ─────────────────────────


async def _drive(w: L2SnapshotWriter, snaps: list[L2Snapshot]) -> None:
    for s in snaps:
        await w.on_snapshot(s)


async def _drive_then_close(w: L2SnapshotWriter, snaps: list[L2Snapshot]) -> None:
    await _drive(w, snaps)
    await w.close()
