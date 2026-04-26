"""Standalone entry point for the high-frequency ingest pipeline.

Wires :class:`~app.highfreq.l2_consumer.L2Consumer` to
:class:`~app.highfreq.aggregator.Aggregator` and runs them until SIGTERM.

Run locally for development::

    python -m app.highfreq.runner

In production it runs as a dedicated Docker service (see
``docker-compose.yml`` → ``highfreq-l2``).

Environment
-----------

* ``DATABASE_URL`` — Postgres DSN (required).
* ``HIGHFREQ_SYMBOLS`` — comma-separated, default ``BTCUSDT``.
* ``HIGHFREQ_DEPTH_LEVELS`` — partial-book depth, default ``20``.
* ``HIGHFREQ_UPDATE_SPEED_MS`` — stream cadence, default ``100``.
* ``HIGHFREQ_FLUSH_BATCH`` — Postgres flush batch size, default ``5``.
* ``LOG_LEVEL`` — default ``INFO``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from app.highfreq.aggregator import Aggregator
from app.highfreq.l2_consumer import L2Consumer
from app.highfreq.l2_snapshot_writer import (
    DEFAULT_FLUSH_BATCH as L2_SNAP_FLUSH_BATCH,
    DEFAULT_SAMPLE_EVERY_N as L2_SNAP_SAMPLE_EVERY,
    DEFAULT_TOP_N as L2_SNAP_TOP_N,
    L2SnapshotWriter,
)
from app.highfreq import metrics as M


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _normalise_dsn(dsn: str) -> str:
    """asyncpg does not accept the ``postgresql://`` URL fragment ``?sslmode=…``
    parameters in older versions; strip them. Also accepts the ``postgres://``
    alias and converts to ``postgresql://`` for clarity in logs.
    """
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    return dsn


async def _main() -> None:
    _configure_logging()
    logger = logging.getLogger("highfreq.runner")

    dsn_raw = os.environ.get("DATABASE_URL")
    if not dsn_raw:
        logger.error("DATABASE_URL is required")
        sys.exit(2)
    dsn = _normalise_dsn(dsn_raw)

    symbols = [
        s.strip().upper()
        for s in os.getenv("HIGHFREQ_SYMBOLS", "BTCUSDT").split(",")
        if s.strip()
    ]
    depth_levels = int(os.getenv("HIGHFREQ_DEPTH_LEVELS", "20"))
    update_speed_ms = int(os.getenv("HIGHFREQ_UPDATE_SPEED_MS", "100"))
    flush_batch = int(os.getenv("HIGHFREQ_FLUSH_BATCH", "5"))

    logger.info(
        "starting highfreq pipeline: symbols=%s depth=%d speed=%dms flush=%d",
        symbols, depth_levels, update_speed_ms, flush_batch,
    )

    aggregator = Aggregator(
        database_url=dsn,
        symbols=symbols,
        depth_levels=min(depth_levels, 10),  # depth_imb computed over top-10
        flush_batch_size=flush_batch,
    )
    await aggregator.start()

    # Optional: also write sub-sampled top-N L2 snapshots into
    # highfreq_l2_snapshots for the heatmap UI + Yandex S3 archival.
    # Toggled via HIGHFREQ_STORE_L2_SNAPSHOTS=1; off by default to keep
    # the ingest box lean for deploys that don't need the heatmap.
    snapshot_writer: L2SnapshotWriter | None = None
    if os.getenv("HIGHFREQ_STORE_L2_SNAPSHOTS", "0") == "1":
        snapshot_writer = L2SnapshotWriter(
            database_url=dsn,
            sample_every_n=int(os.getenv("HIGHFREQ_L2_SAMPLE_EVERY_N",
                                         str(L2_SNAP_SAMPLE_EVERY))),
            top_n_levels=int(os.getenv("HIGHFREQ_L2_TOP_N", str(L2_SNAP_TOP_N))),
            flush_batch_size=int(os.getenv("HIGHFREQ_L2_FLUSH_BATCH",
                                           str(L2_SNAP_FLUSH_BATCH))),
        )
        await snapshot_writer.start()
        logger.info("L2 snapshot writer enabled")

    # Fan-out the WS snapshot dispatch when both subscribers exist.
    if snapshot_writer is not None:
        async def _on_snapshot_fanout(snap):
            await aggregator.on_snapshot(snap)
            await snapshot_writer.on_snapshot(snap)
        snapshot_cb = _on_snapshot_fanout
    else:
        snapshot_cb = aggregator.on_snapshot

    consumer = L2Consumer(
        symbols=symbols,
        on_snapshot=snapshot_cb,
        on_trade=aggregator.on_trade,
        depth_levels=depth_levels,
        update_speed_ms=update_speed_ms,
    )

    # Prometheus /metrics HTTP server. Bound to 127.0.0.1 — Prometheus
    # runs locally on Tokyo and scrapes localhost. Default port 9090
    # (matches Prometheus convention for the main app's exporter port).
    metrics_port = int(os.getenv("HIGHFREQ_INGEST_METRICS_PORT", "9090"))
    try:
        from prometheus_client import start_http_server
        start_http_server(metrics_port, addr="127.0.0.1")
        logger.info("prometheus metrics server on 127.0.0.1:%d", metrics_port)
    except Exception:
        logger.exception(
            "failed to start prometheus /metrics on :%d — continuing without",
            metrics_port,
        )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal(sig: int) -> None:
        logger.info("received signal %d, shutting down", sig)
        consumer.stop()
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
        except NotImplementedError:
            # Windows/threaded test contexts — best effort.
            pass

    consumer_task = asyncio.create_task(consumer.run_forever())

    # Periodic health log: every 30 s emit counters so journalctl shows progress.
    # Also synchronises Prometheus counters from the consumer's live counters
    # (we use _inc_to to mirror absolute values into monotonic counters).
    last_seen = {
        "frames": 0, "reconnects": 0,
        "snaps_per_sym": {s: 0 for s in symbols},
        "trades_per_sym": {s: 0 for s in symbols},
        "ofi_rows_per_sym": {s: 0 for s in symbols},
        "l2_per_sym": {s: 0 for s in symbols},
    }

    def _sync_counter(counter, label_value, current: int, key: str, sub_key: str | None = None):
        bucket = last_seen[key] if sub_key is None else last_seen[key].get(sub_key, 0)
        delta = current - (bucket if sub_key is None else last_seen[key][sub_key])
        if delta > 0:
            if label_value is None:
                counter.inc(delta)
            else:
                counter.labels(symbol=label_value).inc(delta)
        if sub_key is None:
            last_seen[key] = current
        else:
            last_seen[key][sub_key] = current

    async def _health_log() -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass

            # Sync Prometheus counters with consumer/aggregator state.
            _sync_counter(M.ws_frames_total, None, consumer.frames_received, "frames")
            _sync_counter(M.ws_reconnects_total, None, consumer.reconnect_count, "reconnects")
            for sym in symbols:
                _sync_counter(
                    M.snapshots_dispatched_total, sym,
                    consumer.snapshots_per_symbol.get(sym, 0)
                    if hasattr(consumer, "snapshots_per_symbol") else 0,
                    "snaps_per_sym", sym,
                )
                _sync_counter(
                    M.trades_dispatched_total, sym,
                    consumer.trades_per_symbol.get(sym, 0)
                    if hasattr(consumer, "trades_per_symbol") else 0,
                    "trades_per_sym", sym,
                )
                _sync_counter(
                    M.ofi_rows_written_total, sym,
                    aggregator.rows_per_symbol.get(sym, 0)
                    if hasattr(aggregator, "rows_per_symbol") else 0,
                    "ofi_rows_per_sym", sym,
                )
                if snapshot_writer is not None:
                    _sync_counter(
                        M.l2_snapshots_written_total, sym,
                        snapshot_writer._counters.get(sym, 0) // snapshot_writer.sample_every_n,
                        "l2_per_sym", sym,
                    )

            snap_stats = (
                f" l2snaps_written={snapshot_writer.snapshots_written}"
                if snapshot_writer is not None else ""
            )
            logger.info(
                "health: frames=%d snaps=%d trades=%d rows_emitted=%d "
                "rows_written=%d reconnects=%d%s",
                consumer.frames_received, consumer.snapshots_dispatched,
                consumer.trades_dispatched, aggregator.rows_emitted,
                aggregator.rows_written, consumer.reconnect_count, snap_stats,
            )

    health_task = asyncio.create_task(_health_log())

    try:
        await consumer_task
    finally:
        health_task.cancel()
        await aggregator.flush()
        await aggregator.close()
        if snapshot_writer is not None:
            await snapshot_writer.close()
        logger.info(
            "stopped: frames=%d rows_written=%d snapshots_written=%d",
            consumer.frames_received, aggregator.rows_written,
            snapshot_writer.snapshots_written if snapshot_writer else 0,
        )


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
