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

    consumer = L2Consumer(
        symbols=symbols,
        on_snapshot=aggregator.on_snapshot,
        on_trade=aggregator.on_trade,
        depth_levels=depth_levels,
        update_speed_ms=update_speed_ms,
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
    async def _health_log() -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
            logger.info(
                "health: frames=%d snaps=%d trades=%d rows_emitted=%d "
                "rows_written=%d reconnects=%d",
                consumer.frames_received, consumer.snapshots_dispatched,
                consumer.trades_dispatched, aggregator.rows_emitted,
                aggregator.rows_written, consumer.reconnect_count,
            )

    health_task = asyncio.create_task(_health_log())

    try:
        await consumer_task
    finally:
        health_task.cancel()
        await aggregator.flush()
        await aggregator.close()
        logger.info(
            "stopped: frames=%d rows_written=%d",
            consumer.frames_received, aggregator.rows_written,
        )


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
