"""Standalone entry point for the USDM Futures ingest pipeline.

Wires :class:`~app.highfreq.futures_l2_consumer.FuturesL2Consumer` to
:class:`~app.highfreq.futures_aggregator.FuturesAggregator`. Parallel to
:mod:`app.highfreq.runner` for the spot venue (ADR-019).

Run locally for development::

    DATABASE_URL=postgres://... python -m app.highfreq.futures_runner

In production it runs as a dedicated systemd service
``neucast-futures-highfreq.service`` (release S+1).

Environment
-----------

* ``DATABASE_URL`` — Postgres DSN (required).
* ``HIGHFREQ_FUTURES_SYMBOLS`` — comma-separated, default ``BTCUSDT``.
  Use the spot symbol form (no ``.P`` suffix); venue is implicit by
  the table the aggregator writes to (``highfreq_futures_ofi_1s``).
* ``HIGHFREQ_FUTURES_DEPTH_LEVELS`` — partial-book depth, default ``20``.
* ``HIGHFREQ_FUTURES_UPDATE_SPEED_MS`` — stream cadence, default ``100``.
* ``HIGHFREQ_FUTURES_FLUSH_BATCH`` — Postgres flush batch size, default ``5``.
* ``HIGHFREQ_FUTURES_INGEST_METRICS_PORT`` — Prometheus port,
  default ``9100`` (spot uses 9090, stay clear).
* ``LOG_LEVEL`` — default ``INFO``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from app.highfreq.futures_aggregator import FuturesAggregator
from app.highfreq.futures_l2_consumer import FuturesL2Consumer


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _normalise_dsn(dsn: str) -> str:
    """Same normalisation as the spot runner (postgres:// → postgresql://)."""
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    return dsn


async def _main() -> None:
    _configure_logging()
    logger = logging.getLogger("highfreq.futures_runner")

    dsn_raw = os.environ.get("DATABASE_URL")
    if not dsn_raw:
        logger.error("DATABASE_URL is required")
        sys.exit(2)
    dsn = _normalise_dsn(dsn_raw)

    symbols = [
        s.strip().upper()
        for s in os.getenv("HIGHFREQ_FUTURES_SYMBOLS", "BTCUSDT").split(",")
        if s.strip()
    ]
    depth_levels = int(os.getenv("HIGHFREQ_FUTURES_DEPTH_LEVELS", "20"))
    update_speed_ms = int(os.getenv("HIGHFREQ_FUTURES_UPDATE_SPEED_MS", "100"))
    flush_batch = int(os.getenv("HIGHFREQ_FUTURES_FLUSH_BATCH", "5"))

    logger.info(
        "starting futures ingest: symbols=%s depth=%d speed=%dms flush=%d",
        symbols, depth_levels, update_speed_ms, flush_batch,
    )

    aggregator = FuturesAggregator(
        database_url=dsn,
        symbols=symbols,
        depth_levels=min(depth_levels, 10),  # depth_imb computed over top-10
        flush_batch_size=flush_batch,
    )
    await aggregator.start()

    consumer = FuturesL2Consumer(
        symbols=symbols,
        on_snapshot=aggregator.on_snapshot,
        on_trade=aggregator.on_trade,
        on_mark_price=aggregator.on_mark_price,
        depth_levels=depth_levels,
        update_speed_ms=update_speed_ms,
    )

    # Prometheus /metrics HTTP server. Bound to 127.0.0.1 — Prometheus
    # scrapes localhost. Default 9100 (spot uses 9090). Different port
    # so both ingests can coexist on the same host.
    metrics_port = int(
        os.getenv("HIGHFREQ_FUTURES_INGEST_METRICS_PORT", "9100"),
    )
    try:
        from prometheus_client import start_http_server
        start_http_server(metrics_port, addr="127.0.0.1")
        logger.info(
            "prometheus metrics server on 127.0.0.1:%d", metrics_port,
        )
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
            pass

    consumer_task = asyncio.create_task(consumer.run_forever())

    async def _health_log() -> None:
        """Periodic health log (every 30 s) so journalctl shows progress."""
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
            logger.info(
                "futures health: frames=%d snaps=%d trades=%d "
                "marks=%d rows_emitted=%d rows_written=%d reconnects=%d",
                consumer.frames_received,
                consumer.snapshots_dispatched,
                consumer.trades_dispatched,
                consumer.mark_price_dispatched,
                aggregator.rows_emitted,
                aggregator.rows_written,
                consumer.reconnect_count,
            )

    health_task = asyncio.create_task(_health_log())

    try:
        await consumer_task
    finally:
        health_task.cancel()
        await aggregator.flush()
        await aggregator.close()
        logger.info(
            "stopped: frames=%d marks=%d rows_written=%d",
            consumer.frames_received,
            consumer.mark_price_dispatched,
            aggregator.rows_written,
        )


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
