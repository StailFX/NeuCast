"""USDM Futures funding-rate poller (release S phase 3, 2026-04-29).

Why this exists
===============

The futures L2 consumer subscribes to ``<symbol>@markPrice@1s`` for
funding-rate updates, but empirically that stream delivers zero frames
on this Tokyo VPS ↔ Binance Futures route (likely regional / IP-based
filtering at Binance's edge). The depth + trade streams flow normally
on the same connection.

Funding rate changes ~basis-points per minute (it's a slow, mean-
reverting variable that's quoted by the exchange every 8 hours and
interpolated linearly in between), so a 5-minute REST poll is more
than enough operational granularity.

What this does
==============

Once per fire it:

1. Fetches the current funding rate + mark price for each configured
   symbol from ``GET /fapi/v1/premiumIndex`` — Binance's REST endpoint
   that returns the same fields as the ``@markPrice`` stream payload.

2. UPDATEs the most-recent rows of ``highfreq_futures_ofi_1s`` for
   each symbol (``WHERE ts >= now() - interval '5 minutes'``) so that:

   * past rows that arrived while we had no funding data get back-filled,
   * future rows arriving in the next 5 minutes will have the latest
     known funding rate carried forward by the aggregator's
     ``_latest_mark`` cache (we ALSO push directly into that cache via
     a separate codepath when running co-located, but the simple,
     resilient path is just to UPDATE rows in place).

3. Writes a heartbeat metric to the textfile collector so Prometheus
   can detect "funding poller hasn't run in N min" alerts.

Run modes
=========

* CLI one-shot::

      python -m tools.poll_funding_rate --symbols BTCUSDT ETHUSDT BNBUSDT

* Systemd timer (recommended)::

      neucast-futures-funding-poll.timer  →  every 5 minutes
      neucast-futures-funding-poll.service  oneshot

The timer is documented in ``docs/highfreq/deploy/`` alongside the
ingest service.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
from typing import Iterable

logger = logging.getLogger(__name__)


BINANCE_FUTURES_REST_BASE = "https://fapi.binance.com"
PREMIUM_INDEX_PATH = "/fapi/v1/premiumIndex"


def fetch_premium_index(symbol: str, *, timeout_seconds: float = 5.0) -> dict | None:
    """REST GET ``/fapi/v1/premiumIndex?symbol=SYMBOL``.

    Response shape::

        {
          "symbol": "BTCUSDT",
          "markPrice": "65430.10",
          "indexPrice": "65420.50",
          "estimatedSettlePrice": "65425.00",
          "lastFundingRate": "0.00012345",
          "interestRate": "0.0001",
          "nextFundingTime": 1711238400000,
          "time": 1711234567890
        }

    Returns the parsed dict, or ``None`` on transport / parse failure
    (caller logs and continues — a single failed poll shouldn't kill
    the systemd oneshot).
    """
    url = f"{BINANCE_FUTURES_REST_BASE}{PREMIUM_INDEX_PATH}?symbol={symbol}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "neucast-funding-poll/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)
    except Exception as exc:  # noqa: BLE001 — single REST call, log + skip
        logger.warning("premiumIndex fetch failed for %s: %s: %s",
                       symbol, type(exc).__name__, exc)
        return None


def parse_premium_index(data: dict) -> tuple[str, float, float, int] | None:
    """Coerce raw REST payload to ``(symbol, mark_price, funding_rate, next_funding_ms)``.

    All numerics in the REST response are strings — coerce to float / int.
    Returns ``None`` if any required field is missing / malformed.
    """
    try:
        symbol = str(data["symbol"]).upper()
        mark_price = float(data["markPrice"])
        funding_rate = float(data["lastFundingRate"])
        next_funding_ms = int(data["nextFundingTime"])
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("malformed premiumIndex payload (%s): %r", exc, data)
        return None
    return symbol, mark_price, funding_rate, next_funding_ms


async def update_recent_rows(
    dsn: str,
    *,
    symbol: str,
    mark_price: float,
    funding_rate: float,
    next_funding_ms: int,
    backfill_minutes: int = 5,
) -> int:
    """UPDATE ``highfreq_futures_ofi_1s`` rows for ``symbol`` whose
    ``ts >= now() - interval 'N minutes'`` with the latest funding values.

    Returns the number of rows touched. ``0`` means either no recent
    rows exist (ingest down) or the values were already current.

    Idempotent — re-running the poll over the same window is a no-op
    when the values haven't changed.
    """
    import asyncpg

    sql = """
        UPDATE highfreq_futures_ofi_1s
        SET mark_price = $2,
            funding_rate = $3,
            next_funding_ms = $4
        WHERE symbol = $1
          AND ts >= now() - make_interval(mins => $5)
          AND (
              mark_price IS DISTINCT FROM $2
              OR funding_rate IS DISTINCT FROM $3
              OR next_funding_ms IS DISTINCT FROM $4
          )
    """
    conn = await asyncpg.connect(dsn=dsn, command_timeout=10)
    try:
        result = await conn.execute(
            sql, symbol, mark_price, funding_rate, next_funding_ms,
            backfill_minutes,
        )
    finally:
        await conn.close()
    # asyncpg returns "UPDATE <n>" — parse the trailing number.
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        n = 0
    return n


async def poll_once(
    dsn: str,
    *,
    symbols: Iterable[str],
    backfill_minutes: int = 5,
) -> tuple[int, int]:
    """Poll all symbols, UPDATE their recent rows. Returns (n_ok, n_rows_touched).

    Per-symbol failures are logged but don't abort the whole run — we
    want a flaky API on one symbol to not silently kill the others.
    """
    n_ok = 0
    total_rows = 0
    for sym in symbols:
        sym_u = sym.upper()
        data = fetch_premium_index(sym_u)
        if data is None:
            continue
        parsed = parse_premium_index(data)
        if parsed is None:
            continue
        sym_resp, mark, funding, next_ms = parsed
        if sym_resp != sym_u:
            logger.warning(
                "premiumIndex symbol mismatch: requested %s, got %s",
                sym_u, sym_resp,
            )
            continue
        try:
            n = await update_recent_rows(
                dsn,
                symbol=sym_u,
                mark_price=mark,
                funding_rate=funding,
                next_funding_ms=next_ms,
                backfill_minutes=backfill_minutes,
            )
        except Exception:  # noqa: BLE001
            logger.exception("UPDATE failed for %s", sym_u)
            continue
        n_ok += 1
        total_rows += n
        logger.info(
            "%s: mark=%.4f funding_8h=%+.4f bp next_in=%.1f h → updated %d rows",
            sym_u, mark, funding * 1e4,
            max(0.0, (next_ms / 1000 - time.time()) / 3600),
            n,
        )
    return n_ok, total_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.poll_funding_rate",
        description=__doc__,
    )
    p.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        help="symbols to poll (default: 3 majors)",
    )
    p.add_argument(
        "--backfill-minutes", type=int, default=5,
        help="UPDATE rows whose ts is within the last N minutes "
             "(default 5; matches the systemd timer cadence)",
    )
    p.add_argument(
        "--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is required")
        return 2
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]

    started = time.monotonic()
    n_ok, total_rows = asyncio.run(
        poll_once(
            dsn, symbols=args.symbols,
            backfill_minutes=args.backfill_minutes,
        ),
    )
    elapsed = time.monotonic() - started
    logger.info(
        "poll done: %d/%d symbols ok, %d rows updated, elapsed=%.2fs",
        n_ok, len(args.symbols), total_rows, elapsed,
    )

    # Heartbeat for textfile_collector — same convention as
    # cron_metrics.py for the existing nightly cron jobs.
    try:
        from app.highfreq.cron_metrics import write_cron_success
        write_cron_success(
            "neucast_futures_funding_poll_last_success_timestamp_seconds",
            file_stem="neucast_futures_funding_poll",
            labels={},
        )
    except Exception:
        logger.warning("heartbeat write failed", exc_info=True)

    # Exit 0 only if at least one symbol succeeded — systemd then
    # tracks "no successful polls in 30 min" as service failure.
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
