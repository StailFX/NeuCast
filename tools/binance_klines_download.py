"""Download multi-year 1-minute OHLCV from Binance public Klines API.

Why this exists
===============

The HF directional model is trained on ~5 days of OFI seconds-level
data we ingest live. That's enough to fit a CatBoost (sample size
~7000 bars after neutral-band drop) but the trained model's "regime
knowledge" is bounded by what those 5 days happened to look like.
A bull / bear / sideways regime that wasn't in the training window
is, by definition, out-of-distribution for the served model.

The fix is **pre-training on years of historical data**: download
the same symbols' OHLCV from Binance's public Klines endpoint going
back to listing date, fit a CatBoost on the long-horizon TA pipeline
(OHLC + EMA + RSI + Bollinger — which only needs price+volume, not
L2), and then fine-tune on the recent live OFI data via CatBoost's
``init_model`` parameter (incremental learning).

Why Binance directly (not Finam / Kraken)
-----------------------------------------

Originally targeted Finam.ru's xxbtzusd export per user request, but:
* Finam now has anti-bot CAPTCHA (servicepipe.ru challenge) → blocks
  automated downloads at the page-render layer.
* Finam crypto data is *relayed from Kraken*. Kraken doesn't list
  BNB (it's a Binance-native token), so we'd have no source for BNB.
* Distribution mismatch: Kraken vs Binance order-flow differ.

Binance's public Klines endpoint:
* Same symbols (BTC/ETH/BNB USDT pairs) as production — distribution
  matches what the live model serves.
* No registration, no rate-limiting on this endpoint at low rates.
* Goes back to listing date (BTCUSDT: Aug 2017, BNBUSDT: Jul 2017,
  ETHUSDT: Aug 2017).
* OHLCV plus n_trades + taker_buy_volume — enough for long_horizon
  TA features (microprice OHLC + n_updates_sum + trade_imb proxy).

What this writes
================

A parquet file at ``data/historical/<symbol>_1m_klines.parquet``
with the schema expected by ``aggregate_to_minute(...)`` downstream:

    minute, symbol, microprice_open, microprice_close,
    microprice_high, microprice_low, n_updates_sum,
    trade_imb_sum, ofi_sum, spread_bps_mean, ...

Klines doesn't carry OFI / depth / spread. We fill them with neutral
values (0) so the long_horizon pipeline (which doesn't use these
columns anyway) works untouched. When the fine-tune step uses live
data, those columns get real OFI values back.

Run from any host with internet
-------------------------------

::

    python -m tools.binance_klines_download --symbol BTCUSDT --years 2
    python -m tools.binance_klines_download \\
        --symbol BTCUSDT --symbol ETHUSDT --symbol BNBUSDT --years 3
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Klines column ordering per Binance API docs.
KLINES_COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "n_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "_ignore",
]


def fetch_klines_chunk(
    symbol: str, *, start_ms: int, end_ms: int | None = None,
    interval: str = "1m", limit: int = 1000,
    session=None,
) -> list[list]:
    """Fetch one chunk of klines from Binance public REST.

    Limit per request: 1000 bars. For 1m interval that's ~16h of data.
    Caller paginates by repeatedly bumping ``start_ms`` to the
    last_close_ms + 1.
    """
    import requests
    sess = session or requests.Session()
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
        "startTime": start_ms,
    }
    if end_ms is not None:
        params["endTime"] = end_ms
    resp = sess.get(BINANCE_KLINES_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_klines_range(
    symbol: str, *, start: datetime, end: datetime | None = None,
    interval: str = "1m",
) -> Iterator[list[list]]:
    """Generator yielding klines chunks from ``start`` to ``end``.

    Handles pagination automatically. Yields chunks (lists of klines)
    so callers can stream-write to disk for large pulls.
    """
    import requests
    sess = requests.Session()

    if end is None:
        end = datetime.now(tz=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    n_chunks = 0
    n_bars = 0
    while start_ms < end_ms:
        chunk = fetch_klines_chunk(
            symbol, start_ms=start_ms, end_ms=end_ms,
            interval=interval, limit=1000, session=sess,
        )
        if not chunk:
            break
        n_chunks += 1
        n_bars += len(chunk)
        # Advance to one ms past the last close so we don't re-pull.
        last_close_ms = int(chunk[-1][6])  # close_time_ms
        start_ms = last_close_ms + 1
        yield chunk
        # Polite rate limit: Binance allows 1200 req/min on this endpoint
        # but we don't need to push it — 100ms between requests = 600/min,
        # plenty fast and well within limits.
        time.sleep(0.1)
    logger.info(
        "fetched %d chunks (%d bars) for %s [%s..%s]",
        n_chunks, n_bars, symbol, start.isoformat(), end.isoformat(),
    )


def klines_to_minute_df(
    chunks: list[list[list]] | Iterator[list[list]], symbol: str,
) -> pd.DataFrame:
    """Convert raw klines chunks into the ``aggregate_to_minute`` schema.

    Output columns match what ``feature_pipeline_long_horizon.build_long_horizon_features``
    consumes plus the OFI/spread/depth columns filled with neutral
    zeros so the trainer can swap pipelines without schema errors.
    """
    rows: list[list] = []
    for chunk in chunks:
        rows.extend(chunk)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=KLINES_COLUMNS)
    # Convert numeric columns from strings.
    for col in ("open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base_volume",
                "taker_buy_quote_volume"):
        df[col] = pd.to_numeric(df[col])
    df["n_trades"] = df["n_trades"].astype(int)

    # Map to canonical minute-bar schema.
    out = pd.DataFrame()
    out["minute"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    out["symbol"] = symbol.upper()
    # Use OHLC of last-trade price as our "microprice" proxy. For 1-min
    # bars on liquid pairs this is within 1bp of the true microprice
    # (last trade ≈ midpoint of best bid/ask at end-of-minute).
    out["microprice_open"] = df["open"]
    out["microprice_close"] = df["close"]
    out["microprice_high"] = df["high"]
    out["microprice_low"] = df["low"]
    # n_updates_sum: number of trades in the bar (Binance reports this).
    out["n_updates_sum"] = df["n_trades"]
    # trade_imb_sum: 2*taker_buy_volume / total_volume - 1  ∈ [-1, +1]
    # Same convention as live aggregator's trade_imb.
    total = df["volume"].replace(0, pd.NA)
    out["trade_imb_sum"] = (2 * df["taker_buy_base_volume"] / total - 1).fillna(0.0)
    out["trade_imb_abs_sum"] = out["trade_imb_sum"].abs()
    # Klines doesn't carry OFI / depth / spread — fill neutral so the
    # downstream long_horizon pipeline (which doesn't use them anyway)
    # doesn't choke on missing columns.
    out["ofi_sum"] = 0.0
    out["ofi_mean"] = 0.0
    out["ofi_std"] = 0.0
    out["depth_imb_mean"] = 0.0
    out["depth_imb_std"] = 0.0
    out["depth_imb_last"] = 0.0
    out["spread_bps_mean"] = 0.0
    out["spread_bps_last"] = 0.0
    out["spread_bps_max"] = 0.0
    out["seconds_observed"] = 60  # Klines is always a complete bar.
    return out.sort_values("minute").reset_index(drop=True)


def stream_klines_to_parquet(
    symbol: str, *, start: datetime, end: datetime | None,
    interval: str, out_path,
) -> int:
    """Fetch klines paginated and stream-write each chunk to parquet.

    Bounded memory: ~one chunk (1000 bars) in flight at a time. The
    naive ``klines_to_minute_df(list(fetch_klines_range(...)))``
    accumulates all chunks in a Python list, OOM-killing on Tokyo's
    3.8 GB box at ~1.5M bars.

    Uses pyarrow's ``ParquetWriter`` to append chunks to a single
    parquet file — schema inferred from the first chunk and held
    constant. Returns the total number of bars written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None
    n_total = 0
    try:
        for chunk in fetch_klines_range(
            symbol, start=start, end=end, interval=interval,
        ):
            if not chunk:
                continue
            df = klines_to_minute_df([chunk], symbol)
            if df.empty:
                continue
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                # First chunk → open writer with the inferred schema.
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_total += len(df)
    finally:
        if writer is not None:
            writer.close()
    return n_total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", action="append", default=None,
                   help="symbol to fetch; pass multiple times for "
                        "multi-symbol run. Default: BTCUSDT,ETHUSDT,BNBUSDT.")
    p.add_argument("--years", type=float, default=2.0,
                   help="years of history to download (default 2). "
                        "Capped by listing date — Binance auto-truncates.")
    p.add_argument("--out-dir", default="data/historical",
                   help="directory to write <symbol>_1m_klines.parquet")
    p.add_argument("--interval", default="1m",
                   choices=("1m", "3m", "5m", "15m", "30m", "1h"),
                   help="kline interval; long_horizon trainer uses 1m by default")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    symbols = args.symbol or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=int(args.years * 365.25))

    rc = 0
    for sym in symbols:
        sym = sym.upper()
        logger.info("=" * 60)
        logger.info("downloading %s [%s..%s]",
                    sym, start.isoformat(), end.isoformat())
        out_path = out_dir / f"{sym.lower()}_{args.interval}_klines.parquet"
        try:
            n_total = stream_klines_to_parquet(
                sym, start=start, end=end,
                interval=args.interval, out_path=out_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", sym, exc, exc_info=True)
            rc = 1
            continue
        if n_total == 0:
            logger.warning("%s: no bars returned", sym)
            continue
        logger.info(
            "%s: %d bars → %s (streamed)",
            sym, n_total, out_path,
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
