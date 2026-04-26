"""Daily *backup* of ``paper_trades`` to Yandex S3.

Why this exists
===============

``paper_trades`` is the audit trail of every closed simulated trade —
the data that backs the realized-accuracy logger, the cumulative-P&L
sparkline, and the lifetime-stats badge in the UI. Until now it was
the only HF table NOT being shipped to S3. On a Tokyo-VPS death we'd
lose every paper trade ever simulated — for an academic/research
project the entire defence-grade "vs benchmark" P&L story would
evaporate.

This is a **backup**, not an archive: rows are NOT deleted from
Postgres after upload. Two reasons:

1. The table is tiny — ~3-5k rows/day across 3 symbols, ~50 KB
   compressed Parquet/symbol-day. Hot retention is essentially free.
2. The UI's lifetime-P&L widgets query *all* paper trades; deleting
   them after 7 days would silently truncate "trades since model v3"
   sweeps. Better to keep them all hot.

Cron schedule
-------------

Wired to ``neucast-paper-trades-backup.timer`` (daily 02:30 UTC, 30 min
after L2 archive). One firing per day is plenty given how slowly the
table grows; on Tokyo death we lose at most ~24h of paper trades, which
is acceptable when the system itself is sim-only and the trades are
simulations.

Idempotence
-----------

If S3 already has an object for ``paper_trades_backup/<sym>/<day>.parquet``,
we re-fetch the day's rows from Postgres and:

* If the on-S3 row count equals the on-DB row count → skip (already
  caught up, nothing to do).
* If the on-DB count is HIGHER (e.g. a late-arriving trade got
  written after the previous backup) → overwrite the S3 object.

The "trades arriving after their day closed" case is rare but possible
(timer drift, or a runner that buffered before flushing). The check is
cheap: HEAD + a small SELECT COUNT.

Run manually
------------

::

    python -m tools.archive_paper_trades_to_s3
    python -m tools.archive_paper_trades_to_s3 --dry-run
    python -m tools.archive_paper_trades_to_s3 --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def s3_key_for(symbol: str, day: date) -> str:
    """``paper_trades_backup/btcusdt/2026-04-26.parquet`` style key."""
    return f"paper_trades_backup/{symbol.lower()}/{day.isoformat()}.parquet"


def days_to_backup(
    distinct_days: list[date],
    *,
    today_utc: date,
    grace_days: int = 1,
    max_days: Optional[int] = None,
) -> list[date]:
    """Return days eligible for backup (today minus ``grace_days``).

    ``grace_days=1`` means we only back up FULLY-COMPLETE days — not
    today's still-in-progress trades. Otherwise the row-count
    idempotence check would mis-fire all the time as new trades close.
    """
    cutoff = today_utc - timedelta(days=int(grace_days))
    eligible = sorted(d for d in distinct_days if d <= cutoff)
    if max_days is not None and max_days > 0:
        # Process oldest-first (same as archive cron) so a backlog after
        # downtime catches up methodically.
        eligible = eligible[:max_days]
    return eligible


@dataclass
class BackupStats:
    days_uploaded: int = 0
    days_unchanged: int = 0
    days_failed: int = 0
    rows_uploaded: int = 0
    bytes_uploaded: int = 0
    failed_days: list[tuple[str, date]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# DB + S3 layer
# ──────────────────────────────────────────────────────────────────────


def fetch_distinct_days(conn, symbol: str) -> list[date]:
    """Days where this symbol has at least one closed trade.

    We index on ``exit_ts`` (not ``entry_ts``) because the trade
    "belongs" to the day it closed — that's how the UI groups them
    too (``date_trunc('day', exit_ts AT TIME ZONE 'UTC')``).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date_trunc('day', exit_ts AT TIME ZONE 'UTC')::date AS d "
            "FROM paper_trades WHERE symbol = %s ORDER BY d ASC",
            (symbol,),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_day_rowcount(conn, symbol: str, day: date) -> int:
    next_day = day + timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM paper_trades "
            " WHERE symbol = %s "
            "   AND exit_ts >= %s::timestamptz "
            "   AND exit_ts <  %s::timestamptz",
            (symbol, day.isoformat(), next_day.isoformat()),
        )
        return int(cur.fetchone()[0])


def fetch_day_dataframe(conn, symbol: str, day: date):
    import pandas as pd  # noqa: WPS433

    next_day = day + timedelta(days=1)
    sql = (
        "SELECT id, symbol, side, qty, "
        "       entry_ts, entry_price, entry_prob_up, "
        "       exit_ts, exit_price, exit_reason, "
        "       fee_paid_total_usd, pnl_usd, pnl_bps, "
        "       model_version, written_at "
        "  FROM paper_trades "
        " WHERE symbol = %(s)s "
        "   AND exit_ts >= %(d_start)s::timestamptz "
        "   AND exit_ts <  %(d_end)s::timestamptz "
        " ORDER BY exit_ts ASC"
    )
    return pd.read_sql_query(
        sql, conn,
        params={"s": symbol, "d_start": day.isoformat(), "d_end": next_day.isoformat()},
    )


def s3_object_metadata(s3, bucket: str, key: str) -> tuple[bool, int, dict]:
    """Returns ``(exists, content_length, user_metadata)``. ``user_metadata``
    has the ``rows`` field set on upload — used for the row-count
    idempotence check.
    """
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return (
            True,
            int(resp.get("ContentLength") or 0),
            resp.get("Metadata") or {},
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "not found" in msg or "404" in msg or "nosuchkey" in msg:
            return False, 0, {}
        raise


def upload_dataframe_as_parquet(s3, bucket: str, key: str, df) -> int:
    import pyarrow as pa  # noqa: WPS433
    import pyarrow.parquet as pq  # noqa: WPS433

    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    body = buf.getvalue()
    s3.put_object(
        Bucket=bucket, Key=key, Body=body,
        ContentType="application/x-parquet",
        Metadata={
            "uploaded_at": datetime.now(tz=timezone.utc).isoformat(),
            "source": "tokyo-paper-trades-backup",
            "rows": str(len(df)),
        },
    )
    return len(body)


def backup_one(
    conn, s3, *, bucket: str, symbol: str, day: date,
    dry_run: bool, stats: BackupStats,
) -> None:
    """Idempotent: skip when S3's stored row count == DB's current count.

    There is no DELETE here (this is a backup, not archive), so
    idempotence cuts S3 PUT cost on every retry / re-run.
    """
    key = s3_key_for(symbol, day)
    db_count = fetch_day_rowcount(conn, symbol, day)
    if db_count == 0:
        logger.info("(empty) %s %s — nothing to back up, skipping", symbol, day)
        return

    try:
        exists, _size, meta = s3_object_metadata(s3, bucket, key)
    except Exception as exc:
        logger.warning("HEAD failed for s3://%s/%s: %s", bucket, key, exc)
        stats.days_failed += 1
        stats.failed_days.append((symbol, day))
        return

    if exists:
        try:
            s3_count = int(meta.get("rows", "0"))
        except ValueError:
            s3_count = -1  # malformed metadata → force re-upload
        if s3_count == db_count:
            logger.info(
                "unchanged: s3://%s/%s rows=%d — skipping",
                bucket, key, db_count,
            )
            stats.days_unchanged += 1
            return
        logger.info(
            "row count drift on s3://%s/%s: s3=%d db=%d — re-uploading",
            bucket, key, s3_count, db_count,
        )

    df = fetch_day_dataframe(conn, symbol, day)
    if dry_run:
        logger.info(
            "(dry-run) would upload %d rows to s3://%s/%s",
            len(df), bucket, key,
        )
        stats.rows_uploaded += len(df)
        return

    try:
        n_bytes = upload_dataframe_as_parquet(s3, bucket, key, df)
    except Exception as exc:
        logger.warning("upload failed for s3://%s/%s: %s", bucket, key, exc)
        stats.days_failed += 1
        stats.failed_days.append((symbol, day))
        return

    stats.days_uploaded += 1
    stats.rows_uploaded += len(df)
    stats.bytes_uploaded += n_bytes
    logger.info(
        "backed up %d rows (%d KB) → s3://%s/%s",
        len(df), n_bytes // 1024, bucket, key,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grace-days", type=int, default=1,
                   help="don't back up days more recent than today minus N "
                        "(default 1 — only complete days)")
    p.add_argument("--symbol", default=None,
                   help="restrict to one symbol; default: all")
    p.add_argument("--max-days", type=int, default=60,
                   help="max days to process per run (default 60)")
    p.add_argument("--dry-run", action="store_true",
                   help="log what would happen, but don't upload")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        dsn = os.environ["DATABASE_URL"]
        endpoint = os.environ["YANDEX_S3_ENDPOINT"]
        bucket = os.environ["YANDEX_S3_BUCKET"]
        ak = os.environ["YANDEX_S3_ACCESS_KEY_ID"]
        sk = os.environ["YANDEX_S3_SECRET_ACCESS_KEY"]
        region = os.environ.get("YANDEX_S3_REGION", "ru-central1")
    except KeyError as exc:
        logger.error("missing required env var: %s", exc)
        return 2

    import boto3  # noqa: WPS433
    import psycopg2  # noqa: WPS433

    conn = psycopg2.connect(dsn)
    s3 = boto3.client(
        "s3", endpoint_url=endpoint, region_name=region,
        aws_access_key_id=ak, aws_secret_access_key=sk,
    )

    today_utc = datetime.now(tz=timezone.utc).date()

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM paper_trades ORDER BY symbol")
            symbols = [row[0] for row in cur.fetchall()]

    if not symbols:
        logger.info("no symbols in paper_trades — nothing to back up")
        return 0

    logger.info(
        "paper-trades backup run: symbols=%s grace=%d max_days=%d dry_run=%s",
        symbols, args.grace_days, args.max_days, args.dry_run,
    )

    stats = BackupStats()
    for symbol in symbols:
        try:
            distinct_days = fetch_distinct_days(conn, symbol)
        except Exception as exc:
            logger.warning("could not list days for %s: %s", symbol, exc)
            continue
        eligible = days_to_backup(
            distinct_days,
            today_utc=today_utc,
            grace_days=args.grace_days,
            max_days=args.max_days,
        )
        logger.info("%s: %d days eligible (of %d total)",
                    symbol, len(eligible), len(distinct_days))
        for day in eligible:
            backup_one(
                conn, s3, bucket=bucket, symbol=symbol, day=day,
                dry_run=args.dry_run, stats=stats,
            )

    conn.close()

    logger.info(
        "DONE: uploaded=%d days (%d rows, %d KB) | "
        "unchanged=%d | failed=%d",
        stats.days_uploaded, stats.rows_uploaded, stats.bytes_uploaded // 1024,
        stats.days_unchanged,
        stats.days_failed,
    )

    if stats.days_failed > 0:
        return 1

    from app.highfreq.cron_metrics import write_cron_success
    write_cron_success(
        "neucast_hf_paper_trades_backup_last_success_timestamp_seconds",
        file_stem="neucast_hf_paper_trades_backup",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
