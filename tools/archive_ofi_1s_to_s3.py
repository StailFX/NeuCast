"""Daily archival of ``highfreq_ofi_1s`` rows from Postgres → Yandex S3.

Why this exists
===============

``highfreq_ofi_1s`` is the canonical 1-second OFI/microprice/depth
imbalance feed — the only table the trainer reads to fit
``catboost_*_1m.cbm``. Until now, it was the only HF data NOT being
backed up: a Tokyo VPS death meant losing all training data ever
collected, with the only mitigation being "the new ingest will start
collecting again". For an academic/research project that's a hard
sell — every day of unique market microstructure is genuinely
unrecoverable.

This script mirrors the structure of ``tools/archive_l2_to_s3.py``
(verify-before-delete, idempotent, atomic per-(symbol, day) Parquet
upload) but for the OFI table. We DO delete from Postgres after a
successful upload because:

* ~260k rows/day across 3 symbols ≈ 30 MB/day of disk pressure.
* The trainer's default ``--since-hours 72`` means anything older
  than 3 days is already irrelevant for live model fitting.
* Historical recall (e.g. "what did Friday's bars look like?") still
  works via download from S3.

Retention is **7 days** — generous buffer above the trainer's 3-day
window. If you're tuning ``--since-hours`` higher, raise this in
lockstep so the trainer never sees holes.

Cron schedule
-------------

Wired to ``neucast-ofi-archive.timer`` (daily 02:45 UTC, 45 min
after L2 archive so the two don't contend on the same Postgres
maintenance window).

Run manually
------------

::

    # Archive everything older than 7 days, max 30 days per run.
    python -m tools.archive_ofi_1s_to_s3

    # Read-only preview (no S3 upload, no DELETE).
    python -m tools.archive_ofi_1s_to_s3 --dry-run

    # Restrict to one symbol.
    python -m tools.archive_ofi_1s_to_s3 --symbol BTCUSDT
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


# ──────────────────────────────────────────────────────────────────────
# Pure helpers (testable without S3 / Postgres)
# ──────────────────────────────────────────────────────────────────────


def s3_key_for(symbol: str, day: date) -> str:
    """``highfreq_ofi_1s/btcusdt/2026-04-26.parquet`` style key.

    Different prefix from L2 (``highfreq_l2/...``) so a single bucket
    holds both archive types without cross-contamination, and
    ``aws s3 ls`` can scope per-table.
    """
    return f"highfreq_ofi_1s/{symbol.lower()}/{day.isoformat()}.parquet"


def days_to_archive(
    distinct_days: list[date],
    *,
    today_utc: date,
    retention_days: int,
    max_days: Optional[int] = None,
) -> list[date]:
    """Same shape as ``archive_l2_to_s3.days_to_archive`` — see there.

    Re-implemented here (rather than imported) so the two scripts
    can evolve independently if their cadence ever diverges.
    """
    cutoff = today_utc - timedelta(days=int(retention_days))
    eligible = sorted(d for d in distinct_days if d < cutoff)
    if max_days is not None and max_days > 0:
        eligible = eligible[:max_days]
    return eligible


@dataclass
class ArchiveStats:
    """Aggregates per-run counters for the structured DONE log line."""
    days_uploaded: int = 0
    days_already_archived: int = 0
    days_deleted: int = 0
    days_failed: int = 0
    rows_uploaded: int = 0
    rows_deleted: int = 0
    bytes_uploaded: int = 0
    failed_days: list[tuple[str, date]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# DB + S3 layer
# ──────────────────────────────────────────────────────────────────────


def fetch_distinct_days(conn, symbol: str) -> list[date]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date_trunc('day', ts AT TIME ZONE 'UTC')::date AS d "
            "FROM highfreq_ofi_1s WHERE symbol = %s ORDER BY d ASC",
            (symbol,),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_day_dataframe(conn, symbol: str, day: date):
    """Return pandas DataFrame for one (symbol, day). Lazy pandas
    import keeps the pure-helper tests deps-free."""
    import pandas as pd  # noqa: WPS433

    next_day = day + timedelta(days=1)
    sql = (
        "SELECT ts, symbol, ofi, microprice, depth_imb, spread_bps, "
        "       trade_imb, vpin, n_updates, local_recv_ms "
        "  FROM highfreq_ofi_1s "
        " WHERE symbol = %(s)s "
        "   AND ts >= %(d_start)s::timestamptz "
        "   AND ts <  %(d_end)s::timestamptz "
        " ORDER BY ts ASC"
    )
    return pd.read_sql_query(
        sql, conn,
        params={"s": symbol, "d_start": day.isoformat(), "d_end": next_day.isoformat()},
    )


def s3_object_exists(s3, bucket: str, key: str) -> tuple[bool, int]:
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return True, int(resp.get("ContentLength") or 0)
    except Exception as exc:  # boto3 ClientError on 404
        msg = str(exc).lower()
        if "not found" in msg or "404" in msg or "nosuchkey" in msg:
            return False, 0
        raise


def upload_dataframe_as_parquet(s3, bucket: str, key: str, df) -> int:
    """Serialise df → snappy Parquet → PUT to S3. Returns bytes uploaded."""
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
            "source": "tokyo-highfreq-ofi-archive",
            "rows": str(len(df)),
        },
    )
    return len(body)


def delete_day_from_postgres(conn, symbol: str, day: date) -> int:
    """DELETE rows for one (symbol, day). Returns affected count."""
    next_day = day + timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM highfreq_ofi_1s "
            " WHERE symbol = %s AND ts >= %s::timestamptz "
            "                  AND ts <  %s::timestamptz",
            (symbol, day.isoformat(), next_day.isoformat()),
        )
        deleted = cur.rowcount or 0
    conn.commit()
    return deleted


def archive_one(
    conn, s3, *, bucket: str, symbol: str, day: date,
    dry_run: bool, stats: ArchiveStats,
) -> None:
    """Archive (verify-before-delete) one (symbol, day).

    Logic mirrors ``archive_l2_to_s3.archive_one``:
    1. If S3 already has a non-zero object for this key, treat as
       already-archived and proceed straight to DELETE.
    2. Otherwise upload, then verify HEAD reports a non-zero
       ContentLength, then DELETE.
    """
    key = s3_key_for(symbol, day)
    try:
        exists, size = s3_object_exists(s3, bucket, key)
    except Exception as exc:
        logger.warning("HEAD failed for s3://%s/%s: %s", bucket, key, exc)
        stats.days_failed += 1
        stats.failed_days.append((symbol, day))
        return

    if exists and size > 0:
        logger.info("already archived: s3://%s/%s (%d bytes)", bucket, key, size)
        stats.days_already_archived += 1
    else:
        df = fetch_day_dataframe(conn, symbol, day)
        if len(df) == 0:
            logger.info("(empty) %s %s — nothing to upload, skipping", symbol, day)
            return
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
        # Verify the object landed.
        try:
            ok, _ = s3_object_exists(s3, bucket, key)
        except Exception as exc:
            logger.warning("post-upload HEAD failed for s3://%s/%s: %s", bucket, key, exc)
            stats.days_failed += 1
            stats.failed_days.append((symbol, day))
            return
        if not ok:
            logger.warning("post-upload HEAD says object missing: s3://%s/%s", bucket, key)
            stats.days_failed += 1
            stats.failed_days.append((symbol, day))
            return
        stats.days_uploaded += 1
        stats.rows_uploaded += len(df)
        stats.bytes_uploaded += n_bytes
        logger.info(
            "uploaded %d rows (%d KB) → s3://%s/%s",
            len(df), n_bytes // 1024, bucket, key,
        )

    # Delete from Postgres now that S3 has a copy.
    if dry_run:
        logger.info("(dry-run) would DELETE rows for %s %s", symbol, day)
        return
    try:
        deleted = delete_day_from_postgres(conn, symbol, day)
    except Exception as exc:
        logger.warning("DELETE failed for %s %s: %s", symbol, day, exc)
        stats.days_failed += 1
        stats.failed_days.append((symbol, day))
        return
    stats.days_deleted += 1
    stats.rows_deleted += deleted
    logger.info("deleted %d Postgres rows for %s %s", deleted, symbol, day)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retention-days", type=int, default=7,
                   help="archive days older than this (default 7 — matches "
                        "the trainer's ~3-day window with buffer)")
    p.add_argument("--symbol", default=None,
                   help="restrict to one symbol; default: all")
    p.add_argument("--max-days", type=int, default=30,
                   help="max days to process per run (default 30)")
    p.add_argument("--dry-run", action="store_true",
                   help="log what would happen, but don't upload/delete")
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
            cur.execute("SELECT DISTINCT symbol FROM highfreq_ofi_1s ORDER BY symbol")
            symbols = [row[0] for row in cur.fetchall()]

    if not symbols:
        logger.info("no symbols in highfreq_ofi_1s — nothing to archive")
        return 0

    logger.info(
        "ofi-1s archive run: symbols=%s retention=%d max_days=%d dry_run=%s",
        symbols, args.retention_days, args.max_days, args.dry_run,
    )

    stats = ArchiveStats()
    for symbol in symbols:
        try:
            distinct_days = fetch_distinct_days(conn, symbol)
        except Exception as exc:
            logger.warning("could not list days for %s: %s", symbol, exc)
            continue
        eligible = days_to_archive(
            distinct_days,
            today_utc=today_utc,
            retention_days=args.retention_days,
            max_days=args.max_days,
        )
        logger.info("%s: %d days eligible (of %d total)",
                    symbol, len(eligible), len(distinct_days))
        for day in eligible:
            archive_one(
                conn, s3, bucket=bucket, symbol=symbol, day=day,
                dry_run=args.dry_run, stats=stats,
            )

    conn.close()

    logger.info(
        "DONE: uploaded=%d days (%d rows, %d KB) | "
        "already_in_s3=%d | deleted=%d days (%d rows) | failed=%d",
        stats.days_uploaded, stats.rows_uploaded, stats.bytes_uploaded // 1024,
        stats.days_already_archived,
        stats.days_deleted, stats.rows_deleted,
        stats.days_failed,
    )

    if stats.days_failed > 0:
        # Same convention as L2 archive: don't write the heartbeat
        # on partial failure so the cron-stale alert can fire.
        return 1

    from app.highfreq.cron_metrics import write_cron_success
    write_cron_success(
        "neucast_hf_ofi_archive_last_success_timestamp_seconds",
        file_stem="neucast_hf_ofi_archive",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
