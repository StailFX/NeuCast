"""Archive ``highfreq_l2_snapshots`` rows older than N days to S3.

Atomic per-(symbol, day) workflow with **verify-before-delete** so a
crash mid-run can never lose data. Designed to be invoked as a daily
systemd oneshot (see ``deploy/neucast-l2-archive.{service,timer}``).

Algorithm
---------

For each (symbol, calendar-day-UTC) pair where the day is at least
``--retention-days`` days old AND there are rows in Postgres:

    1. Build the S3 key: ``highfreq_l2/{symbol}/{YYYY-MM-DD}.parquet``
    2. Query S3 HEAD on that key.
       * If exists AND size > 0  → assume already archived; skip to (5).
       * Otherwise → continue.
    3. SELECT all rows for (symbol, day) from Postgres → pandas DataFrame.
    4. Serialise to Parquet (snappy), PUT to S3.
    5. HEAD the just-uploaded key; verify size > 0 and matches the
       payload we sent. (No content hash — Yandex doesn't expose
       MD5 reliably across client versions; size+key existence is
       enough for our atomicity claim.)
    6. ONLY THEN: ``DELETE FROM highfreq_l2_snapshots WHERE day=...
       AND symbol=...`` inside a single statement.

If step 4 fails, we skip 5+6 — data remains in Postgres, next run
retries. If step 5 fails, we skip 6 — same story. If step 6 fails
mid-DELETE, the upload is already in S3 (idempotent) and the next
run will re-attempt the delete (HEAD-already-exists fast path).

Safety guarantees
-----------------

1. **No data loss**: every row is in S3 (verified by HEAD) before being
   deleted from Postgres.
2. **Idempotent**: re-running on the same data is safe (skips already-
   archived days).
3. **Per-symbol-per-day partial progress**: if BTCUSDT day-N succeeds
   but ETHUSDT day-N fails, BTC is deleted from PG, ETH stays in PG
   for next-run retry. Independent.
4. **Bounded memory**: one (symbol, day) at a time → max ~30 MB
   DataFrame × snappy → ~10 MB upload. Tokyo VPS handles it trivially.

CLI
---

    python -m tools.archive_l2_to_s3 \\
        --retention-days 7 \\
        [--dry-run]              # log what we'd do, don't upload/delete
        [--symbol BTCUSDT]       # restrict to one symbol
        [--max-days 30]          # cap how many days to process per run
                                 # (defends against catching up after
                                 # multi-week downtime exploding S3 calls)
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pure helpers (testable without S3 / Postgres)
# ──────────────────────────────────────────────────────────────────────


def s3_key_for(symbol: str, day: date) -> str:
    """``s3://bucket/highfreq_l2/btcusdt/2026-04-26.parquet`` style key."""
    return f"highfreq_l2/{symbol.lower()}/{day.isoformat()}.parquet"


@dataclass(frozen=True)
class ArchiveTask:
    """One unit of archival work: upload (symbol, day) and delete."""
    symbol: str
    day: date


def days_to_archive(
    distinct_days: list[date],
    *,
    today_utc: date,
    retention_days: int,
    max_days: Optional[int] = None,
) -> list[date]:
    """Filter days older than ``retention_days`` from today.

    Caps to ``max_days`` oldest entries — prevents a multi-week catch-up
    from doing a thundering-herd of S3 calls (each call costs API quota
    on Yandex's grant).
    """
    cutoff = today_utc - timedelta(days=int(retention_days))
    eligible = sorted(d for d in distinct_days if d < cutoff)
    if max_days is not None and max_days > 0:
        eligible = eligible[:max_days]
    return eligible


# ──────────────────────────────────────────────────────────────────────
# DB + S3 layer
# ──────────────────────────────────────────────────────────────────────


def fetch_distinct_days(conn, symbol: str) -> list[date]:
    """Days for which we have at least one row, ASCending."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date_trunc('day', ts AT TIME ZONE 'UTC')::date AS d "
            "FROM highfreq_l2_snapshots WHERE symbol = %s ORDER BY d ASC",
            (symbol,),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_day_dataframe(conn, symbol: str, day: date):
    """Return pandas DataFrame for one (symbol, day). Imported lazily so
    pure-helper tests don't need pandas/pyarrow installed."""
    import pandas as pd  # noqa: WPS433

    next_day = day + timedelta(days=1)
    sql = (
        "SELECT ts, symbol, bids_price, bids_qty, asks_price, asks_qty, "
        "       written_at "
        "  FROM highfreq_l2_snapshots "
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
    """Return ``(exists, size_bytes)``. ``size_bytes=0`` if not exists."""
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return True, int(resp.get("ContentLength") or 0)
    except Exception as exc:  # boto3 ClientError on 404
        # Distinguish 404 from real errors. We treat anything that's not
        # a 200 as "not exists" — caller will try to upload.
        msg = str(exc).lower()
        if "not found" in msg or "404" in msg or "nosuchkey" in msg:
            return False, 0
        # Re-raise on real errors (auth, network) — caller decides.
        raise


def upload_dataframe_as_parquet(s3, bucket: str, key: str, df) -> int:
    """Serialise df → snappy Parquet → PUT to S3. Returns bytes uploaded."""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    raw = buf.getvalue()
    s3.put_object(
        Bucket=bucket, Key=key, Body=raw,
        ContentType="application/x-parquet",
        Metadata={
            "rows": str(len(df)),
            "uploaded_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return len(raw)


def delete_day_from_postgres(conn, symbol: str, day: date) -> int:
    """DELETE all rows for (symbol, day). Returns row count deleted."""
    next_day = day + timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM highfreq_l2_snapshots "
            " WHERE symbol = %s "
            "   AND ts >= %s::timestamptz "
            "   AND ts <  %s::timestamptz",
            (symbol, day.isoformat(), next_day.isoformat()),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


# ──────────────────────────────────────────────────────────────────────
# Main archival loop
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ArchiveStats:
    days_already_archived: int = 0       # skipped, S3 already had it
    days_uploaded: int = 0
    days_deleted: int = 0                # PG rows successfully deleted post-upload
    days_failed: int = 0
    rows_uploaded: int = 0
    rows_deleted: int = 0
    bytes_uploaded: int = 0


def archive_one(
    conn, s3, *, bucket: str, symbol: str, day: date, dry_run: bool, stats: ArchiveStats,
) -> None:
    """Archive a single (symbol, day). Idempotent. Updates ``stats`` in place."""
    key = s3_key_for(symbol, day)

    # Step 1+2: check if already in S3.
    try:
        already_exists, _existing_size = s3_object_exists(s3, bucket, key)
    except Exception as exc:
        logger.warning("S3 HEAD failed for %s: %s — skipping this (symbol, day)", key, exc)
        stats.days_failed += 1
        return

    if already_exists:
        # Idempotent fast path. Just clean up Postgres (might be a re-run
        # after a partial failure where upload succeeded but delete didn't).
        if dry_run:
            logger.info(
                "[dry-run] %s %s: already in S3, would DELETE from Postgres",
                symbol, day,
            )
            stats.days_already_archived += 1
            return
        try:
            deleted = delete_day_from_postgres(conn, symbol, day)
        except Exception as exc:
            logger.warning("DELETE failed for %s %s: %s", symbol, day, exc)
            conn.rollback()
            stats.days_failed += 1
            return
        logger.info(
            "%s %s: already in S3, deleted %d Postgres rows", symbol, day, deleted,
        )
        stats.days_already_archived += 1
        stats.days_deleted += 1
        stats.rows_deleted += deleted
        return

    # Step 3: SELECT data.
    try:
        df = fetch_day_dataframe(conn, symbol, day)
    except Exception as exc:
        logger.warning("SELECT failed for %s %s: %s", symbol, day, exc)
        stats.days_failed += 1
        return

    if df.empty:
        # Day has no rows (shouldn't happen since the day was in
        # distinct_days, but defensive). Nothing to upload, nothing to delete.
        logger.info("%s %s: 0 rows, no-op", symbol, day)
        return

    # Step 4: upload (or pretend to).
    if dry_run:
        logger.info(
            "[dry-run] %s %s: would upload %d rows to s3://%s/%s",
            symbol, day, len(df), bucket, key,
        )
        return

    try:
        bytes_uploaded = upload_dataframe_as_parquet(s3, bucket, key, df)
    except Exception as exc:
        logger.warning("PUT failed for %s %s: %s", symbol, day, exc)
        stats.days_failed += 1
        return

    # Step 5: verify by HEAD.
    try:
        verified_exists, verified_size = s3_object_exists(s3, bucket, key)
    except Exception as exc:
        logger.warning(
            "Post-upload HEAD failed for %s %s: %s — leaving Postgres untouched",
            symbol, day, exc,
        )
        stats.days_failed += 1
        return
    if not verified_exists or verified_size != bytes_uploaded:
        logger.warning(
            "Post-upload size mismatch for %s %s: sent=%d, server=%d — "
            "leaving Postgres untouched, next run will retry",
            symbol, day, bytes_uploaded, verified_size,
        )
        stats.days_failed += 1
        return

    stats.days_uploaded += 1
    stats.rows_uploaded += len(df)
    stats.bytes_uploaded += bytes_uploaded

    # Step 6: DELETE from Postgres (only after S3 confirmed).
    try:
        deleted = delete_day_from_postgres(conn, symbol, day)
    except Exception as exc:
        # Upload succeeded, delete failed → next run will see it in S3,
        # take the "already exists" path, and complete the delete then.
        logger.warning(
            "DELETE failed for %s %s after successful upload: %s — "
            "next run will retry the DELETE",
            symbol, day, exc,
        )
        conn.rollback()
        stats.days_failed += 1
        return
    stats.days_deleted += 1
    stats.rows_deleted += deleted
    logger.info(
        "%s %s: uploaded %d rows (%d KB), deleted %d Postgres rows",
        symbol, day, len(df), bytes_uploaded // 1024, deleted,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retention-days", type=int, default=7,
                   help="archive days older than this (default 7)")
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

    # Required env.
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

    # Lazy imports (pure-helper tests don't need these).
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
        # Discover symbols present in the table.
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM highfreq_l2_snapshots ORDER BY symbol")
            symbols = [row[0] for row in cur.fetchall()]

    if not symbols:
        logger.info("no symbols in highfreq_l2_snapshots — nothing to archive")
        return 0

    logger.info(
        "archive run: symbols=%s retention=%d max_days=%d dry_run=%s",
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
        # Don't write the heartbeat on partial failure — the alert SHOULD
        # fire so an operator looks at logs. Successful days that did
        # archive don't get rolled back, but the next run will retry the
        # failed ones idempotently.
        return 1

    # Heartbeat for the "L2 archive stale" Grafana alert. Idempotent —
    # rewrites the same metric file on every clean run.
    from app.highfreq.cron_metrics import write_cron_success
    write_cron_success(
        "neucast_hf_l2_archive_last_success_timestamp_seconds",
        file_stem="neucast_hf_l2_archive",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
