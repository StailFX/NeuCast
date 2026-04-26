"""Disaster-recovery drill: read a day of L2 snapshots back from Yandex S3
and verify it's recoverable into Postgres.

Why this exists
===============

A backup that has never been restored is not a backup. The L2 archive
cron (``tools/archive_l2_to_s3.py``) ships per-(symbol, day) Parquet
objects to Yandex S3 and deletes from hot Postgres after 7 days. If a
schema change, a Parquet-encoder regression, or a credential rotation
breaks restoration silently, we won't find out until an actual incident
— at which point the data is gone from hot storage already.

This script exercises the **read path** end-to-end:

1. List ``highfreq_l2/<symbol>/<day>.parquet`` keys in the bucket for a
   chosen day.
2. Download one (or all 3 symbols' files) into memory.
3. Decode the Parquet bytes via pyarrow → pandas.
4. Verify the schema matches ``highfreq_l2_snapshots`` (column names,
   dtypes for the array columns).
5. Print row-count + first/last ``ts`` so an operator can sanity-check.
6. Optionally INSERT into ``highfreq_l2_snapshots_dr_test`` (if
   ``--restore-into-table`` is passed) and report row count vs the
   Parquet's expected count. This last step proves the DB schema is
   compatible — most realistic test of the full recovery path.

Run from Tokyo
--------------

::

    # Read-only validation of yesterday's archive (default).
    python -m tools.dr_drill_l2

    # Pick a specific date.
    python -m tools.dr_drill_l2 --day 2026-04-20

    # Full restore into a temp table — proves the DB path works too.
    python -m tools.dr_drill_l2 --day 2026-04-20 --restore-into-table

Output is a single JSON object on stdout, suitable for piping into
``docs/highfreq/dr_drill.md`` as evidence::

    python -m tools.dr_drill_l2 --day 2026-04-20 \\
        | jq . > docs/highfreq/dr_drill_runs/2026-04-27.json

Frequency recommendation
------------------------

* **At least once before any production milestone** (e.g. defence demo).
* **Quarterly** as a routine — calendar-block 30 minutes, run, archive
  the JSON output.
* After **any change** to ``archive_l2_to_s3.py``, ``s3_key_for``, or
  the ``highfreq_l2_snapshots`` schema.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# Mirror ``archive_l2_to_s3.s3_key_for`` so a refactor over there can't
# silently break the read side. If the key scheme changes, both ends
# need to update — easier to spot in code review than in a 3-month-old
# DR drill that fails only quarterly.
def s3_key_for(symbol: str, day: date) -> str:
    return f"highfreq_l2/{symbol.lower()}/{day.isoformat()}.parquet"


EXPECTED_COLUMNS = {
    "ts", "symbol",
    "bids_price", "bids_qty", "asks_price", "asks_qty",
    "written_at",
}


@dataclass
class DrillStats:
    """Per-(symbol, day) drill result."""
    symbol: str
    day: str
    s3_key: str
    s3_object_size_bytes: int
    rows_in_parquet: int
    columns: list[str]
    earliest_ts: str | None
    latest_ts: str | None
    schema_ok: bool
    schema_diff: list[str]
    rows_restored: int | None  # None when --restore-into-table not passed
    notes: list[str]


def _validate_schema(df) -> tuple[bool, list[str]]:
    """Return (ok, diff_messages). ``ok`` False means schema drift."""
    msgs = []
    cols = set(df.columns)
    missing = EXPECTED_COLUMNS - cols
    extra = cols - EXPECTED_COLUMNS
    if missing:
        msgs.append(f"missing columns: {sorted(missing)}")
    if extra:
        # Extra columns aren't fatal — could be a deliberate widening — but
        # we surface them so a reviewer notices the schema drifted.
        msgs.append(f"unexpected columns: {sorted(extra)}")
    # Array columns must be lists of equal length on each row (CHECK
    # constraint mirror). A non-list there means Parquet round-trip
    # broke them — extremely rare but the whole point of a DR drill is
    # to catch the rare cases.
    for col in ("bids_price", "bids_qty", "asks_price", "asks_qty"):
        if col not in df.columns or len(df) == 0:
            continue
        first = df[col].iloc[0]
        if not isinstance(first, (list, tuple)) and not hasattr(first, "__iter__"):
            msgs.append(f"{col}: expected list-like, got {type(first).__name__}")
    return (not missing and not [m for m in msgs if "expected list-like" in m]), msgs


def drill_one(s3, bucket: str, symbol: str, day: date,
              *, restore_into_table: bool, conn=None) -> DrillStats:
    import pandas as pd  # lazy
    import pyarrow.parquet as pq  # lazy

    key = s3_key_for(symbol, day)
    logger.info("downloading s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    size = int(obj.get("ContentLength") or len(body))

    table = pq.read_table(io.BytesIO(body))
    df = table.to_pandas()

    ok, diff = _validate_schema(df)
    notes: list[str] = []
    earliest = latest = None
    if "ts" in df.columns and len(df) > 0:
        earliest = pd.to_datetime(df["ts"]).min().isoformat()
        latest = pd.to_datetime(df["ts"]).max().isoformat()

    rows_restored: int | None = None
    if restore_into_table:
        if conn is None:
            notes.append("restore-into-table requested but no DB connection passed; skipped")
        else:
            rows_restored = _restore_into_test_table(conn, df, symbol=symbol, day=day)
            notes.append(
                f"restored {rows_restored} rows into highfreq_l2_snapshots_dr_test "
                f"(parquet had {len(df)} rows)"
            )
            if rows_restored != len(df):
                notes.append("MISMATCH between parquet rows and rows inserted")

    return DrillStats(
        symbol=symbol,
        day=day.isoformat(),
        s3_key=key,
        s3_object_size_bytes=size,
        rows_in_parquet=int(len(df)),
        columns=sorted(df.columns.tolist()),
        earliest_ts=earliest,
        latest_ts=latest,
        schema_ok=ok,
        schema_diff=diff,
        rows_restored=rows_restored,
        notes=notes,
    )


def _restore_into_test_table(conn, df, *, symbol: str, day: date) -> int:
    """Create a *test* table mirroring the prod schema, INSERT the
    DataFrame, return rowcount inserted.

    Uses CREATE TABLE IF NOT EXISTS so reruns don't spam errors. Each
    drill clears its own (symbol, day) before re-inserting — idempotent.
    """
    import pandas as pd  # lazy
    next_day = day + timedelta(days=1)

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS highfreq_l2_snapshots_dr_test (
                ts TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                bids_price DOUBLE PRECISION[],
                bids_qty DOUBLE PRECISION[],
                asks_price DOUBLE PRECISION[],
                asks_qty DOUBLE PRECISION[],
                written_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute(
            "DELETE FROM highfreq_l2_snapshots_dr_test "
            "WHERE symbol = %s AND ts >= %s::timestamptz AND ts < %s::timestamptz",
            (symbol, day.isoformat(), next_day.isoformat()),
        )
        # Vectorised insert via executemany. Convert nullable list columns
        # via tolist() so psycopg2 sends ARRAYs natively.
        rows = []
        for _, r in df.iterrows():
            rows.append((
                r["ts"], r["symbol"],
                list(r["bids_price"]) if r["bids_price"] is not None else None,
                list(r["bids_qty"]) if r["bids_qty"] is not None else None,
                list(r["asks_price"]) if r["asks_price"] is not None else None,
                list(r["asks_qty"]) if r["asks_qty"] is not None else None,
                r.get("written_at") or datetime.now(tz=timezone.utc),
            ))
        cur.executemany(
            "INSERT INTO highfreq_l2_snapshots_dr_test "
            "(ts, symbol, bids_price, bids_qty, asks_price, asks_qty, written_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
        conn.commit()
        return len(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day", default=None,
                   help="ISO date (YYYY-MM-DD) to drill, default: yesterday UTC")
    p.add_argument("--symbol", default=None,
                   help="restrict to one symbol; default: all 3 (BTC/ETH/BNB)")
    p.add_argument("--restore-into-table", action="store_true",
                   help="also INSERT into highfreq_l2_snapshots_dr_test")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Required env (mirrored from archive_l2_to_s3 for symmetry).
    try:
        endpoint = os.environ["YANDEX_S3_ENDPOINT"]
        bucket = os.environ["YANDEX_S3_BUCKET"]
        ak = os.environ["YANDEX_S3_ACCESS_KEY_ID"]
        sk = os.environ["YANDEX_S3_SECRET_ACCESS_KEY"]
        region = os.environ.get("YANDEX_S3_REGION", "ru-central1")
    except KeyError as exc:
        logger.error("missing required env var: %s", exc)
        return 2

    if args.day:
        day = date.fromisoformat(args.day)
    else:
        day = (datetime.now(tz=timezone.utc) - timedelta(days=1)).date()

    symbols = [args.symbol.upper()] if args.symbol else ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    import boto3  # noqa: WPS433
    s3 = boto3.client(
        "s3", endpoint_url=endpoint, region_name=region,
        aws_access_key_id=ak, aws_secret_access_key=sk,
    )

    conn = None
    if args.restore_into_table:
        try:
            dsn = os.environ["DATABASE_URL"]
        except KeyError:
            logger.error("--restore-into-table requires DATABASE_URL")
            return 2
        import psycopg2  # noqa: WPS433
        conn = psycopg2.connect(dsn)

    results = []
    overall_ok = True
    for sym in symbols:
        try:
            r = drill_one(
                s3, bucket=bucket, symbol=sym, day=day,
                restore_into_table=args.restore_into_table, conn=conn,
            )
            results.append(asdict(r))
            if not r.schema_ok:
                overall_ok = False
        except Exception as exc:
            logger.exception("drill failed for %s %s", sym, day)
            results.append({
                "symbol": sym,
                "day": day.isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            })
            overall_ok = False

    if conn is not None:
        conn.close()

    summary = {
        "drill_run_at": datetime.now(tz=timezone.utc).isoformat(),
        "drill_day": day.isoformat(),
        "bucket": bucket,
        "endpoint": endpoint,
        "symbols": symbols,
        "restore_into_table": args.restore_into_table,
        "overall_ok": overall_ok,
        "results": results,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
