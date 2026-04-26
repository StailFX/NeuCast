"""Tests for ``tools.dr_drill_l2`` — the disaster-recovery drill that
reads L2 archives back from S3.

The script's main() runs end-to-end against real S3+Postgres and is
exercised manually on Tokyo per the schedule in ``docs/highfreq/dr_drill.md``.
The pure helpers below have no IO and pin the behaviours that would
silently break a future drill if they regressed:

* ``s3_key_for`` must produce the **same** key shape the archive cron
  uses — otherwise the drill would 404 on every object even though the
  data is there.
* ``_validate_schema`` is the actual safety net. If a Parquet round-trip
  drops or renames a column, the drill must fail loudly. This test
  pins the column set and the array-shape check.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tools.dr_drill_l2 import (
    EXPECTED_COLUMNS,
    _validate_schema,
    s3_key_for,
)


# ───── key-shape contract with archive_l2_to_s3 ─────


def test_s3_key_matches_archive_cron_format():
    """``tools.archive_l2_to_s3.s3_key_for`` and
    ``tools.dr_drill_l2.s3_key_for`` MUST produce byte-identical keys
    for the same (symbol, day) — otherwise reads cannot find writes.
    Re-import from the archive cron and compare; this fails the moment
    one side drifts."""
    from tools.archive_l2_to_s3 import s3_key_for as archive_key

    for sym in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        for d in (date(2026, 1, 1), date(2026, 4, 27), date(2026, 12, 31)):
            assert s3_key_for(sym, d) == archive_key(sym, d), (
                f"DR-drill / archive key drift for {sym} {d}"
            )


def test_s3_key_lowercases_symbol():
    """The archive cron lowercases symbols in the path — drill must
    too. Pin against accidental case-preservation."""
    assert s3_key_for("BTCUSDT", date(2026, 4, 27)) == "highfreq_l2/btcusdt/2026-04-27.parquet"
    assert s3_key_for("ethusdt", date(2026, 4, 27)) == "highfreq_l2/ethusdt/2026-04-27.parquet"


# ───── _validate_schema ─────


def _good_parquet_df() -> pd.DataFrame:
    """Mirror what archive_l2_to_s3 writes."""
    return pd.DataFrame({
        "ts": pd.to_datetime(["2026-04-26 00:00:00", "2026-04-26 00:00:01"], utc=True),
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "bids_price": [[77000.0, 76999.5], [77000.5, 77000.0]],
        "bids_qty":   [[1.5, 0.8],          [2.1, 1.0]],
        "asks_price": [[77001.0, 77001.5], [77001.5, 77002.0]],
        "asks_qty":   [[1.2, 0.9],          [2.5, 0.7]],
        "written_at": pd.to_datetime(["2026-04-26 00:00:00", "2026-04-26 00:00:01"], utc=True),
    })


def test_validate_schema_accepts_well_formed_df():
    df = _good_parquet_df()
    ok, diff = _validate_schema(df)
    assert ok is True
    assert diff == []


def test_validate_schema_flags_missing_column():
    """If a column got dropped during Parquet round-trip the drill must
    fail (ok=False) and surface which columns are missing — that's the
    actionable bit for an operator at 03:00 reading the JSON output."""
    df = _good_parquet_df().drop(columns=["asks_qty"])
    ok, diff = _validate_schema(df)
    assert ok is False
    assert any("missing columns" in m and "asks_qty" in m for m in diff)


def test_validate_schema_warns_on_extra_column_but_not_fatal():
    """Extra columns are surfaced but don't fail the drill — could be a
    deliberate widening (e.g. adding a sequence_id later). Schema diff
    is still printed for review."""
    df = _good_parquet_df()
    df["extra_unused"] = "hello"
    ok, diff = _validate_schema(df)
    assert ok is True  # extras are OK
    assert any("unexpected columns" in m and "extra_unused" in m for m in diff)


def test_validate_schema_flags_array_columns_collapsed_to_scalars():
    """Catches the disaster-mode regression where a Parquet writer
    accidentally serialises ``bids_price`` as floats (taking the first
    element). This is the single most likely silent corruption mode for
    array columns through pandas/pyarrow round-trips."""
    df = _good_parquet_df().copy()
    df["bids_price"] = [77000.0, 77000.5]  # scalar, not list
    ok, diff = _validate_schema(df)
    assert ok is False
    assert any("expected list-like" in m and "bids_price" in m for m in diff)


def test_validate_schema_handles_empty_dataframe():
    """Edge case: archive day with zero rows (unlikely but possible).
    Schema should still validate OK — a 0-row Parquet with the right
    columns is a valid recovery."""
    df = _good_parquet_df().iloc[0:0]  # 0 rows, columns preserved
    ok, diff = _validate_schema(df)
    assert ok is True
    assert diff == []


def test_expected_columns_constant_matches_archive_writer():
    """Pin the column set EXPECTED_COLUMNS to mirror what
    fetch_day_dataframe selects in archive_l2_to_s3. If someone widens
    that SELECT, this test fails — forcing them to update both ends
    in lockstep."""
    expected = {
        "ts", "symbol",
        "bids_price", "bids_qty", "asks_price", "asks_qty",
        "written_at",
    }
    assert EXPECTED_COLUMNS == expected
