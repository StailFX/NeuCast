"""Tests for ``tools.archive_l2_to_s3`` — verify-before-delete archival.

We test:
* Pure helpers (``s3_key_for``, ``days_to_archive``) — no I/O.
* The ``archive_one`` orchestrator with mocked Postgres + S3 — covers
  every branch in the verify-before-delete state machine, including
  the resume-after-partial-failure paths that the cron's atomicity
  promise rests on.

We do NOT exercise actual S3 calls or Postgres — those happen in the
Tokyo deploy verification.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.archive_l2_to_s3 import (
    ArchiveStats,
    archive_one,
    days_to_archive,
    s3_key_for,
)


# ───────────────────────── pure helpers ─────────────────────────


def test_s3_key_for_basic():
    assert s3_key_for("BTCUSDT", date(2026, 4, 26)) == \
        "highfreq_l2/btcusdt/2026-04-26.parquet"


def test_s3_key_for_lowercases_symbol():
    assert s3_key_for("ETHUSDT", date(2026, 1, 1)) == \
        "highfreq_l2/ethusdt/2026-01-01.parquet"


def test_days_to_archive_filters_within_retention():
    """Day=today is NOT archived; day=today-7 IS archived (retention=7)."""
    today = date(2026, 4, 26)
    days = [today - timedelta(days=i) for i in range(10)]
    out = days_to_archive(days, today_utc=today, retention_days=7)
    # Days >=7 days old → 4/26 - 7 = 4/19 and earlier.
    expected = sorted(d for d in days if d < today - timedelta(days=7))
    assert out == expected


def test_days_to_archive_returns_oldest_first():
    today = date(2026, 4, 26)
    days = [date(2026, 4, 18), date(2026, 4, 10), date(2026, 4, 15)]
    out = days_to_archive(days, today_utc=today, retention_days=7)
    assert out == [date(2026, 4, 10), date(2026, 4, 15), date(2026, 4, 18)]


def test_days_to_archive_caps_at_max_days():
    """A 30-day backlog with max_days=5 → only 5 oldest processed."""
    today = date(2026, 4, 26)
    days = [today - timedelta(days=i) for i in range(30, 60)]
    out = days_to_archive(days, today_utc=today, retention_days=7, max_days=5)
    assert len(out) == 5
    # Oldest first.
    assert out[0] < out[-1]


def test_days_to_archive_empty_input_returns_empty():
    assert days_to_archive([], today_utc=date(2026, 4, 26), retention_days=7) == []


# ───────────────────────── archive_one branches ─────────────────────────


@pytest.fixture
def mock_conn():
    """Mock psycopg2 connection with cursor() context manager."""
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    cur.rowcount = 100
    conn.cursor.return_value = cur
    return conn


def _mock_s3_head(exists: bool, size: int = 0):
    """Build a side_effect for s3.head_object that returns or raises."""
    if exists:
        def _side(Bucket, Key):
            return {"ContentLength": size}
        return _side
    else:
        def _side(Bucket, Key):
            raise Exception("404 NoSuchKey")
        return _side


def _mock_df(n_rows: int):
    """A minimal DataFrame just to verify it gets passed through."""
    return pd.DataFrame({
        "ts": pd.date_range("2026-04-19", periods=n_rows, freq="s"),
        "symbol": ["BTCUSDT"] * n_rows,
        "bids_price": [[100.0]] * n_rows,
        "bids_qty": [[1.0]] * n_rows,
        "asks_price": [[100.5]] * n_rows,
        "asks_qty": [[1.0]] * n_rows,
        "written_at": pd.date_range("2026-04-19", periods=n_rows, freq="s"),
    })


def test_archive_one_already_in_s3_deletes_postgres(mock_conn):
    """Idempotent fast-path: S3 already has it → just clean up Postgres."""
    s3 = MagicMock()
    s3.head_object.side_effect = _mock_s3_head(exists=True, size=12345)
    stats = ArchiveStats()

    archive_one(
        mock_conn, s3, bucket="b", symbol="BTCUSDT",
        day=date(2026, 4, 19), dry_run=False, stats=stats,
    )

    # No upload happened.
    s3.put_object.assert_not_called()
    # DELETE happened (cursor.execute called).
    mock_conn.cursor.return_value.execute.assert_called()
    mock_conn.commit.assert_called_once()
    assert stats.days_already_archived == 1
    assert stats.days_deleted == 1
    assert stats.rows_deleted == 100  # mock cursor.rowcount


def test_archive_one_dry_run_skips_postgres_delete(mock_conn):
    """Dry-run with S3 hit must NOT touch Postgres."""
    s3 = MagicMock()
    s3.head_object.side_effect = _mock_s3_head(exists=True, size=12345)
    stats = ArchiveStats()

    archive_one(
        mock_conn, s3, bucket="b", symbol="BTCUSDT",
        day=date(2026, 4, 19), dry_run=True, stats=stats,
    )

    s3.put_object.assert_not_called()
    mock_conn.commit.assert_not_called()
    assert stats.days_already_archived == 1
    assert stats.days_deleted == 0  # dry-run: not actually deleted


def test_archive_one_uploads_then_verifies_then_deletes(mock_conn):
    """Happy path: not in S3 → SELECT → upload → HEAD verifies → DELETE."""
    s3 = MagicMock()
    # First HEAD: not exists. Second HEAD (verify): exists with right size.
    head_calls = {"n": 0}
    def head_side(Bucket, Key):
        head_calls["n"] += 1
        if head_calls["n"] == 1:
            raise Exception("404 NoSuchKey")
        return {"ContentLength": 999}  # matches what put_object will report
    s3.head_object.side_effect = head_side

    # Make put_object record the body size we'd send via our mock upload helper.
    captured = {}
    def put_side(Bucket, Key, Body, **kw):
        captured["body_len"] = len(Body)
    s3.put_object.side_effect = put_side

    df = _mock_df(50)
    with patch("tools.archive_l2_to_s3.fetch_day_dataframe", return_value=df), \
         patch("tools.archive_l2_to_s3.upload_dataframe_as_parquet", return_value=999):
        stats = ArchiveStats()
        archive_one(
            mock_conn, s3, bucket="b", symbol="BTCUSDT",
            day=date(2026, 4, 19), dry_run=False, stats=stats,
        )

    assert stats.days_uploaded == 1
    assert stats.rows_uploaded == 50
    assert stats.bytes_uploaded == 999
    assert stats.days_deleted == 1
    mock_conn.commit.assert_called_once()


def test_archive_one_size_mismatch_skips_delete(mock_conn):
    """Post-upload HEAD returns wrong size → DO NOT delete from Postgres."""
    s3 = MagicMock()
    head_calls = {"n": 0}
    def head_side(Bucket, Key):
        head_calls["n"] += 1
        if head_calls["n"] == 1:
            raise Exception("404 NoSuchKey")
        return {"ContentLength": 12345}  # WRONG — claims to differ from upload
    s3.head_object.side_effect = head_side

    df = _mock_df(10)
    with patch("tools.archive_l2_to_s3.fetch_day_dataframe", return_value=df), \
         patch("tools.archive_l2_to_s3.upload_dataframe_as_parquet", return_value=999):
        stats = ArchiveStats()
        archive_one(
            mock_conn, s3, bucket="b", symbol="BTCUSDT",
            day=date(2026, 4, 19), dry_run=False, stats=stats,
        )

    # Verify failed → uploaded counter does NOT increment, days_failed does.
    # Reasoning: "uploaded" means "uploaded AND verified" — so we get an
    # honest "what's actually safe in S3" count from this metric.
    # Postgres is untouched, next run sees object missing and retries upload.
    assert stats.days_uploaded == 0
    assert stats.days_deleted == 0
    assert stats.days_failed == 1
    mock_conn.commit.assert_not_called()


def test_archive_one_upload_failure_skips_delete(mock_conn):
    """PUT throws → don't even attempt DELETE."""
    s3 = MagicMock()
    s3.head_object.side_effect = _mock_s3_head(exists=False)

    df = _mock_df(10)
    with patch("tools.archive_l2_to_s3.fetch_day_dataframe", return_value=df), \
         patch("tools.archive_l2_to_s3.upload_dataframe_as_parquet",
               side_effect=RuntimeError("network down")):
        stats = ArchiveStats()
        archive_one(
            mock_conn, s3, bucket="b", symbol="BTCUSDT",
            day=date(2026, 4, 19), dry_run=False, stats=stats,
        )

    assert stats.days_uploaded == 0
    assert stats.days_deleted == 0
    assert stats.days_failed == 1
    mock_conn.commit.assert_not_called()


def test_archive_one_empty_dataframe_no_op(mock_conn):
    """Day was in distinct_days but SELECT returns 0 rows (race/mismatch)."""
    s3 = MagicMock()
    s3.head_object.side_effect = _mock_s3_head(exists=False)

    with patch("tools.archive_l2_to_s3.fetch_day_dataframe", return_value=pd.DataFrame()):
        stats = ArchiveStats()
        archive_one(
            mock_conn, s3, bucket="b", symbol="BTCUSDT",
            day=date(2026, 4, 19), dry_run=False, stats=stats,
        )

    assert stats.days_uploaded == 0
    assert stats.days_deleted == 0
    assert stats.days_failed == 0  # 0 rows is a no-op, not failure
    s3.put_object.assert_not_called()


def test_archive_one_head_check_fails_skips_run(mock_conn):
    """Initial HEAD throws (auth/network) → skip without uploading."""
    s3 = MagicMock()
    s3.head_object.side_effect = RuntimeError("auth error")

    stats = ArchiveStats()
    archive_one(
        mock_conn, s3, bucket="b", symbol="BTCUSDT",
        day=date(2026, 4, 19), dry_run=False, stats=stats,
    )

    s3.put_object.assert_not_called()
    mock_conn.commit.assert_not_called()
    assert stats.days_failed == 1
