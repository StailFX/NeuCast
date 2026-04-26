"""Tests for ``tools.archive_paper_trades_to_s3``.

Pure-helper tests pin:

* The S3 key shape (``paper_trades_backup/<sym>/<day>.parquet``).
* ``days_to_backup`` — note this uses ``grace_days`` semantics, NOT
  ``retention_days``: backups are produced for any *complete* day
  (today minus grace), and we **never delete** from Postgres. The
  comparison is ``<=`` (inclusive) — backup the cutoff day itself.

The idempotence path (HEAD says metadata.rows == DB count → skip)
is exercised by ``backup_one`` which we cover with mock S3 + DB.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from tools.archive_paper_trades_to_s3 import (
    backup_one,
    BackupStats,
    days_to_backup,
    s3_key_for,
)


# ───── key shape ─────


def test_s3_key_for_basic():
    """Distinct prefix (``paper_trades_backup``) so it can't collide
    with the L2 or OFI archives in the same bucket."""
    assert s3_key_for("BTCUSDT", date(2026, 4, 26)) == \
        "paper_trades_backup/btcusdt/2026-04-26.parquet"


def test_s3_key_for_lowercases_symbol():
    assert s3_key_for("BNBUSDT", date(2026, 1, 1)) == \
        "paper_trades_backup/bnbusdt/2026-01-01.parquet"


# ───── days_to_backup ─────


def test_days_to_backup_includes_cutoff_day():
    """Critical contract: ``<=`` cutoff. The day exactly at
    ``today - grace_days`` IS eligible — it's a complete-yesterday
    that would otherwise be missed forever (the ``<`` form would
    skip it permanently)."""
    today = date(2026, 4, 27)
    days = [
        date(2026, 4, 25),  # 2 days ago → eligible
        date(2026, 4, 26),  # yesterday (cutoff) → eligible (default grace=1)
        date(2026, 4, 27),  # today → NOT eligible (still in progress)
    ]
    out = days_to_backup(days, today_utc=today, grace_days=1)
    assert out == [date(2026, 4, 25), date(2026, 4, 26)]


def test_days_to_backup_excludes_today():
    """Today's still-in-progress trades must NOT be backed up — the
    row-count idempotence check would mis-fire as new trades close
    through the day."""
    today = date(2026, 4, 27)
    days = [today]
    out = days_to_backup(days, today_utc=today, grace_days=1)
    assert out == []


def test_days_to_backup_zero_grace_includes_today():
    """``grace_days=0`` lets the operator force a backup of today's
    partial data (e.g. emergency before maintenance)."""
    today = date(2026, 4, 27)
    days = [today, date(2026, 4, 26)]
    out = days_to_backup(days, today_utc=today, grace_days=0)
    # cutoff = today - 0 = today → today IS eligible (<=).
    assert today in out


def test_days_to_backup_returns_oldest_first():
    today = date(2026, 4, 27)
    days = [date(2026, 4, 20), date(2026, 4, 10), date(2026, 4, 25)]
    out = days_to_backup(days, today_utc=today, grace_days=1)
    assert out == [date(2026, 4, 10), date(2026, 4, 20), date(2026, 4, 25)]


def test_days_to_backup_caps_at_max_days():
    today = date(2026, 4, 27)
    days = [date(2026, 4, d) for d in range(1, 11)]  # 10 days, all complete
    out = days_to_backup(days, today_utc=today, grace_days=1, max_days=3)
    assert out == [date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3)]


def test_days_to_backup_empty_input_returns_empty():
    assert days_to_backup([], today_utc=date(2026, 4, 27), grace_days=1) == []


# ───── backup_one idempotence ─────


@pytest.fixture
def mock_conn():
    """psycopg2-shaped mock with a cursor + commit."""
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cur
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cur


def test_backup_one_skips_when_db_empty(mock_conn):
    """No trades closed on that day → don't even HEAD S3, just bail.
    Important: most operators have weekend/quiet days where a symbol
    closed zero trades."""
    conn, cur = mock_conn
    cur.fetchone.return_value = (0,)
    s3 = MagicMock()
    stats = BackupStats()

    backup_one(conn, s3, bucket="b", symbol="BTCUSDT",
               day=date(2026, 4, 26), dry_run=False, stats=stats)

    s3.head_object.assert_not_called()
    s3.put_object.assert_not_called()
    assert stats.days_uploaded == 0
    assert stats.days_failed == 0


def test_backup_one_skips_when_s3_row_count_matches_db(mock_conn):
    """The pure-idempotence path: HEAD says metadata.rows == DB count
    → no upload. This is what saves us on a cron that fires every
    day on a backlog of 30 days — we don't re-upload everything every
    time."""
    conn, cur = mock_conn
    cur.fetchone.return_value = (42,)  # DB count = 42
    s3 = MagicMock()
    s3.head_object.return_value = {
        "ContentLength": 1024,
        "Metadata": {"rows": "42"},  # S3 count = 42
    }
    stats = BackupStats()

    backup_one(conn, s3, bucket="b", symbol="BTCUSDT",
               day=date(2026, 4, 26), dry_run=False, stats=stats)

    s3.put_object.assert_not_called()
    assert stats.days_unchanged == 1
    assert stats.days_uploaded == 0


def test_backup_one_re_uploads_on_row_count_drift(mock_conn, monkeypatch):
    """A late-arriving trade (e.g. runner buffered before flushing)
    can land in a 'closed' day after the previous backup. The drift
    check catches this and re-uploads."""
    conn, cur = mock_conn
    cur.fetchone.return_value = (50,)  # DB now has 50 (was 42 last backup)
    s3 = MagicMock()
    s3.head_object.return_value = {
        "ContentLength": 1024,
        "Metadata": {"rows": "42"},  # stale
    }

    # Monkeypatch the heavy fetch + upload path to avoid pandas/pyarrow.
    import tools.archive_paper_trades_to_s3 as mod
    monkeypatch.setattr(mod, "fetch_day_dataframe", lambda *a, **k: __import__("pandas").DataFrame({"x": range(50)}))
    monkeypatch.setattr(mod, "upload_dataframe_as_parquet", lambda *a, **k: 2048)

    stats = BackupStats()
    backup_one(conn, s3, bucket="b", symbol="BTCUSDT",
               day=date(2026, 4, 26), dry_run=False, stats=stats)
    assert stats.days_uploaded == 1
    assert stats.rows_uploaded == 50
    assert stats.days_unchanged == 0


def test_backup_one_uploads_when_s3_missing(mock_conn, monkeypatch):
    """First-time backup of a day: HEAD 404 → upload."""
    conn, cur = mock_conn
    cur.fetchone.return_value = (10,)
    s3 = MagicMock()

    # Make HEAD raise a 404-shaped exception.
    class _NotFound(Exception):
        pass
    s3.head_object.side_effect = _NotFound("NoSuchKey")

    import tools.archive_paper_trades_to_s3 as mod
    monkeypatch.setattr(mod, "fetch_day_dataframe", lambda *a, **k: __import__("pandas").DataFrame({"x": range(10)}))
    monkeypatch.setattr(mod, "upload_dataframe_as_parquet", lambda *a, **k: 512)

    stats = BackupStats()
    backup_one(conn, s3, bucket="b", symbol="BTCUSDT",
               day=date(2026, 4, 26), dry_run=False, stats=stats)
    assert stats.days_uploaded == 1
    assert stats.rows_uploaded == 10


def test_backup_one_dry_run_skips_upload(mock_conn, monkeypatch):
    """``--dry-run`` must NEVER mutate S3."""
    conn, cur = mock_conn
    cur.fetchone.return_value = (10,)
    s3 = MagicMock()

    class _NotFound(Exception):
        pass
    s3.head_object.side_effect = _NotFound("NoSuchKey")

    import tools.archive_paper_trades_to_s3 as mod
    monkeypatch.setattr(mod, "fetch_day_dataframe", lambda *a, **k: __import__("pandas").DataFrame({"x": range(10)}))

    stats = BackupStats()
    backup_one(conn, s3, bucket="b", symbol="BTCUSDT",
               day=date(2026, 4, 26), dry_run=True, stats=stats)

    s3.put_object.assert_not_called()
    assert stats.days_uploaded == 0
    assert stats.rows_uploaded == 10  # tracked for "would have uploaded N" log


def test_backup_one_handles_malformed_metadata(mock_conn, monkeypatch):
    """If a previous version wrote a malformed ``rows`` metadata
    (e.g. missing or non-integer), the script must NOT crash — fall
    back to "force re-upload" rather than silently skipping."""
    conn, cur = mock_conn
    cur.fetchone.return_value = (10,)
    s3 = MagicMock()
    s3.head_object.return_value = {
        "ContentLength": 1024,
        "Metadata": {"rows": "not-an-int"},
    }

    import tools.archive_paper_trades_to_s3 as mod
    monkeypatch.setattr(mod, "fetch_day_dataframe", lambda *a, **k: __import__("pandas").DataFrame({"x": range(10)}))
    monkeypatch.setattr(mod, "upload_dataframe_as_parquet", lambda *a, **k: 512)

    stats = BackupStats()
    backup_one(conn, s3, bucket="b", symbol="BTCUSDT",
               day=date(2026, 4, 26), dry_run=False, stats=stats)
    assert stats.days_uploaded == 1  # forced re-upload, no crash
