"""Tests for ``tools.archive_ofi_1s_to_s3`` — pure helpers only.

Same testing strategy as ``tests/test_archive_l2_to_s3.py``: pin the
key shape and the eligibility filter so they cannot drift in a
refactor without us noticing. The IO-heavy ``archive_one`` /
``main`` paths are exercised by the smoke test on the VPS.
"""
from __future__ import annotations

from datetime import date

from tools.archive_ofi_1s_to_s3 import (
    days_to_archive,
    s3_key_for,
)


# ───── s3 key shape ─────


def test_s3_key_for_basic():
    """The OFI prefix is INTENTIONALLY distinct from L2's
    ``highfreq_l2/...`` so a single bucket can hold both archive
    types without name collision and ``aws s3 ls`` can scope per-table."""
    assert s3_key_for("BTCUSDT", date(2026, 4, 26)) == \
        "highfreq_ofi_1s/btcusdt/2026-04-26.parquet"


def test_s3_key_for_lowercases_symbol():
    """Match the lower-case convention used everywhere else (.cbm,
    L2 archive, weights filenames)."""
    assert s3_key_for("ETHUSDT", date(2026, 1, 1)) == \
        "highfreq_ofi_1s/ethusdt/2026-01-01.parquet"


def test_s3_key_for_distinct_from_l2():
    """Pin the prefix difference so a 'helpful' refactor that unifies
    keys can't silently break disaster-recovery for one or the other."""
    from tools.archive_l2_to_s3 import s3_key_for as l2_key
    assert s3_key_for("BTCUSDT", date(2026, 4, 26)) != l2_key("BTCUSDT", date(2026, 4, 26))


# ───── days_to_archive ─────


def test_days_to_archive_filters_within_retention():
    """Days strictly older than ``today - retention_days`` get
    archived; everything within retention is kept hot."""
    today = date(2026, 4, 27)
    days = [
        date(2026, 4, 19),  # 8 days old → archive
        date(2026, 4, 21),  # 6 days old → keep
        date(2026, 4, 26),  # 1 day old → keep
    ]
    out = days_to_archive(days, today_utc=today, retention_days=7)
    assert out == [date(2026, 4, 19)]


def test_days_to_archive_returns_oldest_first():
    """Catch-up runs after downtime should process days in
    chronological order — easier on Postgres locks (no skipping
    around the index) and on operator log-reading."""
    today = date(2026, 4, 27)
    days = [date(2026, 4, 10), date(2026, 4, 5), date(2026, 4, 18)]
    out = days_to_archive(days, today_utc=today, retention_days=7)
    assert out == [date(2026, 4, 5), date(2026, 4, 10), date(2026, 4, 18)]


def test_days_to_archive_caps_at_max_days():
    """``max_days`` keeps a multi-week catch-up from slamming S3 in
    one run. Ordered oldest-first so the cap drops the NEWEST
    overflow, leaving them for the next run."""
    today = date(2026, 4, 27)
    days = [date(2026, 4, d) for d in range(1, 11)]  # 10 days
    out = days_to_archive(days, today_utc=today, retention_days=7, max_days=5)
    assert out == [date(2026, 4, d) for d in range(1, 6)]


def test_days_to_archive_empty_input_returns_empty():
    assert days_to_archive([], today_utc=date(2026, 4, 27), retention_days=7) == []


def test_days_to_archive_default_retention_keeps_recent():
    """Today's data and the past 7 days (inclusive) must NEVER show
    up as eligible for deletion at retention=7. The eligibility filter
    is strict ``<`` against ``today - retention_days``: the day
    EXACTLY at cutoff is kept (=8 days back from today + the day-of
    counts as day 0). The trainer reads ``--since-hours 72`` (3 days)
    by default, so the 7-day buffer leaves 4 days of slack —
    important so a deploy that bumps --since-hours doesn't suddenly
    see holes."""
    from datetime import timedelta
    today = date(2026, 4, 27)
    # 9 days: today through today-8, inclusive. Only today-8 is older
    # than the strict ``< today - 7`` cutoff.
    days = [today - timedelta(days=i) for i in range(9)]
    out = days_to_archive(days, today_utc=today, retention_days=7)
    assert len(out) == 1
    assert out[0] == today - timedelta(days=8)
