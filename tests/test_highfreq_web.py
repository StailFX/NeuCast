"""Unit tests for ``app.highfreq.web`` (Phase A.6 UI scaffold).

These exercise the **pure** logic — JSON sanitisation, age computation,
progress-bar derivation, status assembly — without spinning up FastAPI
or hitting Postgres. Integration tests against a live DB will live in a
separate suite once the trainer pipeline is fully wired.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.highfreq.web import (
    HighfreqStatus,
    MIN_MINUTES_FOR_TRAINING,
    LIVENESS_THRESHOLD_SEC,
    _scrub,
    _to_float,
    assemble_status,
    compute_age_seconds,
    compute_progress_fields,
)


# ── _scrub ────────────────────────────────────────────────────────────────


def test_scrub_replaces_nan_with_none():
    out = _scrub({"a": float("nan"), "b": 1.5})
    assert out == {"a": None, "b": 1.5}


def test_scrub_replaces_inf_with_none():
    out = _scrub({"pos": float("inf"), "neg": float("-inf")})
    assert out == {"pos": None, "neg": None}


def test_scrub_recurses_into_nested_structures():
    out = _scrub({"outer": [{"x": float("nan")}, {"y": 2.0}]})
    assert out == {"outer": [{"x": None}, {"y": 2.0}]}


def test_scrub_serialises_datetime_to_isoformat():
    ts = datetime(2026, 4, 25, 10, 30, 0, tzinfo=timezone.utc)
    out = _scrub({"ts": ts})
    assert out["ts"] == ts.isoformat()


def test_scrub_passes_through_strings_and_ints():
    out = _scrub({"sym": "BTCUSDT", "n": 42})
    assert out == {"sym": "BTCUSDT", "n": 42}


# ── _to_float ─────────────────────────────────────────────────────────────


def test_to_float_handles_none():
    assert _to_float(None) is None


def test_to_float_handles_decimal_strings():
    assert _to_float("3.14") == pytest.approx(3.14)


def test_to_float_returns_none_on_nan():
    assert _to_float(float("nan")) is None


def test_to_float_returns_none_on_inf():
    assert _to_float(float("inf")) is None


def test_to_float_returns_none_on_garbage():
    assert _to_float("not-a-number") is None


# ── compute_progress_fields ───────────────────────────────────────────────


def test_progress_zero_at_start():
    remaining, pct, has_enough = compute_progress_fields(0)
    assert remaining == MIN_MINUTES_FOR_TRAINING
    assert pct == 0.0
    assert has_enough is False


def test_progress_halfway():
    target = MIN_MINUTES_FOR_TRAINING
    remaining, pct, has_enough = compute_progress_fields(target // 2)
    assert remaining == target - target // 2
    assert 49.0 <= pct <= 51.0
    assert has_enough is False


def test_progress_complete():
    remaining, pct, has_enough = compute_progress_fields(MIN_MINUTES_FOR_TRAINING)
    assert remaining == 0
    assert pct == 100.0
    assert has_enough is True


def test_progress_clamps_overshoot():
    remaining, pct, has_enough = compute_progress_fields(
        MIN_MINUTES_FOR_TRAINING * 5,
    )
    assert remaining == 0
    assert pct == 100.0
    assert has_enough is True


def test_progress_handles_negative_input_gracefully():
    remaining, pct, has_enough = compute_progress_fields(-7)
    assert remaining == MIN_MINUTES_FOR_TRAINING
    assert pct == 0.0
    assert has_enough is False


def test_progress_custom_required():
    remaining, pct, has_enough = compute_progress_fields(50, minutes_required=100)
    assert remaining == 50
    assert pct == 50.0
    assert has_enough is False


# ── compute_age_seconds ───────────────────────────────────────────────────


def test_age_seconds_basic():
    now = datetime(2026, 4, 25, 10, 30, 0, tzinfo=timezone.utc)
    ts = now - timedelta(seconds=7)
    assert compute_age_seconds(ts, now) == pytest.approx(7.0)


def test_age_seconds_handles_naive_ts_as_utc():
    """Postgres TIMESTAMPTZ returns aware datetimes, but we should be robust
    to naive ones as well — they're treated as UTC."""
    now = datetime(2026, 4, 25, 10, 30, 0, tzinfo=timezone.utc)
    naive_ts = datetime(2026, 4, 25, 10, 29, 50)
    age = compute_age_seconds(naive_ts, now)
    assert age == pytest.approx(10.0)


def test_age_seconds_clamps_negative_clock_skew():
    # ts in the future (~1s) — possible if VPS clock drifts; should clamp
    # to 0 instead of producing nonsense in the UI.
    now = datetime(2026, 4, 25, 10, 30, 0, tzinfo=timezone.utc)
    future_ts = now + timedelta(seconds=1.2)
    assert compute_age_seconds(future_ts, now) == 0.0


def test_age_seconds_returns_none_when_ts_missing():
    now = datetime(2026, 4, 25, 10, 30, 0, tzinfo=timezone.utc)
    assert compute_age_seconds(None, now) is None


# ── assemble_status ───────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)


def test_assemble_status_no_data_yet():
    status = assemble_status(
        symbol="BTCUSDT",
        last_row=None,
        minutes_accumulated=0,
        rows_last_60s=None,
        now=_now(),
    )
    assert isinstance(status, HighfreqStatus)
    assert status.symbol == "BTCUSDT"
    assert status.snapshot is None
    assert status.is_live is False
    assert status.has_enough_data is False
    assert status.minutes_remaining == MIN_MINUTES_FOR_TRAINING
    assert status.progress_pct == 0.0
    assert status.rows_per_second_estimate is None


def test_assemble_status_live_ingest_fresh_row():
    now = _now()
    last_row = {
        "ts": now - timedelta(seconds=2),
        "symbol": "BTCUSDT",
        "microprice": 67_500.123,
        "ofi": 1.234,
        "depth_imb": 0.045,
        "spread_bps": 0.6,
    }
    status = assemble_status(
        symbol="BTCUSDT",
        last_row=last_row,
        minutes_accumulated=12,
        rows_last_60s=58,
        now=now,
    )
    assert status.snapshot is not None
    assert status.snapshot.microprice == pytest.approx(67_500.123)
    assert status.snapshot.age_seconds == pytest.approx(2.0)
    assert status.is_live is True
    assert status.has_enough_data is False
    # Rate is rounded to 2 decimal places inside assemble_status (UI noise).
    assert status.rows_per_second_estimate == pytest.approx(58 / 60.0, abs=0.01)


def test_assemble_status_stale_row_marks_not_live():
    now = _now()
    last_row = {
        "ts": now - timedelta(seconds=LIVENESS_THRESHOLD_SEC + 5),
        "symbol": "BTCUSDT",
        "microprice": 67_500.0,
        "ofi": 0.0,
        "depth_imb": 0.0,
        "spread_bps": 0.5,
    }
    status = assemble_status(
        symbol="BTCUSDT",
        last_row=last_row,
        minutes_accumulated=3,
        rows_last_60s=0,
        now=now,
    )
    assert status.is_live is False
    assert status.rows_per_second_estimate == 0.0


def test_assemble_status_enough_data_flips_flag():
    status = assemble_status(
        symbol="BTCUSDT",
        last_row=None,
        minutes_accumulated=MIN_MINUTES_FOR_TRAINING + 10,
        rows_last_60s=60,
        now=_now(),
    )
    assert status.has_enough_data is True
    assert status.minutes_remaining == 0
    assert status.progress_pct == 100.0


def test_assemble_status_to_dict_is_json_safe():
    """Output must be RFC-7159 compliant — no NaN / Inf in payload."""
    now = _now()
    last_row = {
        "ts": now - timedelta(seconds=1),
        "symbol": "BTCUSDT",
        "microprice": float("nan"),     # simulate a bad row
        "ofi": float("inf"),
        "depth_imb": 0.5,
        "spread_bps": None,
    }
    status = assemble_status(
        symbol="BTCUSDT",
        last_row=last_row,
        minutes_accumulated=5,
        rows_last_60s=60,
        now=now,
    )
    payload = status.to_dict()

    # _to_float should already have neutralised NaN/Inf into None inside
    # the snapshot, but to_dict adds belt-and-suspenders via _scrub.
    snapshot = payload["snapshot"]
    assert snapshot["microprice"] is None
    assert snapshot["ofi"] is None
    assert snapshot["spread_bps"] is None
    # ts must serialise as ISO-string.
    assert isinstance(snapshot["ts"], str) and "T" in snapshot["ts"]
    # server_time too.
    assert isinstance(payload["server_time"], str)


def test_assemble_status_uppercases_via_router_not_payload():
    """assemble_status itself doesn't uppercase — that's the router's job.
    We verify the symbol is propagated verbatim so the UI can show whatever
    the caller sent."""
    status = assemble_status(
        symbol="ethusdt",
        last_row=None,
        minutes_accumulated=0,
        rows_last_60s=0,
        now=_now(),
    )
    assert status.symbol == "ethusdt"
