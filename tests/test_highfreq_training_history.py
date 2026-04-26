"""Tests for ``app.highfreq.training_history`` — pure shaping logic.

Pin:

* The dataclass → row-dict translation matches the migration's column
  set (so a future column rename in the migration breaks these tests
  rather than silently dropping the field on insert).
* NaN/Inf scrubbing — Postgres JSONB rejects them; the trainer's
  output frequently has NaN dir_acc when no folds were produced.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from app.highfreq.trainer import TrainingReport
from app.highfreq.training_history import (
    _run_dict_for_insert,
    _scrub_nan,
)


def _basic_report(**overrides) -> TrainingReport:
    """Minimal valid TrainingReport for shaping tests."""
    base = dict(
        symbol="BTCUSDT",
        horizon_min=1,
        neutral_band_bps=1.0,
        n_seconds_loaded=132540,
        n_minutes_after_aggregation=2210,
        n_minutes_after_neutral_drop=1115,
        base_rate=0.5009,
        n_folds=0,
        dir_acc_mean=float("nan"),
        dir_acc_ci_low=float("nan"),
        dir_acc_ci_high=float("nan"),
        dir_acc_p_value=float("nan"),
        log_loss_mean=float("nan"),
        elapsed_seconds=8.6,
        weights_path="/opt/neucast/weights/highfreq/btcusdt_1m.cbm",
        frozen_holdout_days=7,
        n_minutes_in_holdout=1115,
        holdout_cutoff_iso="2026-04-20T00:00:00+00:00",
    )
    base.update(overrides)
    return TrainingReport(**base)


# ───── _scrub_nan ─────


def test_scrub_nan_replaces_float_nans_with_none():
    inp = {"a": float("nan"), "b": float("inf"), "c": 1.0, "d": "text"}
    out = _scrub_nan(inp)
    assert out == {"a": None, "b": None, "c": 1.0, "d": "text"}


def test_scrub_nan_recurses_into_nested_structures():
    inp = {
        "outer": [1.0, float("nan"), {"inner": float("inf")}],
        "tup": (float("nan"), 2.0),
    }
    out = _scrub_nan(inp)
    assert out == {
        "outer": [1.0, None, {"inner": None}],
        "tup": [None, 2.0],
    }


def test_scrub_nan_passes_non_floats_through():
    """Strings, ints, bools, None — all unchanged."""
    inp = {"s": "hi", "i": 42, "b": True, "n": None}
    assert _scrub_nan(inp) == inp


# ───── _run_dict_for_insert ─────


def test_run_dict_columns_match_migration_004():
    """Pin: every column in the migration must appear in the row
    dict (modulo db-defaults like written_at). If migration 004 grows
    a column, this test fails until we add it here too — preventing
    silent data-loss inserts."""
    report = _basic_report()
    started = datetime(2026, 4, 27, 4, 0, 0, tzinfo=timezone.utc)
    row = _run_dict_for_insert(report, run_started_at=started)

    expected_columns = {
        "symbol", "run_started_at", "elapsed_seconds",
        "n_seconds_loaded", "n_minutes_after_aggregation",
        "n_minutes_after_neutral_drop",
        "n_folds", "dir_acc_mean", "dir_acc_ci_low",
        "dir_acc_ci_high", "dir_acc_p_value",
        "log_loss_mean", "base_rate",
        "frozen_holdout_days", "n_minutes_in_holdout",
        "weights_path", "full_report_json",
    }
    assert set(row.keys()) == expected_columns


def test_run_dict_nan_floats_become_none():
    """A run with no folds has NaN dir_acc / CI / p / logloss. The
    NUMERIC columns in Postgres reject NaN — must be None."""
    report = _basic_report(
        dir_acc_mean=float("nan"),
        dir_acc_ci_low=float("nan"),
        dir_acc_ci_high=float("nan"),
        dir_acc_p_value=float("nan"),
        log_loss_mean=float("nan"),
        base_rate=float("nan"),
    )
    row = _run_dict_for_insert(
        report, run_started_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )
    for k in ("dir_acc_mean", "dir_acc_ci_low", "dir_acc_ci_high",
              "dir_acc_p_value", "log_loss_mean", "base_rate"):
        assert row[k] is None, f"{k} must be None when NaN, got {row[k]!r}"


def test_run_dict_finite_floats_pass_through():
    """When folds DO exist, the floats land verbatim."""
    report = _basic_report(
        n_folds=4,
        dir_acc_mean=0.547,
        dir_acc_ci_low=0.521,
        dir_acc_ci_high=0.572,
        dir_acc_p_value=0.012,
        log_loss_mean=0.69,
        base_rate=0.5,
    )
    row = _run_dict_for_insert(
        report, run_started_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )
    assert row["dir_acc_mean"] == 0.547
    assert row["dir_acc_ci_low"] == 0.521
    assert row["dir_acc_ci_high"] == 0.572
    assert row["dir_acc_p_value"] == 0.012
    assert row["log_loss_mean"] == 0.69
    assert row["base_rate"] == 0.5
    assert row["n_folds"] == 4


def test_run_dict_full_report_json_is_valid_json():
    """The JSONB column gets a serialised string from the dataclass.
    Must round-trip through json.loads without exceptions."""
    report = _basic_report()
    row = _run_dict_for_insert(
        report, run_started_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )
    parsed = json.loads(row["full_report_json"])
    # NaN scrubbed → None for the report's NaN fields.
    assert parsed["dir_acc_mean"] is None
    assert parsed["symbol"] == "BTCUSDT"
    assert parsed["n_minutes_after_neutral_drop"] == 1115


def test_run_dict_n_minutes_in_holdout_none_passthrough():
    """When the trainer ran with frozen_holdout_days=0, the holdout
    fields are None. Must NOT be coerced to 0 (loses information)."""
    report = _basic_report(frozen_holdout_days=0, n_minutes_in_holdout=None)
    row = _run_dict_for_insert(
        report, run_started_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )
    assert row["frozen_holdout_days"] == 0
    assert row["n_minutes_in_holdout"] is None


def test_run_dict_run_started_at_passes_through_unchanged():
    """The timestamp is the trainer's wall clock at start of run —
    NOT now(). Pin so a future refactor that passes default=now() at
    insert breaks this test."""
    started = datetime(2026, 4, 27, 4, 0, 0, tzinfo=timezone.utc)
    row = _run_dict_for_insert(_basic_report(), run_started_at=started)
    assert row["run_started_at"] == started


def test_run_dict_finite_value_round_trip_to_json():
    """End-to-end: dataclass → row dict → JSONB string → parse back.
    All numeric fields must reappear with their original values."""
    report = _basic_report(
        dir_acc_mean=0.55,
        n_minutes_after_neutral_drop=1500,
    )
    row = _run_dict_for_insert(
        report, run_started_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )
    parsed = json.loads(row["full_report_json"])
    assert parsed["dir_acc_mean"] == 0.55
    assert parsed["n_minutes_after_neutral_drop"] == 1500
