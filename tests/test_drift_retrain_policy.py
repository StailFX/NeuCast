"""Tests for ``app.highfreq.drift_retrain_policy`` (T.22).

Pure-policy tests — pin the safety rails (cooldown + severity gate)
so a refactor can't silently turn the auto-retrain into a runaway
that thrashes the trainer and burns the box.

Reviewer-defensible behaviour:
* warn alone NEVER triggers (intraday regime jitter, often clears).
* high triggers only after cooldown elapses (default 6h).
* unknown severity values fail closed (no retrain).
* missing last-train timestamp fails open (cold-start path).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.highfreq.drift_retrain_policy import (
    RetrainDecision,
    evaluate_drift_retrain_policy,
)


_NOW = datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc)


def _hours_ago(h: float) -> datetime:
    return _NOW - timedelta(hours=h)


# ─────────────────── severity gate ───────────────────


def test_warn_does_not_trigger_by_default():
    """warn-level drift is intentionally below the trigger threshold —
    intraday regime shifts often clear within the hour and a warn-on-
    warn loop would retrain the model on every Asia/US session
    transition."""
    d = evaluate_drift_retrain_policy(
        severity="warn",
        last_train_started_at=_hours_ago(24),
        now=_NOW,
    )
    assert d.should_retrain is False
    assert "not in trigger set" in d.reason


def test_high_triggers_when_cooldown_elapsed():
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at=_hours_ago(8),
        now=_NOW,
        cooldown_hours=6.0,
    )
    assert d.should_retrain is True
    assert "Triggering retrain" in d.reason
    assert d.severity == "high"
    assert pytest.approx(d.hours_since_last_train, rel=1e-3) == 8.0


def test_high_blocked_by_cooldown():
    """Real risk we're guarding against: the drift JSON says high all
    day → without cooldown, we'd retrain every hour, overwriting the
    .cbm with progressively-noisier slices."""
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at=_hours_ago(2),
        now=_NOW,
        cooldown_hours=6.0,
    )
    assert d.should_retrain is False
    assert "cooldown=6.0h" in d.reason
    assert "2.0h ago" in d.reason


def test_ok_severity_does_not_trigger():
    d = evaluate_drift_retrain_policy(
        severity="ok",
        last_train_started_at=_hours_ago(72),  # plenty of time elapsed
        now=_NOW,
    )
    assert d.should_retrain is False


def test_unknown_severity_fails_closed():
    """Defensive: a bad severity value (typo, schema change) MUST
    NOT trigger retrain — fail-closed to avoid retraining on
    nonsense. Ops can fix the upstream and re-run."""
    d = evaluate_drift_retrain_policy(
        severity="critical",  # not in trigger set
        last_train_started_at=_hours_ago(24),
        now=_NOW,
    )
    assert d.should_retrain is False


# ─────────────────── cooldown edge cases ───────────────────


def test_no_prior_training_treated_as_cold_start():
    """Cold start path — no training_runs row yet. Cooldown can't
    restrict what hasn't happened. Trigger immediately."""
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at=None,
        now=_NOW,
    )
    assert d.should_retrain is True
    assert "no prior training run" in d.reason
    assert d.hours_since_last_train is None


def test_cooldown_boundary_just_past():
    """Right at cooldown_hours + epsilon → triggers."""
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at=_hours_ago(6.01),
        now=_NOW,
        cooldown_hours=6.0,
    )
    assert d.should_retrain is True


def test_cooldown_boundary_just_before():
    """Right at cooldown_hours − epsilon → blocked."""
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at=_hours_ago(5.99),
        now=_NOW,
        cooldown_hours=6.0,
    )
    assert d.should_retrain is False


def test_severity_normalisation():
    """Case + whitespace tolerant — drift JSON shouldn't break us
    over a stray space."""
    d = evaluate_drift_retrain_policy(
        severity="  HIGH  ",
        last_train_started_at=_hours_ago(24),
        now=_NOW,
    )
    assert d.should_retrain is True


def test_iso_string_last_train_ts_parsed():
    """drift_check writes ISO strings; policy should accept both
    str and datetime without operator gymnastics."""
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at="2026-05-04T00:00:00Z",
        now=_NOW,
        cooldown_hours=6.0,
    )
    assert d.should_retrain is True
    assert pytest.approx(d.hours_since_last_train, rel=1e-3) == 12.0


def test_iso_string_with_offset_suffix():
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at="2026-05-04T00:00:00+00:00",
        now=_NOW,
    )
    assert d.should_retrain is True


# ─────────────────── custom config ───────────────────


def test_custom_fire_on_severities_can_include_warn():
    """If an operator opts-in to retrain-on-warn (e.g. for a more
    aggressive defence-time demo), the policy honours it."""
    d = evaluate_drift_retrain_policy(
        severity="warn",
        last_train_started_at=_hours_ago(24),
        now=_NOW,
        fire_on_severities=("warn", "high"),
    )
    assert d.should_retrain is True


def test_custom_cooldown_hours():
    """Tighter 1h cooldown for fast-iteration testing."""
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at=_hours_ago(1.5),
        now=_NOW,
        cooldown_hours=1.0,
    )
    assert d.should_retrain is True


def test_returns_frozen_dataclass():
    """Decision is hashable / immutable — safe to pass around the
    log/metric pipeline without defensive copies."""
    d = evaluate_drift_retrain_policy(
        severity="high",
        last_train_started_at=_hours_ago(24),
        now=_NOW,
    )
    assert isinstance(d, RetrainDecision)
    with pytest.raises(Exception):
        d.should_retrain = False  # frozen
