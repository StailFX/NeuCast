"""Tests for ``app.highfreq.actionable_signal`` — pure decision logic.

The endpoint integration is exercised by the production smoke test
(curl /api/highfreq/actionable_signal returns the right shape). Pure
tests focus on the decision math: which gate fires when, how qty +
fee scale, what the skip_reason field reports.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.highfreq.actionable_signal import compute_decision
from app.highfreq.paper_trader import PaperTraderConfig


def _ts() -> datetime:
    return datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)


def _config(**overrides) -> PaperTraderConfig:
    """Helper for building a config with overrides."""
    base = dict(PaperTraderConfig().__dict__)
    base.update(overrides)
    return PaperTraderConfig(**base)


# ───── happy paths: open long / short ─────


def test_decision_opens_long_on_high_prob_up():
    """prob_up=0.65, calibrated, not halted → open long."""
    d = compute_decision(
        prob_up=0.65, microprice=77_000.0, ts=_ts(),
        config=_config(), model_version="3600",
        calibrated=True, demo_mode=False,
    )
    assert d["would_open"] is True
    assert d["side"] == "long"
    assert d["skip_reason"] is None
    assert d["entry_price"] == 77_000.0


def test_decision_opens_short_on_low_prob_up():
    d = compute_decision(
        prob_up=0.30, microprice=77_000.0, ts=_ts(),
        config=_config(), model_version="3600",
        calibrated=True, demo_mode=False,
    )
    assert d["would_open"] is True
    assert d["side"] == "short"


def test_decision_qty_matches_max_qty_usd_at_entry_price():
    """Suggested qty = max_qty_usd / microprice (no vol adjustment in
    this Level-4 surface — that's a runner-side concern). Pin so a
    future config change doesn't silently change UI sizing."""
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(max_qty_usd=1000.0),
        model_version="3600", calibrated=True, demo_mode=False,
    )
    assert d["suggested_qty_native"] == pytest.approx(1000.0 / 77_000.0)
    assert d["suggested_qty_usd"] == pytest.approx(1000.0, rel=1e-6)


def test_decision_fee_estimate_is_roundtrip_two_sides():
    """fee_estimate_usd = 2 × maker_fee_bps × notional. Pin so a
    refactor that drops the 2× silently halves the displayed cost."""
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(max_qty_usd=1000.0, maker_fee_bps_per_side=7.5),
        model_version="3600", calibrated=True, demo_mode=False,
    )
    # 7.5 bp = 0.075 % per side, 0.15 % round-trip → $1.50 on $1000 notional.
    assert d["fee_estimate_usd"] == pytest.approx(1.50, rel=1e-6)


def test_decision_exit_at_iso_is_horizon_minutes_after_ts():
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(horizon_minutes=1),
        model_version="3600", calibrated=True, demo_mode=False,
    )
    assert d["exit_at_iso"] == "2026-04-27T12:01:00+00:00"


# ───── skip paths ─────


def test_decision_skips_neutral_signal_when_in_band():
    """prob_up between thresholds → no trade, neutral_signal."""
    d = compute_decision(
        prob_up=0.50, microprice=77_000.0, ts=_ts(),
        config=_config(), model_version="3600",
        calibrated=True, demo_mode=False,
    )
    assert d["would_open"] is False
    assert d["side"] is None
    assert d["skip_reason"] == "neutral_signal"
    assert d["entry_price"] is None
    assert d["exit_at_iso"] is None


def test_decision_skips_when_not_calibrated_and_gate_on():
    """require_calibrated=True + calibrated=False → skip."""
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(require_calibrated=True),
        model_version="3600", calibrated=False, demo_mode=False,
    )
    assert d["would_open"] is False
    assert d["skip_reason"] == "not_calibrated"


def test_decision_opens_when_demo_mode_disables_calibration_gate():
    """Demo mode flips require_calibrated=False, so an uncalibrated
    model still trades. Pin so the demo-mode wire-up cannot silently
    regress.

    Note: the caller is responsible for passing the right config
    (with require_calibrated=False). compute_decision honours
    whatever it's handed."""
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(require_calibrated=False),
        model_version="pre-calibration-demo",
        calibrated=False, demo_mode=True,
    )
    assert d["would_open"] is True
    assert d["side"] == "long"
    assert d["demo_mode"] is True


def test_decision_skips_when_halted_loss_streak():
    """Halt state is the highest-priority gate — pre-empts even a
    valid signal."""
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(), model_version="3600",
        calibrated=True, demo_mode=False,
        halted_reason="loss_streak",
    )
    assert d["would_open"] is False
    assert d["skip_reason"] == "halted_loss_streak"


def test_decision_skips_when_halted_daily_loss():
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(), model_version="3600",
        calibrated=True, demo_mode=False,
        halted_reason="daily_loss",
    )
    assert d["would_open"] is False
    assert d["skip_reason"] == "halted_daily_loss"


# ───── shape contract for UI ─────


def test_decision_keys_are_pinned():
    """The UI reads specific keys. Pin so a refactor that renames
    them breaks tests instead of breaking the page silently."""
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(), model_version="3600",
        calibrated=True, demo_mode=False,
    )
    expected = {
        "prob_up", "microprice", "ts",
        "would_open", "side", "skip_reason",
        "suggested_qty_native", "suggested_qty_usd",
        "entry_price", "fee_estimate_usd",
        "time_horizon_min", "exit_at_iso",
        "model_version", "calibrated", "demo_mode",
    }
    assert set(d.keys()) == expected


def test_decision_provenance_fields_passed_through():
    """The caveats (model_version, calibrated, demo_mode) must round-
    trip — they're how the UI decides to show the warning banner."""
    d = compute_decision(
        prob_up=0.7, microprice=77_000.0, ts=_ts(),
        config=_config(),
        model_version="pre-calibration-demo",
        calibrated=False, demo_mode=True,
    )
    assert d["model_version"] == "pre-calibration-demo"
    assert d["calibrated"] is False
    assert d["demo_mode"] is True
