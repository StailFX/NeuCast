"""Tests for ``compute_edge_vs_base_rate`` — model vs naive baseline.

The metric is small but defence-grade critical: it surfaces the
"is the model actually adding value?" question explicitly. A bug
that hid a negative edge (model losing to naive) would let a bad
model masquerade as merely "noisy".
"""
from __future__ import annotations

import math

import pytest

from app.highfreq.web import compute_edge_vs_base_rate


# ───── happy paths ─────


def test_positive_edge_when_model_beats_balanced_baseline():
    """50/50 split, model 60% accuracy → edge = +0.10."""
    edge = compute_edge_vs_base_rate(dir_acc_mean=0.60, base_rate=0.50)
    assert edge == pytest.approx(0.10)


def test_zero_edge_at_chance():
    """Model exactly at chance + balanced classes → edge = 0."""
    edge = compute_edge_vs_base_rate(dir_acc_mean=0.50, base_rate=0.50)
    assert edge == pytest.approx(0.0)


def test_negative_edge_when_model_loses_to_majority_class():
    """The defence-grade signal: model UNDERPERFORMS majority class.
    Mirrors the BTC fold-0 finding (dir_acc 0.567, base_rate 0.583
    → naive baseline 0.583, edge = -0.017)."""
    edge = compute_edge_vs_base_rate(dir_acc_mean=0.567, base_rate=0.583)
    assert edge == pytest.approx(-0.016, abs=1e-3)


def test_naive_baseline_uses_majority_class_not_balanced():
    """Pin: with 70/30 base rate, naive baseline = 0.70 (always
    predict majority), NOT 0.50. A model at 65% beats balanced
    chance but LOSES to naive — must be reported as edge < 0."""
    edge = compute_edge_vs_base_rate(dir_acc_mean=0.65, base_rate=0.70)
    assert edge == pytest.approx(-0.05)


def test_naive_baseline_symmetric_around_0_5():
    """base_rate=0.30 (30/70) and 0.70 (70/30) should give same
    naive baseline = 0.70 because the formula is max(p, 1-p)."""
    edge_low  = compute_edge_vs_base_rate(dir_acc_mean=0.65, base_rate=0.30)
    edge_high = compute_edge_vs_base_rate(dir_acc_mean=0.65, base_rate=0.70)
    assert edge_low == pytest.approx(edge_high)


# ───── edge cases / defensive returns ─────


def test_returns_none_when_dir_acc_is_none():
    """Cold-start (no folds yet) → can't compute edge → None.
    The endpoint then renders '—' instead of a misleading 0."""
    assert compute_edge_vs_base_rate(dir_acc_mean=None, base_rate=0.50) is None


def test_returns_none_when_base_rate_is_none():
    assert compute_edge_vs_base_rate(dir_acc_mean=0.55, base_rate=None) is None


def test_returns_none_on_nan():
    """Trainer's no-folds path emits NaN for both fields. Pin."""
    assert compute_edge_vs_base_rate(dir_acc_mean=float("nan"), base_rate=0.50) is None
    assert compute_edge_vs_base_rate(dir_acc_mean=0.55, base_rate=float("nan")) is None


def test_returns_none_on_out_of_range_inputs():
    """Garbage in → None out, NOT a confidently-wrong number."""
    assert compute_edge_vs_base_rate(dir_acc_mean=1.5, base_rate=0.50) is None
    assert compute_edge_vs_base_rate(dir_acc_mean=-0.1, base_rate=0.50) is None
    assert compute_edge_vs_base_rate(dir_acc_mean=0.55, base_rate=2.0) is None


def test_perfect_model_against_balanced_data_gives_max_edge():
    """100% accuracy on 50/50 data → edge = +0.50."""
    edge = compute_edge_vs_base_rate(dir_acc_mean=1.0, base_rate=0.50)
    assert edge == pytest.approx(0.50)


def test_terrible_model_against_balanced_data_gives_min_edge():
    """0% accuracy on 50/50 data → edge = -0.50. (Inverting the
    predictions would give 100% accuracy.)"""
    edge = compute_edge_vs_base_rate(dir_acc_mean=0.0, base_rate=0.50)
    assert edge == pytest.approx(-0.50)
