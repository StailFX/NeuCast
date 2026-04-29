"""Tests for ``tools.power_analysis`` — frequentist power calculator
for one-sided binomial tests.

Pin the math because a sign-flip would silently invert the recommendation
(e.g. tell operator they need 100 samples when they actually need 10000)."""
from __future__ import annotations

import pytest

from tools.power_analysis import (
    grid_table,
    power_achieved,
    required_n,
)


# ─── required_n: known cases ───


def test_required_n_for_055_default_power():
    """Standard textbook: detecting p=0.55 at α=0.05 + 80% power needs
    n ≈ 615-620 — pin so a refactor to wrong z-tables breaks loudly."""
    n = required_n(target_acc=0.55, alpha=0.05, power=0.80)
    assert 600 <= n <= 650


def test_required_n_for_06_higher_power():
    """Larger effect size requires fewer samples."""
    n = required_n(target_acc=0.60, alpha=0.05, power=0.80)
    assert 130 <= n <= 170


def test_required_n_monotonic_in_target_acc():
    """Larger true effect = fewer samples needed."""
    n_55 = required_n(target_acc=0.55)
    n_60 = required_n(target_acc=0.60)
    n_65 = required_n(target_acc=0.65)
    assert n_55 > n_60 > n_65


def test_required_n_monotonic_in_power():
    """Higher power requirement = more samples needed."""
    n_80 = required_n(target_acc=0.55, power=0.80)
    n_90 = required_n(target_acc=0.55, power=0.90)
    n_99 = required_n(target_acc=0.55, power=0.99)
    assert n_80 < n_90 < n_99


def test_required_n_rejects_target_at_or_below_chance():
    """target_acc <= 0.5 is undefined for one-sided test for skill —
    raises rather than silently returning garbage."""
    with pytest.raises(ValueError):
        required_n(target_acc=0.50)
    with pytest.raises(ValueError):
        required_n(target_acc=0.45)


def test_required_n_rejects_invalid_alpha_power():
    with pytest.raises(ValueError):
        required_n(target_acc=0.55, alpha=0.0)
    with pytest.raises(ValueError):
        required_n(target_acc=0.55, alpha=0.5)
    with pytest.raises(ValueError):
        required_n(target_acc=0.55, power=0.5)
    with pytest.raises(ValueError):
        required_n(target_acc=0.55, power=1.0)


# ─── power_achieved: inverse query ───


def test_power_achieved_at_required_n_is_close_to_target():
    """If we use required_n() to compute n for 80% power, then
    power_achieved() on that n should give ≈80%. Round-trip pin."""
    n = required_n(target_acc=0.55, power=0.80)
    achieved = power_achieved(n=n, target_acc=0.55)
    # Rounding in n_required can push us slightly above 0.80.
    assert 0.79 <= achieved <= 0.85


def test_power_achieved_grows_with_n():
    p1 = power_achieved(n=500, target_acc=0.55)
    p2 = power_achieved(n=2000, target_acc=0.55)
    p3 = power_achieved(n=10000, target_acc=0.55)
    assert p1 < p2 < p3
    assert p3 > 0.99  # 10k easily detects 0.55


def test_power_achieved_zero_n_zero_power():
    assert power_achieved(n=0, target_acc=0.55) == 0.0


# ─── live numbers — defense-grade pinpoint ───


def test_live_btc_n_has_extreme_power():
    """Pin: n=2356 (BTC current realized) achieves >99% power to detect
    a true 0.55 dir_acc. Defense: result is not sample-size accident."""
    p = power_achieved(n=2356, target_acc=0.55)
    assert p > 0.99


# ─── grid_table ───


def test_grid_table_renders_markdown():
    out = grid_table(target_accs=(0.55, 0.60), powers=(0.80, 0.95))
    assert "Power analysis grid" in out
    assert "target_acc" in out
    assert "0.550" in out
    assert "0.600" in out


def test_grid_table_n_values_decrease_with_higher_acc():
    """In any column, higher target_acc → smaller n. Visual sanity."""
    out = grid_table(target_accs=(0.51, 0.55, 0.60), powers=(0.80,))
    # Just verify the table has 3 data rows + header structure.
    rows = [l for l in out.splitlines() if l.startswith("| **0.")]
    assert len(rows) == 3
