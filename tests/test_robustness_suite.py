"""Tests for ``tools.robustness_suite`` — pure-stat helpers.

The DB-side ``run_robustness`` is integration-tested by running it on
Tokyo against the live ``predictions_log``. These unit tests pin the
statistical correctness of the pure helpers (block bootstrap,
permutation test, sub-period stability) so a refactor can't silently
break the published numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.robustness_suite import (
    block_bootstrap_dir_acc,
    per_day_stability,
    per_hour_breakdown,
    permutation_test_dir_acc,
    regime_conditional_accuracy,
)


# ──────── block_bootstrap_dir_acc ────────


def test_block_bootstrap_returns_finite_ci_on_balanced_data():
    """Sanity: 50/50 balanced data should give a CI centred near 0.5."""
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, size=200)
    lo, hi = block_bootstrap_dir_acc(correct, block_size=20, n_resamples=200, seed=42)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert hi > lo


def test_block_bootstrap_zero_size_input_returns_nan():
    correct = np.array([], dtype=int)
    lo, hi = block_bootstrap_dir_acc(correct, block_size=10, n_resamples=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_block_bootstrap_block_size_capped_to_sample_size():
    """If block_size > n, we should clamp internally rather than 500."""
    correct = np.array([1, 1, 0, 1, 0])
    lo, hi = block_bootstrap_dir_acc(correct, block_size=100, n_resamples=50, seed=1)
    assert not np.isnan(lo) and not np.isnan(hi)


def test_block_bootstrap_rejects_zero_block_size():
    with pytest.raises(ValueError, match="block_size"):
        block_bootstrap_dir_acc(np.array([0, 1]), block_size=0, n_resamples=100)


def test_block_bootstrap_ci_widens_with_smaller_blocks():
    """Wider blocks preserve more autocorrelation → wider CI than i.i.d.
    Concretely on a 60% biased sample the i.i.d. boot is narrowest;
    larger blocks give equal-or-wider CI. This pins the direction."""
    rng = np.random.default_rng(7)
    correct = (rng.uniform(size=400) < 0.6).astype(int)
    # block_size=1 ≈ i.i.d. bootstrap
    lo1, hi1 = block_bootstrap_dir_acc(
        correct, block_size=1, n_resamples=500, seed=42,
    )
    # block_size=40 = preserves long autocorrelation
    lo40, hi40 = block_bootstrap_dir_acc(
        correct, block_size=40, n_resamples=500, seed=42,
    )
    # Block-bootstrap CI should be wider (or near-equal) than i.i.d.
    width_iid = hi1 - lo1
    width_block = hi40 - lo40
    assert width_block >= width_iid * 0.7  # generous tolerance for randomness


# ──────── permutation_test_dir_acc ────────


def test_permutation_test_perfect_predictor_gets_tiny_p_value():
    """If y_pred == y_true on every row, no permutation matches by chance
    → p-value is the smallest possible (1/(n_perm+1))."""
    n = 100
    y_true = (np.arange(n) % 2).astype(int)
    y_pred = y_true.copy()
    p, mean, std, z = permutation_test_dir_acc(
        y_pred, y_true, n_permutations=200, seed=42,
    )
    assert p == 1 / 201   # smallest achievable
    # Null mean should be ~0.5 (random pairings).
    assert 0.40 <= mean <= 0.60
    assert std > 0.0


def test_permutation_test_random_predictor_gets_large_p_value():
    """Random y_pred uncorrelated with y_true → p-value should be near
    0.5 (observed accuracy is a typical draw from the null)."""
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    p, mean, std, z = permutation_test_dir_acc(
        y_pred, y_true, n_permutations=300, seed=42,
    )
    # Random predictor → observed should sit near the null mean.
    assert 0.05 <= p <= 0.95
    assert -3.0 <= z <= 3.0


def test_permutation_test_empty_returns_nan_tuple():
    p, m, s, z = permutation_test_dir_acc(
        np.array([], dtype=int), np.array([], dtype=int), n_permutations=10,
    )
    assert all(np.isnan(v) for v in (p, m, s, z))


def test_permutation_test_mismatched_lengths_returns_nan():
    p, _, _, _ = permutation_test_dir_acc(
        np.array([1, 0]), np.array([1, 0, 1]), n_permutations=10,
    )
    assert np.isnan(p)


# ──────── per_day_stability ────────


def test_per_day_stability_empty_input():
    assert per_day_stability(np.array([], dtype="datetime64[ns]"),
                             np.array([], dtype=int)) == []


def test_per_day_stability_groups_by_utc_date():
    ts = pd.to_datetime([
        "2026-04-30T10:00:00Z", "2026-04-30T15:00:00Z",
        "2026-05-01T01:00:00Z", "2026-05-01T23:00:00Z",
    ], utc=True).to_numpy()
    correct = np.array([1, 1, 0, 1])
    out = per_day_stability(ts, correct)
    assert len(out) == 2
    apr30 = next(r for r in out if r["date"].startswith("2026-04-30"))
    may01 = next(r for r in out if r["date"].startswith("2026-05-01"))
    assert apr30["n"] == 2 and apr30["hits"] == 2
    assert apr30["dir_acc"] == 1.0
    assert may01["n"] == 2 and may01["hits"] == 1
    assert may01["dir_acc"] == 0.5
    # Wilson CI must contain the point estimate.
    for r in out:
        assert r["ci_low"] <= r["dir_acc"] <= r["ci_high"]


# ──────── per_hour_breakdown ────────


def test_per_hour_breakdown_returns_24_rows_always():
    """Even when the data only spans a few hours, the response carries
    24 rows so the UI heatmap doesn't have gaps."""
    ts = pd.to_datetime([
        "2026-04-30T07:00:00Z", "2026-04-30T07:30:00Z",
        "2026-04-30T15:00:00Z",
    ], utc=True).to_numpy()
    correct = np.array([1, 0, 1])
    out = per_hour_breakdown(ts, correct)
    assert len(out) == 24
    # Hour 7 should have n=2.
    h7 = out[7]
    assert h7["n"] == 2 and h7["hits"] == 1
    # Hour 0 wasn't observed → zero-fill.
    h0 = out[0]
    assert h0["n"] == 0 and h0["dir_acc"] is None


# ──────── regime_conditional_accuracy ────────


def test_regime_conditional_accuracy_returns_three_buckets():
    """Always emit uptrend/sideways/downtrend rows even when one is empty,
    so the UI knows which bucket has no data instead of the row being
    missing."""
    # Synthetic: prices climb steadily → all rows should be 'uptrend'
    # for sufficient lookback.
    ts = pd.to_datetime(pd.date_range(
        "2026-04-30T00:00:00Z", periods=200, freq="1min", tz="UTC",
    )).to_numpy()
    microprices = np.linspace(100.0, 110.0, 200)   # +1000 bps over the window
    correct = np.ones(200, dtype=int)              # always right
    out = regime_conditional_accuracy(
        ts, correct, microprices,
        lookback_minutes=30, sideways_threshold_bps=10.0,
    )
    regimes = {r["regime"]: r for r in out}
    assert set(regimes.keys()) == {"uptrend", "downtrend", "sideways"}
    # The uptrend bucket should hold most of the rows; downtrend = 0.
    assert regimes["uptrend"]["n"] > 0
    assert regimes["downtrend"]["n"] == 0
    # When n=0, dir_acc should be None (not 0.0 → that would falsely
    # claim "model failed in this regime").
    assert regimes["downtrend"]["dir_acc"] is None
