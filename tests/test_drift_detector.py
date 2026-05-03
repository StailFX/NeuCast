"""Tests for ``app.highfreq.drift_detector`` (T.18.b).

Production-grade alerting infra: Kolmogorov-Smirnov test on
feature distributions train vs serve. Pin:

1. compute_per_feature_drift returns one result per common
   numeric column.
2. KS statistic ≈ 0 when reference and recent are the same
   distribution.
3. KS statistic >> 0 when distributions are obviously different
   (different means).
4. Sample-size guard: features with n < 30 in either window are
   skipped (KS unreliable).
5. summarise_drift identifies the worst feature.
6. is_drifted threshold check.
7. alert_payload returns a structured dict with ``severity`` levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.highfreq.drift_detector import (
    DriftSummary,
    FeatureDrift,
    alert_payload,
    compute_per_feature_drift,
    summarise_drift,
)


def _build_synthetic(n: int, *, mean_shift: float = 0.0, seed: int = 0):
    """3-feature DataFrame: 'a' is N(0,1)+shift, 'b' is N(5,2),
    'c' is uniform [0,1]."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(0 + mean_shift, 1, size=n),
        "b": rng.normal(5, 2, size=n),
        "c": rng.uniform(0, 1, size=n),
    })


def test_compute_per_feature_drift_returns_one_per_numeric_column():
    ref = _build_synthetic(1000, seed=0)
    rec = _build_synthetic(500, seed=1)
    drifts = compute_per_feature_drift(ref, rec)
    assert len(drifts) == 3
    # All numeric → all returned.
    feature_names = {d.feature for d in drifts}
    assert feature_names == {"a", "b", "c"}


def test_ks_near_zero_when_distributions_identical():
    """Reference and recent drawn from same distribution → KS small.
    Empirical bound: with n=2000 each, KS < 0.10 in 99% of cases
    (CI of the KS distribution under H0)."""
    ref = _build_synthetic(2000, seed=0)
    rec = _build_synthetic(2000, seed=1)  # different seed, same distribution
    drifts = compute_per_feature_drift(ref, rec)
    for d in drifts:
        assert d.ks_stat < 0.10, (
            f"feature {d.feature} ks={d.ks_stat:.4f} unexpectedly high"
        )


def test_ks_large_when_means_differ():
    """Reference centered at 0, recent centered at 2 → KS > 0.5."""
    ref = _build_synthetic(2000, mean_shift=0.0, seed=0)
    rec = _build_synthetic(2000, mean_shift=2.0, seed=1)
    drifts = compute_per_feature_drift(ref, rec)
    by_name = {d.feature: d for d in drifts}
    # Feature 'a' shifted; 'b' and 'c' didn't.
    assert by_name["a"].ks_stat > 0.50
    # 'b' and 'c' should be small (< 0.10).
    assert by_name["b"].ks_stat < 0.10
    assert by_name["c"].ks_stat < 0.10


def test_skip_features_with_insufficient_samples():
    """KS on n<30 is unreliable — function should skip rather than
    emit a noisy false-alarm."""
    ref = _build_synthetic(20, seed=0)  # too few rows
    rec = _build_synthetic(20, seed=1)
    drifts = compute_per_feature_drift(ref, rec)
    assert drifts == []


def test_skip_non_numeric_columns():
    """A 'symbol' string column shouldn't crash the KS test —
    function filters to numeric only."""
    ref = _build_synthetic(500, seed=0)
    ref["symbol"] = "BTCUSDT"
    rec = _build_synthetic(500, seed=1)
    rec["symbol"] = "BTCUSDT"
    drifts = compute_per_feature_drift(ref, rec)
    feature_names = {d.feature for d in drifts}
    assert "symbol" not in feature_names
    assert {"a", "b", "c"} <= feature_names


def test_handles_columns_missing_in_one_frame():
    """Reference has more cols than recent → only common ones are
    compared. Defensive — pipelines change shape over time."""
    ref = _build_synthetic(500, seed=0)
    ref["d_extra"] = 1.0
    rec = _build_synthetic(500, seed=1)
    drifts = compute_per_feature_drift(ref, rec)
    feature_names = {d.feature for d in drifts}
    assert "d_extra" not in feature_names


def test_calendar_features_excluded_by_default():
    """Calendar columns (day_of_week, hour_of_day, etc.) ALWAYS drift
    between two time windows by construction. They drown the real
    signal — exclude by default so the alert is meaningful."""
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({
        "ofi_sum": rng.normal(0, 1, size=500),
        "day_of_week": np.full(500, 2),  # all Tuesday
        "hour_of_day": np.full(500, 10),
    })
    rec = pd.DataFrame({
        "ofi_sum": rng.normal(0, 1, size=500),
        "day_of_week": np.full(500, 6),  # all Saturday
        "hour_of_day": np.full(500, 14),
    })
    drifts = compute_per_feature_drift(ref, rec)
    feature_names = {d.feature for d in drifts}
    assert "day_of_week" not in feature_names
    assert "hour_of_day" not in feature_names
    assert "ofi_sum" in feature_names


def test_calendar_features_included_when_explicitly_asked():
    """Override path for ablation / debug. Production alerting
    should NEVER pass include_calendar=True."""
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({
        "day_of_week": np.full(500, 2),
        "ofi_sum": rng.normal(0, 1, size=500),
    })
    rec = pd.DataFrame({
        "day_of_week": np.full(500, 6),
        "ofi_sum": rng.normal(0, 1, size=500),
    })
    drifts = compute_per_feature_drift(ref, rec, include_calendar=True)
    feature_names = {d.feature for d in drifts}
    assert "day_of_week" in feature_names


def test_summarise_drift_identifies_worst():
    drifts = [
        FeatureDrift(feature="a", ks_stat=0.20, p_value=0.001,
                     n_reference=1000, n_recent=500,
                     reference_mean=0.0, recent_mean=0.5),
        FeatureDrift(feature="b", ks_stat=0.05, p_value=0.30,
                     n_reference=1000, n_recent=500,
                     reference_mean=5.0, recent_mean=5.1),
        FeatureDrift(feature="c", ks_stat=0.50, p_value=1e-10,
                     n_reference=1000, n_recent=500,
                     reference_mean=0.5, recent_mean=0.9),
    ]
    summary = summarise_drift(drifts, threshold=0.15)
    assert summary.max_ks == 0.50
    assert summary.max_ks_feature == "c"
    assert summary.n_features == 3
    assert summary.n_features_alarming == 2  # 'a' and 'c' >= 0.15
    assert summary.is_drifted() is True
    # Sorted features: worst first.
    assert [f.feature for f in summary.features] == ["c", "a", "b"]


def test_summarise_drift_is_not_drifted_when_below_threshold():
    drifts = [
        FeatureDrift(feature="a", ks_stat=0.05, p_value=0.40,
                     n_reference=1000, n_recent=500,
                     reference_mean=0.0, recent_mean=0.0),
    ]
    summary = summarise_drift(drifts, threshold=0.15)
    assert summary.is_drifted() is False
    assert summary.n_features_alarming == 0


def test_summarise_empty_drifts_safe():
    summary = summarise_drift([], threshold=0.15)
    assert summary.n_features == 0
    assert summary.max_ks == 0.0
    assert summary.is_drifted() is False
    assert summary.features == []


def test_alert_payload_severity_buckets():
    """Verify severity tiers: max_ks ≥ 0.30 → high, ≥ threshold → warn,
    else ok."""
    def _summary(max_ks: float, threshold: float = 0.15):
        d = FeatureDrift(feature="x", ks_stat=max_ks, p_value=0.01,
                         n_reference=1000, n_recent=500,
                         reference_mean=0.0, recent_mean=0.0)
        return summarise_drift([d], threshold=threshold)

    p_ok = alert_payload(_summary(0.05))
    p_warn = alert_payload(_summary(0.20))
    p_high = alert_payload(_summary(0.40))
    assert p_ok["severity"] == "ok"
    assert p_warn["severity"] == "warn"
    assert p_high["severity"] == "high"


def test_alert_payload_includes_top_features():
    """Top-5 features included so operator can triage at a glance."""
    drifts = [
        FeatureDrift(feature=f"feat_{i}", ks_stat=0.5 - i * 0.05,
                     p_value=1e-8, n_reference=1000, n_recent=500,
                     reference_mean=float(i), recent_mean=float(i + 1))
        for i in range(8)
    ]
    summary = summarise_drift(drifts, threshold=0.15)
    payload = alert_payload(summary)
    assert len(payload["top_features"]) == 5
    # Ordered by ks_stat descending.
    ks_values = [f["ks_stat"] for f in payload["top_features"]]
    assert ks_values == sorted(ks_values, reverse=True)
    # Each top entry has the keys the TG message expects.
    for f in payload["top_features"]:
        assert {"feature", "ks_stat", "p_value", "reference_mean",
                "recent_mean", "n_ref", "n_recent"} <= set(f.keys())
