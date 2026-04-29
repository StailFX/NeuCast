"""Tests for ``app.highfreq.calibration`` — Platt scaling + reliability."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.highfreq.calibration import (
    ReliabilityCurve,
    _PassthroughCalibrator,
    apply_calibrator,
    calibrator_path_for,
    compute_reliability_curve,
    fit_platt_calibrator,
    load_calibrator,
    save_calibrator,
)


# ─────────── fit + apply ───────────


def _well_calibrated_data(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate (raw_proba, y_true) where the raw probas are exactly
    calibrated — y is sampled from Bernoulli(raw_proba). Used as
    sanity test: Platt should be near-identity on these inputs."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.1, 0.9, size=n)
    y = rng.binomial(1, raw)
    return raw, y


def _miscalibrated_data(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate data where raw proba is overconfident: a raw 0.7
    actually corresponds to true rate 0.6, etc."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.1, 0.9, size=n)
    # True rate is "compressed toward 0.5" — overconfident model.
    true_rate = 0.5 + 0.6 * (raw - 0.5)  # raw 0.9 → true 0.74; raw 0.1 → true 0.26
    y = rng.binomial(1, true_rate)
    return raw, y


def test_fit_platt_returns_estimator_with_predict_proba():
    raw, y = _well_calibrated_data()
    cal = fit_platt_calibrator(raw, y)
    assert hasattr(cal, "predict_proba")


def test_apply_calibrator_passthrough_when_none():
    """No calibrator → raw input returned unchanged. Critical hot-
    path contract: a missing-calibrator model must NOT degrade."""
    out = apply_calibrator(None, 0.6)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(0.6)


def test_apply_calibrator_handles_scalar_and_array():
    raw, y = _well_calibrated_data()
    cal = fit_platt_calibrator(raw, y)
    # Scalar input.
    out_scalar = apply_calibrator(cal, 0.7)
    assert out_scalar.shape == (1,)
    assert 0.0 <= out_scalar[0] <= 1.0
    # Array input.
    out_array = apply_calibrator(cal, np.array([0.3, 0.5, 0.7]))
    assert out_array.shape == (3,)
    assert np.all((out_array >= 0.0) & (out_array <= 1.0))


def test_calibrator_corrects_overconfidence():
    """Pin: on overconfident raw proba, the calibrator should
    PULL high probs DOWN toward 0.5. The signed test: raw 0.9
    → calibrated should be < 0.9."""
    raw, y = _miscalibrated_data(n=5000)
    cal = fit_platt_calibrator(raw, y)
    # Apply on a high-confidence raw input.
    cal_p = apply_calibrator(cal, 0.9)[0]
    # Should be pulled toward true rate at raw=0.9, which is 0.74.
    # Allow a wide bound — sample noise may push around 0.65-0.80.
    assert cal_p < 0.9, f"calibrator failed to correct overconfidence: {cal_p}"


def test_calibrator_preserves_ordering():
    """Critical: a higher raw probability must still map to a higher
    calibrated probability. If not, the trader's threshold logic
    becomes non-monotonic and unsafe."""
    raw, y = _miscalibrated_data(n=2000)
    cal = fit_platt_calibrator(raw, y)
    test_inputs = np.linspace(0.05, 0.95, 50)
    cal_outputs = apply_calibrator(cal, test_inputs)
    # Calibrated output must be monotonically non-decreasing.
    diffs = np.diff(cal_outputs)
    assert (diffs >= -1e-9).all(), "calibrator output not monotonic"


def test_fit_returns_passthrough_on_single_class():
    """Edge: y_true is all-0 or all-1 (single fold's worth of data).
    LogisticRegression refuses; we return passthrough that yields raw
    probas via inverse-logit."""
    raw = np.linspace(0.1, 0.9, 50)
    y_all_zero = np.zeros(50, dtype=int)
    cal = fit_platt_calibrator(raw, y_all_zero)
    assert isinstance(cal, _PassthroughCalibrator)


def test_fit_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        fit_platt_calibrator(np.array([0.5, 0.6]), np.array([1, 0, 1]))


# ─────────── persistence ───────────


def test_calibrator_path_naming():
    """Pin: ``btcusdt_1m.cbm`` → ``btcusdt_1m_calibrator.pkl``.
    Predictor relies on this convention."""
    p = calibrator_path_for(Path("/x/y/btcusdt_1m.cbm"))
    assert p == Path("/x/y/btcusdt_1m_calibrator.pkl")


def test_save_load_roundtrip(tmp_path):
    raw, y = _well_calibrated_data()
    cal = fit_platt_calibrator(raw, y)
    p = tmp_path / "test_calibrator.pkl"
    save_calibrator(cal, p)
    assert p.exists()
    loaded = load_calibrator(p)
    assert loaded is not None
    # Both produce the same calibrated output on a sample.
    np.testing.assert_allclose(
        apply_calibrator(cal, np.array([0.3, 0.7])),
        apply_calibrator(loaded, np.array([0.3, 0.7])),
        rtol=1e-9,
    )


def test_load_missing_returns_none(tmp_path):
    """Predictor's no-calibrator path: file missing → None → raw passthrough."""
    p = tmp_path / "absent.pkl"
    assert load_calibrator(p) is None


def test_load_corrupt_file_returns_none_not_raise(tmp_path):
    """A corrupt pickle file must NOT crash the predictor — returns
    None, runtime falls back to raw probability."""
    p = tmp_path / "corrupt.pkl"
    p.write_bytes(b"not a real pickle")
    out = load_calibrator(p)
    assert out is None


# ─────────── reliability curve ───────────


def test_reliability_curve_perfect_calibration_brier_low():
    """When raw probas exactly match observed rate, Brier should be
    near base-rate * (1-base-rate). Pin so a regression in the
    binning code shows up."""
    raw, y = _well_calibrated_data(n=5000)
    rc = compute_reliability_curve(raw, y, n_bins=10)
    # Brier upper bound for a balanced y is 0.25 (random guessing).
    # Well-calibrated should be ~0.20-0.24 depending on raw distribution.
    assert rc.brier_score < 0.26
    # ECE should be < 0.05 on 5000 samples.
    assert rc.ece < 0.05


def test_reliability_curve_miscalibration_higher_ece():
    """Miscalibrated input should have larger ECE than well-calibrated."""
    raw_good, y_good = _well_calibrated_data(n=5000)
    raw_bad, y_bad = _miscalibrated_data(n=5000)
    rc_good = compute_reliability_curve(raw_good, y_good, n_bins=10)
    rc_bad = compute_reliability_curve(raw_bad, y_bad, n_bins=10)
    assert rc_bad.ece > rc_good.ece


def test_reliability_curve_empty_input():
    rc = compute_reliability_curve(np.array([]), np.array([]), n_bins=10)
    assert rc.n_bins == 10
    assert all(c == 0 for c in rc.bin_counts)


def test_reliability_curve_bins_have_correct_shape():
    raw, y = _well_calibrated_data(n=500)
    rc = compute_reliability_curve(raw, y, n_bins=10)
    assert len(rc.bin_edges) == 11
    assert len(rc.bin_predicted) == 10
    assert len(rc.bin_observed) == 10
    assert len(rc.bin_counts) == 10
    assert sum(rc.bin_counts) == 500
    # Every populated bin has predicted ∈ [bin_low, bin_high].
    for i, (p_mean, n) in enumerate(zip(rc.bin_predicted, rc.bin_counts)):
        if n == 0:
            continue
        assert rc.bin_edges[i] - 1e-9 <= p_mean <= rc.bin_edges[i + 1] + 1e-9


def test_reliability_curve_to_dict_keys_pinned():
    rc = ReliabilityCurve(
        n_bins=2, bin_edges=[0.0, 0.5, 1.0],
        bin_predicted=[0.25, 0.75], bin_observed=[0.30, 0.70],
        bin_counts=[100, 100],
        brier_score=0.20, ece=0.04,
    )
    d = rc.to_dict()
    assert set(d.keys()) == {
        "n_bins", "bin_edges", "bin_predicted", "bin_observed",
        "bin_counts", "brier_score", "ece",
    }
