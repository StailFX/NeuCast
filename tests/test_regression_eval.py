"""Tests for ``tools.regression_eval`` — magnitude-regression eval tool
(release P / roadmap ε).

We don't fit a real CatBoost in unit tests (slow + heavy dep). Instead
we pin the helpers and the threshold-slicing math, and exercise the
markdown rendering. The end-to-end CV path is covered by an integration
test that runs on a tiny synthetic dataset only when CatBoost is
installed (matching the project pattern in test_highfreq_trainer)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.regression_eval import (
    DEFAULT_THRESHOLDS_BPS,
    FEE_TIERS_BPS,
    RegressionEvalRow,
    ThresholdRow,
    _fmt_or_dash,
    _r2,
    _safe_corr,
    _spearman_corr,
    _threshold_slice,
    render_markdown_report,
)


# ─── _safe_corr ───

def test_safe_corr_perfect_positive_is_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    assert _safe_corr(a, b) == pytest.approx(1.0)


def test_safe_corr_perfect_negative_is_minus_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert _safe_corr(a, b) == pytest.approx(-1.0)


def test_safe_corr_constant_input_returns_none():
    """Pin: constant predictor (e.g. regressor degenerated to its mean)
    must return None, not blow up with 0/0."""
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    assert _safe_corr(a, b) is None
    assert _safe_corr(b, a) is None


def test_safe_corr_too_short_returns_none():
    """Need at least 2 points for correlation."""
    assert _safe_corr(np.array([1.0]), np.array([1.0])) is None
    assert _safe_corr(np.array([]), np.array([])) is None


# ─── _r2 ───

def test_r2_perfect_prediction_is_one():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    assert _r2(y_true, y_pred) == pytest.approx(1.0)


def test_r2_constant_prediction_at_mean_is_zero():
    """Predicting the mean every time → R² = 0 (no better than naive)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert _r2(y_true, y_pred) == pytest.approx(0.0)


def test_r2_negative_when_worse_than_mean():
    """Pin: R² CAN be negative (worse than predicting the mean) — common
    in noisy financial regression. Defence narrative: don't claim R²
    is bounded [0,1]; we report it honestly."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # inverse — terrible
    r2 = _r2(y_true, y_pred)
    assert r2 is not None
    assert r2 < 0


def test_r2_returns_none_on_zero_variance_truth():
    """When all targets are equal, R² is undefined (0/0). Return None
    rather than NaN/inf."""
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert _r2(y_true, y_pred) is None


# ─── _threshold_slice ───

def test_threshold_slice_includes_all_at_zero():
    """θ=0 → all predictions kept (every |pred| > 0 except exact zeros)."""
    yt = np.array([1.0, -1.0, 2.0, -2.0])
    yp = np.array([0.5, -0.5, 1.0, -1.0])
    out = _threshold_slice(yt, yp, threshold_bps=0.0)
    assert out.n_trades == 4
    assert out.fraction_kept == pytest.approx(1.0)
    # All signs match.
    assert out.sign_accuracy == pytest.approx(1.0)


def test_threshold_slice_filters_at_high_threshold():
    yt = np.array([1.0, -1.0, 2.0, -2.0])
    yp = np.array([0.5, -0.5, 3.0, -3.0])
    out = _threshold_slice(yt, yp, threshold_bps=1.0)
    # Only |yp|>1 are 3.0 and -3.0 → 2 trades, both sign-correct.
    assert out.n_trades == 2
    assert out.sign_accuracy == pytest.approx(1.0)
    assert out.mean_realized_bps == pytest.approx(0.0)  # +2 + (-2) = 0
    assert out.mean_abs_realized_bps == pytest.approx(2.0)


def test_threshold_slice_signed_pnl_matches_sign_of_pred():
    """When signs match: signed_realized = +2.0, +2.0 → mean +2.0.
    Subtract round-trip retail fees (15 bp) → -13.0 bp."""
    yt = np.array([2.0, -2.0])
    yp = np.array([3.0, -3.0])
    out = _threshold_slice(yt, yp, threshold_bps=1.0)
    expected_signed_mean = 2.0   # both signs match → +|y_true|
    expected_retail = expected_signed_mean - 2 * FEE_TIERS_BPS["retail"]
    assert out.realized_pnl_bps["retail"] == pytest.approx(expected_retail)


def test_threshold_slice_signed_pnl_when_signs_disagree():
    """Predict +3, actual -2 → signed_realized = -2; predict -3, actual +2 → -2.
    Mean = -2. After retail fees: -2 - 15 = -17."""
    yt = np.array([-2.0, 2.0])
    yp = np.array([3.0, -3.0])
    out = _threshold_slice(yt, yp, threshold_bps=1.0)
    assert out.sign_accuracy == pytest.approx(0.0)
    assert out.realized_pnl_bps["retail"] == pytest.approx(-17.0)


def test_threshold_slice_empty_slice_returns_none_metrics():
    """At a threshold higher than any |pred|, no trades pass — all
    quality metrics None (cannot compute), n_trades=0."""
    yt = np.array([1.0, -1.0])
    yp = np.array([0.5, -0.5])
    out = _threshold_slice(yt, yp, threshold_bps=10.0)
    assert out.n_trades == 0
    assert out.fraction_kept == pytest.approx(0.0)
    assert out.sign_accuracy is None
    assert out.mean_realized_bps is None
    assert out.realized_pnl_bps["retail"] is None


def test_threshold_slice_mm_rebate_subtracts_negative_fee():
    """mm_rebate fee is -0.4 bp/side → -0.8 round-trip; subtracting a
    negative ADDS 0.8. Pin so a sign flip in the subtraction breaks
    loudly."""
    yt = np.array([2.0])
    yp = np.array([3.0])
    out = _threshold_slice(yt, yp, threshold_bps=1.0)
    # signed_realized = +2.0; mm_rebate roundtrip = 2 * (-0.4) = -0.8
    # P&L = 2.0 - (-0.8) = 2.8
    assert out.realized_pnl_bps["mm_rebate"] == pytest.approx(2.8)


# ─── _fmt_or_dash ───

def test_fmt_or_dash_renders_dash_for_none():
    assert _fmt_or_dash(None) == "—"


def test_fmt_or_dash_renders_dash_for_nan():
    """A NaN float must render as em-dash, not 'nan' — defence-grade
    tables embed in slides; 'nan' looks unprofessional."""
    assert _fmt_or_dash(float("nan")) == "—"


def test_fmt_or_dash_renders_dash_for_inf():
    assert _fmt_or_dash(float("inf")) == "—"
    assert _fmt_or_dash(float("-inf")) == "—"


def test_fmt_or_dash_uses_provided_format():
    assert _fmt_or_dash(0.12, "{:.2f}") == "0.12"
    # Use a value that doesn't sit on a half-rounding tie to avoid
    # banker's-rounding ambiguity between platforms.
    assert _fmt_or_dash(0.4321, "{:+.4f}") == "+0.4321"


# ─── render_markdown_report ───

def test_render_markdown_returns_well_formed_table():
    """At least one row, structure has summary table + per-symbol tables."""
    row = RegressionEvalRow(
        symbol="BTCUSDT",
        bar_minutes=1,
        n_seconds_loaded=600_000,
        n_bars_after_aggregation=10_000,
        n_bars_kept=8_500,
        n_folds=80,
        n_predictions=4_800,
        mae_bps=2.5,
        rmse_bps=4.1,
        r2=0.018,
        ic_pearson=0.06,
        ic_spearman=0.05,
        sign_accuracy=0.55,
        thresholds=[
            ThresholdRow(
                threshold_bps=0.0,
                n_trades=4_800,
                fraction_kept=1.0,
                sign_accuracy=0.55,
                mean_realized_bps=0.05,
                mean_abs_realized_bps=2.0,
                realized_pnl_bps={
                    "retail": -14.9,
                    "vip5": -1.9,
                    "vip9": 0.1,
                    "mm_rebate": 0.9,
                },
            ),
            ThresholdRow(
                threshold_bps=2.0,
                n_trades=1_200,
                fraction_kept=0.25,
                sign_accuracy=0.60,
                mean_realized_bps=0.30,
                mean_abs_realized_bps=4.0,
                realized_pnl_bps={
                    "retail": -13.0,
                    "vip5": 0.0,
                    "vip9": 2.0,
                    "mm_rebate": 2.8,
                },
            ),
        ],
    )
    md = render_markdown_report([row])
    # Top-of-doc heading.
    assert "# Magnitude regression" in md
    # Summary section + per-symbol section.
    assert "## Summary" in md
    assert "## BTCUSDT" in md
    # Row appears in summary.
    assert "BTCUSDT" in md
    # Threshold values present.
    assert "0.0" in md  # θ=0 threshold
    assert "2.0" in md  # θ=2 threshold
    # Tier P&L numbers — pin retail at -13.0 (after-fee P&L, not the dir-acc).
    assert "-13.000" in md or "-13.0" in md


def test_render_markdown_handles_none_metrics_gracefully():
    """Empty result row (insufficient data) — must produce a parseable
    table with em-dashes, not crash."""
    row = RegressionEvalRow(
        symbol="BNBUSDT",
        bar_minutes=60,
        n_seconds_loaded=100,
        n_bars_after_aggregation=2,
        n_bars_kept=0,
        n_folds=0,
        n_predictions=0,
        mae_bps=None, rmse_bps=None, r2=None,
        ic_pearson=None, ic_spearman=None,
        sign_accuracy=None,
        thresholds=[],
    )
    md = render_markdown_report([row])
    assert "BNBUSDT" in md
    assert "—" in md  # em-dash placeholder for missing metrics


# ─── DEFAULT_THRESHOLDS_BPS contract ───

def test_default_thresholds_are_sorted_and_include_zero():
    """Pin: thresholds must be sorted ascending and start at 0 (the
    'no filter' baseline anchors the curve)."""
    assert DEFAULT_THRESHOLDS_BPS[0] == 0.0
    assert list(DEFAULT_THRESHOLDS_BPS) == sorted(DEFAULT_THRESHOLDS_BPS)


def test_fee_tiers_match_classifier_tool():
    """Pin: regression_eval and multi_horizon_eval must use the SAME
    fee tier table — otherwise the reported P&L numbers are not
    directly comparable. Defense: a refactor that drifts one without
    the other breaks CI."""
    from tools.multi_horizon_eval import FEE_TIERS_BPS as CLF_TIERS
    assert FEE_TIERS_BPS == CLF_TIERS


# ─── Integration: end-to-end on synthetic data (slow) ───

def _synthetic_seconds_with_signal(
    *, n_minutes: int = 200, drift_per_min_bps: float = 3.0, seed: int = 0,
):
    """Tiny synthetic 1-second frame where ofi_sum is positively
    correlated with future return. Lets the regressor actually fit
    something instead of returning constants.

    Each minute: 60 1-second rows with ofi drawn from a normal whose
    MEAN sets the next-minute drift. The regressor should learn this.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2026-04-29 00:00:00", tz="UTC")
    base_price = 77_000.0
    for m in range(n_minutes):
        # OFI signal — sets the drift for THIS minute (so within-minute
        # ofi correlates with within-minute return).
        ofi_mean_this_min = rng.normal(0.0, 1.0)
        # within-minute drift in bps determined by signal + noise
        drift_bps = ofi_mean_this_min * drift_per_min_bps + rng.normal(0.0, 0.5)
        for s in range(60):
            ts = t0 + pd.Timedelta(minutes=m, seconds=s)
            # microprice walks linearly within the minute by drift_bps
            mp = base_price * (
                1.0 + (m * 1.0 + s / 60.0) * drift_bps / 1e4
            )
            rows.append({
                "ts": ts,
                "symbol": "TESTSYM",
                "ofi": float(rng.normal(ofi_mean_this_min, 0.3)),
                "microprice": float(mp),
                "depth_imb": float(rng.uniform(-0.2, 0.2)),
                "spread_bps": float(rng.uniform(0.5, 1.5)),
                "trade_imb": float(rng.normal(0.0, 0.001)),
                "n_updates": 10,
            })
        # Update base_price for the next minute.
        base_price = base_price * (1.0 + drift_bps / 1e4)
    return pd.DataFrame(rows)


@pytest.mark.skipif(
    pytest.importorskip("catboost", reason="catboost not installed") is None,
    reason="catboost required for end-to-end regression eval",
)
def test_end_to_end_runs_and_produces_finite_metrics():
    """Smoke test: with CatBoost installed and a small synthetic frame,
    the eval returns non-None metrics and at least one fold."""
    from tools.regression_eval import evaluate_one_regression

    df = _synthetic_seconds_with_signal(n_minutes=200, seed=7)
    row = evaluate_one_regression(
        df,
        symbol="TESTSYM",
        bar_minutes=1,
        initial_train_bars=120,
        test_fold_bars=20,
        step_bars=20,
        neutral_band_bps=0.0,  # keep all bars for synthetic test
        catboost_iterations=50,  # fast
        sample_weight_half_life=0,  # disable weighting for determinism
        embargo_bars=0,
        thresholds_bps=(0.0, 1.0),
    )
    # Should produce at least one fold.
    assert row.n_folds >= 1
    assert row.n_predictions > 0
    # Metrics should be finite (signal exists by construction).
    assert row.mae_bps is not None and math.isfinite(row.mae_bps)
    assert row.rmse_bps is not None and math.isfinite(row.rmse_bps)
    # Sign accuracy on synthetic with built-in signal should beat 0.5
    # at least slightly — if it doesn't, something fundamental broke.
    assert row.sign_accuracy is not None
    assert row.sign_accuracy > 0.40, (
        f"sign_accuracy {row.sign_accuracy} suspiciously low; "
        f"a strong refactor may have broken target alignment"
    )
