"""Tests for split-conformal prediction intervals (T.17.b).

The trainer computes nonconformity scores ``s_i = |proba_i - y_i|``
on pooled walk-forward OOS predictions, takes the
``ceil((n+1)(1-α))/n`` quantile to get a 1-α-coverage prediction
band halfwidth ``q``. Predictor stores ``q`` in metrics.json and
exposes ``conformal_interval(prob_up, alpha)`` → (low, high) for
the live forecast endpoint.

Tests pin:
1. ``conformal_quantile`` reads the right field per α.
2. ``conformal_interval`` clamps to [0, 1].
3. None when metrics.json missing the field (legacy model).
4. Trainer integration: q monotonically larger for larger 1-α
   (95% coverage requires wider interval than 90%).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ───────────────────── Predictor.conformal_* ─────────────────────


def _build_predictor_with_metrics(metrics_json: dict, tmp_path: Path):
    """Build a LivePredictor pointing at a tmp metrics.json."""
    from app.highfreq.predictor import LivePredictor

    weights_path = tmp_path / "btcusdt_1m.cbm"
    weights_path.write_bytes(b"placeholder")  # not loaded for these tests
    metrics_path = tmp_path / "btcusdt_1m_metrics.json"
    metrics_path.write_text(json.dumps(metrics_json))
    p = LivePredictor(weights_path=weights_path, metrics_path=metrics_path)
    # Force metrics load.
    p._maybe_reload_metrics()
    return p


def test_conformal_quantile_reads_alpha_0_10(tmp_path):
    p = _build_predictor_with_metrics({
        "conformal_q_alpha_0_10": 0.18,
        "conformal_q_alpha_0_05": 0.25,
    }, tmp_path)
    assert p.conformal_quantile(alpha=0.10) == 0.18


def test_conformal_quantile_reads_alpha_0_05(tmp_path):
    p = _build_predictor_with_metrics({
        "conformal_q_alpha_0_10": 0.18,
        "conformal_q_alpha_0_05": 0.25,
    }, tmp_path)
    assert p.conformal_quantile(alpha=0.05) == 0.25


def test_conformal_quantile_unknown_alpha_returns_none(tmp_path):
    """We only ship α=0.10 and α=0.05. Anything else → None (let
    callers know it's not supported rather than silently return
    a stale value)."""
    p = _build_predictor_with_metrics({
        "conformal_q_alpha_0_10": 0.18,
    }, tmp_path)
    assert p.conformal_quantile(alpha=0.20) is None


def test_conformal_quantile_legacy_metrics_returns_none(tmp_path):
    """A model trained before T.17.b lands has no conformal_q_*
    fields → predictor returns None so the forecast endpoint omits
    the interval rather than emit a fake one."""
    p = _build_predictor_with_metrics({
        "dir_acc_mean": 0.55,  # only legacy fields
    }, tmp_path)
    assert p.conformal_quantile(alpha=0.10) is None
    assert p.conformal_quantile(alpha=0.05) is None


def test_conformal_interval_clamps_to_unit(tmp_path):
    """Interval must be clamped to [0, 1] — probabilities outside
    that range are nonsensical even when prob ± q overflows."""
    p = _build_predictor_with_metrics({
        "conformal_q_alpha_0_10": 0.30,
    }, tmp_path)
    # prob_up = 0.5 → interval ±0.30 → [0.20, 0.80]
    lo, hi = p.conformal_interval(0.5, alpha=0.10)
    assert abs(lo - 0.20) < 1e-9
    assert abs(hi - 0.80) < 1e-9
    # prob_up = 0.05 → interval would be [-0.25, 0.35] → clamped to [0, 0.35]
    lo, hi = p.conformal_interval(0.05, alpha=0.10)
    assert lo == 0.0
    assert abs(hi - 0.35) < 1e-9
    # prob_up = 0.95 → [0.65, 1.25] → clamped to [0.65, 1]
    lo, hi = p.conformal_interval(0.95, alpha=0.10)
    assert abs(lo - 0.65) < 1e-9
    assert hi == 1.0


def test_conformal_interval_returns_none_when_quantile_missing(tmp_path):
    p = _build_predictor_with_metrics({"dir_acc_mean": 0.55}, tmp_path)
    assert p.conformal_interval(0.5, alpha=0.10) is None


# ───────────────── Trainer-side conformal-q computation ─────────────────
#
# Mirror the math from run_training; pin that q@α=0.05 ≥ q@α=0.10
# (95% coverage is strictly wider than 90%) and that q ∈ [0, 1].


def _conformal_q(scores: np.ndarray, alpha: float) -> float:
    """Replicate the trainer's quantile computation for testing.

    Angelopoulos & Bates 2023: q = ceil((n+1)(1-α))/n quantile of
    nonconformity scores, clamped to [0, 1].
    """
    import math
    n = len(scores)
    q_idx = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_idx_clipped = min(1.0, max(0.0, q_idx))
    return float(np.quantile(scores, q_idx_clipped))


def test_conformal_q_monotone_in_coverage():
    """Higher coverage (smaller α) ⇒ wider interval ⇒ larger q.
    Synthetic scores with realistic shape from a calibrated 1-min
    model: most |proba - y| in [0.3, 0.5], some near 0 or 1."""
    rng = np.random.default_rng(0)
    scores = np.clip(rng.beta(2, 2, size=1000), 0, 1)
    q_90 = _conformal_q(scores, alpha=0.10)
    q_95 = _conformal_q(scores, alpha=0.05)
    assert q_95 >= q_90
    assert 0.0 <= q_90 <= 1.0
    assert 0.0 <= q_95 <= 1.0


def test_conformal_q_coverage_holds_empirically():
    """For exchangeable test data, fraction of |proba - y| ≤ q
    must be at least 1 - α. We synthesize 5K calibration scores +
    5K test scores from same distribution and verify."""
    rng = np.random.default_rng(0)
    cal = rng.beta(2, 2, size=5000)
    test = rng.beta(2, 2, size=5000)  # exchangeable with cal
    q = _conformal_q(cal, alpha=0.10)
    coverage = float((test <= q).mean())
    # Allow small slack — exact 1-α coverage is asymptotic; for n=5K
    # we expect within ±2pp of 0.90.
    assert 0.86 <= coverage <= 0.94, f"coverage {coverage} not in [0.86, 0.94]"
