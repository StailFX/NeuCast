"""Probability calibration via Platt scaling.

Why this exists
===============

CatBoost's raw ``predict_proba`` returns scores that are often
**miscalibrated** on small training sets — a "0.60 raw score" might
correspond to a true 0.55 hit rate (model overconfident) or 0.65
(underconfident). This matters for the trader: the entry threshold
``prob_up >= 0.60`` should mean "60 % chance the next minute closes
up", not "raw score in the upper 40 % of the distribution".

Platt scaling fits a 1-parameter logistic regression on
``(raw_proba, y_true)`` from walk-forward CV's pooled OOS predictions.
At serve time we apply this transformation before the trader's
threshold check. Output:

* Calibrated probabilities have meaningful frequency interpretation.
* The trader's threshold is operationally honest.
* The reliability diagram (predicted vs observed across binned proba)
  is a defence-grade visual: a perfectly calibrated model lies on the
  diagonal; deviations show miscalibration.

Why Platt, not isotonic
-----------------------

Isotonic regression is more flexible (monotonic step function) but
needs more data — at our n≈2000 OOS it tends to overfit. Platt is
parametric (logistic curve, 2 params) and well-conditioned on our
sample size. Standard choice for sklearn / financial ML.

Reliability diagram
-------------------

:func:`compute_reliability_curve` bins predicted probabilities into
``n_bins`` equal-width buckets and computes (mean predicted proba,
mean observed y) per bin. Plotting these gives the diagram. We don't
plot here — caller (defence slide / Telegram bot) renders.

Defence narrative
-----------------

> "Raw model output is calibrated post-hoc via Platt scaling on
> walk-forward OOS predictions. The reliability diagram shows
> predicted-vs-observed alignment within ±N pp across all bins —
> the trader's 0.60 threshold corresponds to 60 % empirical hit
> rate, not an arbitrary score percentile."
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin diagnostics. Length-bins lists; bin_predicted[i] is
    the mean predicted proba of bin i, bin_observed[i] is the
    mean realised y in that bin."""
    n_bins: int
    bin_edges: list[float]            # length n_bins + 1
    bin_predicted: list[float | None]  # mean of raw/calibrated proba per bin
    bin_observed: list[float | None]   # mean of y per bin
    bin_counts: list[int]              # n samples per bin
    # Brier score is the canonical scalar calibration metric: mean of
    # (predicted - observed)². Lower is better; perfect calibration
    # gives ≈base-rate × (1−base-rate).
    brier_score: float
    # ECE (expected calibration error) — weighted by bin counts.
    ece: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Fit / apply
# ──────────────────────────────────────────────────────────────────────


def fit_platt_calibrator(
    raw_probas: np.ndarray,
    y_true: np.ndarray,
) -> Any:
    """Fit a Platt scaler.

    Internally a sklearn LogisticRegression on the single feature
    ``logit(raw_proba)`` (logit transform stabilises training when
    the input is itself a probability — a vanilla LR on raw probs
    works too but the logit is the textbook formulation).

    Returns the fitted estimator. Use :func:`apply_calibrator` to
    apply, :func:`save_calibrator` / :func:`load_calibrator` to
    persist.

    Edge case — single-class CV folds: if ``y_true`` is all-0 or
    all-1, LogisticRegression refuses. We return a passthrough
    object that returns the input unchanged.
    """
    raw_probas = np.asarray(raw_probas, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()

    if len(raw_probas) != len(y_true):
        raise ValueError(
            f"length mismatch: raw_probas={len(raw_probas)} vs y_true={len(y_true)}"
        )
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "calibration: y_true has single class only — returning passthrough"
        )
        return _PassthroughCalibrator()

    # Clip to avoid -inf from logit(0) or +inf from logit(1).
    eps = 1e-6
    p = np.clip(raw_probas, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(logits, y_true)
    return clf


class _PassthroughCalibrator:
    """Stand-in returned when fitting can't proceed (single-class
    folds). Calling apply_calibrator on it is a no-op identity.

    Pickle-safe via joblib — it has no state to serialise."""
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Match sklearn convention: shape (n, 2), columns are P(0), P(1).
        # X here is the logit; we return the inverse via sigmoid.
        x = np.asarray(X, dtype=float).ravel()
        sig = 1.0 / (1.0 + np.exp(-x))
        return np.column_stack([1.0 - sig, sig])


def apply_calibrator(calibrator: Any, raw_proba: float | np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator. Accepts scalar or array input,
    always returns numpy array (1-d) for consistency.

    No-op-safe: if ``calibrator`` is None, returns the raw input
    unchanged. Used by the predictor's hot path: a model file with
    no companion calibrator falls back to raw output cleanly.
    """
    arr = np.atleast_1d(np.asarray(raw_proba, dtype=float))
    if calibrator is None:
        return arr
    eps = 1e-6
    p = np.clip(arr, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    proba_2col = calibrator.predict_proba(logits)
    return proba_2col[:, 1]


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────


def calibrator_path_for(weights_path: Path) -> Path:
    """``btcusdt_1m.cbm`` → ``btcusdt_1m_calibrator.pkl`` (sibling)."""
    return weights_path.with_suffix("").with_name(
        weights_path.stem + "_calibrator.pkl"
    )


def save_calibrator(calibrator: Any, path: Path) -> None:
    """Atomic save via tempfile + rename (so a half-written pickle
    can't be loaded)."""
    import joblib
    import tempfile, os
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent),
    )
    os.close(fd)
    try:
        joblib.dump(calibrator, tmp_name)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_calibrator(path: Path) -> Any | None:
    """Returns the fitted calibrator or None if file is missing /
    unreadable. Fail-soft: an unreadable calibrator should NOT take
    down the predictor — it falls back to raw output."""
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        logger.warning("calibrator: load failed for %s: %s", path, exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Reliability diagram
# ──────────────────────────────────────────────────────────────────────


def compute_reliability_curve(
    probas: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Bin predictions into equal-width [0, 1] buckets and compute
    per-bin (predicted_mean, observed_mean, count) plus aggregate
    Brier score and ECE.

    A perfectly calibrated model: bin_predicted[i] ≈ bin_observed[i]
    for every i — the diagonal on the reliability plot.
    """
    probas = np.asarray(probas, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()
    if len(probas) != len(y_true):
        raise ValueError(
            f"length mismatch: probas={len(probas)} vs y_true={len(y_true)}"
        )
    if len(probas) == 0:
        return ReliabilityCurve(
            n_bins=n_bins,
            bin_edges=list(np.linspace(0, 1, n_bins + 1)),
            bin_predicted=[None] * n_bins,
            bin_observed=[None] * n_bins,
            bin_counts=[0] * n_bins,
            brier_score=float("nan"),
            ece=float("nan"),
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probas, edges[1:-1]), 0, n_bins - 1)

    bin_pred: list[float | None] = []
    bin_obs: list[float | None] = []
    bin_n: list[int] = []
    n_total = len(probas)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_idx == i
        n = int(mask.sum())
        if n == 0:
            bin_pred.append(None)
            bin_obs.append(None)
            bin_n.append(0)
            continue
        p_mean = float(probas[mask].mean())
        y_mean = float(y_true[mask].mean())
        bin_pred.append(p_mean)
        bin_obs.append(y_mean)
        bin_n.append(n)
        ece += (n / n_total) * abs(p_mean - y_mean)

    brier = float(((probas - y_true.astype(float)) ** 2).mean())

    return ReliabilityCurve(
        n_bins=n_bins,
        bin_edges=[float(e) for e in edges],
        bin_predicted=bin_pred,
        bin_observed=bin_obs,
        bin_counts=bin_n,
        brier_score=brier,
        ece=float(ece),
    )
