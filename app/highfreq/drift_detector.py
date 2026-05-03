"""Feature-distribution drift detection (T.18.b, 2026-05-03).

Why
===

The trainer fits CatBoost on data from the trainer's window (last
165 hours minus the 3-day frozen holdout, per T.16). At serve
time the predictor sees data from RIGHT NOW. If the live
distribution drifts from the training distribution (a regime shift,
fee change, exchange policy change), the model's predictions
silently degrade — dir_acc on realized OOS data drops, but until
realized labels accumulate (1+ minute lag) we can't detect it from
output alone.

Drift detection compares the **feature distributions** of recent
serve-time bars vs the trainer's reference window using a
two-sample Kolmogorov-Smirnov test per feature. KS is
distribution-free, scale-invariant, and well-conditioned on the
sample sizes we have (~10K reference vs ~500 recent).

What this exports
=================

* :func:`compute_per_feature_drift` — pure function: takes two
  feature DataFrames (reference, recent), returns one KS statistic
  + p-value per feature column.

* :func:`summarise_drift` — aggregates into a single severity score
  (max KS over features) + a list of features sorted by drift.

* :func:`alert_if_drifted` — applies a threshold (default
  KS_max ≥ 0.15 = noticeable shift) and returns a structured
  alert payload that ``tools.drift_check`` can ship to Telegram.

Defence narrative
-----------------

> "We track 18-27 feature distributions in real time. KS-test on
> serve vs train sample raises a Telegram alert if max KS exceeds
> 0.15 across any feature. Operator sees drift signal BEFORE
> realized accuracy degrades — the trader-facing dashboard isn't
> flying blind on a regime shift."

Limitations
-----------

* KS on a single feature ignores joint distribution shifts. A
  truly adversarial drift (correlations rotated while marginals
  preserved) would slip through. For our use case marginal-only
  drift catches most production regime shifts (volatility regime,
  fee tier change, ingest rotation).
* p-values aren't multiplicity-corrected — with 18 features at
  α=0.05, expected false-positive rate is ~60 %. Use the KS
  statistic threshold (0.15), not the p-value, for alerting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureDrift:
    """Per-feature drift result."""
    feature: str
    ks_stat: float
    p_value: float
    n_reference: int
    n_recent: int
    reference_mean: float
    recent_mean: float


@dataclass
class DriftSummary:
    """Aggregate drift result across all features."""
    n_features: int
    max_ks: float                       # worst-feature KS statistic
    max_ks_feature: str                 # feature name with max KS
    n_features_alarming: int            # count with ks ≥ threshold
    threshold: float                    # the alarm threshold used
    features: list[FeatureDrift] = field(default_factory=list)

    def is_drifted(self) -> bool:
        return self.max_ks >= self.threshold


#: Calendar features that ARE expected to drift between any two
#: time windows by construction (they encode the wall-clock day /
#: hour). KS will always fire max=1.0 if reference window covered
#: weekdays and recent window covers a weekend, even if the model's
#: behaviour is fine. Excluded from drift detection by default;
#: caller can override via ``include_calendar=True``.
_CALENDAR_FEATURES: frozenset[str] = frozenset({
    "day_of_week",
    "hour_of_day",
    "hour_of_week",
    "minute_of_hour",
})


def compute_per_feature_drift(
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    include_calendar: bool = False,
) -> list[FeatureDrift]:
    """Two-sample KS test per feature.

    Returns one ``FeatureDrift`` per column. Columns must exist in
    BOTH frames; missing columns are skipped (logged warning).

    Calendar features (``day_of_week``, ``hour_of_day``,
    ``hour_of_week``, ``minute_of_hour``) are excluded by default
    because they ALWAYS drift between two time windows — KS would
    always fire max=1.0 on them and drown out real distribution
    shifts elsewhere. Pass ``include_calendar=True`` to override
    (useful for ablation / debug, never for production alerting).

    KS is distribution-free, no assumption about feature shape.
    """
    if feature_columns is None:
        # Intersect columns; keep only numeric.
        common = [
            c for c in reference.columns
            if c in recent.columns and pd.api.types.is_numeric_dtype(reference[c])
        ]
    else:
        common = [
            c for c in feature_columns
            if c in reference.columns and c in recent.columns
        ]
    if not include_calendar:
        common = [c for c in common if c not in _CALENDAR_FEATURES]
    if not common:
        logger.warning("compute_per_feature_drift: no common numeric columns")
        return []

    # scipy KS — already a transitive dep via sklearn.
    from scipy.stats import ks_2samp

    out: list[FeatureDrift] = []
    for col in common:
        ref_vals = reference[col].dropna().to_numpy(dtype=float)
        rec_vals = recent[col].dropna().to_numpy(dtype=float)
        if len(ref_vals) < 30 or len(rec_vals) < 30:
            # KS on tiny samples is unreliable; skip rather than emit
            # a noisy false-alarm.
            continue
        try:
            ks = ks_2samp(ref_vals, rec_vals)
            ks_stat = float(ks.statistic)
            p_val = float(ks.pvalue)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ks_2samp failed for %s: %s — skipping", col, exc,
            )
            continue
        out.append(FeatureDrift(
            feature=col,
            ks_stat=ks_stat,
            p_value=p_val,
            n_reference=int(len(ref_vals)),
            n_recent=int(len(rec_vals)),
            reference_mean=float(ref_vals.mean()),
            recent_mean=float(rec_vals.mean()),
        ))
    return out


def summarise_drift(
    drifts: list[FeatureDrift],
    *,
    threshold: float = 0.15,
) -> DriftSummary:
    """Aggregate per-feature drift into one summary."""
    if not drifts:
        return DriftSummary(
            n_features=0, max_ks=0.0, max_ks_feature="",
            n_features_alarming=0, threshold=threshold, features=[],
        )
    sorted_drifts = sorted(drifts, key=lambda d: d.ks_stat, reverse=True)
    worst = sorted_drifts[0]
    n_alarm = sum(1 for d in drifts if d.ks_stat >= threshold)
    return DriftSummary(
        n_features=len(drifts),
        max_ks=worst.ks_stat,
        max_ks_feature=worst.feature,
        n_features_alarming=n_alarm,
        threshold=threshold,
        features=sorted_drifts,
    )


def alert_payload(summary: DriftSummary) -> dict[str, Any]:
    """Render a structured payload suitable for Telegram + JSON
    persistence.

    The payload is human-readable + machine-parsable: ``severity``
    is a coarse bucket the operator can grep for; ``top_features``
    surfaces the worst-N for at-a-glance triage.
    """
    severity = "ok"
    if summary.max_ks >= 0.30:
        severity = "high"
    elif summary.max_ks >= summary.threshold:
        severity = "warn"

    top = summary.features[:5]
    return {
        "severity": severity,
        "drifted": summary.is_drifted(),
        "max_ks": summary.max_ks,
        "max_ks_feature": summary.max_ks_feature,
        "threshold": summary.threshold,
        "n_features": summary.n_features,
        "n_features_alarming": summary.n_features_alarming,
        "top_features": [
            {
                "feature": f.feature,
                "ks_stat": f.ks_stat,
                "p_value": f.p_value,
                "reference_mean": f.reference_mean,
                "recent_mean": f.recent_mean,
                "n_ref": f.n_reference,
                "n_recent": f.n_recent,
            }
            for f in top
        ],
    }
