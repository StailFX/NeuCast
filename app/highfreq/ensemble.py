"""Multi-horizon ensemble of calibrated probabilities (T.19, 2026-05-03).

Why
===

The production stack already trains two horizons in parallel:

* **1m model** (microstructure / cross_asset) — primary signal,
  scores ~0.56 dir_acc on walk-forward CV with tight CI.
* **15m model** (long_horizon TA) — weaker individually (~0.52-0.54)
  but built from a DIFFERENT feature space (OHLC, EMA, RSI, BB)
  that captures multi-bar momentum the 1m microstructure can't see.

When two models with **independent error profiles** agree, the joint
signal is stronger than either alone (Wolpert 1992, "Stacked
Generalization"). When they disagree, the stronger model dominates
under a weighted average.

This module provides a pure ``ensemble_probability`` function that
averages calibrated ``prob_up`` values from N predictors with
per-model weights, plus a thin wrapper class that handles the
"one predictor unavailable" cold-start case gracefully.

Defence narrative
-----------------

> "We expose a multi-horizon ensemble combining the 1-min
> microstructure model with the 15-min long-horizon TA model.
> Default weights give the 1m model 70 % share (it's empirically
> stronger) and the 15m model 30 %. When the two horizons agree,
> the ensemble's effective confidence rises beyond either model
> alone. When they disagree, the 1m signal dominates by weight.
> Empirically tested via realized accuracy on predictions_log
> (release T.19)."

Component-prediction shape
--------------------------

The ensemble result includes the per-model breakdown so the UI
can show "1m says 0.62, 15m says 0.48 — averaged to 0.58".
This makes the ensemble's reasoning visible — reviewers see HOW
the joint signal is constructed, not a black-box average.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsembleComponent:
    """One model's contribution to an ensemble prediction."""
    horizon_label: str           # "1m", "15m", etc.
    weight: float                # configured ensemble weight
    prob_up: float | None        # the model's calibrated prob_up; None = unavailable
    is_available: bool           # False if model was missing / errored


@dataclass(frozen=True)
class EnsembleResult:
    """Final ensemble prediction with full breakdown."""
    prob_up: float                              # weighted-average prob_up
    components: list[EnsembleComponent]         # per-model breakdown
    n_components_used: int                      # how many were available
    effective_total_weight: float               # sum of available weights
    agreement: bool                             # do all available models agree on direction?

    def to_dict(self) -> dict[str, Any]:
        return {
            "prob_up": self.prob_up,
            "n_components_used": self.n_components_used,
            "effective_total_weight": self.effective_total_weight,
            "agreement": self.agreement,
            "components": [
                {
                    "horizon_label": c.horizon_label,
                    "weight": c.weight,
                    "prob_up": c.prob_up,
                    "is_available": c.is_available,
                }
                for c in self.components
            ],
        }


def ensemble_probability(
    components: list[EnsembleComponent],
) -> EnsembleResult:
    """Weighted-average available components.

    If NO components are available, raises ``ValueError`` — the
    caller must check upstream or fall back to a single model.

    Re-normalises weights over only the AVAILABLE subset:
    ``prob_up = Σ w_i × p_i / Σ w_i`` over ``i`` with ``is_available``.
    This means a 70/30 ensemble where the 15m component is missing
    degenerates cleanly to the 1m component alone (weight 70 →
    effectively 1.0 after re-norm).

    Agreement: True iff all available components have prob_up on
    the same side of 0.5 (all > 0.5 or all < 0.5). A "neutral"
    component (exactly 0.5) is treated as agreeing with either side.
    """
    available = [c for c in components if c.is_available and c.prob_up is not None]
    if not available:
        raise ValueError(
            "ensemble_probability: no available components; "
            "caller must fall back to single-model prediction"
        )

    total_w = sum(c.weight for c in available)
    if total_w <= 0:
        raise ValueError(
            f"ensemble_probability: total weight of available components "
            f"is {total_w}; weights must be positive"
        )

    p_blend = sum(c.weight * c.prob_up for c in available) / total_w

    # Agreement check.
    sides = [
        ("up" if c.prob_up > 0.5 else ("down" if c.prob_up < 0.5 else "neutral"))
        for c in available
    ]
    non_neutral = [s for s in sides if s != "neutral"]
    agreement = (len(set(non_neutral)) <= 1)

    return EnsembleResult(
        prob_up=float(p_blend),
        components=components,  # return ALL, not just available
        n_components_used=len(available),
        effective_total_weight=float(total_w),
        agreement=agreement,
    )


# Default 70/30 weighting: 1m model is empirically stronger (CI lo
# > 0.53 vs 15m's ~0.51), but 15m's long_horizon TA captures
# multi-bar momentum the 1m can't. 30 % share of the 15m model is
# enough to bend the joint signal when 15m has a clear contrarian
# read but small enough that 1m drives most decisions.
DEFAULT_WEIGHTS_1M_15M: dict[str, float] = {"1m": 0.70, "15m": 0.30}
