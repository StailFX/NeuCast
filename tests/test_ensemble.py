"""Tests for ``app.highfreq.ensemble`` (T.19).

Pure-function ensemble of N model probabilities, weight re-norming
on missing components, agreement flag.
"""
from __future__ import annotations

import pytest

from app.highfreq.ensemble import (
    DEFAULT_WEIGHTS_1M_15M,
    EnsembleComponent,
    EnsembleResult,
    ensemble_probability,
)


def _comp(label: str, weight: float, prob: float | None = None) -> EnsembleComponent:
    return EnsembleComponent(
        horizon_label=label,
        weight=weight,
        prob_up=prob,
        is_available=(prob is not None),
    )


def test_basic_weighted_average():
    """70/30 ensemble of 1m=0.60 and 15m=0.40: result = 0.7×0.6 + 0.3×0.4 = 0.54."""
    components = [_comp("1m", 0.70, 0.60), _comp("15m", 0.30, 0.40)]
    out = ensemble_probability(components)
    assert abs(out.prob_up - 0.54) < 1e-9
    assert out.n_components_used == 2
    assert out.effective_total_weight == 1.0


def test_agreement_flag_true_when_both_above_05():
    components = [_comp("1m", 0.70, 0.60), _comp("15m", 0.30, 0.55)]
    out = ensemble_probability(components)
    assert out.agreement is True


def test_agreement_flag_true_when_both_below_05():
    components = [_comp("1m", 0.70, 0.40), _comp("15m", 0.30, 0.45)]
    out = ensemble_probability(components)
    assert out.agreement is True


def test_agreement_flag_false_when_disagree():
    """1m says up (0.62), 15m says down (0.40) — they DISAGREE."""
    components = [_comp("1m", 0.70, 0.62), _comp("15m", 0.30, 0.40)]
    out = ensemble_probability(components)
    assert out.agreement is False


def test_neutral_component_does_not_break_agreement():
    """A 'neutral' (= 0.5) component agrees with either side. Pin
    so the agreement flag isn't accidentally False on the boundary."""
    components = [_comp("1m", 0.70, 0.62), _comp("15m", 0.30, 0.50)]
    out = ensemble_probability(components)
    assert out.agreement is True


def test_one_component_unavailable_renormalises_weights():
    """If 15m is unavailable, ensemble degenerates to 1m alone with
    re-normalised weight (70 / 70 = 1.0). prob_up = 1m's value."""
    components = [
        _comp("1m", 0.70, 0.62),
        _comp("15m", 0.30, None),  # unavailable
    ]
    out = ensemble_probability(components)
    assert abs(out.prob_up - 0.62) < 1e-9
    assert out.n_components_used == 1
    assert out.effective_total_weight == 0.70


def test_no_components_available_raises():
    """If neither model produced a prediction, the caller MUST handle
    it explicitly. We refuse to silently emit 0.5."""
    components = [
        _comp("1m", 0.70, None),
        _comp("15m", 0.30, None),
    ]
    with pytest.raises(ValueError, match="no available components"):
        ensemble_probability(components)


def test_zero_weight_raises():
    """All available components have weight 0 → can't divide by 0;
    raise rather than emit 0/0 = NaN."""
    components = [_comp("1m", 0.0, 0.60)]
    with pytest.raises(ValueError, match="total weight"):
        ensemble_probability(components)


def test_to_dict_includes_unavailable_components():
    """The breakdown should include ALL components — including
    unavailable ones — so the UI can show "15m: ?" rather than
    silently hide the missing model."""
    components = [
        _comp("1m", 0.70, 0.62),
        _comp("15m", 0.30, None),
    ]
    out = ensemble_probability(components)
    d = out.to_dict()
    assert len(d["components"]) == 2
    by_label = {c["horizon_label"]: c for c in d["components"]}
    assert by_label["15m"]["is_available"] is False
    assert by_label["15m"]["prob_up"] is None


def test_default_weights_sum_to_one():
    """Sanity: if a caller uses DEFAULT_WEIGHTS_1M_15M as is,
    they get a valid probability distribution."""
    assert abs(sum(DEFAULT_WEIGHTS_1M_15M.values()) - 1.0) < 1e-9
    # Both components have positive weight (no degenerate split).
    assert all(w > 0 for w in DEFAULT_WEIGHTS_1M_15M.values())


def test_three_component_ensemble():
    """Generalises beyond 2 components — useful for future
    multi-horizon (1m + 5m + 15m) ablations."""
    components = [
        _comp("1m", 0.5, 0.60),
        _comp("5m", 0.3, 0.55),
        _comp("15m", 0.2, 0.50),
    ]
    out = ensemble_probability(components)
    expected = 0.5 * 0.60 + 0.3 * 0.55 + 0.2 * 0.50
    assert abs(out.prob_up - expected) < 1e-9
    assert out.n_components_used == 3
