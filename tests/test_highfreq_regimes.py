"""Tests for ``app.highfreq.regimes`` — vol-regime classifier."""
from __future__ import annotations

import numpy as np
import pytest

from app.highfreq.regimes import (
    DEFAULT_HIGH_PCT,
    DEFAULT_LOW_PCT,
    classify_regime,
    compute_regime_thresholds,
    label_history,
)


# ───────────────────────── classify_regime ─────────────────────────


def test_classify_regime_below_low_is_calm():
    assert classify_regime(2.0, p_low=5.0, p_high=10.0) == "calm"


def test_classify_regime_between_is_normal():
    assert classify_regime(7.0, p_low=5.0, p_high=10.0) == "normal"


def test_classify_regime_at_or_above_high_is_turbulent():
    assert classify_regime(10.0, p_low=5.0, p_high=10.0) == "turbulent"
    assert classify_regime(15.0, p_low=5.0, p_high=10.0) == "turbulent"


def test_classify_regime_at_low_boundary_is_normal():
    """Boundary semantics: == p_low → normal (only strictly < is calm)."""
    assert classify_regime(5.0, p_low=5.0, p_high=10.0) == "normal"


# ───────────────────────── compute_regime_thresholds ─────────────────────────


def test_compute_thresholds_returns_zeros_on_empty():
    p_low, p_high = compute_regime_thresholds(np.array([]))
    assert p_low == 0.0 and p_high == 0.0


def test_compute_thresholds_uses_default_percentiles():
    """Uniform [0..99] → p33 ≈ 32.7, p66 ≈ 65.3."""
    vols = np.arange(100, dtype=float)
    p_low, p_high = compute_regime_thresholds(vols)
    assert p_low == pytest.approx(32.67, abs=0.5)
    assert p_high == pytest.approx(65.34, abs=0.5)


def test_compute_thresholds_respects_custom_percentiles():
    vols = np.arange(100, dtype=float)
    p_low, p_high = compute_regime_thresholds(vols, low_pct=10, high_pct=90)
    assert p_low == pytest.approx(9.9, abs=0.5)
    assert p_high == pytest.approx(89.1, abs=0.5)


# ───────────────────────── label_history ─────────────────────────


def test_label_history_empty_returns_empty_breakdown():
    b = label_history([])
    assert b.current is None
    assert b.n_bars == 0
    assert b.history == []


def test_label_history_filters_non_positive_vols():
    """vol_bps <= 0 is impossible (price moved zero) — filter rather
    than crash on the percentile computation."""
    history = [
        {"ts": "t1", "vol_bps": 0.0},   # filtered
        {"ts": "t2", "vol_bps": -1.0},  # filtered
        {"ts": "t3", "vol_bps": 5.0},
    ]
    b = label_history(history)
    assert b.n_bars == 1
    assert b.current == "turbulent"  # only one positive bar → it's both p33 and p66


def test_label_history_assigns_three_regimes_proportionally():
    """Uniform vol distribution → ~33% in each regime."""
    history = [{"ts": f"t{i}", "vol_bps": float(i)} for i in range(1, 100)]
    b = label_history(history)
    total = b.counts["calm"] + b.counts["normal"] + b.counts["turbulent"]
    assert total == 99
    # Each bucket should be ~33% (allow 5% slack for rounding).
    for k in ("calm", "normal", "turbulent"):
        share = b.counts[k] / total
        assert 0.28 <= share <= 0.40, f"{k}={share:.3f}"


def test_label_history_current_matches_last_input_bar():
    """The last item in history determines `current`."""
    history = [{"ts": f"t{i}", "vol_bps": float(i)} for i in range(1, 11)]
    b = label_history(history)
    # Last bar is vol=10 → highest of the 10 bars → must be turbulent.
    assert b.current == "turbulent"
    assert b.current_vol_bps == 10.0


def test_label_history_keep_last_n_truncates_output_only():
    """keep_last_n affects the returned history list, NOT the
    percentile thresholds (which use the full input)."""
    history = [{"ts": f"t{i}", "vol_bps": float(i)} for i in range(1, 100)]
    b = label_history(history, keep_last_n=10)
    # Returned history is last 10.
    assert len(b.history) == 10
    # But n_bars = total classified bars.
    assert b.n_bars == 99
    # Counts are over the FULL classified history, not the truncated tail.
    assert b.counts["calm"] + b.counts["normal"] + b.counts["turbulent"] == 99


def test_label_history_thresholds_self_adapt_to_distribution():
    """A high-vol regime's "calm" threshold > a low-vol regime's "calm"
    threshold — labels are relative, not absolute."""
    low_vol = [{"ts": f"a{i}", "vol_bps": float(i)} for i in range(1, 30)]
    high_vol = [{"ts": f"b{i}", "vol_bps": float(i + 50)} for i in range(1, 30)]

    b_low = label_history(low_vol)
    b_high = label_history(high_vol)

    # High-vol regime has shifted thresholds.
    assert b_high.p_low > b_low.p_low
    assert b_high.p_high > b_low.p_high


def test_label_history_to_dict_is_json_safe():
    """Defensive: output must round-trip through json.dumps without
    TypeError (datetimes → ISO strings, RegimeLabel → str)."""
    import json
    history = [{"ts": "2026-04-26T12:34:00+00:00", "vol_bps": 5.0},
               {"ts": "2026-04-26T12:35:00+00:00", "vol_bps": 10.0}]
    b = label_history(history)
    json.dumps(b.to_dict())  # must not raise
