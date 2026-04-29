"""Tests for ``_exponential_sample_weights`` — pure helper for the
recent-favoring weighting introduced in release O.

Pin the math because a sign flip would silently invert the bias
(making OLD bars dominate, the opposite of intent)."""
from __future__ import annotations

import numpy as np
import pytest

from app.highfreq.trainer import _exponential_sample_weights


def test_disabled_when_half_life_zero():
    """half_life=0 disables — returns None signalling 'use uniform'."""
    assert _exponential_sample_weights(n=100, half_life=0) is None


def test_disabled_when_half_life_negative():
    """Defensive: negative half-life is operator error, treat as disabled."""
    assert _exponential_sample_weights(n=100, half_life=-10) is None


def test_disabled_when_half_life_none():
    """Pinned: None half-life is the documented disable sentinel."""
    assert _exponential_sample_weights(n=100, half_life=None) is None  # type: ignore[arg-type]


def test_returns_none_on_empty_input():
    """0-length training set: nothing to weight."""
    assert _exponential_sample_weights(n=0, half_life=720) is None


def test_most_recent_bar_has_weight_one():
    """The LAST element (newest bar) must have weight exactly 1.0.
    Pin so a refactor that flips index ordering fails CI loudly."""
    w = _exponential_sample_weights(n=100, half_life=50)
    assert w is not None
    assert w[-1] == pytest.approx(1.0)


def test_oldest_bar_has_smallest_weight():
    """First element = oldest bar = should have the SMALLEST weight."""
    w = _exponential_sample_weights(n=200, half_life=50)
    assert w is not None
    assert w[0] == w.min()
    assert w[-1] == w.max()


def test_half_life_decay_correct():
    """A bar exactly ``half_life`` away from newest must have weight 0.5.
    Defines what 'half life' means; pin against silent regression."""
    n = 1001
    half_life = 100
    w = _exponential_sample_weights(n=n, half_life=half_life)
    assert w is not None
    # Newest is at index n-1 with weight 1.0.
    # A bar 100 places away (index n-1-100 = 900) should weigh 0.5.
    assert w[n - 1 - half_life] == pytest.approx(0.5)
    # A bar 200 places away should weigh 0.25.
    assert w[n - 1 - 2 * half_life] == pytest.approx(0.25)


def test_weights_monotonically_increasing():
    """Old → new must be monotonically non-decreasing. Critical for
    'recent matters more' semantics."""
    w = _exponential_sample_weights(n=500, half_life=200)
    assert w is not None
    diffs = np.diff(w)
    assert (diffs >= 0).all()


def test_long_half_life_approaches_uniform():
    """As half_life → ∞, weights → uniform. Pin: half_life=10000 on
    n=100 should give weights very close to 1.0 across the board."""
    w = _exponential_sample_weights(n=100, half_life=10_000)
    assert w is not None
    assert w.min() > 0.99


def test_short_half_life_concentrates_recent():
    """As half_life → small, only the very recent bars carry weight.
    half_life=10 on n=1000 should make bar 100 places away weigh ~2^-10
    ≈ 0.001, essentially zero."""
    w = _exponential_sample_weights(n=1000, half_life=10)
    assert w is not None
    # Bar 100 places back has weight 2^-10 = 0.001.
    assert w[1000 - 1 - 100] < 0.01


def test_default_config_half_life_pinned():
    """Document-as-test: 720 bars (=12 h at 1m) is the production
    default. A refactor changing this without updating release notes
    fails CI."""
    from app.highfreq.trainer import WalkForwardConfig
    cfg = WalkForwardConfig()
    assert cfg.sample_weight_half_life_bars == 720


def test_embargo_default_pinned():
    """Embargo default is 1 bar (covers horizon=1 minimum gap)."""
    from app.highfreq.trainer import WalkForwardConfig
    cfg = WalkForwardConfig()
    assert cfg.embargo_bars == 1
    assert cfg.purge_bars == 0
