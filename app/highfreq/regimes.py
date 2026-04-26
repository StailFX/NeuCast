"""Volatility-regime classifier for the HFT slice.

Splits the recent minute-bar history into three regimes (``calm`` /
``normal`` / ``turbulent``) based on the rolling distribution of the
intra-bar high-low range (microprice). No ML fit — just percentile
thresholds on the *symbol's own* recent vol distribution, so the
labels self-adapt as the market shifts.

Why not k-means
---------------
A 2-D k-means on (vol, spread) was the first instinct, but:
* Cluster identities aren't stable (cluster 0 could be calm OR
  turbulent depending on random init).
* Sklearn fit on every poll is wasteful for a 1-D classification.
* Percentile thresholds are interpretable in defense ("the model
  was opened during the *top tercile of intraminute vol* — that's
  why we sized down").

Why this matters for the strategy
---------------------------------
Vol-adjusted sizing (PaperTraderConfig.vol_adjusted_sizing) uses the
per-bar vol directly. The regime label is a SECOND-order signal —
useful to colour UI, slice paper-trade analysis ("Sharpe was 1.2 in
calm regime, 0.4 in turbulent"), and a future opt-in: open trades
ONLY in calm/normal (skip turbulent regimes where the model wasn't
trained).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Regime labels are intentionally string-typed (not int) so JSON output
# and UI display can use them directly without translation.
RegimeLabel = Literal["calm", "normal", "turbulent"]

# Default percentile boundaries: 33rd and 66th. Tuneable but the
# 33/66 split is a strong default — three roughly-equal-frequency
# regimes by definition.
DEFAULT_LOW_PCT: float = 33.0
DEFAULT_HIGH_PCT: float = 66.0


@dataclass(frozen=True)
class RegimeBreakdown:
    """Aggregate state of the vol-regime classifier."""

    current: RegimeLabel | None              # most recent bar's label
    current_vol_bps: float | None            # raw value behind the label
    p_low: float                              # lower percentile boundary
    p_high: float                             # upper percentile boundary
    n_bars: int                               # how many bars considered
    counts: dict[RegimeLabel, int]            # how many bars in each regime
    history: list[dict]                       # [{"ts": str, "vol_bps": float, "regime": str}]

    def to_dict(self) -> dict:
        return {
            "current": self.current,
            "current_vol_bps": self.current_vol_bps,
            "p_low": self.p_low,
            "p_high": self.p_high,
            "n_bars": self.n_bars,
            "counts": dict(self.counts),
            "history": self.history,
        }


def classify_regime(
    vol_bps: float, *, p_low: float, p_high: float,
) -> RegimeLabel:
    """Bucket a single vol value against precomputed percentile thresholds."""
    if vol_bps < p_low:
        return "calm"
    if vol_bps < p_high:
        return "normal"
    return "turbulent"


def compute_regime_thresholds(
    vols_bps: np.ndarray,
    *,
    low_pct: float = DEFAULT_LOW_PCT,
    high_pct: float = DEFAULT_HIGH_PCT,
) -> tuple[float, float]:
    """Compute (p_low, p_high) percentile values from the vol history.

    Returns ``(0.0, 0.0)`` if input is empty (callers should treat as
    "not enough data" — every bar will then be labelled 'turbulent'
    because it'll exceed 0).
    """
    if vols_bps.size == 0:
        return 0.0, 0.0
    p_low = float(np.percentile(vols_bps, low_pct))
    p_high = float(np.percentile(vols_bps, high_pct))
    return p_low, p_high


def label_history(
    history: list[dict],
    *,
    low_pct: float = DEFAULT_LOW_PCT,
    high_pct: float = DEFAULT_HIGH_PCT,
    keep_last_n: int | None = None,
) -> RegimeBreakdown:
    """Classify each bar in ``history`` and return a breakdown.

    Parameters
    ----------
    history
        List of ``{"ts": str|datetime, "vol_bps": float}`` ordered
        chronologically. Vol values <= 0 are filtered out.
    low_pct / high_pct
        Percentile boundaries.
    keep_last_n
        If given, only the last N bars are returned in
        ``RegimeBreakdown.history`` — but the percentile thresholds
        are computed from the FULL input. UI use case: show the last
        60 bars but threshold against the last 24 hours (1440 bars).
    """
    if not history:
        return RegimeBreakdown(
            current=None, current_vol_bps=None,
            p_low=0.0, p_high=0.0, n_bars=0,
            counts={"calm": 0, "normal": 0, "turbulent": 0},
            history=[],
        )

    vols = np.asarray(
        [float(b["vol_bps"]) for b in history if float(b.get("vol_bps") or 0) > 0],
        dtype=float,
    )
    if vols.size == 0:
        return RegimeBreakdown(
            current=None, current_vol_bps=None,
            p_low=0.0, p_high=0.0, n_bars=0,
            counts={"calm": 0, "normal": 0, "turbulent": 0},
            history=[],
        )

    p_low, p_high = compute_regime_thresholds(vols, low_pct=low_pct, high_pct=high_pct)

    labelled: list[dict] = []
    counts: dict[RegimeLabel, int] = {"calm": 0, "normal": 0, "turbulent": 0}
    for b in history:
        v = float(b.get("vol_bps") or 0)
        if v <= 0:
            continue
        regime = classify_regime(v, p_low=p_low, p_high=p_high)
        counts[regime] += 1
        labelled.append({
            "ts": b["ts"] if isinstance(b["ts"], str) else b["ts"].isoformat(),
            "vol_bps": v,
            "regime": regime,
        })

    if keep_last_n is not None and keep_last_n > 0:
        out_history = labelled[-keep_last_n:]
    else:
        out_history = labelled

    last = labelled[-1] if labelled else None
    return RegimeBreakdown(
        current=last["regime"] if last else None,
        current_vol_bps=last["vol_bps"] if last else None,
        p_low=p_low,
        p_high=p_high,
        n_bars=len(labelled),
        counts=counts,
        history=out_history,
    )
