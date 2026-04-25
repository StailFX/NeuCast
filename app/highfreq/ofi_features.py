"""Order-flow imbalance and related microstructure features.

Computes the feature vector that the 1-minute CatBoost model is trained on.
All formulas follow standard market-microstructure conventions; references
are inline.

Features
--------

* **OFI (Order Flow Imbalance)** — Cont, Kukanov, Stoikov (2014).
  Increment to bid depth at the top level minus increment to ask depth.
  Strongly correlated with short-horizon mid-price movements.

* **Microprice** — Stoikov (2018). Depth-weighted mid-price:

  .. math::

      P_\\text{micro} = \\frac{P_b \\cdot Q_a + P_a \\cdot Q_b}{Q_a + Q_b}

  where :math:`P_b, P_a` are best bid / ask and :math:`Q_b, Q_a` the
  corresponding quantities. Better predictor of next trade than the
  arithmetic mid.

* **Depth imbalance** — :math:`(\\sum Q_b - \\sum Q_a) / (\\sum Q_b + \\sum Q_a)`
  over the top N levels. Persistent imbalance indicates directional
  pressure.

* **Spread (bps)** — :math:`(P_a - P_b) / P_\\text{mid} \\cdot 10\\,000`.
  Used as a feature *and* as a regime gate (high spread → wider stops).

* **Trade imbalance** — sum of (signed) trade volume over a 1-second
  window. Sign convention: aggressive buy = +qty, aggressive sell = −qty.
  Computed from the ``is_buyer_maker`` flag in trade frames.

* **VPIN (Volume-synchronized Probability of Informed Trading)** —
  Easley, López de Prado, O'Hara (2012). Bucketed estimate of toxic flow;
  used as a kill-switch in Phase B.

This module only computes features. Aggregation into 1-s / 1-m rows is in
:mod:`app.highfreq.aggregator`.
"""
from __future__ import annotations

# NOTE: Phase-A.0 skeleton. Real implementation in Phase A.3.

from dataclasses import dataclass


@dataclass(frozen=True)
class OFIFeatures1s:
    """Feature vector for one 1-second window."""

    event_time_s: int          # truncated event time, second granularity
    symbol: str
    ofi: float
    microprice: float
    depth_imb: float
    spread_bps: float
    trade_imb: float
    vpin: float
    n_updates: int             # number of L2 updates in window


def compute_ofi(prev_book, curr_book) -> float:
    """Order-flow imbalance between two consecutive top-of-book snapshots.

    Formula: see Cont, Kukanov, Stoikov 2014, eq. (2).

    TODO (Phase A.3): implement multi-level OFI per Kolm, Turiel, Westray 2023.
    """
    raise NotImplementedError("Phase A.3")


def compute_microprice(book) -> float:
    """Depth-weighted mid (Stoikov 2018)."""
    raise NotImplementedError("Phase A.3")
