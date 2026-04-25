"""Order-flow imbalance and microstructure features.

Computes per-frame microstructure quantities from successive
:class:`~app.highfreq.l2_consumer.L2Snapshot` instances. The aggregator
sums / averages these per 1-second window before persisting to Postgres.

References
----------
* Cont, R., Kukanov, A., Stoikov, S. (2014). *The Price Impact of Order
  Book Events.* J. Financial Econometrics 12 (1), 47–88.
* Stoikov, S. (2018). *The Micro-Price.* Quantitative Finance 18 (12).
* Kolm, P., Turiel, J., Westray, N. (2023). *Deep Order Flow Imbalance.*
  Mathematical Finance 33 (4), 1044–1081.
* Easley, D., López de Prado, M., O'Hara, M. (2012). *Flow Toxicity and
  Liquidity in a High-Frequency World.* Review of Financial Studies 25.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.highfreq.l2_consumer import L2Snapshot


@dataclass(frozen=True)
class FrameFeatures:
    """Per-frame features (no aggregation yet)."""

    event_time_ms: int
    symbol: str
    ofi: float            # top-of-book OFI vs previous frame
    microprice: float     # depth-weighted mid
    depth_imb: float      # top-N depth imbalance
    spread_bps: float     # (ask - bid) / mid * 10_000
    mid: float            # arithmetic mid


def compute_ofi(prev: L2Snapshot | None, curr: L2Snapshot) -> float:
    """Top-of-book Order Flow Imbalance (Cont, Kukanov, Stoikov 2014).

    Defined per equation (2) of the paper:

    .. math::

        e_n = \\mathbb{1}[P_b^n \\ge P_b^{n-1}] \\cdot Q_b^n
            - \\mathbb{1}[P_b^n \\le P_b^{n-1}] \\cdot Q_b^{n-1}
            - \\mathbb{1}[P_a^n \\le P_a^{n-1}] \\cdot Q_a^n
            + \\mathbb{1}[P_a^n \\ge P_a^{n-1}] \\cdot Q_a^{n-1}

    Positive OFI ⇒ net upward pressure (bids strengthening or asks weakening).

    Returns 0.0 for the very first frame (no previous reference).
    """
    if prev is None:
        return 0.0
    if not curr.bids or not curr.asks or not prev.bids or not prev.asks:
        return 0.0

    pb_n, qb_n = curr.bids[0]
    pa_n, qa_n = curr.asks[0]
    pb_p, qb_p = prev.bids[0]
    pa_p, qa_p = prev.asks[0]

    bid_term = (qb_n if pb_n >= pb_p else 0.0) - (qb_p if pb_n <= pb_p else 0.0)
    ask_term = (qa_p if pa_n >= pa_p else 0.0) - (qa_n if pa_n <= pa_p else 0.0)
    return bid_term + ask_term


def compute_microprice(snap: L2Snapshot) -> float:
    """Depth-weighted mid (Stoikov 2018).

    .. math::

        P_\\text{micro} = \\frac{P_b \\cdot Q_a + P_a \\cdot Q_b}{Q_a + Q_b}
    """
    pb, qb = snap.bids[0]
    pa, qa = snap.asks[0]
    denom = qa + qb
    if denom <= 0.0:
        return (pa + pb) * 0.5
    return (pb * qa + pa * qb) / denom


def compute_depth_imbalance(snap: L2Snapshot, levels: int = 10) -> float:
    """Top-N depth imbalance.

    .. math::

        D = \\frac{\\sum Q_b - \\sum Q_a}{\\sum Q_b + \\sum Q_a}

    Range: [-1, 1]. Positive ⇒ more bid-side liquidity.
    """
    sum_b = sum(q for _, q in snap.bids[:levels])
    sum_a = sum(q for _, q in snap.asks[:levels])
    denom = sum_b + sum_a
    if denom <= 0.0:
        return 0.0
    return (sum_b - sum_a) / denom


def compute_spread_bps(snap: L2Snapshot) -> tuple[float, float]:
    """Returns (spread_bps, mid). Spread in basis points of mid."""
    pb = snap.bids[0][0]
    pa = snap.asks[0][0]
    mid = (pa + pb) * 0.5
    if mid <= 0.0:
        return 0.0, 0.0
    return (pa - pb) / mid * 10_000.0, mid


def features_from_snapshot(
    snap: L2Snapshot, prev: L2Snapshot | None, *, depth_levels: int = 10
) -> FrameFeatures:
    """Compute the full feature vector for a single L2 frame.

    Pure function — no I/O, no globals. Easily unit-tested.
    """
    ofi = compute_ofi(prev, snap)
    microprice = compute_microprice(snap)
    depth_imb = compute_depth_imbalance(snap, levels=depth_levels)
    spread_bps, mid = compute_spread_bps(snap)
    return FrameFeatures(
        event_time_ms=snap.event_time_ms,
        symbol=snap.symbol,
        ofi=ofi,
        microprice=microprice,
        depth_imb=depth_imb,
        spread_bps=spread_bps,
        mid=mid,
    )
