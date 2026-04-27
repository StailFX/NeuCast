"""Fee-tier P&L re-simulation.

What this module is for
=======================

The paper-trader runs at a single fee assumption (default 7.5 bp
maker per side, BNB-paid retail tier). At that tier, almost every
trade loses to fees alone — typical 1-min BTC return is 3-5 bps,
roundtrip fee is 15 bps.

That's textbook for retail-tier directional ML: even a perfectly
calibrated model can't be profitable. Real HFT firms operate at
**different fee tiers** entirely — VIP 5 (~$50 M/30 d volume) pays
1 bp; VIP 9 + market-maker contract gets a -0.4 bp **rebate** (the
exchange pays them).

This module recomputes a stored ``paper_trades`` row's P&L at any
fee tier by isolating the GROSS price-move profit (which is
fee-independent) and reapplying a different fee assumption. No DB
mutation; pure read-side. Lets the UI / Telegram bot answer "what
would our P&L look like at VIP-tier fees?" without re-running the
trader.

Defence-grade utility
---------------------

The honest defence story is:

> "At retail tier (7.5 bp per side) we lose money on every batch
> of trades. At VIP-9 (0 bp) we'd be ~breakeven. At market-maker
> rebate (-0.4 bp) we'd capture the model's directional edge net
> of zero costs. Edge exists; profitability is determined by the
> fee tier, which is determined by trading volume, which is
> determined by capital — none of which scale with code quality.
> The model + engineering work; the rest is treasury."

This module produces the time-series that backs that paragraph.

Tier definitions
----------------

Pinned constants. Each tier is a real Binance Spot pricing point:

* ``retail`` — 7.5 bp/side. Standard tier with BNB discount; what we
  charge ourselves at by default.
* ``vip5`` — 1.0 bp/side. Reachable at $50 M/30 d volume.
* ``vip9`` — 0.0 bp/side. Reachable at $4 B/30 d volume.
* ``mm_rebate`` — -0.4 bp/side. Negotiated market-maker rebate; the
  exchange pays you to provide liquidity. Requires a separate
  contract on top of VIP 9 with $200 M/30 d volume.

Adding a tier: append to ``FEE_TIERS`` and the UI / Telegram render
picks it up automatically (formatting code reads the dict).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# Tier_name → fee_bps_per_side. Negative = rebate (exchange pays you).
# Order of insertion matters: rendering surfaces use this order so
# "from worst to best".
FEE_TIERS: dict[str, float] = {
    "retail":    7.5,
    "vip5":      1.0,
    "vip9":      0.0,
    "mm_rebate": -0.4,
}


@dataclass(frozen=True)
class TierPnL:
    """Aggregate P&L at one fee tier across N trades.

    All values are derived; nothing in this struct depends on stored
    `pnl_usd` columns (which were computed at the trader's original
    fee setting). The only inputs are the raw trade fields:
    ``qty``, ``entry_price``, ``exit_price``, ``side``.
    """
    tier: str
    fee_bps_per_side: float
    n_trades: int
    n_wins: int
    n_losses: int
    pnl_usd: float
    pnl_bps_avg: float
    pnl_usd_per_trade_avg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recompute_trade_pnl(
    *,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    fee_bps_per_side: float,
) -> tuple[float, float]:
    """Pure: P&L for one trade at the given fee tier.

    Returns ``(pnl_usd, pnl_bps)`` where ``pnl_bps`` normalises by
    the entry notional so trades across BTC / ETH / BNB at very
    different price levels are comparable. Mirrors the trader's
    inline math in ``paper_trader.compute_pnl`` exactly — only the
    fee parameter is variable.

    Negative ``fee_bps_per_side`` (rebate) increases P&L. We accept
    arbitrary negative values; the caller decides whether that's
    physically realistic.
    """
    if entry_price <= 0 or exit_price <= 0 or qty <= 0:
        return 0.0, 0.0
    if side == "long":
        gross = qty * (exit_price - entry_price)
    elif side == "short":
        gross = qty * (entry_price - exit_price)
    else:
        raise ValueError(f"unknown side: {side!r}")
    # Fee is bps × notional; same rate on entry and exit; can be
    # negative (rebate).
    fee_total = qty * (entry_price + exit_price) * (fee_bps_per_side / 1e4)
    pnl_usd = gross - fee_total
    pnl_bps = (pnl_usd / (qty * entry_price)) * 1e4
    return float(pnl_usd), float(pnl_bps)


def summarise_tier(
    trades: list[dict[str, Any]],
    *,
    tier: str,
    fee_bps_per_side: float,
) -> TierPnL:
    """Aggregate one fee-tier's P&L across a list of trade rows.

    ``trades`` is a list of dicts with at minimum: ``side``, ``qty``,
    ``entry_price``, ``exit_price``. Extra keys are ignored.
    """
    n_wins = 0
    n_losses = 0
    pnl_total_usd = 0.0
    pnl_total_bps = 0.0
    n = 0
    for t in trades:
        try:
            pnl_usd, pnl_bps = recompute_trade_pnl(
                side=t["side"],
                qty=float(t["qty"]),
                entry_price=float(t["entry_price"]),
                exit_price=float(t["exit_price"]),
                fee_bps_per_side=fee_bps_per_side,
            )
        except (KeyError, ValueError, TypeError):
            continue
        n += 1
        pnl_total_usd += pnl_usd
        pnl_total_bps += pnl_bps
        if pnl_usd > 0:
            n_wins += 1
        elif pnl_usd < 0:
            n_losses += 1

    if n == 0:
        return TierPnL(
            tier=tier, fee_bps_per_side=float(fee_bps_per_side),
            n_trades=0, n_wins=0, n_losses=0,
            pnl_usd=0.0, pnl_bps_avg=0.0, pnl_usd_per_trade_avg=0.0,
        )

    return TierPnL(
        tier=tier,
        fee_bps_per_side=float(fee_bps_per_side),
        n_trades=n,
        n_wins=n_wins,
        n_losses=n_losses,
        pnl_usd=float(pnl_total_usd),
        pnl_bps_avg=float(pnl_total_bps / n),
        pnl_usd_per_trade_avg=float(pnl_total_usd / n),
    )


def summarise_all_tiers(trades: list[dict[str, Any]]) -> list[TierPnL]:
    """Convenience: produce one row per :data:`FEE_TIERS` entry.

    Output is in the same order as ``FEE_TIERS`` insertion (worst
    → best). UI tables and Telegram messages can iterate
    sequentially.
    """
    return [
        summarise_tier(trades, tier=name, fee_bps_per_side=bps)
        for name, bps in FEE_TIERS.items()
    ]
