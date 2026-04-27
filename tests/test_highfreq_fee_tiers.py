"""Tests for ``app.highfreq.fee_tiers`` — pure P&L recompute math.

Pin the fee re-application formula. Bug here would silently make
the "VIP-tier P&L" column display the wrong number, undermining
the defence story.
"""
from __future__ import annotations

import pytest

from app.highfreq.fee_tiers import (
    FEE_TIERS,
    TierPnL,
    recompute_trade_pnl,
    summarise_all_tiers,
    summarise_tier,
)


# ───── recompute_trade_pnl ─────


def test_long_winner_at_zero_fee_equals_gross():
    """0 fee → gross = qty × (exit - entry). Pin against an
    arithmetic-bug introduction."""
    pnl_usd, pnl_bps = recompute_trade_pnl(
        side="long", qty=0.01,
        entry_price=77_000.0, exit_price=77_100.0,
        fee_bps_per_side=0.0,
    )
    assert pnl_usd == pytest.approx(0.01 * 100.0, rel=1e-9)  # = $1.00
    # bps = pnl / (qty * entry) * 1e4 = 1.0 / 770 * 1e4 ≈ 12.987
    assert pnl_bps == pytest.approx(12.987, abs=0.001)


def test_short_winner_at_zero_fee():
    """Short profits when exit < entry."""
    pnl_usd, _ = recompute_trade_pnl(
        side="short", qty=0.01,
        entry_price=77_000.0, exit_price=76_950.0,
        fee_bps_per_side=0.0,
    )
    assert pnl_usd == pytest.approx(0.01 * 50.0, rel=1e-9)  # = $0.50


def test_fee_reduces_pnl_proportional_to_notional():
    """7.5 bp/side × $1540 notional = $1.155. Round-trip = $2.31."""
    qty = 0.01
    entry = 77_000.0
    exit_p = 77_100.0
    pnl_zero, _ = recompute_trade_pnl(
        side="long", qty=qty, entry_price=entry, exit_price=exit_p,
        fee_bps_per_side=0.0,
    )
    pnl_retail, _ = recompute_trade_pnl(
        side="long", qty=qty, entry_price=entry, exit_price=exit_p,
        fee_bps_per_side=7.5,
    )
    fee_total = qty * (entry + exit_p) * (7.5 / 1e4)
    assert pnl_retail == pytest.approx(pnl_zero - fee_total, rel=1e-9)


def test_negative_fee_rebate_increases_pnl():
    """Rebate (-0.4 bp) means exchange PAYS the trader → P&L goes UP."""
    pnl_zero, _ = recompute_trade_pnl(
        side="long", qty=0.01,
        entry_price=77_000.0, exit_price=77_100.0,
        fee_bps_per_side=0.0,
    )
    pnl_rebate, _ = recompute_trade_pnl(
        side="long", qty=0.01,
        entry_price=77_000.0, exit_price=77_100.0,
        fee_bps_per_side=-0.4,
    )
    assert pnl_rebate > pnl_zero


def test_loser_at_high_fee_loses_more():
    """A trade that's slightly negative gross becomes very negative
    at retail fee. Pin so a sign-flip bug fails CI."""
    pnl_zero, _ = recompute_trade_pnl(
        side="long", qty=0.01,
        entry_price=77_000.0, exit_price=76_990.0,  # tiny down
        fee_bps_per_side=0.0,
    )
    pnl_retail, _ = recompute_trade_pnl(
        side="long", qty=0.01,
        entry_price=77_000.0, exit_price=76_990.0,
        fee_bps_per_side=7.5,
    )
    assert pnl_zero < 0  # already a loser at gross
    assert pnl_retail < pnl_zero  # retail makes it WORSE


def test_zero_or_invalid_inputs_return_zero():
    """Defensive: zero qty / negative price → return (0,0), not raise.
    Real-world data can have rounding-to-zero or NaN-cast-to-zero."""
    assert recompute_trade_pnl(
        side="long", qty=0.0, entry_price=77_000.0, exit_price=77_100.0,
        fee_bps_per_side=7.5,
    ) == (0.0, 0.0)
    assert recompute_trade_pnl(
        side="long", qty=0.01, entry_price=0.0, exit_price=77_100.0,
        fee_bps_per_side=7.5,
    ) == (0.0, 0.0)


def test_unknown_side_raises():
    """An unknown side from the DB (CHECK constraint should prevent
    this) — be loud rather than silently returning 0."""
    with pytest.raises(ValueError, match="unknown side"):
        recompute_trade_pnl(
            side="???", qty=0.01,
            entry_price=77_000.0, exit_price=77_100.0,
            fee_bps_per_side=7.5,
        )


# ───── summarise_tier ─────


def test_summarise_tier_aggregates_w_l_and_pnl():
    trades = [
        {"side": "long", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 77_100.0},
        {"side": "short", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 76_950.0},
        {"side": "long", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 76_990.0},
    ]
    s = summarise_tier(trades, tier="vip9", fee_bps_per_side=0.0)
    assert s.n_trades == 3
    assert s.n_wins == 2  # both winners at 0 fee
    assert s.n_losses == 1
    # Avg P&L per trade = total / 3
    assert s.pnl_usd_per_trade_avg == pytest.approx(s.pnl_usd / 3, rel=1e-9)


def test_summarise_tier_empty_trades_returns_zero_struct():
    s = summarise_tier([], tier="retail", fee_bps_per_side=7.5)
    assert s.n_trades == 0
    assert s.n_wins == 0
    assert s.n_losses == 0
    assert s.pnl_usd == 0.0
    assert s.pnl_bps_avg == 0.0


def test_summarise_tier_skips_malformed_rows():
    """A row missing a required field is silently skipped, not
    raised — defends against legacy-row backfills."""
    trades = [
        {"side": "long", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 77_100.0},
        {"side": "long"},  # missing qty etc.
        {"qty": 0.01, "entry_price": 77_000, "exit_price": 77_100},  # missing side
    ]
    s = summarise_tier(trades, tier="vip9", fee_bps_per_side=0.0)
    assert s.n_trades == 1


# ───── summarise_all_tiers ─────


def test_summarise_all_tiers_returns_one_per_tier():
    trades = [
        {"side": "long", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 77_100.0},
    ]
    out = summarise_all_tiers(trades)
    assert len(out) == len(FEE_TIERS)
    assert [s.tier for s in out] == list(FEE_TIERS.keys())


def test_summarise_all_tiers_pnl_monotonic_in_fee():
    """Critical invariant: as fees DROP, P&L MONOTONICALLY rises.
    A bug that made VIP9 P&L LOWER than retail would silently flip
    the defence-grade "fee tier matters" story upside down."""
    trades = [
        {"side": "long", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 77_100.0},
        {"side": "long", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 77_050.0},
        {"side": "short", "qty": 0.01, "entry_price": 77_000.0, "exit_price": 76_950.0},
    ]
    out = summarise_all_tiers(trades)
    # Tier dict order: retail (worst) → mm_rebate (best). PnL must be
    # non-decreasing across that order.
    pnls = [s.pnl_usd for s in out]
    for prev, nxt in zip(pnls, pnls[1:]):
        assert nxt >= prev, (
            f"non-monotonic P&L across fee tiers: {pnls}"
        )


def test_fee_tiers_constant_includes_expected_entries():
    """Pin the tier set so a defence-grade reviewer sees the same
    table layout in code, UI, and Telegram bot. Adding a tier here
    is fine; renaming or removing one is a UI break."""
    assert "retail" in FEE_TIERS
    assert "vip5" in FEE_TIERS
    assert "vip9" in FEE_TIERS
    assert "mm_rebate" in FEE_TIERS
    assert FEE_TIERS["retail"] == 7.5
    assert FEE_TIERS["vip9"] == 0.0
    assert FEE_TIERS["mm_rebate"] < 0  # rebate sign convention


def test_tier_pnl_to_dict_keys_pinned():
    """The endpoint embeds these dicts directly. Pin the keys so
    a rename here breaks tests instead of silently breaking the UI."""
    s = TierPnL(
        tier="retail", fee_bps_per_side=7.5,
        n_trades=3, n_wins=1, n_losses=2,
        pnl_usd=-2.5, pnl_bps_avg=-150.0,
        pnl_usd_per_trade_avg=-0.83,
    )
    d = s.to_dict()
    assert set(d.keys()) == {
        "tier", "fee_bps_per_side",
        "n_trades", "n_wins", "n_losses",
        "pnl_usd", "pnl_bps_avg", "pnl_usd_per_trade_avg",
    }
