"""Tests for ``app.highfreq.anti_skill_detector`` — pure logic only.

Pin the CI safety net + threshold + halt_close exclusion. The DB
layer is exercised by the production smoke tests on Tokyo (one real
Postgres + the runner's tick loop).
"""
from __future__ import annotations

import pytest

from app.highfreq.anti_skill_detector import (
    DEFAULT_MIN_SAMPLE,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    AntiSkillReport,
    _is_gross_win,
    _wilson_ci,
    compute_anti_skill_from_rows,
    parse_response_policy,
)


def _trade(side: str, entry: float, exit_p: float,
           reason: str = "time_stop") -> dict:
    return {
        "side": side, "entry_price": entry, "exit_price": exit_p,
        "exit_reason": reason,
    }


# ───── _is_gross_win ─────


def test_long_winner_when_exit_above_entry():
    assert _is_gross_win(_trade("long", 77_000, 77_010)) is True


def test_long_loser_when_exit_below_entry():
    assert _is_gross_win(_trade("long", 77_000, 76_990)) is False


def test_short_winner_when_exit_below_entry():
    assert _is_gross_win(_trade("short", 77_000, 76_990)) is True


def test_short_loser_when_exit_above_entry():
    assert _is_gross_win(_trade("short", 77_000, 77_010)) is False


def test_tie_exit_equals_entry_is_loser():
    """No directional move = no win. Pin so a future "count ties as
    half-win" change fails CI loudly."""
    assert _is_gross_win(_trade("long", 77_000, 77_000)) is False


def test_unknown_side_returns_false_does_not_raise():
    """A row with unknown side from DB is not actionable; returning
    False is safer than raising and crashing the runner mid-tick."""
    assert _is_gross_win(_trade("???", 77_000, 77_010)) is False


# ───── compute_anti_skill_from_rows ─────


def test_quiet_until_min_sample_reached():
    """Below ``min_sample`` the detector MUST NOT fire — even if every
    trade is a loss. Defence against false-positive from tiny window."""
    rows = [_trade("long", 77_000, 76_990)] * 5
    r = compute_anti_skill_from_rows(rows, symbol="BTCUSDT", min_sample=30)
    assert r.is_anti_skilled is False
    assert r.n_trades_in_window == 5
    assert "min_sample" in r.note


def test_healthy_winrate_does_not_fire():
    """50%+ winrate → never anti-skilled (by definition)."""
    rows = [_trade("long", 77_000, 77_010)] * 30   # 30 wins
    rows += [_trade("long", 77_000, 76_990)] * 10  # 10 losses
    r = compute_anti_skill_from_rows(rows, symbol="BTCUSDT")
    assert r.is_anti_skilled is False
    assert r.gross_winrate == pytest.approx(0.75)


def test_at_threshold_borderline_does_not_fire_with_wide_ci():
    """30 trades, 12 wins (40%) — point estimate below threshold but
    Wilson CI upper ≈ 0.59. Detector MUST stay quiet — that's the
    CI-safety-net rule."""
    rows = [_trade("long", 77_000, 77_010)] * 12   # 12 wins
    rows += [_trade("long", 77_000, 76_990)] * 18  # 18 losses
    r = compute_anti_skill_from_rows(rows, symbol="BTCUSDT")
    assert r.gross_winrate == pytest.approx(0.40)
    assert r.gross_winrate_ci_high > 0.50
    assert r.is_anti_skilled is False
    assert "borderline" in r.note


def test_clear_anti_skill_fires_when_ci_upper_below_chance():
    """50 trades, 15 wins (30%) — Wilson CI upper ≈ 0.43 < 0.50.
    THIS is when the detector SHOULD fire — both point estimate and
    CI upper are below chance, anti-skill is statistically distinct
    from noise."""
    rows = [_trade("long", 77_000, 77_010)] * 15   # 15 wins
    rows += [_trade("long", 77_000, 76_990)] * 35  # 35 losses
    r = compute_anti_skill_from_rows(rows, symbol="BTCUSDT")
    assert r.gross_winrate == pytest.approx(0.30)
    assert r.gross_winrate_ci_high < 0.50
    assert r.is_anti_skilled is True
    assert "ANTI-SKILL DETECTED" in r.note


def test_window_bounds_to_most_recent_n():
    """If 100 rows passed but window=50, only the FIRST 50 (most
    recent by SQL DESC contract) are evaluated. Catches a regression
    where someone removes the slice and the detector quietly averages
    over too much history."""
    # 50 most recent: all losses (anti-skill territory)
    losers = [_trade("long", 77_000, 76_990)] * 50
    # 50 older: all wins (would dilute if not sliced)
    winners = [_trade("long", 77_000, 77_010)] * 50
    r = compute_anti_skill_from_rows(losers + winners, symbol="BTCUSDT", window=50)
    assert r.n_trades_in_window == 50
    assert r.gross_winrate == pytest.approx(0.0)
    assert r.is_anti_skilled is True


def test_halt_close_trades_excluded_from_sample():
    """halt_close exits are forced by risk caps, not by directional
    calls. They must NOT count toward anti-skill detection — otherwise
    a daily-loss-cap halt sequence would falsely look like model
    failure."""
    # Mix: 20 healthy wins (time_stop) + 30 halt_closes that "lost".
    # Without exclusion winrate = 20/50 = 40 % → would trigger.
    # WITH exclusion winrate = 20/20 = 100 % → healthy.
    rows = [_trade("long", 77_000, 77_010, "time_stop")] * 20
    rows += [_trade("long", 77_000, 76_990, "halt_close")] * 30
    r = compute_anti_skill_from_rows(rows, symbol="BTCUSDT")
    assert r.n_trades_in_window == 20
    assert r.gross_winrate == pytest.approx(1.0)
    assert r.is_anti_skilled is False


def test_under_chance_but_above_threshold_emits_monitoring_note():
    """40+ trades at 45 % winrate — under chance but above 0.42
    threshold. Detector MUST NOT fire (we don't act on this) but
    MUST surface 'monitoring' in the note for operator visibility."""
    rows = [_trade("long", 77_000, 77_010)] * 18   # 18 wins
    rows += [_trade("long", 77_000, 76_990)] * 22  # 22 losses
    r = compute_anti_skill_from_rows(rows, symbol="BTCUSDT")
    assert r.gross_winrate == pytest.approx(0.45)
    assert r.is_anti_skilled is False
    assert "under chance" in r.note or "monitoring" in r.note


def test_to_dict_keys_pinned():
    """Endpoint embeds the dict directly. Pin so a rename here breaks
    tests, not the UI."""
    r = AntiSkillReport(
        symbol="BTCUSDT", window=50, min_sample=30, threshold=0.42,
        n_trades_in_window=40, n_gross_wins=15,
        gross_winrate=0.375, gross_winrate_ci_low=0.225,
        gross_winrate_ci_high=0.55,
        is_anti_skilled=False, note="x",
    )
    d = r.to_dict()
    assert set(d.keys()) == {
        "symbol", "window", "min_sample", "threshold",
        "n_trades_in_window", "n_gross_wins",
        "gross_winrate", "gross_winrate_ci_low",
        "gross_winrate_ci_high",
        "is_anti_skilled", "note",
    }


def test_default_constants_pinned():
    """Defence-grade constants — tests act as documentation for
    future contributors."""
    assert DEFAULT_WINDOW == 50
    assert DEFAULT_MIN_SAMPLE == 30
    assert DEFAULT_THRESHOLD == pytest.approx(0.42)


# ───── parse_response_policy ─────


def test_response_policy_default_when_unset():
    assert parse_response_policy(None) == "alert"
    assert parse_response_policy("") == "alert"


def test_response_policy_recognised_values():
    assert parse_response_policy("alert") == "alert"
    assert parse_response_policy("halt") == "halt"
    assert parse_response_policy("invert") == "invert"


def test_response_policy_case_insensitive():
    assert parse_response_policy("INVERT") == "invert"
    assert parse_response_policy("Halt") == "halt"


def test_response_policy_typo_falls_back_to_alert():
    """A typo in env (e.g. 'inert' instead of 'invert') MUST fall
    back to the safest mode, NOT silently disable protection or
    flip every signal."""
    assert parse_response_policy("inert") == "alert"
    assert parse_response_policy("auto") == "alert"


# ───── Wilson CI ─────


def test_wilson_ci_returns_full_range_on_zero_n():
    lo, hi = _wilson_ci(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_ci_known_case():
    """30 wins / 100 trials at 95 % conf ≈ [0.219, 0.396].
    (Verified by hand: centre 0.307, half-width 0.088.)"""
    lo, hi = _wilson_ci(30, 100)
    assert lo == pytest.approx(0.219, abs=0.005)
    assert hi == pytest.approx(0.396, abs=0.005)
