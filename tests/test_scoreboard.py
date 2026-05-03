"""Tests for ``tools.scoreboard`` (T.20).

The scoreboard is a defence-grade artifact reviewers will read.
Tests pin the document structure + the config-delta tagging
heuristic so refactors don't silently mangle the layout.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tools.scoreboard import (
    TrainingRun,
    _release_tag,
    _to_float,
    render_markdown,
)


def _run(
    *,
    rid: int = 1,
    symbol: str = "BTCUSDT",
    minute_offset: int = 0,
    feature_set: str = "microstructure",
    bar_minutes: int = 1,
    holdout_days: int = 0,
    n_folds: int = 30,
    n_oos: int = 3000,
    dir_acc: float | None = 0.55,
    ci_low: float | None = 0.53,
    ci_high: float | None = 0.57,
    p_value: float | None = 1e-7,
    brier: float | None = 0.25,
    ece: float | None = 0.06,
    conformal_q: float | None = None,
) -> TrainingRun:
    base = datetime(2026, 5, 1, 4, 0, 0, tzinfo=timezone.utc)
    return TrainingRun(
        id=rid, symbol=symbol,
        started_at=base.replace(minute=base.minute + minute_offset),
        feature_set=feature_set, bar_minutes=bar_minutes,
        since_hours=None, frozen_holdout_days=holdout_days,
        n_folds=n_folds, n_minutes_after_neutral_drop=n_oos,
        dir_acc_mean=dir_acc, dir_acc_ci_low=ci_low,
        dir_acc_ci_high=ci_high, dir_acc_p_value=p_value,
        base_rate=0.5,
        calibrator_brier=brier, calibrator_ece=ece,
        conformal_q=conformal_q,
        weights_path="/opt/neucast/weights/highfreq/btcusdt_1m.cbm",
    )


# ──────────────────── _release_tag heuristic ────────────────────


def test_release_tag_initial_run_returns_initial():
    """First run for a symbol has no prev; tagged "(initial)" so the
    table reads cleanly from row 1."""
    r = _run()
    assert _release_tag(r, None) == "(initial)"


def test_release_tag_no_change_returns_empty():
    r1 = _run(rid=1)
    r2 = _run(rid=2)
    assert _release_tag(r2, r1) == ""


def test_release_tag_feature_set_change():
    r1 = _run(feature_set="microstructure")
    r2 = _run(feature_set="cross_asset")
    tag = _release_tag(r2, r1)
    assert "feature_set: microstructure→cross_asset" in tag


def test_release_tag_holdout_days_change():
    r1 = _run(holdout_days=0)
    r2 = _run(holdout_days=3)
    tag = _release_tag(r2, r1)
    assert "holdout_days: 0→3" in tag


def test_release_tag_conformal_added():
    r1 = _run(conformal_q=None)
    r2 = _run(conformal_q=0.674)
    assert "conformal added" in _release_tag(r2, r1)


def test_release_tag_calibrator_change_when_brier_jumps():
    r1 = _run(brier=0.25)
    r2 = _run(brier=0.30)  # > 0.005 jump
    assert "calibrator change" in _release_tag(r2, r1)


def test_release_tag_calibrator_no_change_when_within_tolerance():
    """Tiny Brier jitter (< 0.005) shouldn't trigger a "calibrator
    change" tag — those are run-to-run sample noise, not true
    config changes."""
    r1 = _run(brier=0.250)
    r2 = _run(brier=0.252)
    assert "calibrator change" not in _release_tag(r2, r1)


def test_release_tag_combines_multiple_changes():
    """Real T.13 → T.16 transition: feature_set didn't change but
    holdout_days went 0→3 AND calibrator was retrained. Both must
    surface."""
    r1 = _run(holdout_days=0, brier=0.265)
    r2 = _run(holdout_days=3, brier=0.252)
    tag = _release_tag(r2, r1)
    assert "holdout_days: 0→3" in tag
    assert "calibrator change" in tag
    # Two changes joined with semicolon.
    assert ";" in tag


# ──────────────────── _to_float coercion ────────────────────


def test_to_float_handles_none_and_nan():
    assert _to_float(None) is None
    assert _to_float(float("nan")) is None


def test_to_float_handles_string_numerics():
    assert _to_float("0.5") == 0.5
    assert _to_float("nope") is None


# ──────────────────── render_markdown layout ────────────────────


def test_render_includes_all_three_sections():
    """Document structure: Latest summary → Frozen holdout (if data) →
    Per-symbol timeline → Footer. No section may appear out of order."""
    runs = [_run(symbol="BTCUSDT", rid=1, minute_offset=0)]
    md = render_markdown(runs, weights_dir=None)
    assert "# NeuCast HF — Production Training Scoreboard" in md
    assert "## Latest production metrics" in md
    assert "## BTCUSDT" in md
    assert "_Generated at" in md
    # Order: Latest before per-symbol.
    pos_latest = md.find("## Latest production metrics")
    pos_btc = md.find("## BTCUSDT")
    assert pos_latest < pos_btc, "summary section must precede per-symbol tables"


def test_render_latest_summary_one_row_per_symbol():
    """Three symbols, one row each in the summary block."""
    runs = [
        _run(symbol="BTCUSDT", rid=1, dir_acc=0.55),
        _run(symbol="ETHUSDT", rid=2, dir_acc=0.54),
        _run(symbol="BNBUSDT", rid=3, dir_acc=0.56),
    ]
    md = render_markdown(runs, weights_dir=None)
    summary = md[md.find("## Latest production metrics"):md.find("## BTCUSDT")]
    assert summary.count("| BTCUSDT |") == 1
    assert summary.count("| ETHUSDT |") == 1
    assert summary.count("| BNBUSDT |") == 1


def test_render_per_symbol_table_includes_config_delta_column():
    """The per-symbol table's config-delta column is the headline
    deliverable — pin its presence."""
    runs = [
        _run(symbol="BTCUSDT", rid=1, feature_set="microstructure"),
        _run(symbol="BTCUSDT", rid=2, feature_set="cross_asset"),
    ]
    md = render_markdown(runs, weights_dir=None)
    # Header.
    assert "config delta" in md
    # The 2nd run shows the feature_set transition.
    assert "feature_set: microstructure→cross_asset" in md


def test_render_handles_empty_runs():
    """No data → still produces a valid (mostly-empty) document with
    the header + summary stub. No crash."""
    md = render_markdown([], weights_dir=None)
    assert "# NeuCast HF — Production Training Scoreboard" in md
    assert "## Latest production metrics" in md
    # No per-symbol tables.
    assert "## BTCUSDT" not in md


def test_render_em_dash_for_missing_metrics():
    """Some metrics are None on early runs (before the field was
    added). Rendered as em-dash so the table stays aligned."""
    r = _run(brier=None, ece=None, conformal_q=None)
    md = render_markdown([r], weights_dir=None)
    # All three em-dashes appear in the BTC row.
    assert md.count("| — |") >= 3


def test_render_footer_has_run_count():
    runs = [_run(rid=i) for i in range(7)]
    md = render_markdown(runs, weights_dir=None)
    assert "from 7 training runs" in md
