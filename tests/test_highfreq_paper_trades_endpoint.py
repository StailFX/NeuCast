"""Tests for ``GET /api/highfreq/paper_trades`` (Phase C UI block).

Coverage:

* **Pure stat computation** (``compute_paper_stats``): empty list,
  populated list with mixed pnl, today-vs-yesterday window, halt
  detection (loss-streak + daily-loss), cumulative series shape.
* **Endpoint dispatch**: 200 with empty trades, 200 with trades,
  ``db_status="unavailable"`` graceful fallback, ``limit`` param
  bounds (clamped to MAX, defaulted on missing).

We mock the SQLAlchemy session (via FastAPI ``dependency_overrides``)
rather than spinning up Postgres — the endpoint's logic is mostly
in the pure helper, which we test independently.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.paper_trader import PaperTraderConfig, RiskCaps
from app.highfreq.web import (
    DEFAULT_PAPER_TRADES_LIMIT,
    MAX_PAPER_TRADES_LIMIT,
    _config_to_dict,
    _get_db,
    _serialise_trade,
    compute_paper_stats,
    router,
)


UTC = timezone.utc


# ───────────────────────── helpers ─────────────────────────


def _ts(year: int = 2026, month: int = 4, day: int = 26,
        hour: int = 12, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _trade(
    *,
    pnl_usd: float = 1.0,
    exit_ts: datetime = None,
    side: str = "long",
    pnl_bps: float = 10.0,
    model_version: str = "v1",
) -> dict[str, Any]:
    return {
        "id": 1,
        "symbol": "BTCUSDT",
        "side": side,
        "qty": 0.001,
        "entry_ts": (exit_ts or _ts()) - timedelta(minutes=1),
        "entry_price": 78_000.0,
        "entry_prob_up": 0.7 if side == "long" else 0.3,
        "exit_ts": exit_ts or _ts(),
        "exit_price": 78_010.0 if pnl_usd > 0 else 77_990.0,
        "exit_reason": "time_stop",
        "fee_paid_total_usd": 0.117,
        "pnl_usd": pnl_usd,
        "pnl_bps": pnl_bps,
        "model_version": model_version,
        "written_at": exit_ts or _ts(),
    }


def _make_app() -> FastAPI:
    """Spin up a FastAPI app with just the highfreq router."""
    app = FastAPI()
    app.include_router(router)
    return app


def _override_db(app: FastAPI, *, fetch_returns, count_returns=None) -> None:
    """Override the DB dep to return controlled values.

    ``fetch_returns``: list[dict] for paper_trades query, OR Exception
                       to simulate DB failure.
    ``count_returns``: int total trades for symbol (None → 0).
    """
    fake_db = MagicMock()

    def execute_side_effect(stmt, params=None):
        # Two SELECT shapes: fetch_recent (mappings().all()) +
        # count (.scalar()). Distinguish by looking at the SQL text.
        sql_text = str(stmt)
        result = MagicMock()
        if "COUNT(*)" in sql_text:
            result.scalar.return_value = count_returns
            return result
        # Recent trades query.
        if isinstance(fetch_returns, Exception):
            raise fetch_returns
        result.mappings.return_value.all.return_value = fetch_returns
        return result

    fake_db.execute.side_effect = execute_side_effect
    app.dependency_overrides[_get_db] = lambda: fake_db


# ───────────────────────── compute_paper_stats ─────────────────────────


def test_compute_paper_stats_empty_list():
    s = compute_paper_stats(
        trades_recent=[], total_trades_lifetime=0, now=_ts(),
        config=PaperTraderConfig(), risk_caps=RiskCaps(),
    )
    assert s["total_trades"] == 0
    assert s["trades_today"] == 0
    assert s["today_pnl_usd"] == 0.0
    assert s["wins_today"] == 0
    assert s["losses_today"] == 0
    assert s["consecutive_losses"] == 0
    assert s["halted_reason"] is None
    assert s["last_trade_ts"] is None
    assert s["latest_model_version"] is None
    assert s["cumulative_pnl_series"] == []


def test_compute_paper_stats_today_pnl_sums_only_today():
    """Trades from yesterday must NOT pollute today_pnl_usd."""
    today = _ts(day=26, hour=12)
    yesterday = _ts(day=25, hour=12)
    trades = [
        _trade(pnl_usd=+0.50, exit_ts=today),
        _trade(pnl_usd=+1.00, exit_ts=yesterday),  # not in today
    ]
    s = compute_paper_stats(
        trades_recent=trades, total_trades_lifetime=2, now=today,
        config=PaperTraderConfig(), risk_caps=RiskCaps(),
    )
    assert s["trades_today"] == 1
    assert s["today_pnl_usd"] == 0.5
    assert s["wins_today"] == 1
    assert s["losses_today"] == 0


def test_compute_paper_stats_consecutive_losses_walks_from_newest():
    """Loss streak = count of consecutive losing trades from the most
    recent backwards. A win earlier in the list breaks the streak."""
    now = _ts()
    trades = [
        # Newest first (matches DESC order from the fetch).
        _trade(pnl_usd=-0.20, exit_ts=now),
        _trade(pnl_usd=-0.10, exit_ts=now - timedelta(minutes=1)),
        _trade(pnl_usd=-0.05, exit_ts=now - timedelta(minutes=2)),
        _trade(pnl_usd=+0.30, exit_ts=now - timedelta(minutes=3)),  # breaks streak
        _trade(pnl_usd=-0.10, exit_ts=now - timedelta(minutes=4)),
    ]
    s = compute_paper_stats(
        trades_recent=trades, total_trades_lifetime=5, now=now,
        config=PaperTraderConfig(), risk_caps=RiskCaps(),
    )
    assert s["consecutive_losses"] == 3


def test_compute_paper_stats_halt_loss_streak_at_cap():
    """5 consecutive losses → halted_reason='loss_streak'."""
    now = _ts()
    trades = [_trade(pnl_usd=-0.10, exit_ts=now - timedelta(minutes=i)) for i in range(5)]
    s = compute_paper_stats(
        trades_recent=trades, total_trades_lifetime=5, now=now,
        config=PaperTraderConfig(),
        risk_caps=RiskCaps(max_consecutive_losses=5, max_daily_loss_usd=1e9),
    )
    assert s["consecutive_losses"] == 5
    assert s["halted_reason"] == "loss_streak"


def test_compute_paper_stats_halt_daily_loss():
    """Daily P&L below -cap → halted_reason='daily_loss' (even if streak ok)."""
    now = _ts()
    trades = [
        _trade(pnl_usd=-3.0, exit_ts=now),
        _trade(pnl_usd=+1.0, exit_ts=now - timedelta(minutes=1)),  # win → no streak
    ]
    s = compute_paper_stats(
        trades_recent=trades, total_trades_lifetime=2, now=now,
        config=PaperTraderConfig(),
        risk_caps=RiskCaps(max_consecutive_losses=100, max_daily_loss_usd=1.0),
    )
    assert s["today_pnl_usd"] == -2.0
    assert s["halted_reason"] == "daily_loss"


def test_compute_paper_stats_halt_loss_streak_takes_precedence():
    """If both halts trigger simultaneously, loss_streak wins (matches
    PaperTrader._maybe_engage_risk_caps order)."""
    now = _ts()
    trades = [_trade(pnl_usd=-2.0, exit_ts=now - timedelta(minutes=i)) for i in range(5)]
    s = compute_paper_stats(
        trades_recent=trades, total_trades_lifetime=5, now=now,
        config=PaperTraderConfig(),
        risk_caps=RiskCaps(max_consecutive_losses=5, max_daily_loss_usd=1.0),
    )
    assert s["halted_reason"] == "loss_streak"


def test_compute_paper_stats_cumulative_pnl_oldest_to_newest():
    """The series powers a left-to-right chart; oldest trade first."""
    now = _ts()
    # DESC order from fetch: newest first.
    trades = [
        _trade(pnl_usd=+0.5, exit_ts=now),                            # 3rd chronologically
        _trade(pnl_usd=-0.2, exit_ts=now - timedelta(minutes=1)),     # 2nd
        _trade(pnl_usd=+0.3, exit_ts=now - timedelta(minutes=2)),     # 1st
    ]
    s = compute_paper_stats(
        trades_recent=trades, total_trades_lifetime=3, now=now,
        config=PaperTraderConfig(), risk_caps=RiskCaps(),
    )
    series = s["cumulative_pnl_series"]
    assert len(series) == 3
    # Oldest first.
    assert series[0]["cum_pnl_usd"] == 0.3
    assert series[1]["cum_pnl_usd"] == 0.1   # 0.3 + (-0.2)
    assert series[2]["cum_pnl_usd"] == 0.6   # 0.1 + 0.5


def test_compute_paper_stats_last_trade_ts_is_newest():
    now = _ts()
    trades = [
        _trade(pnl_usd=+0.1, exit_ts=now),
        _trade(pnl_usd=-0.1, exit_ts=now - timedelta(minutes=10)),
    ]
    s = compute_paper_stats(
        trades_recent=trades, total_trades_lifetime=2, now=now,
        config=PaperTraderConfig(), risk_caps=RiskCaps(),
    )
    assert s["last_trade_ts"] == now.isoformat()


# ───────────────────────── _serialise_trade ─────────────────────────


def test_serialise_trade_converts_datetimes_to_iso():
    t = _trade(exit_ts=_ts(hour=12, minute=34))
    out = _serialise_trade(t)
    assert isinstance(out["entry_ts"], str)
    assert isinstance(out["exit_ts"], str)
    assert isinstance(out["written_at"], str)
    assert out["exit_ts"].startswith("2026-04-26T12:34:00")


# ───────────────────────── _config_to_dict ─────────────────────────


def test_config_to_dict_exposes_thresholds_for_ui_labels():
    cfg = PaperTraderConfig(entry_long_threshold=0.6, max_qty_usd=200.0)
    risk = RiskCaps(max_consecutive_losses=7, max_daily_loss_usd=10.0)
    d = _config_to_dict(cfg, risk)
    assert d["entry_long_threshold"] == 0.6
    assert d["max_qty_usd"] == 200.0
    assert d["max_consecutive_losses"] == 7
    assert d["max_daily_loss_usd"] == 10.0


# ───────────────────────── endpoint ─────────────────────────


def test_endpoint_returns_empty_trades_with_200_when_table_empty():
    """Cold-start: no trades yet — must NOT 503; UI needs to render."""
    app = _make_app()
    _override_db(app, fetch_returns=[], count_returns=0)
    client = TestClient(app)

    r = client.get("/api/highfreq/paper_trades")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["db_status"] == "ok"
    assert j["trades"] == []
    assert j["stats"]["total_trades"] == 0
    assert j["config"]["max_consecutive_losses"] == 5  # default
    app.dependency_overrides.clear()


def test_endpoint_returns_populated_trades_and_stats():
    app = _make_app()
    now = _ts(hour=12, minute=10)
    trades = [
        _trade(pnl_usd=+0.50, exit_ts=now),
        _trade(pnl_usd=-0.10, exit_ts=now - timedelta(minutes=1)),
    ]
    _override_db(app, fetch_returns=trades, count_returns=2)
    client = TestClient(app)

    r = client.get("/api/highfreq/paper_trades?symbol=BTCUSDT")
    assert r.status_code == 200
    j = r.json()
    assert len(j["trades"]) == 2
    assert j["stats"]["total_trades"] == 2
    assert j["stats"]["consecutive_losses"] == 0  # newest is a win
    # Trades must be JSON-serialisable (datetimes → strings).
    assert isinstance(j["trades"][0]["exit_ts"], str)
    app.dependency_overrides.clear()


def test_endpoint_returns_db_unavailable_when_db_raises():
    """ProgrammingError / OperationalError → soft-fail, UI keeps rendering."""
    from sqlalchemy.exc import OperationalError
    app = _make_app()
    _override_db(
        app,
        fetch_returns=OperationalError("connection refused", None, None),
    )
    client = TestClient(app)

    r = client.get("/api/highfreq/paper_trades")
    assert r.status_code == 200  # NOT 503 — UI still loads
    j = r.json()
    assert j["ok"] is False
    assert j["db_status"] == "unavailable"
    assert j["trades"] == []
    assert j["stats"] is None
    assert "config" in j  # config still emitted so UI can show thresholds
    app.dependency_overrides.clear()


def test_endpoint_clamps_limit_to_max():
    """?limit=99999999 must not nuke the DB — silently clamp to MAX."""
    app = _make_app()
    captured = {}

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "COUNT(*)" in str(stmt):
            result.scalar.return_value = 0
            return result
        captured["limit_param"] = params.get("limit") if params else None
        result.mappings.return_value.all.return_value = []
        return result

    fake_db = MagicMock()
    fake_db.execute.side_effect = execute_side_effect
    app.dependency_overrides[_get_db] = lambda: fake_db

    client = TestClient(app)
    r = client.get("/api/highfreq/paper_trades?limit=99999999")
    assert r.status_code == 200
    assert captured["limit_param"] == MAX_PAPER_TRADES_LIMIT
    app.dependency_overrides.clear()


def test_endpoint_floors_limit_to_one():
    """?limit=0 or negative must clamp to 1 (not allow 0 → infinite scan)."""
    app = _make_app()
    captured = {}

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "COUNT(*)" in str(stmt):
            result.scalar.return_value = 0
            return result
        captured["limit_param"] = params.get("limit") if params else None
        result.mappings.return_value.all.return_value = []
        return result

    fake_db = MagicMock()
    fake_db.execute.side_effect = execute_side_effect
    app.dependency_overrides[_get_db] = lambda: fake_db

    client = TestClient(app)
    r = client.get("/api/highfreq/paper_trades?limit=-5")
    assert r.status_code == 200
    assert captured["limit_param"] == 1
    app.dependency_overrides.clear()


def test_endpoint_uppercases_symbol():
    """Lowercase symbols in URL must be normalised before SQL."""
    app = _make_app()
    captured = {}

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "COUNT(*)" in str(stmt):
            result.scalar.return_value = 0
            return result
        captured["symbol"] = params.get("symbol") if params else None
        result.mappings.return_value.all.return_value = []
        return result

    fake_db = MagicMock()
    fake_db.execute.side_effect = execute_side_effect
    app.dependency_overrides[_get_db] = lambda: fake_db

    client = TestClient(app)
    r = client.get("/api/highfreq/paper_trades?symbol=btcusdt")
    assert r.status_code == 200
    assert captured["symbol"] == "BTCUSDT"
    app.dependency_overrides.clear()


def test_endpoint_default_limit_is_50():
    """Sanity: omitted ?limit must use DEFAULT_PAPER_TRADES_LIMIT."""
    app = _make_app()
    captured = {}

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "COUNT(*)" in str(stmt):
            result.scalar.return_value = 0
            return result
        captured["limit_param"] = params.get("limit") if params else None
        result.mappings.return_value.all.return_value = []
        return result

    fake_db = MagicMock()
    fake_db.execute.side_effect = execute_side_effect
    app.dependency_overrides[_get_db] = lambda: fake_db

    client = TestClient(app)
    r = client.get("/api/highfreq/paper_trades")
    assert r.status_code == 200
    assert captured["limit_param"] == DEFAULT_PAPER_TRADES_LIMIT
    app.dependency_overrides.clear()


# ───────────────────────── /api/highfreq/training_report ─────────────────────────


def test_training_report_returns_no_report_yet_when_metrics_missing(tmp_path, monkeypatch):
    """Cold-start: trainer never ran → 200 with reason='no_report_yet'."""
    import app.highfreq.web as web_mod
    monkeypatch.setattr(web_mod, "DEFAULT_METRICS_PATH", tmp_path / "absent.json")

    app = _make_app()
    client = TestClient(app)
    r = client.get("/api/highfreq/training_report")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is False
    assert j["reason"] == "no_report_yet"
    assert j["min_bars_for_first_fold"] == 1500


def test_training_report_returns_report_with_fold_progress(tmp_path, monkeypatch):
    """Trainer ran, wrote metrics → endpoint returns it + computed
    fold_ready_pct."""
    import json as _json
    import app.highfreq.web as web_mod

    metrics = tmp_path / "btc_metrics.json"
    metrics.write_text(_json.dumps({
        "symbol": "BTCUSDT",
        "n_seconds_loaded": 94979,
        "n_minutes_after_aggregation": 1583,
        "n_minutes_after_neutral_drop": 726,
        "base_rate": 0.514,
        "n_folds": 0,
        "dir_acc_mean": None,
        "dir_acc_ci_low": None,
        "dir_acc_ci_high": None,
        "low_directional_skill": True,
    }))
    monkeypatch.setattr(web_mod, "DEFAULT_METRICS_PATH", metrics)

    app = _make_app()
    client = TestClient(app)
    r = client.get("/api/highfreq/training_report")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    # 726 / 1500 = 48.4%
    assert j["fold_ready_pct"] == pytest.approx(48.4, abs=0.1)
    assert j["report"]["n_folds"] == 0
    assert j["report"]["n_minutes_after_neutral_drop"] == 726
    assert j["report_age_seconds"] is not None
    assert j["report_age_seconds"] >= 0


def test_training_report_unreadable_file(tmp_path, monkeypatch):
    """Corrupt metrics.json mid-write → 200 with reason='report_unreadable',
    UI keeps rendering."""
    import app.highfreq.web as web_mod

    metrics = tmp_path / "corrupt.json"
    metrics.write_text("{this is not json")
    monkeypatch.setattr(web_mod, "DEFAULT_METRICS_PATH", metrics)

    app = _make_app()
    client = TestClient(app)
    r = client.get("/api/highfreq/training_report")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is False
    assert j["reason"] == "report_unreadable"


def test_training_report_fold_pct_caps_at_100(tmp_path, monkeypatch):
    """If we have MORE than min_bars_for_first_fold (folds completed),
    pct caps at 100% (no overshoot)."""
    import json as _json
    import app.highfreq.web as web_mod

    metrics = tmp_path / "abundant.json"
    metrics.write_text(_json.dumps({
        "n_minutes_after_neutral_drop": 5000,
        "n_folds": 3,
        "dir_acc_mean": 0.547,
        "dir_acc_ci_low": 0.521,
        "dir_acc_ci_high": 0.572,
    }))
    monkeypatch.setattr(web_mod, "DEFAULT_METRICS_PATH", metrics)

    app = _make_app()
    client = TestClient(app)
    j = client.get("/api/highfreq/training_report").json()
    assert j["fold_ready_pct"] == 100.0  # not 333.3
