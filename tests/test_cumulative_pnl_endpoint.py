"""Tests for ``GET /api/highfreq/cumulative_pnl`` (release T.17.c).

Per-symbol cumulative P&L curve by fee tier, sourced from
``paper_trades``. Tests pin:

1. Math: cumul[i] == sum(net_bps_at_tier for trades[:i+1])
2. Tier set matches TIER_DEFS in the UI (one source-of-truth)
3. Win-rate per tier counted correctly
4. Empty DB returns ok=true with empty points + zero finals
5. Downsampling preserves first + last + monotone time order
6. DB error degrades to ok=false
7. limit_points clamped to [10, 1000]
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.highfreq.web import _get_db, router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _override_db(app: FastAPI, rows: list[tuple]) -> None:
    fake_db = MagicMock()
    fake_db.execute.return_value.all.return_value = rows
    app.dependency_overrides[_get_db] = lambda: fake_db


def _ts(minute_offset: int = 0) -> datetime:
    return datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=minute_offset,
    )


def _trade(*, ts, gross_bps: float, side: str = "long") -> tuple:
    """Build a (exit_ts, entry_price, exit_price, side) row that the
    endpoint will reconstruct into ``gross_bps`` (sign-corrected).

    For ``long``: exit = entry * (1 + gross_bps/1e4)
    For ``short``: exit = entry * (1 - gross_bps/1e4)
    """
    entry = 1000.0
    if side == "long":
        exit_ = entry * (1 + gross_bps / 1e4)
    else:
        exit_ = entry * (1 - gross_bps / 1e4)
    return (ts, entry, exit_, side)


def test_cumulative_pnl_happy_path_math():
    """Three trades: +5bp, -3bp, +2bp gross. Net at retail (15bp
    roundtrip) should be -10, -28, -41 cumulative.
    Net at gross (0 fee) should be 5, 2, 4 cumulative.

    Endpoint reconstructs gross from entry/exit/side rather than
    using the stored ``pnl_bps`` which already includes retail fees.
    """
    rows = [
        _trade(ts=_ts(0), gross_bps=5.0),
        _trade(ts=_ts(1), gross_bps=-3.0),
        _trade(ts=_ts(2), gross_bps=2.0),
    ]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    res = client.get("/api/highfreq/cumulative_pnl?symbol=BTCUSDT")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["n_trades"] == 3
    pts = body["points"]
    assert len(pts) == 3
    # Gross cumulative: 5, 2, 4
    assert pts[0]["gross"] == 5.0
    assert pts[1]["gross"] == 2.0
    assert pts[2]["gross"] == 4.0
    # Retail cumulative (15 bp roundtrip subtracted per trade):
    # trade 1: 5 - 15 = -10 cumul=-10
    # trade 2: -3 - 15 = -18 cumul=-28
    # trade 3: 2 - 15 = -13 cumul=-41
    assert pts[0]["retail"] == -10.0
    assert pts[1]["retail"] == -28.0
    assert pts[2]["retail"] == -41.0


def test_cumulative_pnl_tier_definitions_match_ui():
    """TIER keys must match TIER_DEFS in templates/forecast.html so
    the cumulative curve's tier filter lines up with the per-tier
    P&L cards on the same page."""
    rows = [_trade(ts=_ts(0), gross_bps=5.0)]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    body = client.get("/api/highfreq/cumulative_pnl?symbol=BTCUSDT").json()
    tier_keys = {t["key"] for t in body["tiers"]}
    assert tier_keys == {
        "gross", "retail", "vip5", "vip9", "futures", "mm_rebate",
    }


def test_cumulative_pnl_win_rate_correct():
    """Win rate per tier = (trades with net_bps > 0) / n_trades."""
    # 4 trades: gross +5, +5, -1, +0.5
    # At gross (0 fee): wins = 3 (5, 5, 0.5 are > 0; -1 isn't) → 0.75
    # At retail (15bp roundtrip): all 4 net negative → 0.0 wins
    rows = [
        _trade(ts=_ts(0), gross_bps=5.0),
        _trade(ts=_ts(1), gross_bps=5.0),
        _trade(ts=_ts(2), gross_bps=-1.0),
        _trade(ts=_ts(3), gross_bps=0.5),
    ]
    app = _make_app()
    _override_db(app, rows)
    body = TestClient(app).get("/api/highfreq/cumulative_pnl").json()
    by_key = {t["key"]: t for t in body["tiers"]}
    assert by_key["gross"]["win_rate"] == 0.75
    assert by_key["retail"]["win_rate"] == 0.0


def test_cumulative_pnl_empty_db():
    """No trades for this symbol → ok=true, empty points, zero
    finals (not nulls)."""
    app = _make_app()
    _override_db(app, [])
    body = TestClient(app).get("/api/highfreq/cumulative_pnl?symbol=BTCUSDT").json()
    assert body["ok"] is True
    assert body["n_trades"] == 0
    assert body["points"] == []
    # Zero finals, not nulls — chart renders a flat line at 0.
    for t in body["tiers"]:
        assert t["final_bps"] == 0.0


def test_cumulative_pnl_downsampling_preserves_endpoints():
    """When n_trades > limit_points, the response downsamples but
    MUST keep first + last point so the chart's start/end are
    accurate. Also the curve should remain monotone in time."""
    # 1000 trades with constant +1bp gross each.
    rows = [_trade(ts=_ts(i), gross_bps=1.0) for i in range(1000)]
    app = _make_app()
    _override_db(app, rows)
    body = TestClient(app).get(
        "/api/highfreq/cumulative_pnl?limit_points=100"
    ).json()
    pts = body["points"]
    assert body["n_trades"] == 1000
    # Length is approximately limit_points (within ±1 because of stride math).
    assert 90 <= len(pts) <= 110
    # First point is first trade.
    assert pts[0]["ts"].endswith("10:00:00+00:00")
    # Last point is the last trade (index 999 = 999 minutes after start).
    assert pts[-1]["gross"] == 1000.0  # cumulative of 1000 × 1bp
    # Time-monotone.
    for i in range(1, len(pts)):
        assert pts[i]["ts"] > pts[i-1]["ts"]


def test_cumulative_pnl_db_error_returns_ok_false():
    fake_db = MagicMock()
    fake_db.execute.side_effect = OperationalError(
        "SELECT ...", {}, Exception("conn refused"),
    )
    app = _make_app()
    app.dependency_overrides[_get_db] = lambda: fake_db
    res = TestClient(app).get("/api/highfreq/cumulative_pnl")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["db_status"] == "unavailable"


def test_cumulative_pnl_short_side_signs_correctly():
    """Short trades earn gross_bps > 0 when price FALLS. The endpoint
    must sign-correct via the side column so a short on a -3bp move
    gets credited +3bp gross, not -3bp."""
    # Short trade: entry=1000, exit=997 → price moved -3bp → gross +3bp.
    rows = [
        _trade(ts=_ts(0), gross_bps=3.0, side="short"),
        _trade(ts=_ts(1), gross_bps=-2.0, side="short"),  # price went UP, short loses
    ]
    app = _make_app()
    _override_db(app, rows)
    body = TestClient(app).get("/api/highfreq/cumulative_pnl").json()
    pts = body["points"]
    # Cumul gross: 3, 1.
    assert pts[0]["gross"] == 3.0
    assert abs(pts[1]["gross"] - 1.0) < 1e-9


def test_cumulative_pnl_limit_points_clamped():
    rows = [_trade(ts=_ts(i), gross_bps=1.0) for i in range(50)]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    # too small.
    body = client.get("/api/highfreq/cumulative_pnl?limit_points=2").json()
    # 50 trades < 200 (default after clamp), no downsampling needed.
    assert len(body["points"]) == 50
    # too large.
    body = client.get("/api/highfreq/cumulative_pnl?limit_points=99999").json()
    assert len(body["points"]) == 50
