"""Tests for ``GET /api/highfreq/conditional_accuracy`` (release T.12).

The endpoint buckets ``predictions_log`` rows by confidence threshold
``|prob_up - 0.5| >= t`` for t ∈ {0.05, 0.10, 0.15} and computes
realized dir_acc with Wilson 95% CI per (symbol, bucket). Tests pin:

1. Bucket shape: 3 named buckets (conf_55 / conf_60 / conf_65) per
   symbol, nested (each higher threshold is a strict subset of the
   lower one).
2. dir_acc + Wilson CI math matches the SQL aggregates.
3. DB error degrades gracefully to ``ok=False`` (never 500).
4. Empty DB returns ``ok=True`` with empty rows[].
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.highfreq.web import _get_db, router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _override_db_with_per_call_rows(
    app: FastAPI, rows_by_threshold: dict[float, list[tuple]],
) -> None:
    """Mock DB so each call to ``execute(sql, {'t': t}).all()``
    returns the rows mapped to that ``t`` value.

    rows: list of (symbol, n, hits) tuples.
    """
    fake_db = MagicMock()

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        t = (params or {}).get("t")
        rows = rows_by_threshold.get(t, [])
        result.all.return_value = rows
        return result

    fake_db.execute.side_effect = execute_side_effect
    app.dependency_overrides[_get_db] = lambda: fake_db


def test_conditional_accuracy_happy_path_three_buckets_per_symbol():
    """When predictions_log has data for all 3 thresholds, the
    response carries one row per symbol, each with conf_55 /
    conf_60 / conf_65 keys."""
    # Fixture: BTC at conf>=0.55: 1000 calls / 540 hits = 0.54
    #         BTC at conf>=0.60:  500 calls / 290 hits = 0.58
    #         BTC at conf>=0.65:  200 calls / 120 hits = 0.60
    # ETH only has conf>=0.55 (lower threshold).
    rows_by_threshold = {
        0.05: [("BTCUSDT", 1000, 540), ("ETHUSDT", 800, 410)],
        0.10: [("BTCUSDT", 500, 290),  ("ETHUSDT", 400, 215)],
        0.15: [("BTCUSDT", 200, 120),  ("ETHUSDT", 150, 80)],
    }
    app = _make_app()
    _override_db_with_per_call_rows(app, rows_by_threshold)
    client = TestClient(app)
    res = client.get("/api/highfreq/conditional_accuracy")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    rows = body["rows"]
    assert len(rows) == 2

    # Pick BTC.
    btc = next(r for r in rows if r["symbol"] == "BTCUSDT")
    buckets = btc["buckets"]
    assert set(buckets.keys()) == {"conf_55", "conf_60", "conf_65"}
    assert buckets["conf_55"]["n"] == 1000
    assert buckets["conf_55"]["hits"] == 540
    assert abs(buckets["conf_55"]["dir_acc"] - 0.54) < 1e-9
    assert buckets["conf_60"]["dir_acc"] == 0.58
    assert buckets["conf_65"]["dir_acc"] == 0.60
    # Wilson CI sanity: lower bound < dir_acc < upper bound; both in [0,1].
    for key in ("conf_55", "conf_60", "conf_65"):
        b = buckets[key]
        assert 0.0 <= b["ci_low"] <= b["dir_acc"] <= b["ci_high"] <= 1.0
    # threshold values surface (UI uses them for axis labels).
    assert buckets["conf_55"]["threshold"] == 0.05
    assert buckets["conf_60"]["threshold"] == 0.10
    assert buckets["conf_65"]["threshold"] == 0.15
    app.dependency_overrides.clear()


def test_conditional_accuracy_zero_n_bucket_emits_null_acc():
    """Symbol with no rows in a bucket → that bucket gets None for
    dir_acc/CI/p_value (not 0/0 division NaN)."""
    rows_by_threshold = {
        0.05: [("BNBUSDT", 100, 55)],
        0.10: [("BNBUSDT", 0, 0)],   # no high-confidence calls yet
        0.15: [("BNBUSDT", 0, 0)],
    }
    app = _make_app()
    _override_db_with_per_call_rows(app, rows_by_threshold)
    client = TestClient(app)
    res = client.get("/api/highfreq/conditional_accuracy")
    body = res.json()
    bnb = body["rows"][0]
    assert bnb["buckets"]["conf_55"]["dir_acc"] == 0.55
    assert bnb["buckets"]["conf_60"]["dir_acc"] is None
    assert bnb["buckets"]["conf_65"]["dir_acc"] is None
    app.dependency_overrides.clear()


def test_conditional_accuracy_empty_db_returns_ok_with_no_rows():
    """No predictions logged at all → ok=True, rows=[]."""
    rows_by_threshold = {0.05: [], 0.10: [], 0.15: []}
    app = _make_app()
    _override_db_with_per_call_rows(app, rows_by_threshold)
    client = TestClient(app)
    res = client.get("/api/highfreq/conditional_accuracy")
    body = res.json()
    assert body["ok"] is True
    assert body["rows"] == []
    app.dependency_overrides.clear()


def test_conditional_accuracy_db_error_degrades_to_ok_false():
    """Operational error during DB query → ok=False with
    ``db_status='unavailable'``, NOT a 5xx."""
    fake_db = MagicMock()
    fake_db.execute.side_effect = OperationalError(
        "SELECT ...", {}, Exception("connection refused"),
    )
    app = _make_app()
    app.dependency_overrides[_get_db] = lambda: fake_db
    client = TestClient(app)
    res = client.get("/api/highfreq/conditional_accuracy")
    # Endpoint must NOT 500 on transient DB blips — the page polls
    # this on a timer, a short outage shouldn't bubble up as an error.
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["db_status"] == "unavailable"
    app.dependency_overrides.clear()
