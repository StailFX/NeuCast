"""Tests for ``GET /api/highfreq/reliability_diagram`` (release T.16).

The reliability-diagram endpoint buckets ``predictions_log`` by
``prob_up`` and computes ``realized_rate`` per bucket — the standard
calibration plot. A perfectly calibrated model has ``realized_rate ==
predicted_mean`` (points on diagonal). This is one of the canonical
defence-grade plots reviewers ask for.

Tests pin:
1. Bucketing math: prob_up=0.55 lands in bin 5 (out of 10), realized=1
   for an "up"-signal correct call, etc.
2. ECE / Brier accumulation.
3. Empty-bin handling: realized_rate is None (so UI plots a gap).
4. DB error degrades to ok=false (200, not 5xx).
5. n_bins clamped to [3, 30].
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


def _override_db(app: FastAPI, rows: list[tuple]) -> None:
    """Mock the predictions_log SELECT to return ``rows``.
    Each row: (symbol, prob_up, realized_correct, signal)."""
    fake_db = MagicMock()
    fake_db.execute.return_value.all.return_value = rows
    app.dependency_overrides[_get_db] = lambda: fake_db


def test_reliability_diagram_happy_path_buckets_correctly():
    # 10 BTC predictions distributed across the prob_up axis.
    # signal/realized_correct → outcome:
    #   signal='up' + correct → outcome=1 (model said up, was up)
    #   signal='up' + wrong   → outcome=0 (model said up, was down)
    #   signal='down' + correct → outcome=0 (model said down, was down)
    #   signal='down' + wrong   → outcome=1 (model said down, was up)
    rows = [
        # Low-prob bins: model leans down. Realized rate should be ~0.
        ("BTCUSDT", 0.10, True, "down"),    # outcome 0
        ("BTCUSDT", 0.15, True, "down"),    # outcome 0
        ("BTCUSDT", 0.20, False, "down"),   # outcome 1 (one "wrong" downwave)
        # Mid bins.
        ("BTCUSDT", 0.45, False, "down"),   # outcome 1
        ("BTCUSDT", 0.55, True, "up"),      # outcome 1
        ("BTCUSDT", 0.50, True, "up"),      # outcome 1 (lands in bin 5 — 0.5..0.6)
        # High-prob bins: model leans up. Realized rate ~1.
        ("BTCUSDT", 0.75, True, "up"),      # outcome 1
        ("BTCUSDT", 0.80, False, "up"),     # outcome 0
        ("BTCUSDT", 0.90, True, "up"),      # outcome 1
        ("BTCUSDT", 0.95, True, "up"),      # outcome 1
    ]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    res = client.get("/api/highfreq/reliability_diagram?n_bins=10")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    btc = body["rows"][0]
    assert btc["symbol"] == "BTCUSDT"
    assert btc["n_total"] == 10
    assert btc["brier"] is not None and 0.0 < btc["brier"] < 1.0
    assert btc["ece"] is not None and btc["ece"] >= 0
    # Should have 10 buckets (matches n_bins).
    assert len(btc["buckets"]) == 10
    # Bucket 1 (0.10..0.20): 2 entries, both outcome 0 → realized_rate=0.
    b1 = next(b for b in btc["buckets"] if b["bin_idx"] == 1)
    assert b1["n"] == 2
    assert b1["realized_rate"] == 0.0
    # Bucket 9 (0.90..1.0): 2 entries, both outcome 1 → realized_rate=1.0
    b9 = next(b for b in btc["buckets"] if b["bin_idx"] == 9)
    assert b9["n"] == 2
    assert b9["realized_rate"] == 1.0
    app.dependency_overrides.clear()


def test_reliability_diagram_skips_neutral_signals():
    """Neutral signals don't bet on a direction → can't be scored as
    calibration data. They must NOT inflate any bucket."""
    rows = [
        ("BTCUSDT", 0.50, None, "neutral"),
        ("BTCUSDT", 0.51, None, "neutral"),
        ("BTCUSDT", 0.55, True, "up"),
    ]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    res = client.get("/api/highfreq/reliability_diagram")
    body = res.json()
    btc = body["rows"][0]
    # Only the one 'up' signal counts.
    assert btc["n_total"] == 1
    app.dependency_overrides.clear()


def test_reliability_diagram_empty_bins_emit_null_realized_rate():
    """When a bin has 0 predictions, ``realized_rate`` must be None
    (not 0/0 NaN, not 0.0) so the UI plots a gap rather than a
    misleading dot at zero."""
    rows = [("BTCUSDT", 0.55, True, "up")]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    body = client.get("/api/highfreq/reliability_diagram").json()
    btc = body["rows"][0]
    bin_5 = next(b for b in btc["buckets"] if b["bin_idx"] == 5)
    assert bin_5["n"] == 1
    bin_0 = next(b for b in btc["buckets"] if b["bin_idx"] == 0)
    assert bin_0["n"] == 0
    assert bin_0["realized_rate"] is None
    assert bin_0["predicted_mean"] is None
    app.dependency_overrides.clear()


def test_reliability_diagram_db_error_returns_ok_false():
    fake_db = MagicMock()
    fake_db.execute.side_effect = OperationalError(
        "SELECT ...", {}, Exception("connection refused"),
    )
    app = _make_app()
    app.dependency_overrides[_get_db] = lambda: fake_db
    client = TestClient(app)
    res = client.get("/api/highfreq/reliability_diagram")
    # Always 200 so the page poll doesn't crash on a transient DB blip.
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["db_status"] == "unavailable"
    app.dependency_overrides.clear()


def test_reliability_diagram_n_bins_clamped():
    """User-supplied n_bins outside [3, 30] → defaults to 10 (no
    crash)."""
    rows = [("BTCUSDT", 0.55, True, "up")]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    # n_bins too small.
    body = client.get("/api/highfreq/reliability_diagram?n_bins=2").json()
    assert body["n_bins"] == 10
    # n_bins too large.
    body = client.get("/api/highfreq/reliability_diagram?n_bins=100").json()
    assert body["n_bins"] == 10
    # n_bins valid.
    body = client.get("/api/highfreq/reliability_diagram?n_bins=20").json()
    assert body["n_bins"] == 20
    app.dependency_overrides.clear()


def test_reliability_diagram_ece_is_zero_for_perfect_calibration():
    """If realized_rate exactly equals predicted_mean in every bucket,
    ECE = 0. Synthesise that: bin 5 (0.5..0.6) gets predictions
    averaging 0.55 with 55% outcome=1."""
    # 100 predictions: exactly 55 outcome=1 in the 0.5..0.6 bin.
    # Use signal='up' + correct iff outcome=1 (correct=True).
    rows = [
        ("BTCUSDT", 0.55, True, "up") for _ in range(55)
    ] + [
        ("BTCUSDT", 0.55, False, "up") for _ in range(45)
    ]
    app = _make_app()
    _override_db(app, rows)
    client = TestClient(app)
    body = client.get("/api/highfreq/reliability_diagram").json()
    btc = body["rows"][0]
    # All 100 in one bin → predicted_mean=0.55, realized_rate=0.55, ECE=0.
    assert abs(btc["ece"] - 0.0) < 1e-9
    app.dependency_overrides.clear()
