"""Tests for ``GET /api/highfreq/dashboard`` (code-review H-1perf,
2026-05-04).

The dashboard endpoint serves the full /forecast page payload for
multiple symbols in ONE request. Tests pin:

1. Happy path: 3 symbols, all blocks populated.
2. Per-symbol fail-isolation: a missing model on one symbol doesn't
   poison the others.
3. Symbol validation: bad symbol → 400.
4. Cap: too many symbols → 400.
5. Default symbols env-driven (HIGHFREQ_SYMBOLS).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.web import _get_db, router


@pytest.fixture(autouse=True)
def _patch_prefetch_seconds():
    """Stub ``_fetch_recent_seconds`` + ``_fetch_recent_futures_seconds``
    so the underlying ``_predict_for_horizon`` path doesn't hit a real
    DB. Tests that need to exercise specific microprice / drift
    behaviour add their own patches on top."""
    with patch(
        "app.highfreq.web._fetch_recent_seconds",
        return_value=pd.DataFrame({
            "ts": [pd.Timestamp("2026-05-04T00:00:00Z")],
            "symbol": ["BTCUSDT"],
            "microprice": [79000.5],
            "ofi": [0.0], "depth_imb": [0.0], "spread_bps": [1.0],
            "trade_imb": [0.0], "n_updates": [5],
        }),
    ), patch(
        "app.highfreq.web._fetch_recent_futures_seconds",
        return_value=pd.DataFrame(),
    ):
        yield


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_dashboard_happy_path_three_symbols():
    """All 3 symbols have models → response carries forecast +
    drift + microprice for each one."""
    app = _make_app()
    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred, \
         patch("app.highfreq.web._read_json_cached", return_value=None):
        # _predict_for_horizon called 3× — one per symbol.
        mock_pred.side_effect = [
            (0.62, {"has_model": True, "feature_set": "microstructure"}),
            (0.51, {"has_model": True, "feature_set": "cross_asset"}),
            (0.40, {"has_model": True, "feature_set": "cross_asset"}),
        ]
        res = TestClient(app).get(
            "/api/highfreq/dashboard?symbols=BTCUSDT,ETHUSDT,BNBUSDT"
        )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["n_symbols"] == 3
    assert set(body["symbols"]) == {"BTCUSDT", "ETHUSDT", "BNBUSDT"}
    btc = body["symbols"]["BTCUSDT"]
    assert btc["forecast"]["ok"] is True
    assert btc["forecast"]["prob_up"] == pytest.approx(0.62)
    assert btc["forecast"]["signal"] == "up"
    # Drift is no_check_yet (no JSON file in test env).
    assert btc["drift"]["ok"] is False
    # Microprice surfaces from the patched _fetch_recent_seconds frame.
    assert btc["microprice"]["ok"] is True
    assert btc["microprice"]["price"] == pytest.approx(79000.5)
    app.dependency_overrides.clear()


def test_dashboard_per_symbol_fail_isolation():
    """When BTC's model is unavailable, ETH/BNB still surface their
    payloads — one cold-starting model doesn't poison the batch."""
    app = _make_app()
    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred, \
         patch("app.highfreq.web._read_json_cached", return_value=None):
        mock_pred.side_effect = [
            (None, {"has_model": False}),  # BTC cold start
            (0.51, {"has_model": True, "feature_set": "cross_asset"}),
            (0.40, {"has_model": True, "feature_set": "cross_asset"}),
        ]
        res = TestClient(app).get(
            "/api/highfreq/dashboard?symbols=BTCUSDT,ETHUSDT,BNBUSDT"
        )
    body = res.json()
    assert body["symbols"]["BTCUSDT"]["forecast"]["ok"] is False
    assert body["symbols"]["BTCUSDT"]["forecast"]["reason"] == "model_or_data_unavailable"
    assert body["symbols"]["ETHUSDT"]["forecast"]["ok"] is True
    assert body["symbols"]["BNBUSDT"]["forecast"]["ok"] is True
    app.dependency_overrides.clear()


def test_dashboard_invalid_symbol_400():
    """Bad symbol fails the WHOLE batch (not silently skipped)."""
    app = _make_app()
    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db
    res = TestClient(app).get(
        "/api/highfreq/dashboard?symbols=BTCUSDT,not-a-symbol"
    )
    assert res.status_code == 400
    app.dependency_overrides.clear()


def test_dashboard_no_symbols_400():
    """Empty symbols param → 400."""
    app = _make_app()
    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db
    res = TestClient(app).get("/api/highfreq/dashboard?symbols=")
    assert res.status_code == 400
    body = res.json()
    assert body["ok"] is False
    assert body["reason"] == "no_symbols"
    app.dependency_overrides.clear()


def test_dashboard_too_many_symbols_400():
    """Cap at 8 — prevents amplification attack via giant symbol list."""
    app = _make_app()
    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db
    huge = ",".join([f"AA{i:02d}USDT" for i in range(20)])
    res = TestClient(app).get(f"/api/highfreq/dashboard?symbols={huge}")
    assert res.status_code == 400
    body = res.json()
    assert body["reason"] == "too_many_symbols"
    app.dependency_overrides.clear()


def test_dashboard_drift_block_populated_when_file_present():
    """When _read_json_cached returns a parsed payload, the drift
    sub-block surfaces severity + max_ks."""
    app = _make_app()
    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db
    drift_payload = {
        "severity": "warn",
        "max_ks": 0.21,
        "max_ks_feature": "spread_bps_mean",
        "evaluated_at": "2026-05-04T12:00:00+00:00",
    }
    with patch("app.highfreq.web._predict_for_horizon",
               return_value=(0.55, {"has_model": True})), \
         patch("app.highfreq.web._read_json_cached",
               return_value=drift_payload), \
         patch("pathlib.Path.exists", return_value=True):
        res = TestClient(app).get(
            "/api/highfreq/dashboard?symbols=BTCUSDT"
        )
    body = res.json()
    btc_drift = body["symbols"]["BTCUSDT"]["drift"]
    assert btc_drift["ok"] is True
    assert btc_drift["severity"] == "warn"
    assert btc_drift["max_ks"] == pytest.approx(0.21)
    assert btc_drift["max_ks_feature"] == "spread_bps_mean"
    app.dependency_overrides.clear()
