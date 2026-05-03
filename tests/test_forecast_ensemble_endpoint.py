"""Tests for ``GET /api/highfreq/forecast_ensemble`` (T.19).

The endpoint runs inference on both 1m and 15m predictors, blends
their calibrated probabilities via weighted average, and returns
the result with full per-model breakdown.

Tests pin:
1. Both predictors available + both predict → blended response.
2. One predictor unavailable → fallback to the available one,
   weights re-normalised.
3. Both unavailable → 503 with structured reason.
4. Invalid weights → 400.
5. Signal classification (up / down / neutral) matches threshold.
6. Components breakdown includes both horizons.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.web import _get_db, router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _override_predict_for_horizon(app: FastAPI, results_by_horizon: dict):
    """Patch ``_predict_for_horizon`` to return canned values per
    horizon. ``results_by_horizon[h]`` is ``(prob, status_dict)``."""
    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db


def test_both_models_available_returns_blended_prob():
    """1m=0.62, 15m=0.50, weights 0.7/0.3 → blend = 0.7×0.62 + 0.3×0.50 = 0.584."""
    app = _make_app()
    _override_predict_for_horizon(app, {})
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred:
        mock_pred.side_effect = [
            (0.62, {"has_model": True, "model_age_seconds": 100}),
            (0.50, {"has_model": True, "model_age_seconds": 200}),
        ]
        client = TestClient(app)
        res = client.get("/api/highfreq/forecast_ensemble?symbol=BTCUSDT")
        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert abs(body["prob_up"] - 0.584) < 1e-6
        assert body["signal"] == "up"
        assert body["agreement"] is True  # both ≥ 0.5 (15m is exactly 0.5 = neutral, agrees with up)
        assert body["n_components_used"] == 2
        assert len(body["components"]) == 2
    app.dependency_overrides.clear()


def test_15m_unavailable_falls_back_to_1m_alone():
    """When 15m has no model yet (cold start), the ensemble should
    return the 1m prediction with the 15m component marked
    unavailable."""
    app = _make_app()
    _override_predict_for_horizon(app, {})
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred:
        mock_pred.side_effect = [
            (0.62, {"has_model": True}),
            (None, {"has_model": False}),
        ]
        client = TestClient(app)
        body = client.get("/api/highfreq/forecast_ensemble?symbol=BTCUSDT").json()
        assert body["ok"] is True
        assert abs(body["prob_up"] - 0.62) < 1e-9
        assert body["n_components_used"] == 1
        # 15m row in components must exist with is_available=false.
        by_label = {c["horizon_label"]: c for c in body["components"]}
        assert by_label["15m"]["is_available"] is False
        assert by_label["15m"]["prob_up"] is None
    app.dependency_overrides.clear()


def test_both_unavailable_returns_503():
    """No usable signal at all → 503 with structured reason. The
    operator can branch on this in their UI without parsing 500s."""
    app = _make_app()
    _override_predict_for_horizon(app, {})
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred:
        mock_pred.side_effect = [
            (None, {"has_model": False}),
            (None, {"has_model": False}),
        ]
        client = TestClient(app)
        res = client.get("/api/highfreq/forecast_ensemble?symbol=BTCUSDT")
        assert res.status_code == 503
        body = res.json()
        assert body["ok"] is False
        assert body["reason"] == "no_available_components"
    app.dependency_overrides.clear()


def test_invalid_weights_returns_400():
    """Negative or zero weight is a configuration error — refuse
    explicitly rather than divide-by-zero somewhere downstream."""
    app = _make_app()
    _override_predict_for_horizon(app, {})
    client = TestClient(app)
    res = client.get(
        "/api/highfreq/forecast_ensemble?symbol=BTCUSDT&weight_1m=0&weight_15m=1"
    )
    assert res.status_code == 400
    app.dependency_overrides.clear()


def test_signal_classification_threshold():
    """signal='up' iff prob >= 0.55, 'down' iff prob <= 0.45,
    else 'neutral'. Same convention as /forecast."""
    app = _make_app()
    _override_predict_for_horizon(app, {})
    # Build a blend of 0.50 (1m=0.50, 15m=0.50, any weights).
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred:
        mock_pred.side_effect = [(0.50, {}), (0.50, {})]
        body = TestClient(app).get(
            "/api/highfreq/forecast_ensemble?symbol=BTCUSDT"
        ).json()
        assert body["signal"] == "neutral"
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred:
        mock_pred.side_effect = [(0.40, {}), (0.40, {})]
        body = TestClient(app).get(
            "/api/highfreq/forecast_ensemble?symbol=BTCUSDT"
        ).json()
        assert body["signal"] == "down"
    app.dependency_overrides.clear()


def test_disagreement_reflected_in_response():
    """1m says up (0.62), 15m says down (0.40). Blend = 0.7×0.62 + 0.3×0.40 = 0.554.
    agreement should be False (sides differ)."""
    app = _make_app()
    _override_predict_for_horizon(app, {})
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred:
        mock_pred.side_effect = [(0.62, {}), (0.40, {})]
        body = TestClient(app).get(
            "/api/highfreq/forecast_ensemble?symbol=BTCUSDT"
        ).json()
        assert body["agreement"] is False
    app.dependency_overrides.clear()


def test_custom_weights_used_in_blend():
    """A 50/50 ensemble produces a different blend than 70/30."""
    app = _make_app()
    _override_predict_for_horizon(app, {})
    with patch("app.highfreq.web._predict_for_horizon") as mock_pred:
        mock_pred.side_effect = [(0.60, {}), (0.40, {})]
        body = TestClient(app).get(
            "/api/highfreq/forecast_ensemble"
            "?symbol=BTCUSDT&weight_1m=0.5&weight_15m=0.5"
        ).json()
        # 50/50 of (0.60, 0.40) = 0.50.
        assert abs(body["prob_up"] - 0.50) < 1e-9
    app.dependency_overrides.clear()
