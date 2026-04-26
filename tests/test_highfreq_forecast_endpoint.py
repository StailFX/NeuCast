"""Tests for ``GET /api/highfreq/forecast`` (Phase B scaffold).

The endpoint composes three things:

  1. :class:`LivePredictor` — model + metrics state
  2. :func:`_fetch_recent_seconds` — Postgres read
  3. :func:`build_latest_feature_row` — feature transform

Each contributes a distinct 503 reason; we test them as three separate
branches plus the happy 200 path. We mock both the DB session (via
FastAPI's ``dependency_overrides``) and the predictor so the suite
runs without Postgres or CatBoost.

The mapping ``prob_up → signal`` uses thresholds 0.55 / 0.45 and is
exercised separately to pin the contract that the future paper-trader
will read.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.feature_pipeline import FEATURE_COLUMNS
from app.highfreq.predictor import LivePredictor, PredictorStatus
from app.highfreq.web import (
    _fetch_recent_seconds,
    _get_db,
    _get_forecast_predictor,
    router,
)


# ───────────────────────── fixtures ─────────────────────────


def _make_app() -> FastAPI:
    """Spin up a minimal FastAPI app with just the highfreq router.

    Avoids importing app.main, which has its own DB / config dependencies.
    """
    app = FastAPI()
    app.include_router(router)
    return app


def _stub_status(
    *,
    has_model: bool = True,
    is_calibrated: bool = True,
    dir_acc_ci_low: float | None = 0.55,
    dir_acc_mean: float | None = 0.57,
    dir_acc_p_value: float | None = 0.01,
) -> PredictorStatus:
    return PredictorStatus(
        has_model=has_model,
        model_path="/fake/path/btcusdt_1m.cbm",
        model_age_seconds=120.0 if has_model else None,
        is_calibrated=is_calibrated,
        dir_acc_mean=dir_acc_mean,
        dir_acc_ci_low=dir_acc_ci_low,
        dir_acc_p_value=dir_acc_p_value,
        metrics_age_seconds=120.0 if has_model else None,
        n_features_expected=len(FEATURE_COLUMNS),
    )


def _stub_predictor(
    *,
    status: PredictorStatus | None = None,
    proba_up: float | None = 0.62,
) -> MagicMock:
    """A MagicMock matching the LivePredictor surface we use in the endpoint."""
    p = MagicMock(spec=LivePredictor)
    p.status.return_value = status or _stub_status()
    p.predict.return_value = proba_up
    return p


def _seconds_frame_two_complete_minutes() -> pd.DataFrame:
    """Build a per-second frame with two COMPLETE minutes + a partial
    third minute. ``build_latest_feature_row`` should yield a row from
    minute-2 (the latest fully-closed one)."""
    rng = np.random.default_rng(0)
    rows = []
    t0 = pd.Timestamp("2026-04-25 12:00:00", tz="UTC")
    # Minutes 0 and 1: full 60s each.
    for m in range(2):
        for s in range(60):
            rows.append({
                "ts": t0 + pd.Timedelta(minutes=m, seconds=s),
                "symbol": "BTCUSDT",
                "ofi": float(rng.normal(0, 0.5)),
                "microprice": 77_000.0 + m * 5.0 + rng.normal(0, 1),
                "depth_imb": float(rng.uniform(-0.1, 0.1)),
                "spread_bps": float(rng.uniform(0.5, 1.0)),
                "trade_imb": float(rng.normal(0, 0.001)),
                "n_updates": 10,
            })
    # Partial minute 2: only 15 seconds (the in-flight one).
    for s in range(15):
        rows.append({
            "ts": t0 + pd.Timedelta(minutes=2, seconds=s),
            "symbol": "BTCUSDT",
            "ofi": 0.0,
            "microprice": 77_010.0,
            "depth_imb": 0.0,
            "spread_bps": 0.7,
            "trade_imb": 0.0,
            "n_updates": 10,
        })
    return pd.DataFrame(rows)


# Sentinel — compared by `is`, so DataFrames and None can pass through
# without triggering pandas' "truth value is ambiguous" exception.
_USE_REAL_FETCH = object()


def _override_deps(
    app: FastAPI,
    *,
    predictor: Any,
    fetch_returns: Any = _USE_REAL_FETCH,
):
    """Wire DI overrides for both the DB session and the predictor.

    ``fetch_returns`` semantics:
      * DataFrame    → mock _fetch_recent_seconds to return it
      * None         → mock fetch to return None (DB unreachable)
      * _USE_REAL_FETCH → leave _fetch_recent_seconds as real function
                          (caller must override _get_db)
    """
    app.dependency_overrides[_get_forecast_predictor] = lambda: predictor

    fake_db = MagicMock()
    app.dependency_overrides[_get_db] = lambda: fake_db

    if fetch_returns is _USE_REAL_FETCH:
        return None

    # Monkey-patch the module-level helper so the endpoint sees our
    # canned DataFrame regardless of what fake_db looks like.
    import app.highfreq.web as web_mod
    original_fetch = web_mod._fetch_recent_seconds
    web_mod._fetch_recent_seconds = lambda db, sym, lookback_seconds=180: fetch_returns
    # Stash the original so the test can restore it (pytest fixture
    # would be cleaner but this keeps tests inline-readable).
    return original_fetch


# ───────────────────────── 503 paths ─────────────────────────


def test_forecast_503_when_no_model_yet():
    app = _make_app()
    predictor = _stub_predictor(status=_stub_status(has_model=False, is_calibrated=False))
    original_fetch = _override_deps(app, predictor=predictor)  # use real fetch — won't be called

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 503
        body = r.json()
        assert body["ok"] is False
        assert body["reason"] == "no_model_yet"
        assert body["symbol"] == "BTCUSDT"
        # Status block must be embedded so the UI can show "model age" etc.
        assert body["model"]["has_model"] is False
        assert body["model"]["n_features_expected"] == len(FEATURE_COLUMNS)
        # _fetch_recent_seconds must NOT be called when there's no model
        # (saves a DB round-trip on every call during the 2-week ramp-up).
        predictor.predict.assert_not_called()
    finally:
        if original_fetch is not None:
            import app.highfreq.web as web_mod
            web_mod._fetch_recent_seconds = original_fetch


def test_forecast_503_when_db_unavailable():
    """Model loaded, but DB query failed (None from _fetch_recent_seconds)."""
    app = _make_app()
    predictor = _stub_predictor()
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=None)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 503
        body = r.json()
        assert body["reason"] == "database_unavailable"
        assert body["model"]["has_model"] is True  # model is fine, DB is down
        predictor.predict.assert_not_called()
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


def test_forecast_503_when_not_enough_recent_data():
    """DB returned an empty frame → no complete-minute bar to score."""
    app = _make_app()
    predictor = _stub_predictor()
    empty = pd.DataFrame(columns=[
        "ts", "symbol", "ofi", "microprice", "depth_imb",
        "spread_bps", "trade_imb", "n_updates",
    ])
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=empty)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 503
        body = r.json()
        assert body["reason"] == "not_enough_recent_data"
        assert body["rows_seen"] == 0
        predictor.predict.assert_not_called()
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


def test_forecast_503_when_only_partial_current_minute():
    """30 seconds of data in the current in-flight minute = no complete bar."""
    app = _make_app()
    predictor = _stub_predictor()

    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2026-04-25 12:00:00", tz="UTC")
    rows = [{
        "ts": t0 + pd.Timedelta(seconds=s),
        "symbol": "BTCUSDT",
        "ofi": float(rng.normal(0, 0.5)),
        "microprice": 77_000.0,
        "depth_imb": 0.0,
        "spread_bps": 0.7,
        "trade_imb": 0.0,
        "n_updates": 10,
    } for s in range(30)]
    only_partial = pd.DataFrame(rows)

    original_fetch = _override_deps(
        app, predictor=predictor, fetch_returns=only_partial,
    )
    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 503
        body = r.json()
        assert body["reason"] == "not_enough_recent_data"
        assert body["rows_seen"] == 30
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


def test_forecast_503_when_predictor_returns_none(monkeypatch):
    """Defensive branch: status said has_model=True but predict() returned None.
    Treated as transient — 503 with reason model_unavailable so the
    caller retries instead of caching a bad response."""
    app = _make_app()
    predictor = _stub_predictor(proba_up=None)  # quirky predictor
    df = _seconds_frame_two_complete_minutes()
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=df)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 503
        body = r.json()
        assert body["reason"] == "model_unavailable"
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


# ───────────────────────── 200 happy-path ─────────────────────────


def test_forecast_200_with_complete_data_and_model():
    app = _make_app()
    predictor = _stub_predictor(proba_up=0.62)
    df = _seconds_frame_two_complete_minutes()
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=df)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["symbol"] == "BTCUSDT"
        assert body["horizon_minutes"] == 1
        assert body["prob_up"] == pytest.approx(0.62)
        # 0.62 ≥ 0.55 → "up"
        assert body["signal"] == "up"
        assert body["calibrated"] is True
        assert body["model"]["has_model"] is True
        assert "ts" in body
        # predict was called exactly once with a feature row.
        assert predictor.predict.call_count == 1
        # The argument must be a Series whose index is FEATURE_COLUMNS
        # (canary for any future regression that breaks the contract).
        called_with = predictor.predict.call_args[0][0]
        assert isinstance(called_with, pd.Series)
        assert list(called_with.index) == FEATURE_COLUMNS
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


@pytest.mark.parametrize("proba,expected_signal", [
    (0.95, "up"),
    (0.55, "up"),       # exactly at threshold → up (predicate ≥)
    (0.54, "neutral"),  # just below
    (0.50, "neutral"),
    (0.46, "neutral"),  # just above lower
    (0.45, "down"),     # exactly at lower threshold → down (predicate ≤)
    (0.05, "down"),
])
def test_forecast_signal_thresholds(proba, expected_signal):
    app = _make_app()
    predictor = _stub_predictor(proba_up=proba)
    df = _seconds_frame_two_complete_minutes()
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=df)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 200
        assert r.json()["signal"] == expected_signal
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


def test_forecast_200_uppercases_symbol():
    app = _make_app()
    predictor = _stub_predictor()
    df = _seconds_frame_two_complete_minutes()
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=df)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast?symbol=btcusdt")
        assert r.status_code == 200
        assert r.json()["symbol"] == "BTCUSDT"
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


def test_forecast_200_calibrated_false_when_low_skill():
    """A model exists but bootstrap CI is below threshold — UI must
    surface this so users don't trade on noise."""
    app = _make_app()
    predictor = _stub_predictor(
        status=_stub_status(is_calibrated=False, dir_acc_ci_low=0.49),
    )
    df = _seconds_frame_two_complete_minutes()
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=df)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        body = r.json()
        assert r.status_code == 200
        assert body["calibrated"] is False
        assert body["model"]["dir_acc_ci_low"] == pytest.approx(0.49)
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


def test_forecast_response_is_json_safe():
    """RFC-7159: no NaN / Inf must escape into the response body. The
    endpoint wraps the success path in _scrub() — verify by injecting
    a NaN into the predictor's status block."""
    app = _make_app()
    nan_status = _stub_status(dir_acc_mean=float("nan"))
    predictor = _stub_predictor(status=nan_status)
    df = _seconds_frame_two_complete_minutes()
    original_fetch = _override_deps(app, predictor=predictor, fetch_returns=df)

    try:
        client = TestClient(app)
        r = client.get("/api/highfreq/forecast")
        assert r.status_code == 200
        body = r.json()
        # _scrub neutralises NaN to None.
        assert body["model"]["dir_acc_mean"] is None
    finally:
        import app.highfreq.web as web_mod
        web_mod._fetch_recent_seconds = original_fetch


# ───────────────────────── _fetch_recent_seconds direct ─────────────────────────


def test_fetch_recent_seconds_returns_dataframe_on_success():
    """Direct unit test of the helper — without going through the endpoint."""
    fake_db = MagicMock()
    rows = [{
        "ts": pd.Timestamp("2026-04-25 12:00:00", tz="UTC"),
        "symbol": "BTCUSDT",
        "ofi": 0.5, "microprice": 77000.0, "depth_imb": 0.1,
        "spread_bps": 0.6, "trade_imb": 0.0, "n_updates": 10,
    }]
    # SQLAlchemy chain: db.execute(text, params).mappings().all() → rows
    fake_db.execute.return_value.mappings.return_value.all.return_value = rows

    df = _fetch_recent_seconds(fake_db, "BTCUSDT", lookback_seconds=60)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "BTCUSDT"


def test_fetch_recent_seconds_returns_none_on_db_error():
    from sqlalchemy.exc import OperationalError

    fake_db = MagicMock()
    fake_db.execute.side_effect = OperationalError("stmt", {}, Exception("nope"))

    df = _fetch_recent_seconds(fake_db, "BTCUSDT")
    assert df is None
