"""Tests for ``GET /api/highfreq/drift_status`` (T.21).

Reads the per-symbol drift JSON written by the hourly cron and
surfaces severity + max-KS feature for the UI badge.

Tests pin:
1. File present + valid → ok=True with severity / max_ks fields.
2. File missing → ok=False with reason=no_check_yet (NOT 5xx —
   drift cron is best-effort).
3. Malformed file → ok=False with reason=malformed.
4. Top-features list passthrough preserved (UI shows top-3 in
   tooltip).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.web import router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_drift_status_happy_path(tmp_path, monkeypatch):
    """Drift JSON present → endpoint returns severity + max_ks +
    top_features."""
    payload = {
        "severity": "warn",
        "drifted": True,
        "max_ks": 0.42,
        "max_ks_feature": "spread_bps_mean",
        "threshold": 0.15,
        "n_features": 14,
        "n_features_alarming": 7,
        "top_features": [
            {"feature": "spread_bps_mean", "ks_stat": 0.42},
            {"feature": "depth_imb_std", "ks_stat": 0.38},
        ],
        "evaluated_at": "2026-05-03T16:30:00+00:00",
        "feature_set": "microstructure",
    }
    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    (weights_dir / "btcusdt_drift.json").write_text(json.dumps(payload))

    monkeypatch.chdir(tmp_path)  # endpoint reads relative path
    client = TestClient(_make_app())
    res = client.get("/api/highfreq/drift_status?symbol=BTCUSDT")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["severity"] == "warn"
    assert body["max_ks"] == 0.42
    assert body["max_ks_feature"] == "spread_bps_mean"
    assert len(body["top_features"]) == 2
    assert body["top_features"][0]["feature"] == "spread_bps_mean"


def test_drift_status_no_file_returns_no_check_yet(tmp_path, monkeypatch):
    """Drift cron hasn't run yet → ok=False, reason=no_check_yet.
    UI renders this as muted "—", not alarming red."""
    monkeypatch.chdir(tmp_path)
    client = TestClient(_make_app())
    res = client.get("/api/highfreq/drift_status?symbol=BTCUSDT")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["reason"] == "no_check_yet"


def test_drift_status_malformed_file_returns_malformed(tmp_path, monkeypatch):
    """Corrupt JSON file → ok=False with reason. Doesn't 5xx so the
    UI can show "—" without breaking page load."""
    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    (weights_dir / "btcusdt_drift.json").write_text("not valid json {[")

    monkeypatch.chdir(tmp_path)
    client = TestClient(_make_app())
    res = client.get("/api/highfreq/drift_status?symbol=BTCUSDT")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["reason"] == "malformed"


def test_drift_status_severity_ok_when_no_drift(tmp_path, monkeypatch):
    """Common path: hourly cron ran, no drift detected. UI renders
    a green dot."""
    payload = {
        "severity": "ok",
        "drifted": False,
        "max_ks": 0.05,
        "max_ks_feature": "ofi_mean",
        "threshold": 0.15,
        "n_features": 14,
        "n_features_alarming": 0,
        "top_features": [],
        "evaluated_at": "2026-05-03T16:30:00+00:00",
        "feature_set": "microstructure",
    }
    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    (weights_dir / "btcusdt_drift.json").write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    body = TestClient(_make_app()).get(
        "/api/highfreq/drift_status?symbol=BTCUSDT"
    ).json()
    assert body["ok"] is True
    assert body["severity"] == "ok"
    assert body["drifted"] is False


def test_drift_status_uppercases_symbol(tmp_path, monkeypatch):
    """Symbol must be normalised to uppercase to match DB convention,
    but the file is read as lowercase. Ensures both sides agree."""
    payload = {"severity": "ok", "max_ks": 0.05, "drifted": False}
    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    (weights_dir / "ethusdt_drift.json").write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    body = TestClient(_make_app()).get(
        "/api/highfreq/drift_status?symbol=ethusdt"
    ).json()
    assert body["ok"] is True
    assert body["symbol"] == "ETHUSDT"
