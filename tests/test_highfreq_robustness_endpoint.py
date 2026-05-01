"""Tests for ``GET /api/highfreq/robustness?symbol=...`` (release T.14).

The endpoint serves the JSON written by ``tools.robustness_suite``.
Tests pin:

1. Missing file → 200 + ok=False + actionable hint (NEVER 5xx).
2. Malformed JSON → 200 + ok=False (defensive).
3. Happy path → 200 + ok=True + payload echoed under ``report`` key.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.web import router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client_with_cwd(tmp_path, monkeypatch):
    """Run the endpoint with cwd pointing at a temp dir so we control
    whether ``weights/highfreq/<sym>_1m_robustness.json`` exists."""
    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    app = _make_app()
    return TestClient(app), weights_dir


def test_robustness_missing_file_returns_ok_false_with_hint(client_with_cwd):
    client, _ = client_with_cwd
    res = client.get("/api/highfreq/robustness?symbol=BTCUSDT")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["reason"] == "no_robustness_run_yet"
    # The hint must guide the operator toward the fix command.
    assert "tools.robustness_suite" in body["hint"]
    assert "BTCUSDT" in body["hint"]


def test_robustness_malformed_file_returns_ok_false(client_with_cwd):
    client, weights_dir = client_with_cwd
    (weights_dir / "btcusdt_1m_robustness.json").write_text("{NOT VALID JSON")
    res = client.get("/api/highfreq/robustness?symbol=BTCUSDT")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["reason"] == "malformed"


def test_robustness_happy_path_echoes_full_payload(client_with_cwd):
    client, weights_dir = client_with_cwd
    payload = {
        "symbol": "BTCUSDT",
        "n_predictions": 1500,
        "dir_acc": 0.566,
        "block_bootstrap_ci_low": 0.541,
        "block_bootstrap_ci_high": 0.591,
        "permutation_p_value": 0.0009,
        "permutation_z_score": 4.2,
        "per_day": [
            {"date": "2026-04-28", "n": 380, "hits": 215,
             "dir_acc": 0.566, "ci_low": 0.51, "ci_high": 0.62},
        ],
        "per_hour": [{"hour_utc": h, "n": 0, "hits": 0,
                      "dir_acc": None, "ci_low": None, "ci_high": None}
                     for h in range(24)],
        "per_regime": [
            {"regime": "uptrend", "n": 600, "hits": 350,
             "dir_acc": 0.583, "ci_low": 0.54, "ci_high": 0.62},
        ],
        "per_day_min": 0.566,
        "per_day_max": 0.566,
        "per_day_std": 0.0,
        "per_day_all_above_chance": True,
    }
    (weights_dir / "btcusdt_1m_robustness.json").write_text(json.dumps(payload))
    res = client.get("/api/highfreq/robustness?symbol=BTCUSDT")
    body = res.json()
    assert body["ok"] is True
    assert body["report"]["dir_acc"] == 0.566
    assert body["report"]["permutation_p_value"] == 0.0009
    assert body["report"]["per_day_all_above_chance"] is True
    # The 24-bucket per-hour scaffold is preserved (UI relies on
    # always-24-rows for the heatmap to be gap-free).
    assert len(body["report"]["per_hour"]) == 24


def test_robustness_symbol_case_insensitive(client_with_cwd):
    """Both ``btcusdt`` and ``BTCUSDT`` query params must reach the
    same file. The endpoint upper-cases the symbol but reads the
    lowercase filename — pin the contract."""
    client, weights_dir = client_with_cwd
    (weights_dir / "ethusdt_1m_robustness.json").write_text(
        json.dumps({"symbol": "ETHUSDT", "n_predictions": 1000,
                    "dir_acc": 0.55, "per_day": [], "per_hour": [],
                    "per_regime": []})
    )
    res = client.get("/api/highfreq/robustness?symbol=ethusdt")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["report"]["symbol"] == "ETHUSDT"
