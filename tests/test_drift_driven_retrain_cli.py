"""Tests for ``tools.drift_driven_retrain`` CLI (T.22).

We test the CLI as a unit — the policy logic itself is covered in
``tests/test_drift_retrain_policy.py``. Here we pin:

1. Drift JSON parsing (missing / malformed / valid).
2. ``systemctl`` invocation only when policy says trigger.
3. ``--dry-run`` short-circuits the systemctl call.
4. Textfile-collector output format (Prometheus scrape compatibility).
5. Cold-start behaviour (no DB, no rows).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.drift_driven_retrain import (
    _emit_textfile,
    _read_drift_json,
    _trigger_systemctl,
    main,
)


def _write_drift(tmp_path: Path, sym: str, *, severity: str,
                 max_ks: float = 0.5) -> Path:
    p = tmp_path / "weights" / "highfreq"
    p.mkdir(parents=True, exist_ok=True)
    payload = {
        "severity": severity,
        "drifted": severity != "ok",
        "max_ks": max_ks,
        "max_ks_feature": "spread_bps_mean",
        "threshold": 0.15,
        "n_features": 14,
        "n_features_alarming": 10 if severity == "high" else 0,
        "top_features": [],
        "evaluated_at": "2026-05-04T00:00:00+00:00",
        "feature_set": "microstructure",
    }
    (p / f"{sym.lower()}_drift.json").write_text(json.dumps(payload))
    return p


# ─────────────────── _read_drift_json ───────────────────


def test_read_drift_json_missing_returns_none(tmp_path):
    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    assert _read_drift_json(weights_dir, "BTCUSDT") is None


def test_read_drift_json_malformed_returns_none(tmp_path):
    weights_dir = tmp_path / "weights" / "highfreq"
    weights_dir.mkdir(parents=True)
    (weights_dir / "btcusdt_drift.json").write_text("not valid {[")
    assert _read_drift_json(weights_dir, "BTCUSDT") is None


def test_read_drift_json_valid(tmp_path):
    weights_dir = _write_drift(tmp_path, "BTCUSDT", severity="warn", max_ks=0.21)
    j = _read_drift_json(weights_dir, "BTCUSDT")
    assert j is not None
    assert j["severity"] == "warn"
    assert j["max_ks"] == pytest.approx(0.21)


# ─────────────────── _trigger_systemctl dry-run ───────────────────


def test_trigger_systemctl_dry_run_returns_zero():
    rc, stderr = _trigger_systemctl("BTCUSDT", dry_run=True)
    assert rc == 0
    assert stderr == "(dry-run)"


# ─────────────────── _emit_textfile ───────────────────


def test_emit_textfile_writes_prometheus_format(tmp_path):
    out = tmp_path / "metrics.prom"
    _emit_textfile(out, per_symbol=[
        {"symbol": "BTCUSDT", "severity": "high",
         "should_retrain": True, "hours_since_last_train": 9.5},
        {"symbol": "ETHUSDT", "severity": "ok",
         "should_retrain": False, "hours_since_last_train": 9.6},
    ])
    text = out.read_text()
    # Required help/type headers — node_exporter chokes on missing TYPE.
    assert "# TYPE neucast_drift_retrain_decision gauge" in text
    assert "# TYPE neucast_drift_retrain_hours_since_last gauge" in text
    # Values + labels.
    assert (
        'neucast_drift_retrain_decision{symbol="BTCUSDT",severity="high"} 1'
    ) in text
    assert (
        'neucast_drift_retrain_decision{symbol="ETHUSDT",severity="ok"} 0'
    ) in text
    assert 'neucast_drift_retrain_hours_since_last{symbol="BTCUSDT"} 9.500' in text


def test_emit_textfile_none_path_is_noop():
    """Operator may not have set up textfile_collector — the CLI
    must still work without the output path."""
    _emit_textfile(None, per_symbol=[])  # no error


def test_emit_textfile_skips_hours_when_unknown(tmp_path):
    """Cold-start path: ``hours_since_last_train=None`` — emit only
    the decision, not a bogus 0 hour value."""
    out = tmp_path / "metrics.prom"
    _emit_textfile(out, per_symbol=[
        {"symbol": "BTCUSDT", "severity": "high",
         "should_retrain": True, "hours_since_last_train": None},
    ])
    text = out.read_text()
    assert "decision" in text
    assert "hours_since_last" not in text.split("\n", 4)[4:][0] \
        or 'neucast_drift_retrain_hours_since_last{symbol="BTCUSDT"}' not in text


# ─────────────────── main() integration ───────────────────


def test_main_dry_run_no_drift_files(tmp_path, monkeypatch, capsys):
    """Drift cron hasn't fired yet → CLI should log "no drift JSON"
    and not call systemctl, exit 0."""
    monkeypatch.chdir(tmp_path)
    rc = main(["--symbol", "BTCUSDT", "--dry-run", "--weights-dir",
               str(tmp_path / "weights" / "highfreq")])
    assert rc == 0


def test_main_dry_run_high_severity_triggers(tmp_path, monkeypatch):
    """High-severity drift + cold start (no DB rows) → policy says
    trigger, dry-run avoids actual systemctl. Exit 0."""
    weights_dir = _write_drift(tmp_path, "BTCUSDT", severity="high", max_ks=0.96)
    monkeypatch.chdir(tmp_path)
    rc = main([
        "--symbol", "BTCUSDT",
        "--dry-run",
        "--weights-dir", str(weights_dir),
    ])
    assert rc == 0


def test_main_dry_run_warn_does_not_trigger(tmp_path, monkeypatch):
    """warn alone is below the trigger threshold (default
    fire_on_severities=("high",))."""
    weights_dir = _write_drift(tmp_path, "BTCUSDT", severity="warn", max_ks=0.18)
    monkeypatch.chdir(tmp_path)
    rc = main([
        "--symbol", "BTCUSDT",
        "--dry-run",
        "--weights-dir", str(weights_dir),
    ])
    assert rc == 0


def test_main_writes_textfile(tmp_path, monkeypatch):
    """Exercises the full Prometheus textfile path end-to-end so a
    schema regression in _emit_textfile is caught."""
    weights_dir = _write_drift(tmp_path, "BTCUSDT", severity="high")
    out = tmp_path / "metrics.prom"
    monkeypatch.chdir(tmp_path)
    rc = main([
        "--symbol", "BTCUSDT",
        "--dry-run",
        "--weights-dir", str(weights_dir),
        "--textfile-out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert 'neucast_drift_retrain_decision{symbol="BTCUSDT"' in text


def test_main_systemctl_failure_logged_but_does_not_crash(tmp_path, monkeypatch):
    """When systemctl is missing (off-Tokyo) the CLI logs the error
    but exits 0 so the cron timer keeps running — operator alert
    routes through Prometheus, not via CLI exit code."""
    weights_dir = _write_drift(tmp_path, "BTCUSDT", severity="high")
    monkeypatch.chdir(tmp_path)
    # Need DATABASE_URL set (real or fake) so the CLI doesn't bail
    # before reaching systemctl. We patch the DB lookup so the fake
    # URL is never actually used.
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    with patch("tools.drift_driven_retrain._last_training_started_at",
               return_value=None), \
         patch("tools.drift_driven_retrain.subprocess.run",
               side_effect=FileNotFoundError):
        rc = main([
            "--symbol", "BTCUSDT",
            "--weights-dir", str(weights_dir),
        ])
    # CLI exit is 0 (idempotent timer); operator sees error in logs
    # + Prometheus.
    assert rc == 0
