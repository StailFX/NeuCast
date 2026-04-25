"""Tests for ``app.highfreq.predictor`` — Phase B live-inference layer.

Coverage matrix:

* **No-model path.** Until the trainer ships a ``.cbm``, the predictor
  must return ``None`` from :meth:`predict` and ``has_model=False``
  from :meth:`status`. This is the contract Phase A operates under for
  ~2 weeks while data accumulates; if it ever returned garbage instead,
  the forecast endpoint would 200 with noise and we'd discover it on
  the UI.

* **Hot-reload.** When the trainer atomically writes a new ``.cbm``,
  the next ``predict()`` call must pick it up via mtime-bump. Tested
  with a stub model (no real CatBoost) by monkeypatching
  ``LivePredictor._load_model``.

* **Calibration gate.** The forecast UI shows "calibrated?" badge based
  on metrics-JSON ``dir_acc_ci_low > 0.5``. Bracket cases tested.

* **Feature reordering.** A pd.Series with shuffled index must be
  reordered to ``FEATURE_COLUMNS`` order before being fed to CatBoost.
  This is the silent-killer test — if the trainer fits column order
  ``[a, b, c]`` and the predictor passes ``[b, a, c]``, every forecast
  is wrong but predict_proba returns valid floats.

CatBoost is NOT imported here — we stub the model object. This keeps
the test suite runnable without the catboost wheel and matches the
predictor's lazy-import design.

Style: synchronous tests; ``tmp_path`` fixture for filesystem isolation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.highfreq.feature_pipeline import FEATURE_COLUMNS
from app.highfreq.predictor import (
    CALIBRATED_DIR_ACC_THRESHOLD,
    LivePredictor,
    PredictorStatus,
    _maybe_float,
    get_predictor,
    reset_predictor,
)


# ───────────────────────── stub model ─────────────────────────


class _StubModel:
    """Minimal stand-in for CatBoostClassifier.

    Records the last input vector for assertions, returns a controllable
    ``proba_up`` value so tests can verify the model output flows
    end-to-end through ``LivePredictor.predict``.
    """

    def __init__(self, proba_up: float = 0.6) -> None:
        self.proba_up = proba_up
        self.last_x: np.ndarray | None = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.last_x = np.asarray(x).copy()
        # Shape (n_rows, 2): [P(down), P(up)].
        n = x.shape[0] if x.ndim == 2 else 1
        return np.tile([1.0 - self.proba_up, self.proba_up], (n, 1))


def _write_dummy_weights(path: Path) -> None:
    """Touch the file so :meth:`_maybe_reload_model` sees a valid mtime.
    Content doesn't matter — we monkeypatch ``_load_model`` to bypass
    actual CatBoost deserialization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"DUMMY_CATBOOST_BLOB")


def _write_metrics(
    path: Path, *, dir_acc_mean: float = 0.55, dir_acc_ci_low: float = 0.51,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dir_acc_mean": dir_acc_mean,
        "dir_acc_ci_low": dir_acc_ci_low,
        "dir_acc_ci_high": 0.59,
        "n_samples": 1234,
    }))


def _patch_loader(monkeypatch, model: _StubModel) -> None:
    """Replace ``LivePredictor._load_model`` with a function returning ``model``."""
    monkeypatch.setattr(
        LivePredictor, "_load_model", staticmethod(lambda path: model)
    )


# ───────────────────────── no-model path ─────────────────────────


def test_predict_returns_none_when_no_weights_file(tmp_path):
    """Phase A.* contract: predictor degrades gracefully without weights."""
    p = LivePredictor(
        weights_path=tmp_path / "missing.cbm",
        metrics_path=tmp_path / "missing.json",
    )
    fake_features = pd.Series([0.0] * len(FEATURE_COLUMNS), index=FEATURE_COLUMNS)
    assert p.predict(fake_features) is None


def test_status_when_no_model(tmp_path):
    p = LivePredictor(
        weights_path=tmp_path / "absent.cbm",
        metrics_path=tmp_path / "absent.json",
    )
    s = p.status()
    assert isinstance(s, PredictorStatus)
    assert s.has_model is False
    assert s.is_calibrated is False
    assert s.model_age_seconds is None
    assert s.metrics_age_seconds is None
    assert s.dir_acc_mean is None
    assert s.dir_acc_ci_low is None
    assert s.n_features_expected == len(FEATURE_COLUMNS)
    # to_dict must be JSON-friendly (Path → str).
    d = s.to_dict()
    assert isinstance(d["model_path"], str)
    assert d["has_model"] is False


def test_is_calibrated_false_with_no_metrics(tmp_path):
    p = LivePredictor(
        weights_path=tmp_path / "x.cbm",
        metrics_path=tmp_path / "x_metrics.json",
    )
    assert p.is_calibrated() is False


def test_model_age_seconds_none_when_no_model(tmp_path):
    p = LivePredictor(
        weights_path=tmp_path / "x.cbm",
        metrics_path=tmp_path / "x_metrics.json",
    )
    assert p.model_age_seconds() is None


# ───────────────────────── happy-path ─────────────────────────


def test_predict_returns_proba_when_model_present(tmp_path, monkeypatch):
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    stub = _StubModel(proba_up=0.73)
    _patch_loader(monkeypatch, stub)

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "btc_metrics.json")

    feats = pd.Series([1.0] * len(FEATURE_COLUMNS), index=FEATURE_COLUMNS)
    out = p.predict(feats)
    assert out == pytest.approx(0.73)
    # Stub recorded a (1, N_FEATURES) input.
    assert stub.last_x is not None
    assert stub.last_x.shape == (1, len(FEATURE_COLUMNS))


def test_predict_reorders_shuffled_series(tmp_path, monkeypatch):
    """Silent-killer guard: a Series with shuffled index must be reordered."""
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    stub = _StubModel(proba_up=0.5)
    _patch_loader(monkeypatch, stub)

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")

    # Construct a Series where each feature carries its column-ordinal
    # value (a ↔ 0, b ↔ 1, ...). After reordering by FEATURE_COLUMNS the
    # vector must be [0, 1, 2, ...] in canonical order.
    canonical = {c: float(i) for i, c in enumerate(FEATURE_COLUMNS)}
    shuffled = list(canonical.items())
    # Reverse → guaranteed not in FEATURE_COLUMNS order.
    shuffled.reverse()
    series_shuffled = pd.Series(dict(shuffled))

    p.predict(series_shuffled)

    expected = np.arange(len(FEATURE_COLUMNS), dtype=float)
    np.testing.assert_array_equal(stub.last_x.ravel(), expected)


def test_predict_accepts_dict_input(tmp_path, monkeypatch):
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    stub = _StubModel(proba_up=0.4)
    _patch_loader(monkeypatch, stub)

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")

    feats_dict = {c: float(i) for i, c in enumerate(FEATURE_COLUMNS)}
    out = p.predict(feats_dict)

    assert out == pytest.approx(0.4)
    np.testing.assert_array_equal(
        stub.last_x.ravel(), np.arange(len(FEATURE_COLUMNS), dtype=float),
    )


def test_predict_accepts_ndarray_input(tmp_path, monkeypatch):
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    stub = _StubModel(proba_up=0.55)
    _patch_loader(monkeypatch, stub)

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")

    arr = np.arange(len(FEATURE_COLUMNS), dtype=float)
    out = p.predict(arr)
    assert out == pytest.approx(0.55)


def test_predict_rejects_wrong_size_ndarray(tmp_path, monkeypatch):
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    _patch_loader(monkeypatch, _StubModel())

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")

    with pytest.raises(ValueError, match="expected"):
        p.predict(np.zeros(len(FEATURE_COLUMNS) - 1))


def test_predict_scrubs_nan_and_inf_in_features(tmp_path, monkeypatch):
    """The aggregator scrubs NaN at write time, but defensive predictor
    code matters because feature_pipeline.build_features could in theory
    leak inf via div-by-zero (e.g. spread_now/spread_mean with mean=0)."""
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    stub = _StubModel(proba_up=0.5)
    _patch_loader(monkeypatch, stub)

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")

    feats = pd.Series([float("nan")] * len(FEATURE_COLUMNS), index=FEATURE_COLUMNS)
    feats.iloc[0] = float("inf")
    feats.iloc[1] = float("-inf")
    p.predict(feats)

    assert np.all(np.isfinite(stub.last_x)), "NaN/inf must be scrubbed before model"


# ───────────────────────── calibration ─────────────────────────


def test_is_calibrated_true_above_threshold(tmp_path):
    weights = tmp_path / "x.cbm"  # not loaded — irrelevant for is_calibrated
    metrics = tmp_path / "x_metrics.json"
    _write_metrics(metrics, dir_acc_ci_low=CALIBRATED_DIR_ACC_THRESHOLD + 0.02)

    p = LivePredictor(weights_path=weights, metrics_path=metrics)
    assert p.is_calibrated() is True


def test_is_calibrated_false_at_or_below_threshold(tmp_path):
    weights = tmp_path / "x.cbm"
    metrics = tmp_path / "x_metrics.json"
    # Exactly 0.5 must NOT clear the gate (predicate is `> threshold`).
    _write_metrics(metrics, dir_acc_ci_low=CALIBRATED_DIR_ACC_THRESHOLD)

    p = LivePredictor(weights_path=weights, metrics_path=metrics)
    assert p.is_calibrated() is False


def test_is_calibrated_handles_missing_field(tmp_path):
    weights = tmp_path / "x.cbm"
    metrics = tmp_path / "x_metrics.json"
    metrics.write_text(json.dumps({"some_other_field": 1.0}))
    p = LivePredictor(weights_path=weights, metrics_path=metrics)
    assert p.is_calibrated() is False


def test_is_calibrated_handles_corrupt_json(tmp_path):
    weights = tmp_path / "x.cbm"
    metrics = tmp_path / "x_metrics.json"
    metrics.write_text("{this is not json")
    p = LivePredictor(weights_path=weights, metrics_path=metrics)
    # Must not crash; calibration default-to-False is the safer UX.
    assert p.is_calibrated() is False


# ───────────────────────── hot-reload ─────────────────────────


def test_hot_reload_picks_up_new_weights(tmp_path, monkeypatch):
    """Trainer wrote a new model overnight → next predict() reloads."""
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)

    stub_v1 = _StubModel(proba_up=0.30)
    stub_v2 = _StubModel(proba_up=0.80)

    # Track invocations so we can flip the loader mid-test.
    state = {"call": 0}

    def loader(path):
        state["call"] += 1
        return stub_v1 if state["call"] == 1 else stub_v2

    monkeypatch.setattr(LivePredictor, "_load_model", staticmethod(loader))

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")
    feats = pd.Series([0.0] * len(FEATURE_COLUMNS), index=FEATURE_COLUMNS)
    assert p.predict(feats) == pytest.approx(0.30)

    # Trainer "overwrites" the file with a newer mtime. We bump it
    # explicitly because writing the same content within ~1ms can leave
    # mtime unchanged on coarse-resolution filesystems.
    _write_dummy_weights(weights)
    new_mtime = time.time() + 5
    import os as _os
    _os.utime(weights, (new_mtime, new_mtime))

    # Next predict() must hit the loader again and serve from stub_v2.
    assert p.predict(feats) == pytest.approx(0.80)
    assert state["call"] == 2


def test_hot_reload_keeps_old_model_on_load_failure(tmp_path, monkeypatch):
    """A corrupted .cbm mid-deploy must NOT degrade the live serve to None.
    The predictor logs and keeps the previously-good model in memory."""
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)

    stub_v1 = _StubModel(proba_up=0.30)
    state = {"call": 0}

    def loader(path):
        state["call"] += 1
        if state["call"] == 1:
            return stub_v1
        raise RuntimeError("corrupted CatBoost blob")

    monkeypatch.setattr(LivePredictor, "_load_model", staticmethod(loader))

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")
    feats = pd.Series([0.0] * len(FEATURE_COLUMNS), index=FEATURE_COLUMNS)
    assert p.predict(feats) == pytest.approx(0.30)

    # Bump mtime → trigger reload that will raise.
    _write_dummy_weights(weights)
    import os as _os
    _os.utime(weights, (time.time() + 5, time.time() + 5))

    # predict() must still serve via stub_v1, not return None.
    assert p.predict(feats) == pytest.approx(0.30)
    assert state["call"] == 2


def test_hot_reload_drops_model_when_file_disappears(tmp_path, monkeypatch):
    """If the weights file is deleted (operator error / disk issue), the
    predictor must report has_model=False on next status() — not keep
    serving stale predictions indefinitely."""
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    _patch_loader(monkeypatch, _StubModel(proba_up=0.5))

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")
    feats = pd.Series([0.0] * len(FEATURE_COLUMNS), index=FEATURE_COLUMNS)
    p.predict(feats)
    assert p.status().has_model is True

    # Operator removes the file.
    weights.unlink()

    # Next status check must drop the cached model.
    s = p.status()
    assert s.has_model is False
    assert p.predict(feats) is None


# ───────────────────────── status / metrics roundtrip ─────────────────────────


def test_status_contains_metrics_when_present(tmp_path, monkeypatch):
    weights = tmp_path / "btc.cbm"
    metrics = tmp_path / "btc_metrics.json"
    _write_dummy_weights(weights)
    _write_metrics(metrics, dir_acc_mean=0.567, dir_acc_ci_low=0.521)
    _patch_loader(monkeypatch, _StubModel())

    p = LivePredictor(weights_path=weights, metrics_path=metrics)
    s = p.status()
    assert s.has_model is True
    assert s.is_calibrated is True
    assert s.dir_acc_mean == pytest.approx(0.567)
    assert s.dir_acc_ci_low == pytest.approx(0.521)
    assert s.model_age_seconds is not None and s.model_age_seconds >= 0
    assert s.metrics_age_seconds is not None and s.metrics_age_seconds >= 0


def test_status_to_dict_is_json_serialisable(tmp_path, monkeypatch):
    weights = tmp_path / "btc.cbm"
    metrics = tmp_path / "btc_metrics.json"
    _write_dummy_weights(weights)
    _write_metrics(metrics)
    _patch_loader(monkeypatch, _StubModel())

    p = LivePredictor(weights_path=weights, metrics_path=metrics)
    d = p.status().to_dict()
    # Must round-trip through json.dumps without TypeError.
    json.dumps(d)


def test_model_age_seconds_advances_with_clock(tmp_path, monkeypatch):
    weights = tmp_path / "btc.cbm"
    _write_dummy_weights(weights)
    _patch_loader(monkeypatch, _StubModel())

    p = LivePredictor(weights_path=weights, metrics_path=tmp_path / "m.json")

    # Force the predictor to load the model before computing age, otherwise
    # _weights_mtime is None and model_age_seconds returns None.
    p.predict(pd.Series([0.0] * len(FEATURE_COLUMNS), index=FEATURE_COLUMNS))

    # Inject a known mtime by overriding the cached value.
    import os as _os
    fixed = time.time() - 10.0
    _os.utime(weights, (fixed, fixed))
    p._weights_mtime = fixed  # type: ignore[attr-defined]

    age = p.model_age_seconds(now=fixed + 30.0)
    assert age == pytest.approx(30.0)


# ───────────────────────── singleton ─────────────────────────


def test_get_predictor_returns_same_instance():
    reset_predictor()
    a = get_predictor()
    b = get_predictor()
    assert a is b
    reset_predictor()


def test_reset_predictor_drops_singleton():
    reset_predictor()
    a = get_predictor()
    reset_predictor()
    b = get_predictor()
    assert a is not b
    reset_predictor()


# ───────────────────────── _maybe_float helper ─────────────────────────


def test_maybe_float_passes_finite():
    assert _maybe_float(0.5) == 0.5
    assert _maybe_float("0.5") == 0.5
    assert _maybe_float(0) == 0.0


def test_maybe_float_returns_none_on_garbage():
    assert _maybe_float(None) is None
    assert _maybe_float("nope") is None
    assert _maybe_float(float("nan")) is None
    assert _maybe_float(float("inf")) is None
