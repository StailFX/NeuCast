"""Live 1-minute directional predictor (Phase B).

Owns the in-memory CatBoost model loaded from
``weights/highfreq/<symbol>_1m.cbm`` and exposes :meth:`LivePredictor.predict`
to return :math:`P(\\text{up})` for a single feature row.

Why this lives outside the trainer
----------------------------------
The trainer is a one-shot Celery / systemd job that may run for minutes;
the predictor must score in milliseconds inside FastAPI request handlers.
Keeping them separate lets us:

* boot FastAPI without importing CatBoost cold (lazy ``import catboost``
  inside ``_load_model`` — saves ~200 ms cold-start);
* run unit tests without CatBoost at all (parsers + I/O paths only);
* hot-reload a freshly trained ``.cbm`` without restarting the web app.

Graceful degradation contract
-----------------------------
**Never crash on missing model.** During Phase A.* data accumulation the
trainer correctly exits with code 1 (no folds yet) and never writes a
``.cbm``. The web layer must keep serving — the predictor handles this
by returning :class:`PredictorStatus` with ``has_model=False`` and
``predict()`` returning ``None``. Callers (forecast endpoint, future
paper-trader) decide whether to surface a 503 / hide the UI block / etc.

Hot-reload
----------
On every :meth:`predict` call we ``stat()`` the weights file (cheap,
microseconds) and reload only if mtime advanced. The trainer's atomic
write (CatBoost ``save_model`` writes to a temp file then ``os.rename``
on POSIX, which is atomic) guarantees we never observe a torn file.

Calibration gate
----------------
The metrics report sits next to the weights (``..._metrics.json``).
We expose :meth:`is_calibrated` — ``True`` only when the lower bound of
the bootstrap CI on ``dir_acc`` exceeds 0.5. Forecast UI uses this to
surface a "model is not statistically reliable yet" badge so users
don't trade on noise. ADR-007.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.highfreq.feature_pipeline import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Defaults
# ────────────────────────────────────────────────────────────────────────────

#: Lower bound of the 95% bootstrap CI on directional accuracy required
#: for the model to be considered "statistically calibrated". Anything
#: at or below 0.5 means the model has not statistically beaten random;
#: serving forecasts in that regime would be UX malpractice.
CALIBRATED_DIR_ACC_THRESHOLD: float = 0.50


def _default_weights_path() -> Path:
    """Resolve the default ``.cbm`` path (env-overridable).

    Used by the legacy single-predictor ``get_predictor()`` call. New
    multi-symbol callers should use :func:`weights_path_for_symbol`
    which derives the path purely from the symbol (no env lookup) so
    different symbols don't collide on the same env override.
    """
    return Path(os.getenv(
        "HIGHFREQ_WEIGHTS_PATH",
        "weights/highfreq/btcusdt_1m.cbm",
    ))


def weights_path_for_symbol(symbol: str) -> Path:
    """Trainer-convention weights path for ``symbol``.

    e.g. ``"BTCUSDT"`` → ``weights/highfreq/btcusdt_1m.cbm``.
    The base directory is overridable via ``HIGHFREQ_WEIGHTS_DIR`` env
    (default ``weights/highfreq``) so dev / prod / test can split.
    """
    base = Path(os.getenv("HIGHFREQ_WEIGHTS_DIR", "weights/highfreq"))
    return base / f"{symbol.lower()}_1m.cbm"


def _metrics_path_for(weights_path: Path) -> Path:
    """``btcusdt_1m.cbm`` → ``btcusdt_1m_metrics.json`` (trainer convention)."""
    return weights_path.with_suffix("").with_name(weights_path.stem + "_metrics.json")


# ────────────────────────────────────────────────────────────────────────────
# Status payload (JSON-friendly)
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictorStatus:
    """Snapshot of the predictor's loadable state.

    Returned by :meth:`LivePredictor.status` for the forecast UI / API.
    All fields are JSON-friendly (no ``Path``, no ``datetime``) so the
    caller can embed it directly in a JSONResponse.
    """

    has_model: bool                   # ``.cbm`` is loaded right now
    model_path: str                   # absolute path the predictor watches
    model_age_seconds: float | None   # since file mtime; None when no model
    is_calibrated: bool               # dir_acc_ci_low > threshold
    dir_acc_mean: float | None        # last training run's bootstrap point
    dir_acc_ci_low: float | None      # bootstrap 95% lower bound
    #: One-sided binomial p-value testing H₀ "no skill" vs H_a "p > 0.5".
    #: Complementary to ``dir_acc_ci_low``: even when the CI lower bound
    #: brushes 0.5, a small p-value (< 0.05) is the canonical academic
    #: way to claim significance. ``None`` until the trainer produces
    #: at least one fold's worth of pooled predictions.
    dir_acc_p_value: float | None
    metrics_age_seconds: float | None # since metrics-file mtime
    n_features_expected: int          # len(FEATURE_COLUMNS) for sanity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────────────
# LivePredictor
# ────────────────────────────────────────────────────────────────────────────


class LivePredictor:
    """In-memory model holder with hot-reload + graceful no-model fallback.

    Lifecycle::

        predictor = LivePredictor()                      # no I/O yet
        prob = predictor.predict(features_row)           # loads on first call
        # ...trainer writes a new .cbm overnight...
        prob = predictor.predict(features_row)           # auto-reloads

    Thread safety
    -------------
    A single ``threading.Lock`` guards the reload-and-swap path. Scoring
    itself is read-only on ``self._model`` — CatBoost's ``predict_proba``
    is thread-safe — and we accept that two concurrent requests in the
    middle of a reload may briefly score against the previous model.
    Both models are valid; the staleness window is the duration of one
    ``CatBoostClassifier.load_model`` call (~10ms).
    """

    def __init__(
        self,
        weights_path: str | os.PathLike[str] | None = None,
        metrics_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.weights_path: Path = (
            Path(weights_path) if weights_path is not None else _default_weights_path()
        )
        self.metrics_path: Path = (
            Path(metrics_path) if metrics_path is not None
            else _metrics_path_for(self.weights_path)
        )
        # In-memory state.
        self._model: Any | None = None
        self._weights_mtime: float | None = None
        self._metrics: dict[str, Any] | None = None
        self._metrics_mtime: float | None = None
        self._reload_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def predict(self, features_row: pd.Series | dict[str, float] | np.ndarray) -> float | None:
        """Return :math:`P(\\text{up}) \\in [0, 1]`, or ``None`` if no model.

        ``features_row`` may be:
          * a :class:`pd.Series` indexed by :data:`FEATURE_COLUMNS`,
          * a ``dict[str, float]`` keyed by the same names,
          * a 1-D :class:`numpy.ndarray` already in :data:`FEATURE_COLUMNS` order.

        We always reorder by :data:`FEATURE_COLUMNS` to match the trainer's
        fitted column order — passing a Series with shuffled index would
        otherwise silently produce garbage forecasts.
        """
        self._maybe_reload_model()
        if self._model is None:
            return None

        x = self._prepare_features(features_row)
        # CatBoost predict_proba returns shape (1, 2): [[P(0), P(1)]].
        proba = self._model.predict_proba(x.reshape(1, -1))
        # Defensive: unusual classifiers may return shape (1,) instead of (1, 2).
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(np.asarray(proba).ravel()[-1])

    def feature_importance(self) -> list[dict[str, Any]] | None:
        """Per-feature importance from the loaded CatBoost model.

        Returns a list of ``{"feature": str, "importance": float}`` sorted
        by importance descending, or ``None`` when no model is loaded.

        Uses CatBoost's default ``PredictionValuesChange`` method —
        average change in prediction when the feature is shuffled,
        normalised to sum to 100. Constant for a given .cbm file, so
        the result can be cached in the caller (we don't cache here
        because it's cheap on 14 features and the predictor doesn't
        own UI-cache lifetime).

        Stub-friendly: if the model object exposes ``get_feature_importance``
        (CatBoost convention), we use it. Otherwise return ``None`` —
        keeps this method usable in tests with a minimal model stub.
        """
        self._maybe_reload_model()
        if self._model is None:
            return None
        if not hasattr(self._model, "get_feature_importance"):
            return None
        try:
            raw = self._model.get_feature_importance()
        except Exception as exc:
            logger.warning("get_feature_importance() failed: %s", exc)
            return None
        from app.highfreq.feature_pipeline import FEATURE_COLUMNS as _FC
        if len(raw) != len(_FC):
            logger.warning(
                "feature_importance length mismatch: model=%d expected=%d",
                len(raw), len(_FC),
            )
            return None
        pairs = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(_FC, raw)
        ]
        pairs.sort(key=lambda p: p["importance"], reverse=True)
        return pairs

    def is_calibrated(self) -> bool:
        """``True`` when the latest bootstrap CI clears the threshold."""
        self._maybe_reload_metrics()
        if self._metrics is None:
            return False
        ci_low = self._metrics.get("dir_acc_ci_low")
        if ci_low is None:
            return False
        try:
            return float(ci_low) > CALIBRATED_DIR_ACC_THRESHOLD
        except (TypeError, ValueError):
            return False

    def model_age_seconds(self, now: float | None = None) -> float | None:
        """Seconds since the weights file was last written, or ``None``."""
        self._maybe_reload_model()
        if self._weights_mtime is None:
            return None
        ts = time.time() if now is None else now
        return max(0.0, ts - self._weights_mtime)

    def status(self, now: float | None = None) -> PredictorStatus:
        """JSON-friendly snapshot for the forecast endpoint / UI."""
        # Trigger reload pass once so the snapshot is consistent.
        self._maybe_reload_model()
        self._maybe_reload_metrics()
        ts = time.time() if now is None else now

        metrics_age: float | None = None
        if self._metrics_mtime is not None:
            metrics_age = max(0.0, ts - self._metrics_mtime)

        weights_age: float | None = None
        if self._weights_mtime is not None:
            weights_age = max(0.0, ts - self._weights_mtime)

        dir_acc_mean: float | None = None
        dir_acc_ci_low: float | None = None
        dir_acc_p_value: float | None = None
        if self._metrics is not None:
            dir_acc_mean = _maybe_float(self._metrics.get("dir_acc_mean"))
            dir_acc_ci_low = _maybe_float(self._metrics.get("dir_acc_ci_low"))
            # ``.get`` so older metrics.json files written before
            # the p-value field was added still load cleanly (returns
            # None → UI shows "—").
            dir_acc_p_value = _maybe_float(self._metrics.get("dir_acc_p_value"))

        return PredictorStatus(
            has_model=self._model is not None,
            model_path=str(self.weights_path),
            model_age_seconds=weights_age,
            is_calibrated=self.is_calibrated(),
            dir_acc_mean=dir_acc_mean,
            dir_acc_ci_low=dir_acc_ci_low,
            dir_acc_p_value=dir_acc_p_value,
            metrics_age_seconds=metrics_age,
            n_features_expected=len(FEATURE_COLUMNS),
        )

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _prepare_features(
        self, features_row: pd.Series | dict[str, float] | np.ndarray
    ) -> np.ndarray:
        """Coerce the input into a 1-D float array matching FEATURE_COLUMNS."""
        if isinstance(features_row, pd.Series):
            try:
                ordered = features_row.reindex(FEATURE_COLUMNS)
            except Exception:  # noqa: BLE001
                # Fall back to positional if reindex is impossible (no matching index).
                ordered = pd.Series(features_row.values, index=FEATURE_COLUMNS)
            arr = ordered.to_numpy(dtype=float, copy=False)
        elif isinstance(features_row, dict):
            arr = np.array(
                [float(features_row.get(c, 0.0)) for c in FEATURE_COLUMNS],
                dtype=float,
            )
        else:
            arr = np.asarray(features_row, dtype=float).ravel()
            if arr.shape[0] != len(FEATURE_COLUMNS):
                raise ValueError(
                    f"feature vector has {arr.shape[0]} elements, "
                    f"expected {len(FEATURE_COLUMNS)} (FEATURE_COLUMNS)"
                )

        # Scrub NaN / inf — same contract as `aggregator._safe`.
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return arr

    def _maybe_reload_model(self) -> None:
        """Reload the model if the file is new or has a newer mtime."""
        try:
            stat = self.weights_path.stat()
        except FileNotFoundError:
            # No model on disk yet — clear cached state if we ever loaded.
            if self._model is not None:
                logger.warning(
                    "LivePredictor: weights vanished from %s, dropping cached model",
                    self.weights_path,
                )
            self._model = None
            self._weights_mtime = None
            return
        except OSError as e:
            logger.warning("LivePredictor: stat() failed on %s: %s", self.weights_path, e)
            return

        if self._weights_mtime is not None and stat.st_mtime <= self._weights_mtime:
            return  # already up-to-date — fast path, ~microseconds

        # Slow path: load (or reload) under lock.
        with self._reload_lock:
            # Re-check inside lock — another thread may have just loaded.
            try:
                stat = self.weights_path.stat()
            except OSError:
                return
            if self._weights_mtime is not None and stat.st_mtime <= self._weights_mtime:
                return
            try:
                model = self._load_model(self.weights_path)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "LivePredictor: failed to load model from %s — keeping previous",
                    self.weights_path,
                )
                return  # keep serving on the old model rather than degrading
            self._model = model
            self._weights_mtime = stat.st_mtime
            logger.info(
                "LivePredictor: loaded model from %s (mtime=%s, age=%.1fs)",
                self.weights_path,
                stat.st_mtime,
                max(0.0, time.time() - stat.st_mtime),
            )

    def _maybe_reload_metrics(self) -> None:
        """Reload metrics JSON when the file mtime advances. No lock needed —
        worst case is a request seeing the previous metrics for a few µs."""
        try:
            stat = self.metrics_path.stat()
        except FileNotFoundError:
            self._metrics = None
            self._metrics_mtime = None
            return
        except OSError:
            return

        if self._metrics_mtime is not None and stat.st_mtime <= self._metrics_mtime:
            return
        try:
            with open(self.metrics_path) as f:
                self._metrics = json.load(f)
            self._metrics_mtime = stat.st_mtime
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("LivePredictor: cannot read metrics %s: %s", self.metrics_path, e)
            self._metrics = None

    @staticmethod
    def _load_model(path: Path) -> Any:
        """Lazy CatBoost import — keeps the predictor module importable
        in test contexts that don't have catboost installed."""
        from catboost import CatBoostClassifier  # noqa: WPS433

        model = CatBoostClassifier()
        model.load_model(str(path))
        return model


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _maybe_float(x: Any) -> float | None:
    """``None`` if ``x`` is ``None`` or not coercible; otherwise ``float(x)``."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


# ────────────────────────────────────────────────────────────────────────────
# Process-wide singleton (FastAPI worker)
# ────────────────────────────────────────────────────────────────────────────

_predictor: LivePredictor | None = None
_predictors_by_symbol: dict[str, LivePredictor] = {}
_predictor_lock = threading.Lock()


def get_predictor(symbol: str | None = None) -> LivePredictor:
    """Return a :class:`LivePredictor` instance.

    * ``symbol=None`` (legacy / default) → process-wide singleton tied
      to the env-overridable ``HIGHFREQ_WEIGHTS_PATH``. Kept for
      backward compatibility with the single-symbol code path.
    * ``symbol="BTCUSDT"`` (multi-symbol mode) → per-symbol cached
      instance whose weights/metrics paths are derived purely from the
      symbol via :func:`weights_path_for_symbol`. Each FastAPI worker
      process holds at most one predictor per symbol.

    The two caches are independent — calling ``get_predictor()`` then
    ``get_predictor("BTCUSDT")`` yields two different instances (they
    happen to share the same weights file in default config, but the
    objects are distinct).
    """
    global _predictor
    if symbol is None:
        if _predictor is not None:
            return _predictor
        with _predictor_lock:
            if _predictor is None:
                _predictor = LivePredictor()
        return _predictor

    sym = symbol.upper()
    cached = _predictors_by_symbol.get(sym)
    if cached is not None:
        return cached
    with _predictor_lock:
        cached = _predictors_by_symbol.get(sym)
        if cached is None:
            wp = weights_path_for_symbol(sym)
            cached = LivePredictor(weights_path=wp)
            _predictors_by_symbol[sym] = cached
    return cached


def reset_predictor() -> None:
    """Drop ALL cached predictors — used by tests to force a clean state."""
    global _predictor
    with _predictor_lock:
        _predictor = None
        _predictors_by_symbol.clear()
