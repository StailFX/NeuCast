"""FastAPI router for the high-frequency `/highfreq` page (Phase A.6).

Read-only UI surface over the data the L2 ingestor writes into Postgres:

* ``GET /highfreq``                 — HTML page (live microprice + countdown)
* ``GET /api/highfreq/status``      — JSON poll endpoint (2-second cadence)
* ``GET /api/highfreq/health``      — quick liveness probe (200 / 503)

Design notes
------------

The L2 ingestor is a long-running asyncio service writing rows via
``asyncpg`` (see :mod:`app.highfreq.aggregator`). The FastAPI process
intentionally does **not** share that connection pool — it issues short
synchronous read queries on the main SQLAlchemy engine, matching the
pattern used by the rest of ``app.main``.

This means the UI is a "thin observer" of the same Postgres tables and
will keep working even if the ingest service is restarting.

Phase A scope is sim-only — the page must visibly reflect ADR-005:
**no live trading, paper P&L only**. See ``docs/highfreq/architecture.md``.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.orm import Session

from app.highfreq.feature_pipeline import build_latest_feature_row
from app.highfreq.paper_trader import PaperTraderConfig, RiskCaps
from app.highfreq.predictor import LivePredictor, get_predictor

# NOTE: ``app.db`` reads ``DATABASE_URL`` at import time, so we **defer**
# importing ``SessionLocal`` to the request-scoped dependency below. This
# keeps ``app.highfreq.web`` importable in test contexts without env config
# (matches the pattern used by ``app.highfreq.trainer.run_training``).

logger = logging.getLogger(__name__)


# ── Constants tied to architecture decisions ──────────────────────────────
# The walk-forward trainer needs ≥(initial_train_minutes + test_fold_minutes)
# minutes of data before it can produce its first fold report. With current
# defaults (initial=5, test=60) the minimum is 65 minutes; we surface this
# to users as a countdown so they know when training becomes feasible.
MIN_MINUTES_FOR_TRAINING: int = 65

DEFAULT_SYMBOL: str = os.getenv("HIGHFREQ_DEFAULT_SYMBOL", "BTCUSDT")


def _available_symbols() -> list[str]:
    """Symbols the UI dropdown should expose.

    Read from ``HIGHFREQ_SYMBOLS`` (comma-separated, same env the ingest
    runner uses — keeps the UI and ingest in lock-step). Falls back to
    just ``DEFAULT_SYMBOL`` so a fresh deploy without explicit env set
    still works (single-symbol mode).
    """
    raw = os.getenv("HIGHFREQ_SYMBOLS", DEFAULT_SYMBOL)
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return out or [DEFAULT_SYMBOL]

# A row is considered "fresh" if its ts is within this many seconds of now.
# The aggregator emits at 1-second cadence, so anything older than ~10s
# is a clear sign the ingest pipeline has stalled.
LIVENESS_THRESHOLD_SEC: int = 15


# ── Pure-logic dataclasses (testable without a DB) ────────────────────────


@dataclass(frozen=True)
class HighfreqSnapshot:
    """Most recent 1-second row joined with derived freshness signals."""

    ts: Optional[datetime]
    symbol: str
    microprice: Optional[float]
    ofi: Optional[float]
    depth_imb: Optional[float]
    spread_bps: Optional[float]
    age_seconds: Optional[float]


@dataclass(frozen=True)
class HighfreqStatus:
    """Full payload returned by ``GET /api/highfreq/status``.

    All fields are JSON-friendly (no NaN / Inf — see :func:`_scrub`).
    """

    symbol: str
    snapshot: Optional[HighfreqSnapshot]
    minutes_accumulated: int
    minutes_required: int
    minutes_remaining: int
    progress_pct: float                  # 0.0 .. 100.0
    is_live: bool                         # latest row is fresh
    has_enough_data: bool                 # ≥ MIN_MINUTES_FOR_TRAINING
    rows_per_second_estimate: Optional[float]  # last 60s
    server_time: datetime

    def to_dict(self) -> dict[str, Any]:
        return _scrub(asdict(self))


# ── JSON sanitisation ─────────────────────────────────────────────────────


def _scrub(value: Any) -> Any:
    """Recursively replace NaN / Inf with ``None`` for RFC-7159 compliance.

    Same pattern used in ``app.highfreq.trainer.TrainingReport.to_json``.
    """
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


# ── Pure-logic helpers (unit-tested without spinning up FastAPI) ──────────


def compute_progress_fields(
    minutes_accumulated: int,
    minutes_required: int = MIN_MINUTES_FOR_TRAINING,
) -> tuple[int, float, bool]:
    """Derive (remaining, progress_pct, has_enough_data) from raw count.

    ``minutes_accumulated`` may exceed the requirement; ``progress_pct`` is
    clamped to ``100.0`` and ``minutes_remaining`` floors at zero.
    """
    accumulated = max(0, int(minutes_accumulated))
    required = max(1, int(minutes_required))
    remaining = max(0, required - accumulated)
    progress = min(100.0, 100.0 * accumulated / required)
    has_enough = accumulated >= required
    return remaining, progress, has_enough


def compute_age_seconds(ts: Optional[datetime], now: datetime) -> Optional[float]:
    """Seconds elapsed between ``ts`` and ``now`` (UTC-aware), or ``None``."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = (now - ts).total_seconds()
    # Clamp tiny clock-skew negatives — they confuse the UI but are not bugs.
    return max(0.0, float(delta))


def assemble_status(
    *,
    symbol: str,
    last_row: Optional[dict[str, Any]],
    minutes_accumulated: int,
    rows_last_60s: Optional[int],
    now: datetime,
) -> HighfreqStatus:
    """Pure function that turns raw DB results into the status payload.

    Split out so unit tests can exercise the field-derivation logic without
    a live database. The router :func:`get_status` composes the three
    queries and calls this.
    """
    if last_row is not None:
        ts = last_row.get("ts")
        age = compute_age_seconds(ts, now)
        snapshot = HighfreqSnapshot(
            ts=ts,
            symbol=str(last_row.get("symbol", symbol)),
            microprice=_to_float(last_row.get("microprice")),
            ofi=_to_float(last_row.get("ofi")),
            depth_imb=_to_float(last_row.get("depth_imb")),
            spread_bps=_to_float(last_row.get("spread_bps")),
            age_seconds=age,
        )
        is_live = age is not None and age <= LIVENESS_THRESHOLD_SEC
    else:
        snapshot = None
        is_live = False

    remaining, progress, has_enough = compute_progress_fields(minutes_accumulated)

    rate: Optional[float]
    if rows_last_60s is None:
        rate = None
    else:
        rate = round(float(rows_last_60s) / 60.0, 2)

    return HighfreqStatus(
        symbol=symbol,
        snapshot=snapshot,
        minutes_accumulated=int(minutes_accumulated),
        minutes_required=MIN_MINUTES_FOR_TRAINING,
        minutes_remaining=remaining,
        progress_pct=round(progress, 2),
        is_live=is_live,
        has_enough_data=has_enough,
        rows_per_second_estimate=rate,
        server_time=now,
    )


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ── DB layer (synchronous, defensive) ─────────────────────────────────────


def _fetch_last_row(db: Session, symbol: str) -> Optional[dict[str, Any]]:
    """Return the most recent ``highfreq_ofi_1s`` row or ``None``.

    Returns ``None`` when the table is empty or does not exist (Phase A
    not yet deployed — the page renders gracefully in that case).
    """
    try:
        row = db.execute(
            text(
                "SELECT ts, symbol, microprice, ofi, depth_imb, spread_bps "
                "FROM highfreq_ofi_1s "
                "WHERE symbol = :symbol "
                "ORDER BY ts DESC LIMIT 1"
            ),
            {"symbol": symbol},
        ).mappings().first()
    except (ProgrammingError, OperationalError) as exc:
        # Table missing / DB unreachable — treat as "not initialised yet".
        logger.warning("highfreq_ofi_1s read failed (%s): %s", symbol, exc)
        return None
    return dict(row) if row is not None else None


def _fetch_minutes_accumulated(db: Session, symbol: str) -> int:
    """How many distinct minute-buckets exist for ``symbol``.

    Used to drive the "data ready in N minutes" countdown. Counted against
    ``highfreq_ofi_1s`` (1 row / sec) rather than ``highfreq_features_1m``
    so the bar moves smoothly even before any minute-aggregation runs.
    """
    try:
        result = db.execute(
            text(
                "SELECT COUNT(DISTINCT date_trunc('minute', ts)) AS n "
                "FROM highfreq_ofi_1s WHERE symbol = :symbol"
            ),
            {"symbol": symbol},
        ).scalar()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("highfreq_ofi_1s minute-count failed (%s): %s", symbol, exc)
        return 0
    return int(result or 0)


def _fetch_rows_last_60s(db: Session, symbol: str) -> Optional[int]:
    """Count of 1-second rows in the last 60 s — proxy for ingest health."""
    try:
        result = db.execute(
            text(
                "SELECT COUNT(*) AS n FROM highfreq_ofi_1s "
                "WHERE symbol = :symbol AND ts > now() - interval '60 seconds'"
            ),
            {"symbol": symbol},
        ).scalar()
    except (ProgrammingError, OperationalError):
        return None
    return int(result or 0)


# Forecast endpoint reads enough seconds of history to assemble at
# least one COMPLETE 1-minute feature bar (60s) plus margin so the
# build_latest_feature_row() helper can drop the in-flight current minute.
# 180s = ~3 minutes = comfortable headroom even right after a reconnect.
_FORECAST_LOOKBACK_SECONDS: int = 180


def _fetch_recent_seconds(
    db: Session, symbol: str, lookback_seconds: int = _FORECAST_LOOKBACK_SECONDS,
) -> Optional[pd.DataFrame]:
    """Fetch the last ``lookback_seconds`` of 1-s rows for ``symbol``.

    Returns
    -------
    pd.DataFrame
        Columns matching ``highfreq_ofi_1s`` (ts, symbol, ofi, microprice,
        depth_imb, spread_bps, trade_imb, n_updates). Sorted by ``ts``
        ascending. May be empty if the ingestor is down.
    None
        If the query failed (DB unreachable / table missing). Caller
        should surface this as a 503, distinct from "model loaded but
        nothing to score on".
    """
    try:
        rows = db.execute(
            text(
                "SELECT ts, symbol, ofi, microprice, depth_imb, "
                "spread_bps, trade_imb, n_updates "
                "FROM highfreq_ofi_1s "
                "WHERE symbol = :symbol "
                "  AND ts > now() - make_interval(secs => :secs) "
                "ORDER BY ts ASC"
            ),
            {"symbol": symbol, "secs": int(lookback_seconds)},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("highfreq_ofi_1s recent-rows fetch failed (%s): %s", symbol, exc)
        return None
    return pd.DataFrame([dict(r) for r in rows])


# ── Router ────────────────────────────────────────────────────────────────

# Templates dir is repo-root/templates; resolved relative to this file so the
# router works whether the FastAPI app is launched from the repo root or
# anywhere else (tests, IDEs).
_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "templates",
)
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


router = APIRouter(tags=["highfreq"])


def _get_db():
    """SQLAlchemy session dependency — mirrors ``app.main.get_db``.

    Deferred import: ``app.db`` requires ``DATABASE_URL`` at import time,
    so we delay until the first request. This keeps the module importable
    under pytest without a live database.
    """
    from app.db import SessionLocal  # noqa: WPS433 — intentional lazy import

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# NOTE — release T.4 (2026-04-29):
# The ``/highfreq`` and ``/forecast`` HTML routes used to live HERE
# (Tokyo). They've been MOVED to Finland's ``app/main.py`` so the
# auth gate (login required for /forecast, admin-only for /highfreq)
# can use Finland's existing session middleware.
#
# Tokyo now serves ONLY the JSON ``/api/highfreq/*`` endpoints.
# Finland's nginx still proxies ``/api/highfreq/*`` to this router via
# the WireGuard tunnel; the rendered HTML pages are produced on
# Finland and pull data through that proxy.
#
# Defence-in-depth: if a future visitor somehow hits Tokyo directly
# (e.g. via WG IP), they get a 404 on /highfreq + /forecast rather
# than an unauthenticated render of the operator UI.


@router.get("/api/highfreq/status")
async def get_status(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Return the current ingest snapshot + training-readiness countdown."""
    symbol = symbol.upper()
    last_row = _fetch_last_row(db, symbol)
    minutes_accumulated = _fetch_minutes_accumulated(db, symbol)
    rows_last_60s = _fetch_rows_last_60s(db, symbol)

    payload = assemble_status(
        symbol=symbol,
        last_row=last_row,
        minutes_accumulated=minutes_accumulated,
        rows_last_60s=rows_last_60s,
        now=datetime.now(tz=timezone.utc),
    )
    return JSONResponse(content=payload.to_dict())


@router.get("/api/highfreq/health")
async def get_health(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Liveness probe: 200 if rows are flowing, 503 otherwise.

    Used by uptime monitors and the systemd watchdog. Cheap (one indexed
    query) so it can be polled at high frequency without DB pressure.
    """
    symbol = symbol.upper()
    rows_last_60s = _fetch_rows_last_60s(db, symbol)

    if rows_last_60s is None:
        # DB or table missing — still report what we know without crashing.
        return JSONResponse(
            status_code=503,
            content={"ok": False, "reason": "database_unavailable", "symbol": symbol},
        )

    if rows_last_60s == 0:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "no_rows_in_last_60s",
                "symbol": symbol,
                "rows_last_60s": 0,
            },
        )

    return JSONResponse(
        content={
            "ok": True,
            "symbol": symbol,
            "rows_last_60s": rows_last_60s,
        }
    )


# ── Forecast endpoint (Phase B scaffold) ──────────────────────────────────


def _get_forecast_predictor(
    symbol: str = DEFAULT_SYMBOL,
    horizon: int = 1,
) -> LivePredictor:
    """DI-friendly per-(symbol, horizon) predictor accessor — overrideable
    in tests.

    FastAPI shares query params across the dependency tree, so the
    ``?symbol=`` and ``?horizon=`` on the endpoint flow in here — each
    request gets the cached predictor for that (symbol, horizon) pair
    via :func:`get_predictor`. Default horizon=1 preserves the original
    contract: callers that don't specify get the 1m model.
    """
    return get_predictor(symbol, horizon_minutes=horizon)


@router.get("/api/highfreq/forecast")
async def get_forecast(
    symbol: str = DEFAULT_SYMBOL,
    horizon: int = 1,
    db: Session = Depends(_get_db),
    predictor: LivePredictor = Depends(_get_forecast_predictor),
) -> JSONResponse:
    """Latest directional forecast for ``symbol`` × ``horizon`` (release T.6).

    Default ``horizon=1`` preserves the original 1-minute contract;
    callers can pass ``?horizon=15`` (or 5/60) to score against a
    long-horizon model when one has been trained.

    The endpoint dispatches the live-inference helper based on the
    predictor's recorded ``feature_set`` (microstructure for 1m,
    long_horizon for ≥5m) so the feature-row column count matches the
    model's expectation — same dispatch the paper-trader runner uses.

    Returns 200 with ``prob_up`` once the trainer has produced a
    ``.cbm`` AND there's a complete recent bar to score on.  Until
    then, returns 503 with a structured ``reason``:

    * ``no_model_yet`` — trainer hasn't run yet / no weights file
    * ``database_unavailable`` — DB unreachable
    * ``not_enough_recent_data`` — no complete bar in the recent
      window (cold-start, or reconnect just happened)

    The response always includes the predictor ``model`` block so
    clients can show "model age" / "calibrated?" badges even on 503.
    """
    symbol = symbol.upper()
    if horizon <= 0:
        horizon = 1
    status = predictor.status()

    # Branch 1: no model on disk → trainer hasn't shipped weights yet.
    if not status.has_model:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "no_model_yet",
                "symbol": symbol,
                "horizon_minutes": horizon,
                "model": status.to_dict(),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    # Branch 2: model loaded but DB unreachable (e.g. ingest restart).
    # For long-horizon models we need MUCH more lookback (the long-
    # horizon helper needs ≥20 complete bars to bootstrap EMA(20) /
    # Bollinger(20) / RSI(14)). For cross_asset we need ≥4 complete
    # bars so lag3 is defined for the latest row.
    fs = predictor.feature_set()
    bm = predictor.bar_minutes()
    if fs == "long_horizon" and bm > 1:
        # 25 bars × bar_minutes × 60s + 60s headroom for in-flight drop.
        lookback = 25 * bm * 60 + 60
    elif fs == "cross_asset":
        # 6 bars × bar_minutes × 60s + 60s headroom = ≥4 complete bars
        # after the in-flight drop, plenty for lag3 + cross-asset join.
        lookback = max(_FORECAST_LOOKBACK_SECONDS, 6 * bm * 60 + 60)
    else:
        lookback = _FORECAST_LOOKBACK_SECONDS
    df_seconds = _fetch_recent_seconds(db, symbol, lookback_seconds=lookback)
    if df_seconds is None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "database_unavailable",
                "symbol": symbol,
                "horizon_minutes": horizon,
                "model": status.to_dict(),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    # Branch 3: not enough recent data for a complete bar.
    if fs == "long_horizon":
        from app.highfreq.feature_pipeline_long_horizon import (
            build_latest_inference_bar_long_horizon,
        )
        inference = build_latest_inference_bar_long_horizon(
            df_seconds, bar_minutes=bm,
        )
        feature_row = inference[0] if inference is not None else None
    elif fs == "cross_asset":
        # ETH/BNB pull BTC seconds as a reference. Same wall-clock
        # window as the target seconds — both already share the
        # bar-minute alignment after aggregation. BTC's own model has
        # no reference (would be identity).
        from app.highfreq.feature_pipeline_cross_asset import (
            build_latest_inference_bar_cross_asset,
        )
        sym_upper = symbol.upper()
        ref_symbol = None if sym_upper == "BTCUSDT" else "BTCUSDT"
        ref_seconds: pd.DataFrame | None = None
        if ref_symbol is not None:
            ref_seconds = _fetch_recent_seconds(
                db, ref_symbol, lookback_seconds=lookback,
            )
            # If reference fetch fails we don't fail the request —
            # build_latest_inference_bar_cross_asset handles missing
            # reference by zero-filling those columns.
        inference = build_latest_inference_bar_cross_asset(
            df_seconds,
            bar_minutes=bm,
            reference_df_seconds=ref_seconds,
            reference_symbol=ref_symbol,
        )
        feature_row = inference[0] if inference is not None else None
    elif fs == "microstructure_v2":
        # T.18.c: base microstructure + 4 trade-flow rolling features.
        # Same single-symbol contract as base — no reference seconds.
        # Needs ≥4 complete bars so lag1 + 3-bar rolling are defined.
        from app.highfreq.feature_pipeline_microstructure_v2 import (
            build_latest_inference_bar_microstructure_v2,
        )
        inference = build_latest_inference_bar_microstructure_v2(df_seconds)
        feature_row = inference[0] if inference is not None else None
    else:
        feature_row = build_latest_feature_row(df_seconds)

    if feature_row is None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "not_enough_recent_data",
                "symbol": symbol,
                "horizon_minutes": horizon,
                "model": status.to_dict(),
                "rows_seen": int(len(df_seconds)),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    prob_up = predictor.predict(feature_row)
    if prob_up is None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "model_unavailable",
                "symbol": symbol,
                "horizon_minutes": horizon,
                "model": status.to_dict(),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    # Map probability to a human-readable signal so the UI doesn't have to
    # re-derive thresholds. Same convention as the future paper-trader:
    # prob > 0.55 = "up" tilt, prob < 0.45 = "down" tilt, else "neutral".
    if prob_up >= 0.55:
        signal = "up"
    elif prob_up <= 0.45:
        signal = "down"
    else:
        signal = "neutral"

    # Conformal prediction interval (T.17.b) — wraps prob_up with a
    # 90 %-coverage band ``[max(0, p - q), min(1, p + q)]`` where q is
    # the split-conformal nonconformity quantile from training. The
    # UI shows this as "0.62 ± [0.55, 0.69]" so reviewers see the
    # uncertainty alongside the point estimate.
    conformal_90: dict[str, Any] | None = None
    conformal_95: dict[str, Any] | None = None
    try:
        ci_90 = predictor.conformal_interval(float(prob_up), alpha=0.10)
        if ci_90 is not None:
            conformal_90 = {
                "alpha": 0.10,
                "low": float(ci_90[0]),
                "high": float(ci_90[1]),
                "halfwidth": float(predictor.conformal_quantile(alpha=0.10) or 0.0),
            }
        ci_95 = predictor.conformal_interval(float(prob_up), alpha=0.05)
        if ci_95 is not None:
            conformal_95 = {
                "alpha": 0.05,
                "low": float(ci_95[0]),
                "high": float(ci_95[1]),
                "halfwidth": float(predictor.conformal_quantile(alpha=0.05) or 0.0),
            }
    except Exception:  # noqa: BLE001
        # Conformal is best-effort — never fail the forecast over it.
        conformal_90 = conformal_95 = None

    return JSONResponse(
        content=_scrub({
            "ok": True,
            "symbol": symbol,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "horizon_minutes": horizon,
            "prob_up": float(prob_up),
            "signal": signal,
            "calibrated": bool(status.is_calibrated),
            "conformal_90": conformal_90,
            "conformal_95": conformal_95,
            "model": status.to_dict(),
        })
    )


def _predict_for_horizon(
    symbol: str, horizon: int, db: Session,
) -> tuple[float | None, dict]:
    """Helper for the ensemble endpoint — runs the same inference
    chain ``/api/highfreq/forecast`` does for one (symbol, horizon)
    pair. Returns ``(prob_up, status_dict)``: prob_up is None when
    the model isn't ready or features can't be built.

    Factored out as a thin local helper rather than refactoring
    ``get_forecast`` to avoid a sweeping rewrite of the existing
    branching logic; the ensemble endpoint calls this twice (1m + 15m)
    and combines via ``ensemble_probability``.
    """
    from app.highfreq.predictor import get_predictor as _get_predictor

    predictor = _get_predictor(symbol, horizon_minutes=horizon)
    status = predictor.status()
    status_dict = status.to_dict()
    if not status.has_model:
        return None, status_dict

    fs = predictor.feature_set()
    bm = predictor.bar_minutes()
    if fs == "long_horizon" and bm > 1:
        lookback = 25 * bm * 60 + 60
    elif fs == "cross_asset":
        lookback = max(_FORECAST_LOOKBACK_SECONDS, 6 * bm * 60 + 60)
    else:
        lookback = _FORECAST_LOOKBACK_SECONDS
    df_seconds = _fetch_recent_seconds(db, symbol, lookback_seconds=lookback)
    if df_seconds is None:
        return None, status_dict

    if fs == "long_horizon":
        from app.highfreq.feature_pipeline_long_horizon import (
            build_latest_inference_bar_long_horizon,
        )
        inference = build_latest_inference_bar_long_horizon(
            df_seconds, bar_minutes=bm,
        )
        feature_row = inference[0] if inference is not None else None
    elif fs == "cross_asset":
        from app.highfreq.feature_pipeline_cross_asset import (
            build_latest_inference_bar_cross_asset,
        )
        sym_upper = symbol.upper()
        ref_symbol = None if sym_upper == "BTCUSDT" else "BTCUSDT"
        ref_seconds = (
            _fetch_recent_seconds(db, ref_symbol, lookback_seconds=lookback)
            if ref_symbol is not None else None
        )
        inference = build_latest_inference_bar_cross_asset(
            df_seconds, bar_minutes=bm,
            reference_df_seconds=ref_seconds,
            reference_symbol=ref_symbol,
        )
        feature_row = inference[0] if inference is not None else None
    elif fs == "microstructure_v2":
        from app.highfreq.feature_pipeline_microstructure_v2 import (
            build_latest_inference_bar_microstructure_v2,
        )
        inference = build_latest_inference_bar_microstructure_v2(df_seconds)
        feature_row = inference[0] if inference is not None else None
    else:
        feature_row = build_latest_feature_row(df_seconds)

    if feature_row is None:
        return None, status_dict

    prob = predictor.predict(feature_row)
    if prob is None:
        return None, status_dict
    return float(prob), status_dict


@router.get("/api/highfreq/forecast_ensemble")
async def get_forecast_ensemble(
    symbol: str = DEFAULT_SYMBOL,
    weight_1m: float = 0.70,
    weight_15m: float = 0.30,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Multi-horizon ensemble forecast (T.19, 2026-05-03).

    Combines the 1-minute model (microstructure / cross_asset) with
    the 15-minute model (long_horizon TA) via a weighted average of
    calibrated probabilities. Default 70/30 — the 1m model is
    empirically stronger but the 15m model captures multi-bar
    momentum the 1m can't see.

    When two models agree, the ensemble's effective confidence rises
    beyond either alone. When they disagree, the 1m signal dominates
    by weight.

    Response shape::

        {
          "ok": true,
          "symbol": "BTCUSDT",
          "ts": "...",
          "prob_up": 0.58,            // weighted blend
          "signal": "up" | "down" | "neutral",
          "agreement": true,
          "components": [
            {"horizon_label": "1m",  "weight": 0.70, "prob_up": 0.62, "is_available": true},
            {"horizon_label": "15m", "weight": 0.30, "prob_up": 0.50, "is_available": true}
          ],
          "n_components_used": 2,
          "models": {
            "1m": {...status dict...},
            "15m": {...status dict...}
          }
        }

    Backward-compat: ``/api/highfreq/forecast`` (no _ensemble) keeps
    returning 1m-only — this is a separate endpoint so existing
    consumers aren't affected.
    """
    from app.highfreq.ensemble import (
        EnsembleComponent,
        ensemble_probability,
    )

    symbol = symbol.upper()
    if not (0.0 < weight_1m and 0.0 < weight_15m):
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "reason": "invalid_weights",
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    prob_1m, status_1m = _predict_for_horizon(symbol, 1, db)
    prob_15m, status_15m = _predict_for_horizon(symbol, 15, db)

    components = [
        EnsembleComponent(
            horizon_label="1m",
            weight=float(weight_1m),
            prob_up=prob_1m,
            is_available=(prob_1m is not None),
        ),
        EnsembleComponent(
            horizon_label="15m",
            weight=float(weight_15m),
            prob_up=prob_15m,
            is_available=(prob_15m is not None),
        ),
    ]
    if not any(c.is_available for c in components):
        # Both models unavailable. Cold-start path — no usable signal.
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "no_available_components",
                "symbol": symbol,
                "models": {"1m": status_1m, "15m": status_15m},
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    result = ensemble_probability(components)
    if result.prob_up >= 0.55:
        signal = "up"
    elif result.prob_up <= 0.45:
        signal = "down"
    else:
        signal = "neutral"

    return JSONResponse(content=_scrub({
        "ok": True,
        "symbol": symbol,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "prob_up": result.prob_up,
        "signal": signal,
        "agreement": result.agreement,
        "n_components_used": result.n_components_used,
        "components": [
            {
                "horizon_label": c.horizon_label,
                "weight": c.weight,
                "prob_up": c.prob_up,
                "is_available": c.is_available,
            }
            for c in result.components
        ],
        "models": {"1m": status_1m, "15m": status_15m},
    }))


# ── Paper trades endpoint (Phase C UI block) ──────────────────────────────


# Default page size for the recent-trades list. Big enough that the UI's
# 24h sparkline + recent table don't both need separate queries; small
# enough that even after months of trading the response is a few KB.
DEFAULT_PAPER_TRADES_LIMIT: int = 50

# Hard cap to prevent a `?limit=99999999` browser-bug from melting the DB.
MAX_PAPER_TRADES_LIMIT: int = 500


def _fetch_recent_paper_trades(
    db: Session, symbol: str, limit: int,
) -> Optional[list[dict[str, Any]]]:
    """Return the ``limit`` most recent closed paper trades for ``symbol``.

    ``None`` distinguishes "DB unreachable / table missing" (caller
    surfaces 503) from "no trades yet" (caller returns empty list — a
    valid 200).
    """
    try:
        rows = db.execute(
            text(
                "SELECT id, symbol, side, qty, "
                "entry_ts, entry_price, entry_prob_up, "
                "exit_ts, exit_price, exit_reason, "
                "fee_paid_total_usd, pnl_usd, pnl_bps, "
                "model_version, written_at "
                "FROM paper_trades "
                "WHERE symbol = :symbol "
                "ORDER BY exit_ts DESC "
                "LIMIT :limit"
            ),
            {"symbol": symbol, "limit": int(limit)},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("paper_trades fetch failed (%s): %s", symbol, exc)
        return None
    return [dict(r) for r in rows]


def _fetch_paper_trades_total(db: Session, symbol: str) -> Optional[int]:
    """Lifetime trade count (for the "X trades total" badge). ``None``
    on DB error — caller treats as unknown without 503-ing the page."""
    try:
        return int(db.execute(
            text("SELECT COUNT(*) FROM paper_trades WHERE symbol = :symbol"),
            {"symbol": symbol},
        ).scalar() or 0)
    except (ProgrammingError, OperationalError):
        return None


def compute_paper_stats(
    *,
    trades_recent: list[dict[str, Any]],
    total_trades_lifetime: Optional[int],
    now: datetime,
    config: PaperTraderConfig,
    risk_caps: RiskCaps,
) -> dict[str, Any]:
    """Pure function — derive the stats block shown above the trades table.

    Designed to mirror what ``PaperTrader.status()`` would compute in
    the runner process, but reconstructed from history. There's a
    small race window when the runner has just opened a position (the
    DB doesn't see it until close) — the UI accepts that and labels
    the *open* position state via a separate snapshot endpoint if we
    ever need it. For Phase C this is fine: the UI is a *read-only
    log*, not a control surface.

    ``trades_recent`` must be sorted DESC by ``exit_ts`` (matches what
    :func:`_fetch_recent_paper_trades` returns).
    """
    # "Today" is computed in UTC since exit_ts is TIMESTAMPTZ stored as UTC.
    today_utc = now.astimezone(timezone.utc).date()

    trades_today = [
        t for t in trades_recent
        if t["exit_ts"].astimezone(timezone.utc).date() == today_utc
    ]
    today_pnl_usd = sum(float(t["pnl_usd"]) for t in trades_today)
    wins_today = sum(1 for t in trades_today if float(t["pnl_usd"]) >= 0)
    losses_today = len(trades_today) - wins_today

    # Consecutive losses ending at the most recent trade. Walk newest →
    # oldest; stop at the first non-losing trade.
    consecutive_losses = 0
    for t in trades_recent:
        if float(t["pnl_usd"]) < 0:
            consecutive_losses += 1
        else:
            break

    halted_reason: Optional[str] = None
    if consecutive_losses >= risk_caps.max_consecutive_losses:
        halted_reason = "loss_streak"
    elif today_pnl_usd <= -risk_caps.max_daily_loss_usd:
        halted_reason = "daily_loss"

    last_trade_ts = trades_recent[0]["exit_ts"] if trades_recent else None
    latest_model_version = (
        trades_recent[0]["model_version"] if trades_recent else None
    )

    # Cumulative P&L for the last N trades (newest first → reverse to
    # plot left-to-right). One pass; cheap.
    cumulative_pnl_series: list[dict[str, Any]] = []
    running = 0.0
    for t in reversed(trades_recent):
        running += float(t["pnl_usd"])
        cumulative_pnl_series.append({
            "exit_ts": t["exit_ts"].isoformat(),
            "cum_pnl_usd": round(running, 6),
        })

    return {
        "total_trades": total_trades_lifetime,
        "trades_today": len(trades_today),
        "today_pnl_usd": round(today_pnl_usd, 4),
        "wins_today": wins_today,
        "losses_today": losses_today,
        "consecutive_losses": consecutive_losses,
        "halted_reason": halted_reason,
        "last_trade_ts": last_trade_ts.isoformat() if last_trade_ts else None,
        "latest_model_version": latest_model_version,
        "cumulative_pnl_series": cumulative_pnl_series,
    }


def _serialise_trade(t: dict[str, Any]) -> dict[str, Any]:
    """asyncpg/sqlalchemy give us datetimes — JSON-encode them as ISO."""
    out = dict(t)
    for k in ("entry_ts", "exit_ts", "written_at"):
        v = out.get(k)
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    # Decimal → float for JSON.
    for k in (
        "qty", "entry_price", "entry_prob_up",
        "exit_price", "fee_paid_total_usd", "pnl_usd", "pnl_bps",
    ):
        v = out.get(k)
        if v is not None:
            out[k] = float(v)
    return out


@router.get("/api/highfreq/paper_trades")
async def get_paper_trades(
    symbol: str = DEFAULT_SYMBOL,
    limit: int = DEFAULT_PAPER_TRADES_LIMIT,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Recent closed paper trades + computed running stats.

    Returns 200 always (never 503): empty trades list is a valid state
    during the cold-start period before the trainer ships a model.
    Database unavailability degrades to ``trades=[]`` with a structured
    ``db_status`` flag — the UI can show "DB hiccup" without losing the
    page entirely.
    """
    symbol = symbol.upper()
    limit = max(1, min(int(limit), MAX_PAPER_TRADES_LIMIT))

    config = PaperTraderConfig()
    risk_caps = RiskCaps()

    trades_raw = _fetch_recent_paper_trades(db, symbol, limit)
    if trades_raw is None:
        # DB unreachable — return a soft-failed 200 so the UI keeps
        # rendering the rest of the page.
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "trades": [],
            "stats": None,
            "config": _config_to_dict(config, risk_caps),
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    total_lifetime = _fetch_paper_trades_total(db, symbol)
    now = datetime.now(tz=timezone.utc)

    stats = compute_paper_stats(
        trades_recent=trades_raw,
        total_trades_lifetime=total_lifetime,
        now=now,
        config=config,
        risk_caps=risk_caps,
    )

    return JSONResponse(content=_scrub({
        "ok": True,
        "db_status": "ok",
        "symbol": symbol,
        "trades": [_serialise_trade(t) for t in trades_raw],
        "stats": stats,
        "config": _config_to_dict(config, risk_caps),
        "ts": now.isoformat(),
    }))


def _config_to_dict(
    config: PaperTraderConfig, risk_caps: RiskCaps,
) -> dict[str, Any]:
    """Expose the trader's tunables to the UI so it can label thresholds
    correctly (e.g. "halted: ≥5 consecutive losses")."""
    return {
        "entry_long_threshold": config.entry_long_threshold,
        "entry_short_threshold": config.entry_short_threshold,
        "horizon_minutes": config.horizon_minutes,
        "max_qty_usd": config.max_qty_usd,
        "maker_fee_bps_per_side": config.maker_fee_bps_per_side,
        "max_consecutive_losses": risk_caps.max_consecutive_losses,
        "max_daily_loss_usd": risk_caps.max_daily_loss_usd,
    }


@router.get("/api/highfreq/realized_accuracy")
async def get_realized_accuracy(
    symbol: str = DEFAULT_SYMBOL,
    window: int = 100,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Rolling realized directional accuracy of the paper trader.

    Complementary to ``training_report`` — that endpoint surfaces
    walk-forward CV accuracy on **historical** data; this one shows
    accuracy on **trades that actually closed in production**.
    Defence-prep: if CV says 0.55 but realized is 0.49, a reviewer's
    obvious next question ("does it work in production?") gets a
    direct numeric answer instead of "trust the CV".

    Returns 200 even on cold-start (no trades yet) — the JSON includes
    a clear ``ok=true``/``n_eligible=0`` shape and the UI renders "—".
    Returns 503 only when the DB is unreachable.
    """
    from app.highfreq.realized_accuracy import (
        DEFAULT_WINDOW_SIZES,
        fetch_rolling_accuracy_sync,
    )

    symbol = symbol.upper()
    # Allow only sane window sizes — accept the documented defaults
    # (50 / 100) plus anything between 10 and the same hard cap we use
    # for the trades endpoint. Prevents ?window=99999 from scanning
    # the full table.
    if window <= 0:
        window = max(DEFAULT_WINDOW_SIZES)
    window = max(10, min(int(window), MAX_PAPER_TRADES_LIMIT))

    try:
        snap = fetch_rolling_accuracy_sync(db, symbol=symbol, window=window)
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("realized_accuracy fetch failed (%s): %s", symbol, exc)
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "db_status": "unavailable",
                "symbol": symbol,
                "window": window,
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    return JSONResponse(content={
        "ok": True,
        "db_status": "ok",
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        **snap.to_dict(),
    })


# ── Training report endpoint (Phase C.5 calibration progress widget) ──────


# Minimum minutes (post-neutral-band drop) for one walk-forward fold.
# Mirrors the trainer's check; surfaced to the UI so users can see
# "X / Y bars accumulated" without inspecting metrics.json by hand.
MIN_BARS_FOR_FIRST_FOLD: int = 1500

# Path to the metrics JSON the trainer writes alongside the .cbm.
# Single-symbol legacy path — overridable via env. Multi-symbol callers
# go through ``metrics_path_for_symbol`` instead.
DEFAULT_METRICS_PATH = Path(
    os.getenv("HIGHFREQ_METRICS_PATH", "weights/highfreq/btcusdt_1m_metrics.json")
)


def metrics_path_for_symbol(symbol: str, *, horizon: int = 1) -> Path:
    """Per-(symbol × horizon) metrics path — mirrors trainer's --report
    convention.

    e.g. ``("BTCUSDT", horizon=1)``  → ``weights/highfreq/btcusdt_1m_metrics.json``
         ``("BTCUSDT", horizon=15)`` → ``weights/highfreq/btcusdt_15m_metrics.json``

    Default horizon=1 preserves the original single-horizon contract.
    Lives next to the per-(symbol, horizon) .cbm; both paths share
    ``HIGHFREQ_WEIGHTS_DIR`` if overridden.
    """
    base = Path(os.getenv("HIGHFREQ_WEIGHTS_DIR", "weights/highfreq"))
    return base / f"{symbol.lower()}_{int(horizon)}m_metrics.json"


# Z-score for 95% two-sided confidence — standard normal inverse at 0.975.
# Inlined as a constant rather than scipy.stats import to avoid pulling
# scipy onto the slim Tokyo deploy.
_Z_95: float = 1.959963984540054


def wilson_ci(*, successes: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Better than the naive normal approximation when ``n`` is small or
    the proportion is close to 0/1 (which is exactly the regime we're
    in for per-fold dir_acc with n_test ≈ 60). Returns ``(lo, hi)`` in
    [0, 1].

    Reference: Wilson 1927; widely used in baseball SABR + clinical
    trials for exactly this "directional accuracy with small sample"
    case.
    """
    if n <= 0:
        return 0.0, 1.0
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _enrich_folds_with_ci(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add per-fold ``dir_acc_ci_low`` / ``dir_acc_ci_high`` via Wilson.

    The trainer's bootstrap CI is over ALL folds combined (one number
    for the whole report). For the calibration plot we want per-fold
    error bars to see if the model is consistently calibrated or just
    averages out across high-variance folds. Wilson is analytical from
    ``dir_acc`` + ``n_test`` so no extra inference passes needed.
    """
    out = []
    for f in folds:
        n_test = int(f.get("n_test") or 0)
        dir_acc = float(f.get("dir_acc") or 0.0)
        successes = int(round(dir_acc * n_test))
        ci_lo, ci_hi = wilson_ci(successes=successes, n=n_test)
        out.append({**f, "dir_acc_ci_low": ci_lo, "dir_acc_ci_high": ci_hi})
    return out


def compute_edge_vs_base_rate(
    dir_acc_mean: float | None,
    base_rate: float | None,
) -> float | None:
    """Model's directional accuracy minus the naive majority-class
    baseline.

    The naive classifier "always predict the more common class"
    achieves ``max(base_rate, 1 - base_rate)`` accuracy by definition.
    Subtracting that from the model's ``dir_acc_mean`` gives the
    edge — how much MORE the model captures than a 1-line if-else
    that ignores features entirely.

    Defence-grade interpretation:
    *   ``> 0`` : model has measurable directional skill beyond
        the trivial "guess the dominant class" classifier.
    *   ``≈ 0`` : model is no better than naive baseline. Could
        still be useful (provides probabilities, not just labels)
        but the directional claim is weak.
    *   ``< 0`` : model UNDERPERFORMS naive baseline. This is what
        we observed on BTC fold-0 (dir_acc 0.567 vs base_rate 0.583
        → edge -0.017). At small n it's noise; at large n it's a
        red flag that suggests feature engineering or training
        regression.

    Both inputs accept ``None`` / NaN; the helper returns ``None``
    in those cases so the API can render "—" rather than a
    misleading number.
    """
    if dir_acc_mean is None or base_rate is None:
        return None
    if not (0.0 <= dir_acc_mean <= 1.0) or not (0.0 <= base_rate <= 1.0):
        return None
    if math.isnan(dir_acc_mean) or math.isnan(base_rate):
        return None
    naive_baseline = max(base_rate, 1.0 - base_rate)
    return float(dir_acc_mean - naive_baseline)


# ── Microprice history endpoint (live chart, Phase C.5c) ──────────────


# Default chart window: 5 minutes = 300 seconds = 300 rows at 1 row/sec.
# Configurable per-request via ?seconds= but capped at 1 hour to keep
# the response under ~50 KB even on the slowest dial-up.
DEFAULT_HISTORY_SECONDS: int = 300
MAX_HISTORY_SECONDS: int = 3600


def _fetch_microprice_history(
    db: Session, symbol: str, seconds: int,
) -> Optional[list[dict[str, Any]]]:
    """Return ``[{ts, microprice}, ...]`` for the last ``seconds``.

    Lightweight (only 2 columns) — designed for the live chart's
    2-second polling cadence. Sorted ASC by ts so the chart can
    append without re-sorting.

    ``None`` distinguishes DB unavailable from "no rows yet".
    """
    try:
        rows = db.execute(
            text(
                "SELECT ts, microprice "
                "FROM highfreq_ofi_1s "
                "WHERE symbol = :symbol "
                "  AND ts > now() - make_interval(secs => :secs) "
                "ORDER BY ts ASC"
            ),
            {"symbol": symbol, "secs": int(seconds)},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("microprice_history fetch failed (%s): %s", symbol, exc)
        return None
    return [
        {
            "ts": r["ts"].isoformat() if isinstance(r["ts"], datetime) else r["ts"],
            "microprice": _to_float(r["microprice"]),
        }
        for r in rows
    ]


@router.get("/api/highfreq/microprice_history")
async def get_microprice_history(
    symbol: str = DEFAULT_SYMBOL,
    seconds: int = DEFAULT_HISTORY_SECONDS,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Recent microprice trajectory for the live chart on /highfreq.

    Returns up to ``seconds`` worth of 1-s rows. Empty list during
    cold-start / DB hiccup is a valid 200 — the chart just shows its
    empty-state placeholder.
    """
    symbol = symbol.upper()
    seconds = max(10, min(int(seconds), MAX_HISTORY_SECONDS))
    rows = _fetch_microprice_history(db, symbol, seconds)
    if rows is None:
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "rows": [],
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })
    return JSONResponse(content={
        "ok": True,
        "symbol": symbol,
        "rows": rows,
        "seconds": seconds,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    })


# ── Orderbook heatmap endpoint (Phase D.1) ────────────────────────────


# Default heatmap window: 5 minutes = 300 snapshots at 1 Hz. Matches
# the microprice chart's window so users compare side-by-side.
DEFAULT_OB_HEATMAP_SECONDS: int = 300
MAX_OB_HEATMAP_SECONDS: int = 1800  # 30 min — keeps response < 1 MB


def _fetch_orderbook_window(
    db: Session, symbol: str, seconds: int,
) -> Optional[list[dict[str, Any]]]:
    """Recent ``highfreq_l2_snapshots`` rows. Powers the canvas heatmap.

    Returns one dict per snapshot with the four price/qty arrays cast to
    Python lists (asyncpg/psycopg2 handle PG arrays natively, but
    SQLAlchemy returns them as Python lists already so no conversion
    needed).

    ``None`` distinguishes DB unavailable from "no rows yet" (cold-start
    or HIGHFREQ_STORE_L2_SNAPSHOTS=0 deploy).
    """
    try:
        rows = db.execute(
            text(
                "SELECT ts, bids_price, bids_qty, asks_price, asks_qty "
                "FROM highfreq_l2_snapshots "
                "WHERE symbol = :symbol "
                "  AND ts > now() - make_interval(secs => :secs) "
                "ORDER BY ts ASC"
            ),
            {"symbol": symbol, "secs": int(seconds)},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("orderbook fetch failed (%s): %s", symbol, exc)
        return None
    out = []
    for r in rows:
        out.append({
            "ts": r["ts"].isoformat() if isinstance(r["ts"], datetime) else r["ts"],
            "bids_price": list(r["bids_price"] or []),
            "bids_qty": list(r["bids_qty"] or []),
            "asks_price": list(r["asks_price"] or []),
            "asks_qty": list(r["asks_qty"] or []),
        })
    return out


@router.get("/api/highfreq/orderbook")
async def get_orderbook(
    symbol: str = DEFAULT_SYMBOL,
    seconds: int = DEFAULT_OB_HEATMAP_SECONDS,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Recent L2 snapshots for the heatmap UI.

    Returns 200 always; empty rows during cold-start / writer-disabled
    is a valid state surfaced via empty ``rows`` array. The UI shows
    its own "ждём данные…" placeholder.
    """
    symbol = symbol.upper()
    seconds = max(10, min(int(seconds), MAX_OB_HEATMAP_SECONDS))

    rows = _fetch_orderbook_window(db, symbol, seconds)
    if rows is None:
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "rows": [],
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })
    return JSONResponse(content={
        "ok": True,
        "symbol": symbol,
        "seconds": seconds,
        "n_snapshots": len(rows),
        "rows": rows,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    })


# ── Regime detection endpoint (Phase C.5e) ────────────────────────────


def _fetch_minute_vol_history(
    db: Session, symbol: str, hours: float,
) -> Optional[list[dict[str, Any]]]:
    """Per-minute aggregated vol (microprice high-low in bps) for the
    last ``hours``. Drives the regime classifier — separate from the
    per-second microprice history endpoint to keep that one cheap.

    Computes vol on-the-fly via SQL: (max - min) / first × 10000 bps,
    grouped by date_trunc('minute', ts). One row per minute.
    """
    try:
        rows = db.execute(
            text(
                "WITH per_min AS ("
                "  SELECT date_trunc('minute', ts) AS minute, "
                "         min(microprice) AS lo, max(microprice) AS hi, "
                "         (array_agg(microprice ORDER BY ts ASC))[1] AS first_mp "
                "  FROM highfreq_ofi_1s "
                "  WHERE symbol = :symbol "
                # make_interval(hours=>int) is strict; multiply seconds
                # to support fractional hours without explicit cast.
                "    AND ts > now() - make_interval(secs => :secs) "
                "  GROUP BY 1"
                ") "
                "SELECT minute AS ts, "
                "       CASE WHEN first_mp > 0 "
                "            THEN ((hi - lo) / first_mp) * 1e4 "
                "            ELSE 0 END AS vol_bps "
                "  FROM per_min "
                " ORDER BY minute ASC"
            ),
            {"symbol": symbol, "secs": int(float(hours) * 3600)},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("minute vol history fetch failed (%s): %s", symbol, exc)
        return None
    return [
        {
            "ts": r["ts"].isoformat() if isinstance(r["ts"], datetime) else r["ts"],
            "vol_bps": _to_float(r["vol_bps"]) or 0.0,
        }
        for r in rows
    ]


@router.get("/api/highfreq/regimes")
async def get_regimes(
    symbol: str = DEFAULT_SYMBOL,
    hours: float = 24.0,
    keep_last_n: int = 60,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Vol-regime breakdown for the last ``hours``.

    UI shows: current regime label (calm/normal/turbulent) coloured
    pill, last ``keep_last_n`` bars as a tiny stripe chart, and the
    percentile thresholds the classifier is using.

    Returns 200 always; empty history (cold-start) is a valid state
    surfaced via ``current=null``.
    """
    from app.highfreq.regimes import label_history

    symbol = symbol.upper()
    hours = max(1.0, min(168.0, float(hours)))   # 1h .. 7d
    keep_last_n = max(10, min(500, int(keep_last_n)))

    history = _fetch_minute_vol_history(db, symbol, hours)
    if history is None:
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "regime": None,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    breakdown = label_history(history, keep_last_n=keep_last_n)
    return JSONResponse(content=_scrub({
        "ok": True,
        "symbol": symbol,
        "hours_lookback": hours,
        "regime": breakdown.to_dict(),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


# ── Feature importance endpoint (Phase C.5d) ──────────────────────────


@router.get("/api/highfreq/feature_importance")
async def get_feature_importance(
    symbol: str = DEFAULT_SYMBOL,
    predictor: LivePredictor = Depends(_get_forecast_predictor),
) -> JSONResponse:
    """Per-feature importance from the loaded CatBoost model.

    Used by the UI to render a horizontal bar chart — "which features
    is the model actually using?". Constant for a given ``.cbm`` so
    the UI can poll once on page load + on model-mtime change rather
    than every 5 s.

    Returns 200 always: empty/None ``importance`` is a valid no-model
    state surfaced via ``ok=False``.
    """
    symbol = symbol.upper()
    pairs = predictor.feature_importance()
    status = predictor.status()
    if pairs is None:
        return JSONResponse(content={
            "ok": False,
            "reason": "no_model" if not status.has_model else "unsupported",
            "symbol": symbol,
            "importance": [],
            "model": status.to_dict(),
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })
    return JSONResponse(content=_scrub({
        "ok": True,
        "symbol": symbol,
        "importance": pairs,
        "model": status.to_dict(),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


@router.get("/api/highfreq/training_report")
async def get_training_report(
    symbol: str = DEFAULT_SYMBOL,
    horizon: int = 1,
    lite: int = 0,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Return the trainer's last metrics report + computed fold-readiness.

    Used by the UI to render two things: (a) the calibration status
    badge ("dir_acc 0.547 [0.521, 0.572]" or "low directional skill"),
    (b) the "X / Y bars for next fold" progress widget — so the user
    sees ramp-up progress visually rather than digging into journals.

    Real-time fold readiness
    ------------------------

    Older revisions of this endpoint sourced ``fold_ready_pct`` from the
    trainer's ``metrics.json`` — meaning the progress bar was frozen
    until the next daily firing. We now compute a **live inventory**
    via :func:`app.highfreq.data_inventory.fetch_live_inventory` (cached
    30 s per symbol) and surface it as ``live_inventory`` in the response.

    ``lite=1`` query parameter (release T.3, 2026-04-29)
    -----------------------------------------------------

    The live-inventory query reads up to 7 days of 1-second rows from
    Postgres on a cache miss (~600k rows). Through the WG tunnel from
    Finland to Tokyo this can hit 10-20 seconds of TTFB on a cold
    cache, blocking page-render for any UI that fans out
    training_report calls per-symbol.

    The public-facing /forecast page only needs metrics.json (dir_acc,
    p-value, n_folds) — it has no progress widget. Pass ``?lite=1`` to
    skip the live_inventory computation entirely; response drops the
    ``live_inventory`` and ``fold_ready_pct`` fields. Operator-facing
    /highfreq omits the flag (gets the full payload as before).

    Returns 200 with ``ok=False, reason="no_report_yet"`` if the
    trainer hasn't written its first ``metrics.json`` yet (cold-start).
    Never 503 — the page must keep rendering through trainer outages.
    """
    symbol = symbol.upper()
    skip_live = bool(lite)
    if horizon <= 0:
        horizon = 1

    # Live inventory ALWAYS computed when lite=0 — the progress widget
    # on /highfreq should work even before the trainer has fired its
    # first run. Failure of the live computation degrades gracefully
    # (None) without 503-ing the whole endpoint.
    live: dict[str, Any] | None = None
    if not skip_live:
        try:
            from app.highfreq.data_inventory import fetch_live_inventory
            live_snap = fetch_live_inventory(db, symbol=symbol)
            live = live_snap.to_dict()
        except (ProgrammingError, OperationalError) as exc:
            logger.warning("live_inventory query failed (%s): %s", symbol, exc)
        except Exception as exc:
            logger.warning("live_inventory unexpected failure (%s): %s", symbol, exc)

    # Use per-(symbol, horizon) path. Default horizon=1 preserves the
    # legacy single-horizon contract for callers that don't pass it.
    path = metrics_path_for_symbol(symbol, horizon=horizon)
    if not path.exists() and horizon == 1:
        # Try the legacy override path as a last resort (catches cases
        # where the trainer was invoked with --report at a custom path).
        if DEFAULT_METRICS_PATH.exists():
            path = DEFAULT_METRICS_PATH
    if not path.exists():
        # Cold start: no report file yet, but live_inventory may still
        # be populated. Compute fold_ready_pct from live data when
        # available so the UI shows the progress bar from minute 1.
        live_pct: float | None = None
        if live is not None:
            live_pct = min(
                100.0,
                100.0 * int(live.get("n_eligible_for_training") or 0) / MIN_BARS_FOR_FIRST_FOLD,
            )
        return JSONResponse(content=_scrub({
            "ok": False,
            "reason": "no_report_yet",
            "metrics_path": str(path),
            "min_bars_for_first_fold": MIN_BARS_FOR_FIRST_FOLD,
            "live_inventory": live,
            "fold_ready_pct": round(live_pct, 1) if live_pct is not None else 0.0,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }))

    try:
        report = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("training_report read failed: %s", exc)
        return JSONResponse(content={
            "ok": False,
            "reason": "report_unreadable",
            "metrics_path": str(path),
            "live_inventory": live,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    # Fold readiness — TOTAL bars collected so far, regardless of how
    # they'd split into train vs holdout. This is the user-facing
    # "are we there yet?" number; the holdout-split detail goes into
    # ``live_inventory`` for the curious.
    #
    # Why not use ``n_eligible_for_training``: during the first 7 days
    # of operation, the entire dataset is recent enough to fall inside
    # the holdout window — eligible=0 would mean the bar shows 0 % all
    # week, which is unhelpful and makes the system look broken.
    # Showing collection progress against the same denominator the
    # trainer uses (``MIN_BARS_FOR_FIRST_FOLD``) lets users watch the
    # ramp-up minute by minute.
    if live is not None:
        bars_for_pct = int(live.get("n_minutes_after_neutral_drop") or 0)
    else:
        bars_for_pct = int(report.get("n_minutes_after_neutral_drop") or 0)
    fold_ready_pct = min(100.0, 100.0 * bars_for_pct / MIN_BARS_FOR_FIRST_FOLD)

    # Enrich per-fold dir_acc with Wilson CI for the calibration plot.
    folds_raw = report.get("folds") or []
    if folds_raw:
        report = {**report, "folds": _enrich_folds_with_ci(folds_raw)}

    # Age of the report — UI shows "trained 4 hours ago" so users can
    # tell stale weights from fresh ones.
    try:
        report_age_seconds = max(
            0.0, datetime.now(tz=timezone.utc).timestamp() - path.stat().st_mtime,
        )
    except OSError:
        report_age_seconds = None

    # Edge over naive majority-class baseline. Computed on the report
    # values (not live_inventory — naive baseline is a model-skill
    # metric, not a data-collection metric).
    edge_vs_base_rate = compute_edge_vs_base_rate(
        dir_acc_mean=report.get("dir_acc_mean"),
        base_rate=report.get("base_rate"),
    )

    return JSONResponse(content=_scrub({
        "ok": True,
        "report": report,
        "report_age_seconds": report_age_seconds,
        "min_bars_for_first_fold": MIN_BARS_FOR_FIRST_FOLD,
        "fold_ready_pct": round(fold_ready_pct, 1),
        "live_inventory": live,
        "edge_vs_base_rate": edge_vs_base_rate,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


@router.get("/api/highfreq/realized_accuracy_full")
async def get_realized_accuracy_full(
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Full realized directional accuracy with Wilson CI + p-value
    per symbol, computed over ALL minute-level predictions in
    ``predictions_log``.

    Critical defence-grade endpoint: the walk-forward CV in the
    trainer reports skill on n≈120 (2 folds × 60 OOS bars), giving
    a wide CI that crosses 0.5 — looks like noise. The realized
    accuracy on the FULL ``predictions_log`` operates on n>2000,
    tightening the CI by 3-4× and revealing whether the model has
    statistically detectable directional skill.

    Excludes neutral signals (no directional claim made) and rows
    where the t+1 microprice hasn't yet been backfilled.

    Returns 200 always; DB error degrades to ``ok=False``.
    """
    import math
    try:
        from scipy.stats import binomtest as _binomtest
    except ImportError:
        _binomtest = None

    sql = text(
        "SELECT symbol, "
        "       COUNT(*) FILTER (WHERE realized_correct IS NOT NULL) AS directional, "
        "       COUNT(*) FILTER (WHERE realized_correct IS TRUE)     AS hits "
        "  FROM predictions_log "
        " GROUP BY symbol "
        " ORDER BY symbol"
    )
    try:
        raw_rows = db.execute(sql).all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("realized_accuracy_full failed: %s", exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    def _wilson(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return 0.0, 1.0
        p = k / n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, centre - half), min(1.0, centre + half)

    out: list[dict[str, Any]] = []
    for r in raw_rows:
        sym = r[0]
        n = int(r[1] or 0)
        hits = int(r[2] or 0)
        if n == 0:
            out.append({
                "symbol": sym, "n": 0, "hits": 0,
                "dir_acc": None, "ci_low": None, "ci_high": None,
                "p_value": None,
                "verdict": "no_data",
            })
            continue
        acc = hits / n
        lo, hi = _wilson(hits, n)
        p_val: float | None
        if _binomtest is not None:
            p_val = float(_binomtest(k=hits, n=n, p=0.5, alternative="greater").pvalue)
        else:
            p_val = None
        # Verdict per defence-grade convention
        if p_val is not None and p_val < 0.001 and lo > 0.5:
            verdict = "definitive_skill"
        elif lo > 0.5:
            verdict = "borderline_skill"
        elif p_val is not None and p_val > 0.99:
            verdict = "anti_skill"
        else:
            verdict = "noise"
        out.append({
            "symbol": sym,
            "n": n,
            "hits": hits,
            "dir_acc": acc,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p_val,
            "verdict": verdict,
        })

    return JSONResponse(content=_scrub({
        "ok": True,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "rows": out,
    }))


@router.get("/api/highfreq/conditional_accuracy")
async def get_conditional_accuracy(
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Realized directional accuracy bucketed by confidence threshold.

    The unconditional dir_acc on a calibrated 1-min direction model
    typically lands at 53-56% — close-to-chance because half the
    predictions are mild-confidence noise. The CALIBRATED model's
    Platt-fit posteriors mean: on bars where ``prob_up`` is far from
    0.5, the realized hit-rate climbs sharply. This endpoint surfaces
    that "when the model is confident, how often is it right?" answer.

    Buckets are nested (a bar in the 65% bucket is also in 60% / 55%
    counts) so the UI can show a monotone descending series:
    "conf ≥ 65% → 57.2% acc; conf ≥ 60% → 54.8%; conf ≥ 55% → 53.4%".

    Returns 200 always; DB error degrades to ``ok=False``.
    """
    import math
    try:
        from scipy.stats import binomtest as _binomtest
    except ImportError:
        _binomtest = None

    # Confidence thresholds → "abs(prob_up - 0.5) >= threshold". Each
    # bucket is the FULL set of bars at-or-above that confidence
    # (nested, not exclusive).
    THRESHOLDS = [0.05, 0.10, 0.15]   # = prob ≥ 0.55 / 0.60 / 0.65
    LABELS = ["conf_55", "conf_60", "conf_65"]

    sql = text(
        "SELECT symbol, "
        "       COUNT(*) FILTER (WHERE realized_correct IS NOT NULL "
        "                        AND ABS(prob_up - 0.5) >= :t) AS n, "
        "       COUNT(*) FILTER (WHERE realized_correct IS TRUE "
        "                        AND ABS(prob_up - 0.5) >= :t) AS hits "
        "  FROM predictions_log "
        " GROUP BY symbol "
        " ORDER BY symbol"
    )

    def _wilson(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return 0.0, 1.0
        p = k / n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, centre - half), min(1.0, centre + half)

    by_symbol: dict[str, dict[str, Any]] = {}
    try:
        for label, t in zip(LABELS, THRESHOLDS):
            for r in db.execute(sql, {"t": t}).all():
                sym = r[0]
                n = int(r[1] or 0)
                hits = int(r[2] or 0)
                if sym not in by_symbol:
                    by_symbol[sym] = {"symbol": sym, "buckets": {}}
                if n == 0:
                    by_symbol[sym]["buckets"][label] = {
                        "threshold": t, "n": 0, "hits": 0,
                        "dir_acc": None, "ci_low": None, "ci_high": None,
                        "p_value": None,
                    }
                    continue
                acc = hits / n
                lo, hi = _wilson(hits, n)
                p_val: float | None = None
                if _binomtest is not None:
                    p_val = float(
                        _binomtest(k=hits, n=n, p=0.5, alternative="greater").pvalue
                    )
                by_symbol[sym]["buckets"][label] = {
                    "threshold": t,
                    "n": n,
                    "hits": hits,
                    "dir_acc": acc,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p_val,
                }
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("conditional_accuracy failed: %s", exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    return JSONResponse(content=_scrub({
        "ok": True,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "rows": list(by_symbol.values()),
    }))


@router.get("/api/highfreq/cumulative_pnl")
async def get_cumulative_pnl(
    symbol: str = DEFAULT_SYMBOL,
    limit_points: int = 200,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Cumulative P&L curve from ``paper_trades``, by fee tier.

    Returns a chronological time-series suitable for a line chart on
    /forecast: "this is what bp-by-bp the model has done since paper-
    trader spawn, per fee tier".

    Per trade we already have ``pnl_bps`` (gross of fees). For each
    fee tier we recompute net by subtracting ``2 × fee_bps`` (round-
    trip cost), then accumulate. ``limit_points`` (default 200) caps
    the curve length — we evenly downsample to keep the JSON payload
    small without distorting the shape.

    Tier definitions match the UI's ``TIER_DEFS`` so the legend lines
    up. Server-side recomputation matters because we want a single
    source-of-truth for the fee math (the UI currently does the same
    math client-side; server replication makes the test surface
    smaller).

    Returns 200 always; DB error → ``ok=False``.
    """
    symbol = symbol.upper()
    if not (10 <= limit_points <= 1000):
        limit_points = 200

    # Mirrors templates/forecast.html TIER_DEFS — keep in sync.
    TIERS = [
        {"key": "gross",     "fee_bps": 0.0,   "name": "Без комиссии"},
        {"key": "retail",    "fee_bps": 7.5,   "name": "Spot retail"},
        {"key": "vip5",      "fee_bps": 1.0,   "name": "Spot VIP-5"},
        {"key": "vip9",      "fee_bps": 0.0,   "name": "Spot VIP-9"},
        {"key": "futures",   "fee_bps": 2.0,   "name": "Futures maker"},
        {"key": "mm_rebate", "fee_bps": -0.4,  "name": "MM rebate"},
    ]

    # Pull entry/exit prices + side so we can RECONSTRUCT gross_bps
    # — the stored ``pnl_bps`` already has retail fees deducted (see
    # ``app.highfreq.paper_trader.compute_pnl``), so subtracting fees
    # again here would double-count. The UI's ``meanGrossBps`` does
    # the same reconstruction.
    sql = text(
        "SELECT exit_ts, entry_price, exit_price, side "
        "  FROM paper_trades "
        " WHERE symbol = :symbol "
        "   AND exit_ts IS NOT NULL "
        "   AND entry_price IS NOT NULL AND entry_price > 0 "
        "   AND exit_price IS NOT NULL "
        "   AND side IN ('long', 'short') "
        " ORDER BY exit_ts ASC"
    )
    try:
        raw_rows = db.execute(sql, {"symbol": symbol}).all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("cumulative_pnl failed: %s", exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    n_trades = len(raw_rows)
    if n_trades == 0:
        return JSONResponse(content={
            "ok": True,
            "symbol": symbol,
            "n_trades": 0,
            "first_trade_ts": None,
            "last_trade_ts": None,
            "tiers": [
                {**t, "final_bps": 0.0, "win_rate": None}
                for t in TIERS
            ],
            "points": [],
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    # Compute per-tier cumulative series.
    cumul = {t["key"]: 0.0 for t in TIERS}
    wins = {t["key"]: 0 for t in TIERS}
    full_points: list[dict[str, Any]] = []
    for r in raw_rows:
        ts_iso = r[0].isoformat() if r[0] is not None else None
        entry = float(r[1])
        exit_ = float(r[2])
        side = str(r[3])
        # Gross bps return on the trade, sign-corrected for direction.
        move_bps = ((exit_ - entry) / entry) * 1e4
        side_mul = 1.0 if side == "long" else -1.0
        gross_bps = side_mul * move_bps
        point = {"ts": ts_iso}
        for t in TIERS:
            net_bps = gross_bps - 2.0 * t["fee_bps"]
            cumul[t["key"]] += net_bps
            if net_bps > 0:
                wins[t["key"]] += 1
            point[t["key"]] = round(cumul[t["key"]], 4)
        full_points.append(point)

    # Downsample if needed — keep first + last, evenly stride the rest.
    if len(full_points) > limit_points:
        # Stride preserves order + always includes the last point.
        # Picks indices [0, S, 2S, ..., n-1] where S = ceil(n / limit).
        n = len(full_points)
        stride = max(1, (n - 1) // (limit_points - 1))
        idx = list(range(0, n, stride))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        points = [full_points[i] for i in idx]
    else:
        points = full_points

    tiers_out = [
        {
            **t,
            "final_bps": round(cumul[t["key"]], 4),
            "win_rate": round(wins[t["key"]] / n_trades, 4),
        }
        for t in TIERS
    ]

    return JSONResponse(content=_scrub({
        "ok": True,
        "symbol": symbol,
        "n_trades": n_trades,
        "n_points": len(points),
        "first_trade_ts": full_points[0]["ts"],
        "last_trade_ts": full_points[-1]["ts"],
        "tiers": tiers_out,
        "points": points,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


@router.get("/api/highfreq/reliability_diagram")
async def get_reliability_diagram(
    db: Session = Depends(_get_db),
    n_bins: int = 10,
) -> JSONResponse:
    """Calibration / reliability curve per symbol.

    For a calibrated probabilistic classifier, ``P(realized=1 | prob=p)``
    should equal ``p`` for every p ∈ [0, 1]. The reliability diagram
    plots empirical realized-rate vs predicted-probability across
    equal-width bins — a perfectly calibrated model lies on the
    diagonal y = x.

    This is the canonical defence-grade plot for "is your model
    actually telling the truth about its uncertainty?". A model with
    high dir_acc but a hump-shaped curve is overconfident; one with
    low dir_acc but on-diagonal is honest. Both matter for trading.

    Bucketing logic: equal-width bins of ``prob_up`` from 0 to 1.
    Per bin: count predictions, count realized=1, compute realized
    rate. Bins with n=0 are returned with realized_rate=null so the
    UI can plot a gap (don't pretend we have data we don't).

    Brier score and ECE (expected calibration error) per symbol are
    also returned for the academic-defence-grade single-number
    summary alongside the curve.

    Returns 200 always; DB error degrades to ``ok=False``.
    """
    import math

    if not (3 <= n_bins <= 30):
        n_bins = 10

    # Pull ALL predictions+realizations per symbol. We stream at the
    # python level rather than try to bucket in SQL — the bucket
    # boundaries are clean numerical operations on the prob_up float.
    sql = text(
        "SELECT symbol, prob_up, realized_correct, signal "
        "  FROM predictions_log "
        " WHERE realized_correct IS NOT NULL"
    )
    try:
        raw_rows = db.execute(sql).all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("reliability_diagram failed: %s", exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    # Build per-symbol bucketing.
    by_symbol: dict[str, dict[str, Any]] = {}
    for r in raw_rows:
        sym = r[0]
        prob = float(r[1])
        # ``realized_correct`` is bool; convert to {0, 1} of "realized
        # direction matched the predicted direction". We need
        # realized_outcome ∈ {0, 1} where 1 = price went UP. That's
        # different from realized_correct (which is True if model's
        # prediction matched). Reconstruct: signal='up' AND correct →
        # outcome=1; signal='down' AND correct → outcome=0;
        # signal='up' AND wrong → outcome=0; signal='down' AND wrong
        # → outcome=1. Skip 'neutral' (no direction was bet on).
        signal = (r[3] or "").lower()
        if signal not in ("up", "down"):
            continue
        correct = bool(r[2])
        if signal == "up":
            outcome = 1 if correct else 0
        else:  # signal == "down"
            outcome = 0 if correct else 1
        # The prob we're calibrating is P(price goes up | x). The
        # predictor.signal is "up" iff prob_up >= 0.55, "down" iff
        # prob_up <= 0.45. Either way, prob_up reflects the model's
        # belief in price-up — that's what we plot vs realized "up"
        # outcome.
        if sym not in by_symbol:
            by_symbol[sym] = {
                "symbol": sym,
                "n_total": 0,
                "buckets": [
                    {
                        "bin_idx": i,
                        "p_lo": i / n_bins,
                        "p_hi": (i + 1) / n_bins,
                        "p_mid": (i + 0.5) / n_bins,
                        "n": 0, "n_pos": 0,
                        "predicted_mean": 0.0,
                        "_p_sum": 0.0,
                    }
                    for i in range(n_bins)
                ],
                "_brier_sum": 0.0,
            }
        sym_state = by_symbol[sym]
        # Bucket index. Clamp prob to [0, 1) so bin = n_bins - 1 for prob = 1.
        bin_idx = min(n_bins - 1, max(0, int(prob * n_bins)))
        b = sym_state["buckets"][bin_idx]
        b["n"] += 1
        b["n_pos"] += outcome
        b["_p_sum"] += prob
        sym_state["n_total"] += 1
        sym_state["_brier_sum"] += (prob - outcome) ** 2

    # Finalise bucket aggregates + ECE.
    out_rows: list[dict[str, Any]] = []
    for sym, state in by_symbol.items():
        n_total = state["n_total"]
        if n_total == 0:
            out_rows.append({
                "symbol": sym, "n_total": 0,
                "brier": None, "ece": None,
                "buckets": [
                    {
                        "bin_idx": b["bin_idx"],
                        "p_mid": b["p_mid"],
                        "p_lo": b["p_lo"], "p_hi": b["p_hi"],
                        "n": 0, "realized_rate": None,
                        "predicted_mean": None,
                    }
                    for b in state["buckets"]
                ],
            })
            continue
        ece = 0.0
        for b in state["buckets"]:
            if b["n"] > 0:
                b["realized_rate"] = b["n_pos"] / b["n"]
                b["predicted_mean"] = b["_p_sum"] / b["n"]
                # ECE contribution: (n_bin / n_total) * |realized - predicted|
                ece += (b["n"] / n_total) * abs(
                    b["realized_rate"] - b["predicted_mean"]
                )
            else:
                b["realized_rate"] = None
                b["predicted_mean"] = None
            # Drop the internal accumulator before returning.
            b.pop("_p_sum", None)
        brier = state["_brier_sum"] / n_total
        out_rows.append({
            "symbol": sym,
            "n_total": n_total,
            "brier": brier,
            "ece": ece,
            "buckets": state["buckets"],
        })

    out_rows.sort(key=lambda r: r["symbol"])
    return JSONResponse(content=_scrub({
        "ok": True,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "n_bins": n_bins,
        "rows": out_rows,
    }))


@router.get("/api/highfreq/robustness")
async def get_robustness(
    symbol: str = DEFAULT_SYMBOL,
) -> JSONResponse:
    """Robustness suite results: block bootstrap CI + permutation test
    + per-day stability + per-hour heatmap + regime-conditional accuracy.

    The trainer's walk-forward CI uses i.i.d. bootstrap (assumes
    independent samples → too narrow for autocorrelated time series)
    and a binomial p-value (tests against a fair-coin null, the
    weakest possible H₀).  This endpoint serves the output of
    :mod:`tools.robustness_suite` which closes both gaps:

    * **Block bootstrap CI** (60-min blocks) — preserves
      autocorrelation; canonical Politis-Romano estimator.
    * **Permutation test** — shuffles ``y_true`` 1000× to build a
      null distribution of dir_acc.  Compares the observed point.
      p < 0.01 means "this dir_acc couldn't plausibly come from a
      labels-permuted run".
    * **Per-day / per-hour / per-regime accuracy** — slices the
      realized predictions to expose regime sensitivity.

    File contract: ``weights/highfreq/<symbol>_1m_robustness.json``
    written by ``python -m tools.robustness_suite --symbol ...``.
    Returns 200 always; missing file → ``ok=False`` with reason.
    """
    sym = symbol.upper()
    path = Path(f"weights/highfreq/{sym.lower()}_1m_robustness.json")
    if not path.exists():
        return JSONResponse(content={
            "ok": False,
            "reason": "no_robustness_run_yet",
            "symbol": sym,
            "hint": (
                "run `python -m tools.robustness_suite --symbol "
                f"{sym}` to generate"
            ),
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("robustness file read failed for %s: %s", sym, exc)
        return JSONResponse(content={
            "ok": False,
            "reason": "malformed",
            "symbol": sym,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })
    return JSONResponse(content=_scrub({
        "ok": True,
        "report": payload,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


@router.get("/api/highfreq/anti_skill")
async def get_anti_skill(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Anti-skill detector — surfaces "model is statistically WORSE
    than chance on directional calls".

    Returns the same :class:`AntiSkillReport` shape the runner uses
    each tick. UI / Telegram / operators can read this without
    needing live access to the runner's process state.

    Never 503: DB error degrades to ``ok=False, db_status="unavailable"``
    so the page keeps rendering.
    """
    from app.highfreq.anti_skill_detector import fetch_anti_skill_sync

    symbol = symbol.upper()
    try:
        report = fetch_anti_skill_sync(db, symbol=symbol)
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("anti_skill fetch failed (%s): %s", symbol, exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })
    return JSONResponse(content=_scrub({
        "ok": True,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        **report.to_dict(),
    }))


@router.get("/api/highfreq/pnl_by_fee_tier")
async def get_pnl_by_fee_tier(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """P&L of all closed paper trades, recomputed at each fee tier.

    Defence-grade artefact: at retail tier the trader loses money on
    every trade due to 15 bp roundtrip fees on 3-5 bp moves. At
    VIP-9 (0 bp) or market-maker rebate (-0.4 bp) the same trade
    sequence flips to positive (or breakeven). Shows that the model's
    directional skill exists; profitability is determined by fee
    tier (which scales with capital, not code).

    Returns 200 always:
    * cold start (no trades) → ``ok=True, tiers=[]`` so the UI can
      render "no data" without breaking.
    * DB error → ``ok=False, db_status="unavailable"``.
    """
    from app.highfreq.fee_tiers import summarise_all_tiers

    symbol = symbol.upper()
    try:
        rows = db.execute(
            text(
                "SELECT side, qty, entry_price, exit_price "
                "  FROM paper_trades "
                " WHERE symbol = :symbol "
                " ORDER BY exit_ts ASC"
            ),
            {"symbol": symbol},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("pnl_by_fee_tier fetch failed (%s): %s", symbol, exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    trades = [dict(r) for r in rows]
    summaries = summarise_all_tiers(trades)
    return JSONResponse(content=_scrub({
        "ok": True,
        "symbol": symbol,
        "n_trades": len(trades),
        "tiers": [s.to_dict() for s in summaries],
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


@router.get("/api/highfreq/actionable_signal")
async def get_actionable_signal(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Level-4 trade-signal payload — what the trader WOULD do right
    now with the latest predictor output.

    Composes:
    * Latest row from ``predictions_log`` (the predictor's last word).
    * Trader's threshold logic + position sizer (pure).
    * Calibration / demo-mode flags from env + predictor status.

    Returns 200 always:
    * cold start (no predictions yet) → ``ok=False, reason="no_predictions_yet"``.
    * DB error → ``ok=False, db_status="unavailable"``.
    * normal → full decision object.

    The UI renders this as an actionable "trade card" alongside the
    existing Forecast block — defence-grade artefact showing the
    complete decision pipeline end-to-end.
    """
    import os
    from app.highfreq.actionable_signal import (
        compute_decision,
        fetch_latest_prediction_sync,
    )
    from app.highfreq.paper_trader import PaperTraderConfig

    symbol = symbol.upper()

    try:
        latest = fetch_latest_prediction_sync(db, symbol=symbol)
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("actionable_signal fetch failed (%s): %s", symbol, exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    if latest is None:
        return JSONResponse(content={
            "ok": False,
            "reason": "no_predictions_yet",
            "symbol": symbol,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    # Demo mode → require_calibrated=False (mirrors paper_trader_runner).
    demo_mode = os.environ.get("HF_PAPER_DEMO_MODE", "").strip() in ("1", "true", "yes")
    config = PaperTraderConfig(require_calibrated=not demo_mode)

    # Trader's calibration state lives in the predictor; we approximate
    # via the model_version tag on the prediction (pre-calibration-demo
    # ⇔ uncalibrated). Future improvement: include a calibrated flag in
    # predictions_log directly.
    calibrated = (latest["model_version"] != "pre-calibration-demo")

    decision = compute_decision(
        prob_up=float(latest["prob_up"]),
        microprice=float(latest["microprice"]),
        ts=latest["ts"],
        config=config,
        model_version=latest["model_version"],
        calibrated=calibrated,
        demo_mode=demo_mode,
        halted_reason=None,  # halt state is per-runner; not surfaced here yet
    )

    return JSONResponse(content=_scrub({
        "ok": True,
        "symbol": symbol,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "decision": decision,
        "config": {
            "entry_long_threshold": config.entry_long_threshold,
            "entry_short_threshold": config.entry_short_threshold,
            "horizon_minutes": config.horizon_minutes,
            "max_qty_usd": config.max_qty_usd,
            "maker_fee_bps_per_side": config.maker_fee_bps_per_side,
            "require_calibrated": config.require_calibrated,
        },
    }))


@router.get("/api/highfreq/predictions_history")
async def get_predictions_history(
    symbol: str = DEFAULT_SYMBOL,
    since_minutes: int = 60,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Last N minutes of model predictions for the live signal tape.

    Backed by ``predictions_log`` (migration 005). One row per minute
    per symbol — captures EVERY forecast, not just the ones the
    paper-trader acted on. UI uses this to render a tape of recent
    signals with timestamps + price; future use cases include the
    Telegram bot replay on user start, and "model skill vs trader
    skill" cohort analysis (predictions that the trader skipped due
    to halt or neutral signal still reflect model directional
    output).

    Read-only, never 503 — degrades to ``rows=[]`` on DB error.
    """
    from app.highfreq.predictions_log import fetch_history_sync

    symbol = symbol.upper()
    # Sane bounds: 1 min .. 24 h. Past 24h is not realistically useful
    # for a "recent tape" widget; the query would also return 1500+
    # rows and slow the page.
    since_minutes = max(1, min(int(since_minutes), 24 * 60))

    try:
        rows = fetch_history_sync(db, symbol=symbol, since_minutes=since_minutes)
    except (ProgrammingError, OperationalError) as exc:
        logger.warning(
            "predictions_history fetch failed (%s): %s", symbol, exc,
        )
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "since_minutes": since_minutes,
            "rows": [],
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    return JSONResponse(content=_scrub({
        "ok": True,
        "db_status": "ok",
        "symbol": symbol,
        "since_minutes": since_minutes,
        "rows": rows,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


@router.get("/api/highfreq/training_history")
async def get_training_history(
    symbol: str = DEFAULT_SYMBOL,
    since_days: int = 7,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Append-only history of trainer runs, for the UI's evolution
    sparkline.

    Backed by ``training_runs`` (migration 004). One row per
    trainer run per symbol — gives a time-series view of how
    ``dir_acc_mean``, the bar count, and ``n_folds`` evolved as data
    accumulated. Defence-grade artefact: "model improved
    monotonically over 14 days; here's the chart."

    Read-only, never 503 — degrades to ``runs=[]`` on DB error so
    the UI keeps rendering.
    """
    from app.highfreq.training_history import fetch_history_sync

    symbol = symbol.upper()
    # Sane bounds: since_days in [1, 90]. Past 90 days is way more
    # than the trainer's training history is meaningful for at
    # current cadences.
    since_days = max(1, min(int(since_days), 90))

    try:
        runs = fetch_history_sync(db, symbol=symbol, since_days=since_days)
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("training_history fetch failed (%s): %s", symbol, exc)
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "since_days": since_days,
            "runs": [],
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    return JSONResponse(content=_scrub({
        "ok": True,
        "db_status": "ok",
        "symbol": symbol,
        "since_days": since_days,
        "runs": runs,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))


# ── Regression metrics endpoint (release T.8, 2026-04-30) ─────────────
#
# Reads the latest output of ``tools/regression_eval.py`` and returns
# per-symbol MAE / RMSE / R² / IC_pearson / IC_spearman / sign_accuracy.
#
# This is the "prediction-page-style" metrics block the user asked for:
# our 1-minute production model is a CLASSIFIER (predicts direction),
# but the regression tool fits a CatBoostRegressor on continuous
# return_bps target with the same walk-forward CV.  The resulting
# regression metrics are the canonical MAPE/RMSE/R²-style numbers that
# every classical ML eval surface reports.
#
# The tool writes ``weights/highfreq/regression_eval.json`` on each
# manual run (or via a future systemd timer).  This endpoint just reads
# that file — never recomputes (the eval takes 1-3 minutes per symbol).
@router.get("/api/highfreq/regression_metrics")
async def get_regression_metrics() -> JSONResponse:
    """Latest CatBoostRegressor metrics from ``tools.regression_eval``.

    Returns 200 always:
    * file present + parseable → ok=True with rows[]
    * file missing (tool never run) → ok=False, reason="no_eval_yet"
    * file malformed → ok=False, reason="malformed"

    Response shape::

        {
          "ok": true,
          "generated_at": "2026-04-29T...",
          "rows": [
            {
              "symbol": "BTCUSDT", "bar_minutes": 1, "n_predictions": 2460,
              "mae_bps": 4.067, "rmse_bps": 5.657, "r2": -0.0849,
              "ic_pearson": 0.0205, "ic_spearman": 0.0949,
              "sign_accuracy": 0.5508
            }, ...
          ]
        }
    """
    path = Path("weights/highfreq/regression_eval.json")
    if not path.exists():
        return JSONResponse(content={
            "ok": False,
            "reason": "no_eval_yet",
            "hint": "run `python -m tools.regression_eval` to generate",
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("regression_eval.json read failed: %s", exc)
        return JSONResponse(content={
            "ok": False,
            "reason": "malformed",
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    # Flatten the eval rows into a UI-friendly shape; drop the
    # threshold curves (the page surfaces fee-tier P&L separately).
    rows_out = []
    for r in payload.get("rows", []):
        rows_out.append({
            "symbol": r.get("symbol"),
            "bar_minutes": r.get("bar_minutes"),
            "n_predictions": r.get("n_predictions"),
            "n_folds": r.get("n_folds"),
            "mae_bps": r.get("mae_bps"),
            "rmse_bps": r.get("rmse_bps"),
            "r2": r.get("r2"),
            "ic_pearson": r.get("ic_pearson"),
            "ic_spearman": r.get("ic_spearman"),
            "sign_accuracy": r.get("sign_accuracy"),
        })

    return JSONResponse(content=_scrub({
        "ok": True,
        "generated_at": payload.get("generated_at"),
        "config": payload.get("config", {}),
        "rows": rows_out,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))
