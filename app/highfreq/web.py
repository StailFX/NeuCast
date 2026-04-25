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

import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.orm import Session

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


@router.get("/highfreq", response_class=HTMLResponse)
async def highfreq_page(request: Request) -> HTMLResponse:
    """Render the live ``/highfreq`` page.

    The HTML is intentionally minimal — actual data is filled in client-side
    by polling :func:`get_status` every 2 seconds. This keeps server-side
    rendering cheap and lets the page degrade gracefully if the ingest
    service is briefly down.
    """
    return templates.TemplateResponse(
        "highfreq.html",
        {
            "request": request,
            "symbol": DEFAULT_SYMBOL,
            "minutes_required": MIN_MINUTES_FOR_TRAINING,
        },
    )


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
