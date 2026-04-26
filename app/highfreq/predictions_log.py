"""Append-only log of every minute's model prediction.

Separation of concerns: the paper-trader's `paper_trades` table only
captures ACTIONED predictions (those that opened a position). The
predictor however emits a forecast every minute regardless of whether
the trader acted — and we want all of those for the UI signal tape,
the Telegram-bot replay, and a future "model skill ≠ trader skill"
analysis.

Schema lives in ``app/highfreq/migrations/005_predictions_log.sql``.

Two write modes
---------------

* :func:`log_prediction_async` — async path, used by
  ``paper_trader_runner`` (which already holds an asyncpg pool).
  Idempotent on (ts, symbol) via ``ON CONFLICT DO NOTHING`` so a
  runner restart that re-processes the same bar doesn't duplicate.
* :func:`fetch_history_sync` — sync path for the FastAPI endpoint.

Both built on the same SQL contract; tests pin the column shape.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionRow:
    """JSON-friendly shape for the UI / endpoint.

    Keys mirror ``predictions_log`` columns exactly so a future
    column add (e.g. ``confidence``) needs only one place to change.
    """
    id: int
    ts: str               # ISO-8601 (frontend doesn't want raw datetime)
    symbol: str
    prob_up: float
    signal: str           # 'up' | 'down' | 'neutral'
    microprice: float
    model_version: str
    realized_microprice_1m: float | None
    realized_correct: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Async writer (runner)
# ──────────────────────────────────────────────────────────────────────


async def log_prediction_async(
    pool: Any,  # asyncpg.Pool (typed Any to avoid hard dep in tests)
    *,
    ts: datetime,
    symbol: str,
    prob_up: float,
    signal: str,
    microprice: float,
    model_version: str,
) -> int | None:
    """INSERT one prediction. Returns id, or None if conflict (already
    logged for this minute) or DB error.

    Fail-soft: a successful prediction that couldn't be logged is still
    a successful prediction. The predictor cached the result in-memory
    and the trader already saw it. Logging failure is observability
    degradation, not correctness.
    """
    sql = """
        INSERT INTO predictions_log (
            ts, symbol, prob_up, signal, microprice, model_version
        ) VALUES (
            $1, $2, $3, $4, $5, $6
        )
        ON CONFLICT (ts, symbol) DO NOTHING
        RETURNING id
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                ts, symbol, float(prob_up), signal,
                float(microprice), model_version,
            )
        return int(row["id"]) if row is not None else None
    except Exception as exc:
        logger.warning(
            "predictions_log INSERT failed (symbol=%s ts=%s): %s",
            symbol, ts, exc,
        )
        return None


# ──────────────────────────────────────────────────────────────────────
# Sync fetcher (FastAPI endpoint)
# ──────────────────────────────────────────────────────────────────────


def fetch_history_sync(
    db_session: Any,
    *,
    symbol: str,
    since_minutes: int = 60,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return last ``since_minutes`` predictions for ``symbol``,
    most-recent first.

    Hard cap on ``limit`` (500) keeps a malformed
    ``?since_minutes=99999`` query from streaming millions of rows.
    """
    from sqlalchemy import text  # local import — keeps non-web modules light

    sql = text(
        "SELECT id, ts, symbol, prob_up, signal, microprice, model_version, "
        "       realized_microprice_1m, realized_correct "
        "  FROM predictions_log "
        " WHERE symbol = :symbol "
        "   AND ts >= now() - (:mins * interval '1 minute') "
        " ORDER BY ts DESC "
        " LIMIT :limit"
    )
    res = db_session.execute(sql, {
        "symbol": symbol,
        "mins": int(since_minutes),
        "limit": int(limit),
    })
    out: list[dict[str, Any]] = []
    for r in res:
        d = dict(r._mapping) if hasattr(r, "_mapping") else dict(r)
        if d.get("ts") is not None:
            d["ts"] = d["ts"].isoformat()
        out.append(d)
    return out
