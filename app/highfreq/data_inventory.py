"""Live, on-demand data inventory for the ``/highfreq`` UI.

What this is for
================

The trainer fires once a day. Between firings, the *only* place the
UI sees "how many bars do we have?" is in the on-disk
``metrics.json`` written by the previous trainer run. So the
"Готовность к первому walk-forward fold" progress widget freezes at
~14% for hours and looks broken even though the ingest is happily
collecting fresh microstructure data.

This module fixes that by computing the same fold-eligibility number
the trainer would compute, on demand, **without** doing a CatBoost
fit. The pipeline is the trainer's pure pipeline (``aggregate_to_minute
→ build_target → neutral-band drop``) — no model. Cost: one SQL query
+ ~100ms of pandas. Cached per-symbol with a short TTL so opening the
page 50× a minute doesn't hammer Postgres.

Why a separate module
---------------------

* Same separation of concerns as ``app.highfreq.realized_accuracy``
  — the **pure** computation is independent of any DB, FastAPI, or
  asyncio context, so it's unit-testable with a hand-built DataFrame.
* The cache is module-level (not per-request) so it survives across
  FastAPI request handlers without needing application state plumbing.
* If the trainer ever changes its eligibility filter (new minimum-coverage
  rule, different horizon, etc.), updating ``feature_pipeline.make_supervised``
  is enough — the live inventory and the trainer share the same code
  path so they cannot drift apart.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.highfreq.feature_pipeline import (
    aggregate_to_minute,
    make_supervised,
)

logger = logging.getLogger(__name__)


#: Default TTL for the in-memory inventory cache. Live data only changes
#: minute-by-minute so 30 s is plenty fresh for a UI that polls every 5 s.
DEFAULT_CACHE_TTL_SECONDS: float = 30.0

#: How many hours of seconds the live inventory loads from Postgres.
#: Mirrors the trainer's ``--since-hours`` default — same window, same
#: bars. If the trainer's window changes, change this in lockstep.
DEFAULT_SINCE_HOURS: float = 72.0


@dataclass(frozen=True)
class LiveInventory:
    """JSON-friendly snapshot of the *live* data state.

    All counts are recomputed from the current contents of
    ``highfreq_ofi_1s`` — they reflect what the trainer WOULD see
    if it ran right now, not what it saw at last firing.
    """

    symbol: str
    #: Wall-clock when this snapshot was computed (server-side).
    computed_at: str
    #: Total 1-second rows loaded (last ``since_hours`` for ``symbol``).
    n_seconds_loaded: int
    #: Minute-level bars after coverage filter (>=30s observed per minute).
    n_minutes_after_aggregation: int
    #: Bars left after neutral-band + unobservable-future drop. **This is
    #: the number the trainer would feed into walk-forward CV.**
    n_minutes_after_neutral_drop: int
    #: Bars eligible for **training**: ``n_minutes_after_neutral_drop`` minus
    #: those reserved by the frozen holdout (see ``WalkForwardConfig.
    #: frozen_holdout_days``).  Equals the readiness-progress numerator.
    n_eligible_for_training: int
    #: Bars currently sitting in the frozen holdout (last
    #: ``frozen_holdout_days`` of data the trainer is forbidden from
    #: looking at). 0 when the holdout is disabled.
    n_in_holdout: int
    #: How many hours of data this inventory represents.
    since_hours: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_live_inventory_from_seconds(
    df_secs: pd.DataFrame,
    *,
    symbol: str,
    frozen_holdout_days: int = 7,
    since_hours: float = DEFAULT_SINCE_HOURS,
    now: datetime | None = None,
) -> LiveInventory:
    """Pure: compute the live inventory from an already-loaded
    seconds DataFrame.

    ``now`` is overridable for tests. Defaults to UTC ``now()``.
    """
    now = now if now is not None else datetime.now(tz=timezone.utc)

    if df_secs.empty:
        return LiveInventory(
            symbol=symbol,
            computed_at=now.isoformat(),
            n_seconds_loaded=0,
            n_minutes_after_aggregation=0,
            n_minutes_after_neutral_drop=0,
            n_eligible_for_training=0,
            n_in_holdout=0,
            since_hours=since_hours,
        )

    minute_df = aggregate_to_minute(df_secs)
    n_min = int(len(minute_df))

    # ``make_supervised`` applies build_target (sign-of-1m-return), drops
    # the unobservable trailing bar, and drops bars in the neutral band —
    # exactly what the trainer feeds into walk-forward CV.
    X, _y, meta = make_supervised(df_secs)
    n_kept = int(len(X))

    if frozen_holdout_days > 0 and not meta.empty:
        cutoff = now - timedelta(days=int(frozen_holdout_days))
        in_holdout = (meta["minute"] >= pd.Timestamp(cutoff)).sum()
        n_in_holdout = int(in_holdout)
        n_eligible = max(0, n_kept - n_in_holdout)
    else:
        n_in_holdout = 0
        n_eligible = n_kept

    return LiveInventory(
        symbol=symbol,
        computed_at=now.isoformat(),
        n_seconds_loaded=int(len(df_secs)),
        n_minutes_after_aggregation=n_min,
        n_minutes_after_neutral_drop=n_kept,
        n_eligible_for_training=n_eligible,
        n_in_holdout=n_in_holdout,
        since_hours=since_hours,
    )


# ──────────────────────────────────────────────────────────────────────
# DB layer (sync — used by the FastAPI training_report endpoint)
# ──────────────────────────────────────────────────────────────────────


# Module-level cache. Keyed by symbol; value is (timestamp, snapshot).
# Guarded by a Lock so concurrent requests don't both run the heavy
# query in the same TTL window.
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, LiveInventory]] = {}


def _fetch_seconds_sync(
    db_session: Any,
    *,
    symbol: str,
    since_hours: float,
) -> pd.DataFrame:
    """Sync seconds-loader for the FastAPI endpoint.

    Mirrors ``trainer.load_seconds`` but uses an existing SQLAlchemy
    Session rather than spinning up its own engine. Same query shape
    so the trainer and the live endpoint see byte-identical data.
    """
    from sqlalchemy import text  # local import (web.py already imports it)
    sql = text(
        "SELECT ts, symbol, ofi, microprice, depth_imb, spread_bps, "
        "       trade_imb, vpin, n_updates, local_recv_ms "
        "  FROM highfreq_ofi_1s "
        " WHERE symbol = :symbol "
        "   AND ts >= now() - (:hours * interval '1 hour') "
        " ORDER BY ts ASC"
    )
    df = pd.read_sql(sql, db_session.connection(), params={"symbol": symbol, "hours": since_hours})
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def fetch_live_inventory(
    db_session: Any,
    *,
    symbol: str,
    since_hours: float = DEFAULT_SINCE_HOURS,
    frozen_holdout_days: int = 7,
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> LiveInventory:
    """Cached fetch + compute. Hits the DB at most once per ``ttl_seconds``
    per symbol.

    The cache is shared across requests to the same uvicorn worker. With
    a 30 s TTL and a UI poll cadence of ~5 s, ~6 of 6 requests inside a
    cache window are free; only 1 hits Postgres. Across multiple workers
    the worst case is N caches × 1 query / TTL — still trivial.
    """
    now_unix = time.time()
    with _cache_lock:
        cached = _cache.get(symbol)
        if cached is not None and (now_unix - cached[0]) < ttl_seconds:
            return cached[1]

    # Outside the lock: heavy work (DB query + pandas). We accept that
    # two concurrent first-cache-miss requests on the SAME symbol may
    # both compute. Still bounded; not worth the lock contention to
    # serialise them.
    df_secs = _fetch_seconds_sync(db_session, symbol=symbol, since_hours=since_hours)
    snap = compute_live_inventory_from_seconds(
        df_secs,
        symbol=symbol,
        frozen_holdout_days=frozen_holdout_days,
        since_hours=since_hours,
    )

    with _cache_lock:
        _cache[symbol] = (now_unix, snap)
    return snap


def clear_cache_for_tests() -> None:
    """Test helper — drops the cache between tests so each one sees a
    deterministic fresh-cache state. Production callers should never
    invoke this."""
    with _cache_lock:
        _cache.clear()
