"""Append-only persistence of every trainer run.

Why this exists
===============

The trainer's on-disk ``metrics.json`` is overwritten every run, so we
lose all prior context: how did ``dir_acc`` evolve as the dataset
grew? When did the first walk-forward fold land? Did
``feature_importance`` shift after a regime change?

This module logs every trainer run to the ``training_runs`` Postgres
table (migration ``004_training_runs.sql``). One row per run, append
only. Used by:

* ``/api/highfreq/training_history`` — surfaces the time series for
  the UI's "model evolution" sparkline.
* Defence story — concrete evidence of "the model improved
  monotonically over 14 days as data accumulated".

Pure separation
---------------

Same shape as ``realized_accuracy.py`` and ``data_inventory.py`` — the
SQL-touching code is here, the trainer simply imports
:func:`persist_run_sync` and calls it. Tests cover the row-mapping
logic (``_run_dict_for_insert``) without a real DB.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.highfreq.trainer import TrainingReport


def _scrub_nan(o: Any) -> Any:
    """Recursively replace NaN/Inf with None so the JSONB encoder
    doesn't reject the payload. Same scrubber the trainer uses for
    its on-disk metrics.json."""
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _scrub_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_scrub_nan(v) for v in o]
    return o


def _run_dict_for_insert(
    report: "TrainingReport",
    *,
    run_started_at: datetime,
) -> dict[str, Any]:
    """Pure: shape a TrainingReport into the column dict.

    Pinned by tests so a future column rename in the migration
    breaks the test instead of silently dropping data on insert.
    """
    full_report = _scrub_nan(asdict(report))

    def _none_if_nan(x: float | None) -> float | None:
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)

    return {
        "symbol": report.symbol,
        "run_started_at": run_started_at,
        "elapsed_seconds": float(report.elapsed_seconds),
        "n_seconds_loaded": int(report.n_seconds_loaded),
        "n_minutes_after_aggregation": int(report.n_minutes_after_aggregation),
        "n_minutes_after_neutral_drop": int(report.n_minutes_after_neutral_drop),
        "n_folds": int(report.n_folds),
        "dir_acc_mean": _none_if_nan(report.dir_acc_mean),
        "dir_acc_ci_low": _none_if_nan(report.dir_acc_ci_low),
        "dir_acc_ci_high": _none_if_nan(report.dir_acc_ci_high),
        "dir_acc_p_value": _none_if_nan(report.dir_acc_p_value),
        "log_loss_mean": _none_if_nan(report.log_loss_mean),
        "base_rate": _none_if_nan(report.base_rate),
        "frozen_holdout_days": int(report.frozen_holdout_days),
        "n_minutes_in_holdout": (
            int(report.n_minutes_in_holdout)
            if report.n_minutes_in_holdout is not None else None
        ),
        "weights_path": report.weights_path,
        "full_report_json": json.dumps(full_report, default=str),
    }


def persist_run_sync(
    database_url: str,
    report: "TrainingReport",
    *,
    run_started_at: datetime,
) -> int | None:
    """INSERT one row into ``training_runs``. Returns the new ``id``,
    or ``None`` on failure (logged at WARNING; never raises).

    Why fail-soft: a successful trainer run that couldn't log itself to
    Postgres is **still successful**. The .cbm + metrics.json are
    already on disk; the predictor will hot-reload them. Losing the
    history row is observability degradation, not a correctness bug.
    """
    payload = _run_dict_for_insert(report, run_started_at=run_started_at)
    try:
        from sqlalchemy import create_engine, text  # local import keeps CLI light

        eng = create_engine(database_url, future=True)
        with eng.begin() as conn:
            row = conn.execute(
                text("""
                    INSERT INTO training_runs (
                        symbol, run_started_at, elapsed_seconds,
                        n_seconds_loaded, n_minutes_after_aggregation,
                        n_minutes_after_neutral_drop,
                        n_folds, dir_acc_mean, dir_acc_ci_low,
                        dir_acc_ci_high, dir_acc_p_value,
                        log_loss_mean, base_rate,
                        frozen_holdout_days, n_minutes_in_holdout,
                        weights_path, full_report
                    ) VALUES (
                        :symbol, :run_started_at, :elapsed_seconds,
                        :n_seconds_loaded, :n_minutes_after_aggregation,
                        :n_minutes_after_neutral_drop,
                        :n_folds, :dir_acc_mean, :dir_acc_ci_low,
                        :dir_acc_ci_high, :dir_acc_p_value,
                        :log_loss_mean, :base_rate,
                        :frozen_holdout_days, :n_minutes_in_holdout,
                        :weights_path, CAST(:full_report_json AS JSONB)
                    ) RETURNING id
                """),
                payload,
            ).fetchone()
        new_id = int(row[0]) if row else None
        logger.info(
            "training_runs row written: id=%s symbol=%s n_folds=%d",
            new_id, report.symbol, report.n_folds,
        )
        return new_id
    except Exception as exc:
        logger.warning(
            "training_runs INSERT failed for symbol=%s: %s",
            report.symbol, exc,
        )
        return None


def fetch_history_sync(
    db_session: Any,
    *,
    symbol: str,
    since_days: int = 7,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """SELECT recent training runs for the UI sparkline.

    Returns column dicts (NOT the full_report JSON — too verbose for
    a list view). UI fetches details via a separate `?id=N` endpoint
    if needed (not yet implemented).
    """
    from sqlalchemy import text
    sql = text(
        "SELECT id, symbol, run_started_at, elapsed_seconds, "
        "       n_seconds_loaded, n_minutes_after_aggregation, "
        "       n_minutes_after_neutral_drop, "
        "       n_folds, dir_acc_mean, dir_acc_ci_low, dir_acc_ci_high, "
        "       dir_acc_p_value, log_loss_mean, base_rate, "
        "       frozen_holdout_days, n_minutes_in_holdout, weights_path "
        "  FROM training_runs "
        " WHERE symbol = :symbol "
        "   AND run_started_at >= now() - (:days * interval '1 day') "
        " ORDER BY run_started_at ASC "
        " LIMIT :limit"
    )
    res = db_session.execute(sql, {
        "symbol": symbol, "days": since_days, "limit": int(limit),
    })
    out: list[dict[str, Any]] = []
    for row in res:
        d = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
        # Convert datetime → ISO for JSON serialisation.
        if d.get("run_started_at") is not None:
            d["run_started_at"] = d["run_started_at"].isoformat()
        out.append(d)
    return out
