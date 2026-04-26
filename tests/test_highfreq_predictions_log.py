"""Tests for ``app.highfreq.predictions_log`` — sync fetch path.

The async ``log_prediction_async`` is exercised by the production
smoke test (the runner writes a row every minute, easily verifiable
via SELECT). Pure-helper tests focus on the SQL shape via the sync
fetcher's mock-session contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.highfreq.predictions_log import (
    PredictionRow,
    fetch_history_sync,
)


# ───── PredictionRow shape ─────


def test_prediction_row_to_dict_matches_endpoint_keys():
    """The endpoint embeds these dicts directly. Pin so a future
    field rename breaks tests, not the UI silently."""
    r = PredictionRow(
        id=1,
        ts="2026-04-27T12:00:00+00:00",
        symbol="BTCUSDT",
        prob_up=0.55,
        signal="up",
        microprice=77_000.0,
        model_version="3600",
        realized_microprice_1m=None,
        realized_correct=None,
    )
    d = r.to_dict()
    assert set(d.keys()) == {
        "id", "ts", "symbol", "prob_up", "signal",
        "microprice", "model_version",
        "realized_microprice_1m", "realized_correct",
    }


# ───── sync fetcher ─────


class _FakeRow:
    def __init__(self, **kw): self.__dict__.update(kw)
    @property
    def _mapping(self): return self.__dict__


class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def __iter__(self): return iter(self._rows)


class _FakeSession:
    """Records executed SQL + params, returns canned rows."""
    def __init__(self, rows):
        self._rows = rows
        self.last_sql_text: str | None = None
        self.last_params: dict[str, Any] | None = None

    def execute(self, sql, params):
        self.last_sql_text = str(getattr(sql, "text", sql))
        self.last_params = params
        return _FakeResult(self._rows)


def test_fetch_history_sync_executes_correct_sql():
    """Pin: filtered by symbol, time-windowed by minutes, ordered
    DESC, LIMITed. A regression that drops the WHERE silently
    streams the entire table."""
    session = _FakeSession(rows=[
        _FakeRow(
            id=1,
            ts=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            prob_up=0.55, signal="up", microprice=77_000.0,
            model_version="3600",
            realized_microprice_1m=None, realized_correct=None,
        ),
    ])
    rows = fetch_history_sync(session, symbol="BTCUSDT", since_minutes=60)

    sql_str = session.last_sql_text.upper()
    assert "PREDICTIONS_LOG" in sql_str
    assert "WHERE" in sql_str and "SYMBOL" in sql_str
    assert "ORDER BY" in sql_str
    assert "DESC" in sql_str
    assert "LIMIT" in sql_str
    assert session.last_params == {
        "symbol": "BTCUSDT", "mins": 60, "limit": 500,
    }
    assert len(rows) == 1


def test_fetch_history_sync_iso_serialises_ts():
    """The endpoint streams JSON; raw datetimes must be ISO strings
    so json.dumps doesn't fall back to default=str (which has subtle
    timezone-handling differences)."""
    session = _FakeSession(rows=[
        _FakeRow(
            id=1,
            ts=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            prob_up=0.55, signal="up", microprice=77_000.0,
            model_version="3600",
            realized_microprice_1m=None, realized_correct=None,
        ),
    ])
    rows = fetch_history_sync(session, symbol="BTCUSDT", since_minutes=60)
    assert rows[0]["ts"] == "2026-04-27T12:00:00+00:00"


def test_fetch_history_sync_empty_window():
    """No rows in the window → empty list, not None or 503."""
    session = _FakeSession(rows=[])
    rows = fetch_history_sync(session, symbol="BTCUSDT", since_minutes=60)
    assert rows == []


def test_fetch_history_sync_caps_limit():
    """Hard cap on LIMIT (500) inside the function. Pin so a
    ``?since_minutes=99999`` doesn't stream millions of rows."""
    session = _FakeSession(rows=[])
    fetch_history_sync(session, symbol="BTCUSDT", since_minutes=10000)
    assert session.last_params["limit"] == 500
