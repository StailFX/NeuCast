"""Tests for ``app.highfreq.data_inventory`` — the live, on-demand
fold-readiness counter that backs the /highfreq progress widget.

Two layers as in ``realized_accuracy``:

1. **Pure logic** (``compute_live_inventory_from_seconds``) takes a
   pandas DataFrame and produces a snapshot. No DB. Pinned: empty
   input, frozen-holdout split arithmetic, equality with the
   trainer's ``make_supervised`` output.

2. **Cache** (``fetch_live_inventory``) — module-level dict guarded
   by a Lock, TTL'd by wall-clock. Tests use ``clear_cache_for_tests``
   between cases so each starts clean.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.highfreq.data_inventory import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_SINCE_HOURS,
    LiveInventory,
    clear_cache_for_tests,
    compute_live_inventory_from_seconds,
    fetch_live_inventory,
)


# ────────────── fixtures ──────────────


def _seconds_frame(
    n_minutes: int = 5,
    seconds_per_minute: int = 60,
    base_price: float = 77_000.0,
    drift_per_min_bps: float = 5.0,  # ensure non-trivial returns so neutral-band drop is partial
    symbol: str = "BTCUSDT",
    seed: int = 0,
    start: str = "2026-04-25 00:00:00",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp(start, tz="UTC")
    for m in range(n_minutes):
        mid = base_price * (1.0 + (m * drift_per_min_bps) / 1e4)
        for s in range(seconds_per_minute):
            ts = t0 + pd.Timedelta(minutes=m, seconds=s)
            tick = mid + rng.normal(0.0, 0.5)
            rows.append({
                "ts": ts,
                "symbol": symbol,
                "ofi": float(rng.normal(0.0, 0.5)),
                "microprice": float(tick),
                "depth_imb": float(rng.uniform(-0.2, 0.2)),
                "spread_bps": float(rng.uniform(0.5, 1.5)),
                "trade_imb": float(rng.normal(0.0, 0.001)),
                "vpin": 0.0,
                "n_updates": 10,
                "local_recv_ms": 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _wipe_cache():
    """Every test starts with a clean cache so cross-test pollution
    can't make tests order-dependent."""
    clear_cache_for_tests()
    yield
    clear_cache_for_tests()


# ────────────── pure-logic tests ──────────────


def test_compute_live_inventory_returns_zeros_on_empty_input():
    """Cold start (predictor running before any seconds rows
    landed) — endpoint must produce a valid LiveInventory with all
    zeros, NOT raise. Pin so a defensive future change can't
    silently 500 the endpoint."""
    snap = compute_live_inventory_from_seconds(
        pd.DataFrame(columns=["ts", "symbol", "microprice"]),
        symbol="BTCUSDT",
    )
    assert isinstance(snap, LiveInventory)
    assert snap.symbol == "BTCUSDT"
    assert snap.n_seconds_loaded == 0
    assert snap.n_minutes_after_aggregation == 0
    assert snap.n_minutes_after_neutral_drop == 0
    assert snap.n_eligible_for_training == 0
    assert snap.n_in_holdout == 0


def test_compute_live_inventory_full_pipeline_counts_match():
    """The point of this module: live counts must match what the
    trainer would compute on the SAME data. Run both, compare."""
    df = _seconds_frame(n_minutes=10, drift_per_min_bps=10.0)
    snap = compute_live_inventory_from_seconds(
        df, symbol="BTCUSDT", frozen_holdout_days=0,
    )

    # Cross-check against trainer pipeline.
    from app.highfreq.feature_pipeline import (
        aggregate_to_minute, make_supervised,
    )
    minute_df = aggregate_to_minute(df)
    X, _y, _meta = make_supervised(df)

    assert snap.n_seconds_loaded == 600
    assert snap.n_minutes_after_aggregation == len(minute_df)
    assert snap.n_minutes_after_neutral_drop == len(X)


def test_compute_live_inventory_excludes_holdout_window():
    """The eligibility numerator must EXCLUDE bars the trainer is
    forbidden from seeing. Pin the cutoff arithmetic — getting it
    wrong by one minute breaks defence-grade "no leak" claims."""
    # Two days of data: today and yesterday. With holdout=1 day,
    # only yesterday's bars are eligible for training.
    now = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
    yesterday = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    today = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

    df_old = _seconds_frame(n_minutes=15, drift_per_min_bps=10.0,
                             start=yesterday, seed=1)
    df_new = _seconds_frame(n_minutes=15, drift_per_min_bps=10.0,
                             start=today, seed=2)
    df = pd.concat([df_old, df_new], ignore_index=True)

    snap = compute_live_inventory_from_seconds(
        df, symbol="BTCUSDT", frozen_holdout_days=1, now=now,
    )
    # Some bars must be in holdout (the recent ones).
    assert snap.n_in_holdout > 0
    # Eligible = total - holdout.
    assert (
        snap.n_eligible_for_training
        == snap.n_minutes_after_neutral_drop - snap.n_in_holdout
    )
    # Eligible must be >= 0 (trivially).
    assert snap.n_eligible_for_training >= 0


def test_compute_live_inventory_holdout_zero_disables_split():
    """frozen_holdout_days=0 → ``n_in_holdout`` must be 0 and
    ``n_eligible == n_minutes_after_neutral_drop``. Pin so the
    bootstrap-mode escape hatch behaves cleanly."""
    df = _seconds_frame(n_minutes=10, drift_per_min_bps=10.0)
    snap = compute_live_inventory_from_seconds(
        df, symbol="BTCUSDT", frozen_holdout_days=0,
    )
    assert snap.n_in_holdout == 0
    assert snap.n_eligible_for_training == snap.n_minutes_after_neutral_drop


def test_to_dict_shape_is_json_safe():
    """The endpoint embeds ``snap.to_dict()`` directly. Pin the
    keys so a future field rename breaks tests, not the UI."""
    df = _seconds_frame(n_minutes=3)
    snap = compute_live_inventory_from_seconds(df, symbol="BTCUSDT")
    d = snap.to_dict()
    assert set(d.keys()) == {
        "symbol", "computed_at",
        "n_seconds_loaded",
        "n_minutes_after_aggregation",
        "n_minutes_after_neutral_drop",
        "n_eligible_for_training",
        "n_in_holdout",
        "since_hours",
    }


# ────────────── cache layer tests ──────────────


class _FakeSession:
    """Stand-in for SQLAlchemy Session — only ``connection()`` is touched
    by ``_fetch_seconds_sync`` (passed straight through to pandas read_sql).

    We don't try to mock Postgres; instead, we monkeypatch
    ``_fetch_seconds_sync`` itself to control what the cache layer sees.
    """
    def __init__(self):
        self.calls: list[Any] = []


def test_fetch_live_inventory_caches_within_ttl(monkeypatch):
    """Repeated calls within TTL hit the function once. Pin so a
    refactor that loses the cache doesn't suddenly hammer Postgres
    on every page render."""
    df = _seconds_frame(n_minutes=5)
    call_count = {"n": 0}

    def _fake_fetch(db, *, symbol, since_hours):
        call_count["n"] += 1
        return df

    monkeypatch.setattr(
        "app.highfreq.data_inventory._fetch_seconds_sync", _fake_fetch,
    )

    s1 = fetch_live_inventory(_FakeSession(), symbol="BTCUSDT")
    s2 = fetch_live_inventory(_FakeSession(), symbol="BTCUSDT")
    s3 = fetch_live_inventory(_FakeSession(), symbol="BTCUSDT")

    assert call_count["n"] == 1, "all 3 calls within TTL must share the cache"
    # Same snapshot identity — proves it's the same cached object.
    assert s1 is s2
    assert s2 is s3


def test_fetch_live_inventory_refreshes_after_ttl(monkeypatch):
    """After TTL elapses, next call recomputes."""
    df = _seconds_frame(n_minutes=5)
    call_count = {"n": 0}

    def _fake_fetch(db, *, symbol, since_hours):
        call_count["n"] += 1
        return df

    monkeypatch.setattr(
        "app.highfreq.data_inventory._fetch_seconds_sync", _fake_fetch,
    )

    fetch_live_inventory(_FakeSession(), symbol="BTCUSDT", ttl_seconds=0.01)
    import time
    time.sleep(0.05)  # exceed the tiny TTL
    fetch_live_inventory(_FakeSession(), symbol="BTCUSDT", ttl_seconds=0.01)

    assert call_count["n"] == 2


def test_fetch_live_inventory_keys_cache_by_symbol(monkeypatch):
    """Different symbols don't share cache entries (otherwise BTC's
    snapshot would mistakenly serve as ETH's)."""
    df_btc = _seconds_frame(n_minutes=5, symbol="BTCUSDT", seed=1)
    df_eth = _seconds_frame(n_minutes=8, symbol="ETHUSDT", seed=2)
    by_sym = {"BTCUSDT": df_btc, "ETHUSDT": df_eth}

    def _fake_fetch(db, *, symbol, since_hours):
        return by_sym[symbol]

    monkeypatch.setattr(
        "app.highfreq.data_inventory._fetch_seconds_sync", _fake_fetch,
    )

    btc = fetch_live_inventory(_FakeSession(), symbol="BTCUSDT")
    eth = fetch_live_inventory(_FakeSession(), symbol="ETHUSDT")
    assert btc.n_seconds_loaded == 300
    assert eth.n_seconds_loaded == 480
    assert btc.symbol == "BTCUSDT"
    assert eth.symbol == "ETHUSDT"


def test_default_ttl_and_since_hours_pinned():
    """Document-as-test: defence-grade work assumes 30 s TTL and 72 h
    window. Future refactor that bumps the TTL to 5 minutes (=stale UI)
    breaks this loudly."""
    assert DEFAULT_CACHE_TTL_SECONDS == 30.0
    assert DEFAULT_SINCE_HOURS == 72.0
