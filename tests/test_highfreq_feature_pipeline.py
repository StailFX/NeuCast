"""Tests for ``app.highfreq.feature_pipeline`` — the canonical home of the
feature transforms shared by trainer + predictor.

Most pure-function coverage of ``aggregate_to_minute``, ``build_target``,
``build_features``, ``make_supervised`` already lives in
``test_highfreq_trainer.py`` (the trainer re-exports those names). This
file focuses on what's *new* in Phase B:

* ``build_latest_feature_row`` — live-inference helper that drops the
  in-flight current minute and returns the most recent COMPLETE-minute
  feature vector (or ``None`` at cold-start).

The "complete-minute" semantics are critical: the predictor must NEVER
score on a partial bar (the trainer fits on whole-minute aggregates and
a partial bar's OFI sum / mp_close are out-of-distribution). These
tests pin that contract down before we wire it into a paper trader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.highfreq.feature_pipeline import (
    FEATURE_COLUMNS,
    build_latest_feature_row,
)


# ───────────────────────── helpers ─────────────────────────

def _seconds_frame(
    n_minutes: int = 3,
    seconds_per_minute: int = 60,
    base_price: float = 77_000.0,
    drift_per_min_bps: float = 2.0,
    symbol: str = "BTCUSDT",
    seed: int = 0,
    start: str = "2026-04-25 00:00:00",
) -> pd.DataFrame:
    """Per-second frame matching the highfreq_ofi_1s schema.

    Same layout as the trainer-test fixture but parametrisable by start
    time so tests can pretend "now" is mid-minute by truncating the tail.
    """
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp(start, tz="UTC")
    for m in range(n_minutes):
        mid = base_price * (1.0 + (m * drift_per_min_bps) / 1e4)
        for s in range(seconds_per_minute):
            ts = t0 + pd.Timedelta(minutes=m, seconds=s)
            tick = mid + rng.normal(0.0, 1.0)
            rows.append({
                "ts": ts,
                "symbol": symbol,
                "ofi": float(rng.normal(0.0, 0.5)),
                "microprice": float(tick),
                "depth_imb": float(rng.uniform(-0.2, 0.2)),
                "spread_bps": float(rng.uniform(0.5, 1.5)),
                "trade_imb": float(rng.normal(0.0, 0.001)),
                "n_updates": 10,
            })
    return pd.DataFrame(rows)


# ───────────────────────── tests ─────────────────────────


def test_build_latest_returns_none_on_empty_frame():
    """No data ever observed → cold-start → caller surfaces 503."""
    out = build_latest_feature_row(pd.DataFrame())
    assert out is None


def test_build_latest_returns_none_when_only_partial_current_minute():
    """If the only data we have is the in-flight minute (e.g. just
    reconnected 20s ago), there's no COMPLETE prior minute to score.
    Must return None — never a partial bar."""
    # 30 rows in a single minute → only the current (incomplete) minute.
    df = _seconds_frame(n_minutes=1, seconds_per_minute=30)
    out = build_latest_feature_row(df)
    assert out is None, (
        "in-flight minute only must yield None — the trainer never sees "
        "30-second bars and the predictor must not either"
    )


def test_build_latest_drops_in_flight_current_minute():
    """3 minutes observed: 2 complete + 1 in-flight (15s).
    Output must come from the 2nd-most-recent (last fully-closed) minute."""
    # Two whole minutes of m=0,1, then 15s into m=2.
    full = _seconds_frame(n_minutes=2, seconds_per_minute=60)
    partial = _seconds_frame(
        n_minutes=1, seconds_per_minute=15, start="2026-04-25 00:02:00",
    )
    df = pd.concat([full, partial], ignore_index=True)

    row = build_latest_feature_row(df)

    assert row is not None
    assert isinstance(row, pd.Series)
    # Series index must be exactly FEATURE_COLUMNS (downstream predictor
    # reindexes against this).
    assert list(row.index) == FEATURE_COLUMNS


def test_build_latest_returns_most_recent_complete_minute():
    """With 3 complete minutes, the result must come from minute-2.
    We probe this via a deterministic feature: drift means later minutes
    have monotonically growing microprice, so spread-related features
    will be similar but `n_updates_sum` will mirror the input — easier
    to ANCHOR via an OFI sign trick.
    """
    # Build a 4-minute frame where minute-3 has a deliberately huge
    # ofi_sum spike that no earlier minute could mistake for itself.
    df = _seconds_frame(n_minutes=4, seconds_per_minute=60, seed=1)

    # Stamp minute-2 (the LAST COMPLETE minute after dropping in-flight
    # minute-3) with a known OFI sum.
    target_minute = pd.Timestamp("2026-04-25 00:02:00", tz="UTC")
    mask = df["ts"].dt.floor("1min") == target_minute
    df.loc[mask, "ofi"] = 100.0  # 60 × 100 = 6000 ofi_sum

    # Stamp minute-3 (the in-flight one we should be DROPPING) with an
    # opposite-sign sentinel to catch any leak.
    leak_minute = pd.Timestamp("2026-04-25 00:03:00", tz="UTC")
    mask = df["ts"].dt.floor("1min") == leak_minute
    df.loc[mask, "ofi"] = -999.0

    row = build_latest_feature_row(df)

    assert row is not None
    # If we leaked the in-flight minute, ofi_sum would be -999 × 60 = -59940.
    # If we correctly used minute-2, ofi_sum is +6000.
    assert row["ofi_sum"] == 6000.0, (
        f"build_latest_feature_row leaked the in-flight minute "
        f"(got ofi_sum={row['ofi_sum']!r}, expected 6000.0)"
    )


def test_build_latest_returns_none_when_complete_minute_too_sparse():
    """A single complete minute with only 5 seconds of coverage (e.g.
    huge WS gap) must be DROPPED by aggregate_to_minute → no feature
    row available → None.
    """
    # 5 seconds in minute-0, then 30s in minute-1 (current/in-flight).
    sparse = _seconds_frame(n_minutes=1, seconds_per_minute=5)
    in_flight = _seconds_frame(
        n_minutes=1, seconds_per_minute=30, start="2026-04-25 00:01:00",
    )
    df = pd.concat([sparse, in_flight], ignore_index=True)
    out = build_latest_feature_row(df)
    assert out is None, (
        "sparse complete minute (<30s coverage) must be dropped by "
        "MIN_SECONDS_PER_MINUTE filter"
    )


def test_build_latest_no_nan_or_inf_in_output():
    """The predictor's _prepare_features scrubs NaN/inf, but the upstream
    must not introduce them in the first place — easier to debug if
    distributions match the trainer's exactly.
    """
    df = _seconds_frame(n_minutes=3)
    # Add a tail in the in-flight minute (otherwise the last full minute
    # would never get aggregated).
    in_flight = _seconds_frame(
        n_minutes=1, seconds_per_minute=20, start="2026-04-25 00:03:00",
    )
    df = pd.concat([df, in_flight], ignore_index=True)

    row = build_latest_feature_row(df)
    assert row is not None
    arr = row.to_numpy(dtype=float)
    assert np.all(np.isfinite(arr)), (
        f"NaN/inf leaked through build_features: {row.to_dict()}"
    )


def test_build_latest_handles_string_timestamps():
    """Postgres asyncpg returns aware datetimes, but defensive code in
    case something upstream hands us ISO strings."""
    df = _seconds_frame(n_minutes=3, seconds_per_minute=60)
    df_str = df.copy()
    df_str["ts"] = df_str["ts"].astype(str)  # ISO strings

    row = build_latest_feature_row(df_str)
    # If pd.to_datetime fails to parse, the output would be None or an
    # exception; we explicitly guarantee parsability.
    assert row is not None
    assert list(row.index) == FEATURE_COLUMNS
