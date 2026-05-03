"""Drift-driven retrain decision policy (T.22, 2026-05-04).

When the hourly KS-test drift detector (T.18.b) flags ``severity=high``,
the operator's manual response was: SSH to Tokyo, ``systemctl start
neucast-highfreq-trainer@<sym>.service``, wait, verify. T.22 closes
that feedback loop automatically — but with **two safety rails**:

1. **Cooldown**: don't retrigger within ``cooldown_hours`` of the
   last training run. Otherwise a stuck-high drift JSON would force
   a retrain every hour, melting the box and overwriting the model
   with progressively-noisier slices.
2. **Severity gate**: only fire on ``high`` (KS ≥ 0.30). ``warn``
   alone is too noisy — KS-test is sensitive on intraday calendar
   regimes (Asia open, US open) and we already explicitly filter
   calendar features. A persistent ``high`` is the clear signal.

Pure module — the policy decision lives here so it's unit-testable
without subprocess / DB / filesystem mocks. The thin CLI in
``tools/drift_driven_retrain.py`` wires it to the live drift JSON
+ ``systemctl``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class RetrainDecision:
    """Outcome of evaluating the policy for one symbol.

    ``should_retrain`` is the binary trigger; ``reason`` is a short
    human-readable string for logging / Telegram. ``severity`` and
    ``hours_since_last_train`` are passed through so the CLI can log
    the full context without re-fetching.
    """

    should_retrain: bool
    reason: str
    severity: str
    hours_since_last_train: float | None


def _parse_iso8601(ts: str) -> datetime:
    """Tolerant ISO-8601 parser. Accepts both ``...Z`` (UTC) and
    explicit ``+00:00`` suffixes; returns timezone-aware UTC dt."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def evaluate_drift_retrain_policy(
    *,
    severity: str,
    last_train_started_at: str | datetime | None,
    now: datetime | None = None,
    cooldown_hours: float = 6.0,
    fire_on_severities: tuple[str, ...] = ("high",),
) -> RetrainDecision:
    """Decide whether a drift-driven retrain should fire.

    Parameters
    ----------
    severity
        Latest drift severity bucket ("ok", "warn", "high").
        Comparison is case-insensitive and whitespace-trimmed so a
        miswritten ``" High "`` doesn't silently fall through.
    last_train_started_at
        ``training_runs.started_at`` for the most recent production
        training run for this symbol. ``None`` means "never trained" —
        which is unusual on a live system but is treated as "always
        retrain" (the cooldown can't restrict what hasn't happened).
    now
        Override for testing; defaults to ``datetime.now(UTC)``.
    cooldown_hours
        Minimum elapsed hours since the last training run before a
        new retrain may be triggered. Default 6h matches the gap
        between the daily 04:00 trainer + a mid-day drift response,
        without thrashing.
    fire_on_severities
        Tuple of severity values that trigger retrain. Default
        ``("high",)`` — ``warn`` is intentionally **not** included
        because Phase-1 deployment showed warn-level KS spikes are
        often transient (intraday regime shifts) and clear within
        an hour.

    Returns
    -------
    RetrainDecision
        ``should_retrain=True`` only when both gates clear.
        ``reason`` always populated; useful for telemetry / TG alerts.
    """
    sev = (severity or "").strip().lower()

    if sev not in fire_on_severities:
        return RetrainDecision(
            should_retrain=False,
            reason=f"severity={sev!r} not in trigger set {fire_on_severities}",
            severity=sev,
            hours_since_last_train=None,
        )

    now = now or datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if last_train_started_at is None:
        # Never trained — we definitely want to retrain (Phase-1
        # cold-start path, almost never hit in production).
        return RetrainDecision(
            should_retrain=True,
            reason=f"severity={sev}, no prior training run on record",
            severity=sev,
            hours_since_last_train=None,
        )

    last_dt = (
        last_train_started_at
        if isinstance(last_train_started_at, datetime)
        else _parse_iso8601(last_train_started_at)
    )
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)

    elapsed = (now - last_dt).total_seconds() / 3600.0
    if elapsed < cooldown_hours:
        return RetrainDecision(
            should_retrain=False,
            reason=(
                f"severity={sev} but last train was {elapsed:.1f}h ago "
                f"(cooldown={cooldown_hours}h). Skipping to avoid retrain storm."
            ),
            severity=sev,
            hours_since_last_train=elapsed,
        )

    return RetrainDecision(
        should_retrain=True,
        reason=(
            f"severity={sev}, last train {elapsed:.1f}h ago "
            f"(> cooldown={cooldown_hours}h). Triggering retrain."
        ),
        severity=sev,
        hours_since_last_train=elapsed,
    )
