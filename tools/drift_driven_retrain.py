"""Drift-driven retrain CLI (T.22, 2026-05-04).

Closes the feedback loop between the hourly drift detector
(``tools/drift_check.py`` / T.18.b) and the daily trainer
(``neucast-highfreq-trainer@<sym>.service``).

Wire-up
-------

* Reads ``weights/highfreq/<sym>_drift.json`` for each requested
  symbol.
* Reads the latest ``training_runs.started_at`` for that symbol from
  Postgres (DATABASE_URL).
* Asks ``app.highfreq.drift_retrain_policy.evaluate_drift_retrain_policy``
  whether to fire.
* If yes → ``systemctl start neucast-highfreq-trainer@<sym>.service``.
* Always logs the decision + reason to stdout (and to the optional
  textfile-collector path so Prometheus can scrape it).

Idempotent — safe to run on a schedule (proposed: every 30 min via
``neucast-drift-retrain.timer``) without thrashing because the
cooldown rail in the policy module guarantees ≥6h between starts.

Example
-------

::

    sudo -u stailfx --preserve-env=DATABASE_URL \\
        /opt/neucast/venv/bin/python -m tools.drift_driven_retrain \\
        --symbol BTCUSDT --symbol ETHUSDT --symbol BNBUSDT \\
        --textfile-out /var/lib/node_exporter/textfile_collector/drift_retrain.prom

Dry-run::

    python -m tools.drift_driven_retrain --symbol BTCUSDT --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_drift_json(weights_dir: Path, symbol: str) -> dict[str, Any] | None:
    """Returns the parsed drift JSON or ``None`` if missing/malformed."""
    p = weights_dir / f"{symbol.lower()}_drift.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("malformed drift JSON for %s: %s", symbol, exc)
        return None


def _last_training_started_at(database_url: str, symbol: str) -> datetime | None:
    """Most-recent ``training_runs.run_started_at`` for ``symbol``.

    Returns ``None`` if no row exists (cold-start path) — the policy
    interprets this as "trigger immediately".

    NOTE: the column is named ``run_started_at`` in the production
    schema (not ``started_at``). Earlier T.22 deploy hit a fallback
    when this query failed; the fallback (cold-start = trigger) is
    correct only when the table really is empty, not when we have a
    schema mismatch. Both branches now agree.
    """
    from sqlalchemy import create_engine, text

    eng = create_engine(database_url, future=True)
    with eng.connect() as conn:
        row = conn.execute(text(
            """SELECT run_started_at FROM training_runs
               WHERE symbol = :sym
               ORDER BY run_started_at DESC LIMIT 1"""
        ), {"sym": symbol.upper()}).fetchone()
    return row[0] if row else None


def _trigger_systemctl(symbol: str, *, dry_run: bool) -> tuple[int, str]:
    """Fire the per-symbol trainer unit. Returns (rc, captured stderr).

    ``dry_run`` short-circuits with rc=0 for tests + ad-hoc operator
    runs that just want to see "what would happen?".
    """
    sym_lower = symbol.lower()
    cmd = [
        "systemctl", "start",
        f"neucast-highfreq-trainer@{sym_lower}.service",
    ]
    if dry_run:
        logger.info("[dry-run] would exec: %s", shlex.join(cmd))
        return 0, "(dry-run)"
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return proc.returncode, (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "systemctl not on PATH (running off-Tokyo?)"


def _emit_textfile(path: Path | None, *,
                   per_symbol: list[dict[str, Any]]) -> None:
    """Write Prometheus textfile-collector output. Safe-no-op on None.

    Format follows node_exporter's convention; metrics:
      * ``neucast_drift_retrain_decision`` — 1 / 0 per symbol+severity.
      * ``neucast_drift_retrain_hours_since_last`` — gauge.
    """
    if path is None:
        return
    lines = [
        "# HELP neucast_drift_retrain_decision Drift-driven retrain trigger fired (1) / suppressed (0).",
        "# TYPE neucast_drift_retrain_decision gauge",
        "# HELP neucast_drift_retrain_hours_since_last Hours since last training_runs row.",
        "# TYPE neucast_drift_retrain_hours_since_last gauge",
    ]
    for r in per_symbol:
        sym = r["symbol"]
        sev = r["severity"]
        lines.append(
            f'neucast_drift_retrain_decision{{symbol="{sym}",severity="{sev}"}} '
            f'{1 if r["should_retrain"] else 0}'
        )
        h = r.get("hours_since_last_train")
        if h is not None:
            lines.append(
                f'neucast_drift_retrain_hours_since_last{{symbol="{sym}"}} {h:.3f}'
            )
    # Atomic write — write tempfile, rename. node_exporter only cares
    # about the final file content, but this prevents partial reads.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", action="append", default=None,
                   help="symbol to evaluate; pass multiple times")
    p.add_argument("--weights-dir", default="weights/highfreq",
                   help="directory holding <sym>_drift.json files")
    p.add_argument("--cooldown-hours", type=float, default=6.0)
    p.add_argument("--severity", action="append", default=None,
                   help='severities that trigger retrain (default: "high"). '
                        "Repeat to allow multiple: --severity warn --severity high.")
    p.add_argument("--dry-run", action="store_true",
                   help="evaluate + log; do NOT exec systemctl")
    p.add_argument("--textfile-out", default=None,
                   help="optional Prometheus textfile-collector path")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from app.highfreq.drift_retrain_policy import evaluate_drift_retrain_policy

    symbols = [s.upper() for s in (args.symbol or ["BTCUSDT", "ETHUSDT", "BNBUSDT"])]
    fire_on = tuple(s.lower() for s in (args.severity or ["high"]))
    weights_dir = Path(args.weights_dir)
    dsn = os.getenv("DATABASE_URL")
    if not dsn and not args.dry_run:
        logger.error("DATABASE_URL is required (set --dry-run to skip DB lookup)")
        return 2

    now = datetime.now(tz=timezone.utc)
    summary: list[dict[str, Any]] = []
    triggered = 0

    for sym in symbols:
        drift = _read_drift_json(weights_dir, sym)
        if not drift:
            logger.info("[%s] no drift JSON yet — skipping", sym)
            summary.append({
                "symbol": sym, "severity": "no_check_yet",
                "should_retrain": False,
                "hours_since_last_train": None,
                "reason": "no_drift_json",
            })
            continue

        severity = str(drift.get("severity", "ok"))
        last_started: datetime | None
        if dsn:
            try:
                last_started = _last_training_started_at(dsn, sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] DB lookup failed: %s — treating as cold start",
                               sym, exc)
                last_started = None
        else:
            last_started = None

        decision = evaluate_drift_retrain_policy(
            severity=severity,
            last_train_started_at=last_started,
            now=now,
            cooldown_hours=args.cooldown_hours,
            fire_on_severities=fire_on,
        )
        logger.info(
            "[%s] severity=%s should_retrain=%s reason=%s",
            sym, severity, decision.should_retrain, decision.reason,
        )

        rc = 0
        stderr = ""
        if decision.should_retrain:
            rc, stderr = _trigger_systemctl(sym, dry_run=args.dry_run)
            if rc == 0:
                triggered += 1
                logger.info("[%s] trainer unit started OK", sym)
            else:
                logger.error("[%s] systemctl rc=%d stderr=%s", sym, rc, stderr)

        summary.append({
            "symbol": sym,
            "severity": severity,
            "should_retrain": decision.should_retrain,
            "hours_since_last_train": decision.hours_since_last_train,
            "reason": decision.reason,
            "systemctl_rc": rc if decision.should_retrain else None,
        })

    _emit_textfile(
        Path(args.textfile_out) if args.textfile_out else None,
        per_symbol=summary,
    )

    # Plain-stdout summary line for cron logs.
    logger.info("drift_driven_retrain: triggered=%d / %d symbols evaluated",
                triggered, len(symbols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
