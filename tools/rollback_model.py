"""Rollback a production model to an archived snapshot (T.17.d).

Usage
-----

List available snapshots for BTC::

    python -m tools.rollback_model --symbol BTCUSDT --list

Rollback to a specific timestamp::

    python -m tools.rollback_model --symbol BTCUSDT \\
        --ts 20260503T040000Z

Interactive (default) — prints menu, asks operator to pick::

    python -m tools.rollback_model --symbol BTCUSDT

The rollback action:
1. Archives the CURRENT live weights (so a misclick is undoable).
2. Copies the chosen snapshot's .cbm + metrics.json + calibrator.pkl
   back over the live names.
3. Paper-trader's mtime-watcher picks up the change within ~60 s
   (no service restart needed).

Production safety
-----------------

This tool is meant for ad-hoc operator action. After a rollback:
* Verify metrics: ``curl /api/highfreq/training_report?symbol=...``
* Watch realized accuracy: ``curl /api/highfreq/conditional_accuracy``
* If the rollback also regresses, roll forward to the latest
  archived (the rollback ITSELF created an archive of what you
  just clobbered).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _weights_path_for(symbol: str, *, root: Path, bar_minutes: int) -> Path:
    """Mirror predictor.weights_path_for_symbol convention."""
    sym_lower = symbol.lower()
    return root / "weights" / "highfreq" / f"{sym_lower}_{bar_minutes}m.cbm"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True,
                   help="trading pair (e.g. BTCUSDT)")
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--root", default=os.getenv("NEUCAST_ROOT", "/opt/neucast"),
                   help="repo root containing weights/highfreq/")
    p.add_argument("--list", action="store_true",
                   help="just print available snapshots and exit")
    p.add_argument("--ts", default=None,
                   help="snapshot timestamp to roll back to "
                        "(YYYYMMDDTHHMMSSZ); if not provided, prompts "
                        "interactively")
    p.add_argument("--yes", "-y", action="store_true",
                   help="skip the interactive confirmation prompt")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from app.highfreq.model_archive import (
        list_snapshots, rollback_to,
    )

    weights_path = _weights_path_for(
        args.symbol, root=Path(args.root), bar_minutes=args.bar_minutes,
    )
    snapshots = list_snapshots(weights_path)
    if not snapshots:
        logger.error("no snapshots in archive for %s", weights_path)
        return 1

    print(f"Available snapshots for {args.symbol} ({weights_path.name}):")  # noqa: T201
    for i, snap in enumerate(snapshots, start=1):
        size_kb = snap["size_bytes"] / 1024
        has_metrics = "✓" if snap["metrics_path"] else " "
        has_cal = "✓" if snap["calibrator_path"] else " "
        print(f"  [{i}] {snap['ts_iso']}  cbm={size_kb:.1f}KB  "  # noqa: T201
              f"metrics={has_metrics}  calibrator={has_cal}")

    if args.list:
        return 0

    # Pick target.
    if args.ts is not None:
        target_ts = args.ts
    else:
        # Interactive prompt.
        try:
            choice = input("\nRollback to snapshot # (enter to abort): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\naborted")  # noqa: T201
            return 1
        if not choice:
            print("aborted")  # noqa: T201
            return 1
        try:
            idx = int(choice) - 1
            target_ts = snapshots[idx]["ts_iso"]
        except (ValueError, IndexError):
            logger.error("invalid choice %r", choice)
            return 2

    # Confirmation.
    if not args.yes:
        try:
            confirm = input(
                f"Rollback {args.symbol} to {target_ts}? "
                f"(current weights will be archived first) [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\naborted")  # noqa: T201
            return 1
        if confirm not in ("y", "yes"):
            print("aborted")  # noqa: T201
            return 1

    try:
        result = rollback_to(weights_path, target_ts)
    except FileNotFoundError as exc:
        logger.error("rollback failed: %s", exc)
        return 2

    print(f"\n✓ Rollback complete:")  # noqa: T201
    for k, v in result.items():
        print(f"   {k}: {v}")  # noqa: T201
    print("\nPaper-trader will pick up the change within 60 seconds.")  # noqa: T201
    print("Verify with:")  # noqa: T201
    print(f"  curl https://neucast.ru/api/highfreq/forecast?symbol={args.symbol}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
