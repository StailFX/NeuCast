"""Auto-regenerated scoreboard of production model metrics over time.

Reads ``training_runs`` from Postgres + frozen-holdout JSON files
from disk, emits a clean markdown table of every spot-production
training run with the headline metrics. Annotates rows with the
release tag (``T.*``) inferred from the (feature_set, since_hours,
frozen_holdout_days) config trio that changed at each release
boundary.

The markdown is written to ``docs/highfreq/scoreboard.md`` and
checked into git on each notable release. After each new release
re-run this tool to refresh the file:

    python -m tools.scoreboard

The defence committee reads this artifact to see the FULL history
— which experiment caused which metric movement — without
hand-tracking commits.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingRun:
    id: int
    symbol: str
    started_at: datetime
    feature_set: str
    bar_minutes: int
    since_hours: float | None        # parsed from full_report.config if present
    frozen_holdout_days: int
    n_folds: int
    n_minutes_after_neutral_drop: int
    dir_acc_mean: float | None
    dir_acc_ci_low: float | None
    dir_acc_ci_high: float | None
    dir_acc_p_value: float | None
    base_rate: float | None
    calibrator_brier: float | None
    calibrator_ece: float | None
    conformal_q: float | None
    weights_path: str


def _fetch_runs(database_url: str, *, limit_per_symbol: int = 80) -> list[TrainingRun]:
    """Pull the recent spot-production training runs across the 3
    main symbols. ``limit_per_symbol`` caps memory; we expect ≤
    1 run/day per symbol per cron, so 80 covers ~80 days."""
    from sqlalchemy import create_engine, text

    sql = text("""
        SELECT id, symbol, run_started_at,
               n_folds, n_minutes_after_neutral_drop,
               dir_acc_mean, dir_acc_ci_low, dir_acc_ci_high,
               dir_acc_p_value, base_rate, frozen_holdout_days,
               weights_path, full_report
          FROM training_runs
         WHERE n_folds > 0
           AND symbol = ANY(:symbols)
           AND weights_path ~ '/weights/highfreq/[^/]+_[0-9]+m\\.cbm$'
         ORDER BY run_started_at ASC
    """)
    eng = create_engine(database_url, future=True)
    with eng.connect() as conn:
        rows = conn.execute(
            sql, {"symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"]},
        ).all()
    out: list[TrainingRun] = []
    for r in rows:
        full = r.full_report or {}
        # Some metric fields aren't promoted to columns (calibrator_*,
        # conformal_q_*) — pull from JSONB.
        out.append(TrainingRun(
            id=int(r.id),
            symbol=str(r.symbol),
            started_at=r.run_started_at,
            feature_set=str(full.get("feature_set", "?")),
            bar_minutes=int(full.get("bar_minutes", 1)),
            since_hours=None,  # not currently persisted; trainer config not in JSONB
            frozen_holdout_days=int(r.frozen_holdout_days or 0),
            n_folds=int(r.n_folds),
            n_minutes_after_neutral_drop=int(r.n_minutes_after_neutral_drop or 0),
            dir_acc_mean=_to_float(r.dir_acc_mean),
            dir_acc_ci_low=_to_float(r.dir_acc_ci_low),
            dir_acc_ci_high=_to_float(r.dir_acc_ci_high),
            dir_acc_p_value=_to_float(r.dir_acc_p_value),
            base_rate=_to_float(r.base_rate),
            calibrator_brier=_to_float(full.get("calibrator_brier")),
            calibrator_ece=_to_float(full.get("calibrator_ece")),
            conformal_q=_to_float(full.get("conformal_q_alpha_0_10")),
            weights_path=str(r.weights_path or ""),
        ))
    return out


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _release_tag(run: TrainingRun, prev: TrainingRun | None) -> str:
    """Heuristic: tag a run with a T.* release marker when its
    config differs from the previous run for that symbol.

    The ordering of conditions matters — we pick the FIRST change
    that explains this run's config delta. Anchored to the user-
    facing release tags so the scoreboard reads the same way as
    docs/highfreq/experiments.md.
    """
    if prev is None:
        return "(initial)"
    tags: list[str] = []
    if run.feature_set != prev.feature_set:
        tags.append(f"feature_set: {prev.feature_set}→{run.feature_set}")
    if run.frozen_holdout_days != prev.frozen_holdout_days:
        tags.append(
            f"holdout_days: {prev.frozen_holdout_days}→{run.frozen_holdout_days}"
        )
    if run.bar_minutes != prev.bar_minutes:
        tags.append(f"bar_minutes: {prev.bar_minutes}→{run.bar_minutes}")
    if (run.calibrator_brier is not None and prev.calibrator_brier is not None
            and abs(run.calibrator_brier - prev.calibrator_brier) > 0.005):
        tags.append("calibrator change")
    if (run.conformal_q is None) != (prev.conformal_q is None):
        tags.append("conformal added" if run.conformal_q else "conformal removed")
    if not tags:
        return ""
    return "; ".join(tags)


def _read_holdout_json(weights_path: Path, symbol: str) -> dict | None:
    """Read frozen-holdout eval JSON if it exists for the model
    pointed-to by this training run. The eval is run independently
    by ``tools.eval_frozen_holdout``; we surface its dir_acc as a
    separate column so the scoreboard distinguishes walk-forward
    CV from true-OOS holdout."""
    p = weights_path.with_name(f"{symbol.lower()}_1m_holdout.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def render_markdown(
    runs: list[TrainingRun], *, weights_dir: Path | None = None,
) -> str:
    """Render runs as a markdown scoreboard, grouped by symbol.

    Document structure (top to bottom):
      1. Title + intro
      2. Latest production metrics summary (1 row per symbol)
      3. Frozen-holdout numbers (gold-standard OOS, if available)
      4. Per-symbol full timeline tables (oldest → newest)
      5. Footer: timestamp + total run count
    """
    by_symbol: dict[str, list[TrainingRun]] = {}
    for r in runs:
        by_symbol.setdefault(r.symbol, []).append(r)

    lines: list[str] = []

    # ── 1. Header ──────────────────────────────────────────────
    lines.append("# NeuCast HF — Production Training Scoreboard\n")
    lines.append(
        "Auto-regenerated from the ``training_runs`` Postgres table on Tokyo. "
        "One row per spot-production trainer run (excludes /tmp experiments + "
        "futures-side runs). Single-glance view of metric evolution across "
        "the T.* releases.\n"
    )
    lines.append(
        "Re-generate via ``python -m tools.scoreboard`` (after each release).\n"
    )

    # ── 2. Latest production metrics summary ───────────────────
    lines.append("## Latest production metrics (one row per symbol)\n")
    lines.append(
        "| symbol | feature_set | dir_acc | CI [lo, hi] | p | Brier | ECE | conformal_q | n_oos |"
    )
    lines.append(
        "|--------|-------------|---------|-------------|---|-------|-----|-------------|-------|"
    )
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        if symbol not in by_symbol or not by_symbol[symbol]:
            continue
        latest = by_symbol[symbol][-1]
        ci = "—" if (latest.dir_acc_ci_low is None or latest.dir_acc_ci_high is None) else (
            f"[{latest.dir_acc_ci_low:.4f}, {latest.dir_acc_ci_high:.4f}]"
        )
        acc = "—" if latest.dir_acc_mean is None else f"{latest.dir_acc_mean:.4f}"
        p = "—" if latest.dir_acc_p_value is None else f"{latest.dir_acc_p_value:.2g}"
        brier = "—" if latest.calibrator_brier is None else f"{latest.calibrator_brier:.4f}"
        ece = "—" if latest.calibrator_ece is None else f"{latest.calibrator_ece:.4f}"
        cq = "—" if latest.conformal_q is None else f"{latest.conformal_q:.3f}"
        lines.append(
            f"| {symbol} | {latest.feature_set} | {acc} | {ci} | {p} | "
            f"{brier} | {ece} | {cq} | {latest.n_minutes_after_neutral_drop} |"
        )
    lines.append("")

    # ── 3. Frozen-holdout block (gold-standard OOS) ────────────
    if weights_dir is not None:
        holdout_rows: list[str] = []
        for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
            holdout = _read_holdout_json(weights_dir / "btcusdt_1m.cbm", symbol)
            if not holdout:
                continue
            n = holdout.get("n_eligible") or holdout.get("n_minutes_after_neutral_drop")
            acc = holdout.get("dir_acc")
            lo = holdout.get("dir_acc_ci_low")
            hi = holdout.get("dir_acc_ci_high")
            p = holdout.get("dir_acc_p_value")
            cutoff = holdout.get("holdout_cutoff_iso", "?")
            if acc is None:
                continue
            holdout_rows.append(
                f"| {symbol} | {acc:.4f} | "
                f"[{lo:.4f}, {hi:.4f}] | {p:.2g} | {n} | {cutoff} |"
            )
        if holdout_rows:
            lines.append(
                "## Frozen holdout (untouched by trainer — gold-standard OOS)\n"
            )
            lines.append(
                "| symbol | dir_acc | CI [lo, hi] | p | n_holdout | cutoff |"
            )
            lines.append(
                "|--------|---------|-------------|---|-----------|--------|"
            )
            lines.extend(holdout_rows)
            lines.append("")

    # ── 4. Per-symbol full timeline tables ─────────────────────
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        if symbol not in by_symbol:
            continue
        sym_runs = by_symbol[symbol]
        lines.append(f"## {symbol}\n")
        lines.append(
            "| run_started | feature_set | hd | n_folds | n_oos | "
            "dir_acc | CI low | CI high | p-value | Brier | ECE | conformal_q | config delta |"
        )
        lines.append(
            "|-------------|-------------|----|---------|-------|"
            "---------|--------|---------|---------|-------|-----|-------------|--------------|"
        )
        prev: TrainingRun | None = None
        for run in sym_runs:
            delta = _release_tag(run, prev)
            ts = run.started_at.strftime("%Y-%m-%d %H:%M")
            acc = "—" if run.dir_acc_mean is None else f"{run.dir_acc_mean:.4f}"
            lo = "—" if run.dir_acc_ci_low is None else f"{run.dir_acc_ci_low:.4f}"
            hi = "—" if run.dir_acc_ci_high is None else f"{run.dir_acc_ci_high:.4f}"
            p = "—" if run.dir_acc_p_value is None else f"{run.dir_acc_p_value:.2g}"
            brier = "—" if run.calibrator_brier is None else f"{run.calibrator_brier:.4f}"
            ece = "—" if run.calibrator_ece is None else f"{run.calibrator_ece:.4f}"
            cq = "—" if run.conformal_q is None else f"{run.conformal_q:.3f}"
            lines.append(
                f"| {ts} | {run.feature_set} | {run.frozen_holdout_days} | "
                f"{run.n_folds} | {run.n_minutes_after_neutral_drop} | "
                f"{acc} | {lo} | {hi} | {p} | {brier} | {ece} | {cq} | {delta} |"
            )
            prev = run
        lines.append("")

    # ── 5. Footer ──────────────────────────────────────────────
    lines.append(
        f"_Generated at {datetime.now(tz=timezone.utc).isoformat()} "
        f"from {len(runs)} training runs._"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/highfreq/scoreboard.md",
                   help="output markdown path")
    p.add_argument("--weights-dir", default="weights/highfreq",
                   help="path to weights dir (for reading holdout JSONs)")
    p.add_argument("--print", action="store_true",
                   help="also print the markdown to stdout")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is required")
        return 2

    runs = _fetch_runs(dsn)
    logger.info("loaded %d training runs", len(runs))
    md = render_markdown(runs, weights_dir=Path(args.weights_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    logger.info("wrote %s (%d bytes)", out_path, len(md))
    if args.print:
        print(md)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
