"""Compare candidate fine-tuned models to production weights.

Use after running pretrain → fine-tune. Loads:
* Production ``weights/highfreq/<sym>_1m.cbm`` + metrics.json
* Candidate ``weights/highfreq/candidate/<sym>_1m.cbm`` + metrics.json
* Optionally robustness JSON for each (block bootstrap CI + permutation).

Prints a markdown comparison table — green-light deploy if EVERY symbol
shows BOTH:
1. dir_acc_mean(candidate) >= dir_acc_mean(production) - ε (tolerance)
2. CI lower bound (block bootstrap if available, else walk-forward)
   improves OR stays within ε

Otherwise red-light — keep production. Honest no-deploy is a feature,
not a failure.

Run from any host with the JSON files
-------------------------------------

::

    python -m tools.compare_candidate_models \\
        --production-dir weights/highfreq \\
        --candidate-dir weights/highfreq/candidate \\
        --tolerance 0.005
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ComparisonRow:
    symbol: str
    n_pred_prod: int = 0
    n_pred_cand: int = 0
    dir_acc_prod: float | None = None
    dir_acc_cand: float | None = None
    ci_low_prod: float | None = None
    ci_low_cand: float | None = None
    feature_set_prod: str | None = None
    feature_set_cand: str | None = None
    delta: float | None = None
    verdict: str = "unknown"
    notes: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("malformed JSON at %s: %s", path, exc)
        return None


def compare_one(
    symbol: str, *,
    production_dir: Path, candidate_dir: Path,
    tolerance: float = 0.005,
) -> ComparisonRow:
    """Compare a single symbol's production vs candidate metrics."""
    prod = _read_json(production_dir / f"{symbol.lower()}_1m_metrics.json")
    cand = _read_json(candidate_dir / f"{symbol.lower()}_1m_metrics.json")

    row = ComparisonRow(symbol=symbol)

    if prod:
        row.n_pred_prod = int(prod.get("n_minutes_after_neutral_drop", 0))
        row.dir_acc_prod = prod.get("dir_acc_mean")
        row.ci_low_prod = prod.get("dir_acc_ci_low")
        row.feature_set_prod = prod.get("feature_set")
    if cand:
        row.n_pred_cand = int(cand.get("n_minutes_after_neutral_drop", 0))
        row.dir_acc_cand = cand.get("dir_acc_mean")
        row.ci_low_cand = cand.get("dir_acc_ci_low")
        row.feature_set_cand = cand.get("feature_set")

    if row.dir_acc_prod is None or row.dir_acc_cand is None:
        row.verdict = "missing"
        row.notes.append("one of the metrics.json was missing or malformed")
        return row

    row.delta = row.dir_acc_cand - row.dir_acc_prod

    # Decision rule (conservative): candidate must NOT regress dir_acc
    # by more than tolerance, AND its CI lower bound must NOT drop
    # below production's by more than tolerance.
    acc_ok = row.delta >= -tolerance
    ci_ok = (
        row.ci_low_prod is None or row.ci_low_cand is None
        or (row.ci_low_cand >= row.ci_low_prod - tolerance)
    )
    if acc_ok and ci_ok and row.delta > 0:
        row.verdict = "deploy"
    elif acc_ok and ci_ok:
        row.verdict = "tied"
    else:
        row.verdict = "keep_production"
        if not acc_ok:
            row.notes.append(
                f"dir_acc regressed by {(-row.delta) * 100:.2f}pp (tolerance "
                f"{tolerance * 100:.2f}pp)"
            )
        if not ci_ok and row.ci_low_prod and row.ci_low_cand:
            row.notes.append(
                f"CI lower regressed by "
                f"{(row.ci_low_prod - row.ci_low_cand) * 100:.2f}pp"
            )
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production-dir", default="weights/highfreq",
                   help="path to production weights+metrics.json")
    p.add_argument("--candidate-dir", default="weights/highfreq/candidate",
                   help="path to candidate (fine-tuned) weights+metrics.json")
    p.add_argument("--symbol", action="append", default=None,
                   help="symbols to compare; default BTCUSDT,ETHUSDT,BNBUSDT")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="ε for the no-regression decision rule (default 0.5pp)")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    prod_dir = Path(args.production_dir)
    cand_dir = Path(args.candidate_dir)
    symbols = args.symbol or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    rows = [
        compare_one(
            sym, production_dir=prod_dir, candidate_dir=cand_dir,
            tolerance=args.tolerance,
        )
        for sym in symbols
    ]

    # Markdown table for human consumption.
    print()  # noqa: T201
    print("| symbol | prod | candidate | Δ | prod CI_lo | cand CI_lo | feature_set | verdict |")  # noqa: T201
    print("|--------|------|-----------|---|------------|------------|-------------|---------|")  # noqa: T201

    def _fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.4f}"

    def _fmt_pp(v: float | None) -> str:
        if v is None:
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v * 100:.2f}pp"

    for r in rows:
        prod_acc = _fmt(r.dir_acc_prod)
        cand_acc = _fmt(r.dir_acc_cand)
        delta = _fmt_pp(r.delta)
        prod_cilo = _fmt(r.ci_low_prod)
        cand_cilo = _fmt(r.ci_low_cand)
        fs = (
            f"{r.feature_set_prod} → {r.feature_set_cand}"
            if r.feature_set_prod != r.feature_set_cand
            else (r.feature_set_prod or "?")
        )
        print(  # noqa: T201
            f"| {r.symbol} | {prod_acc} | {cand_acc} | {delta} | "
            f"{prod_cilo} | {cand_cilo} | {fs} | **{r.verdict}** |"
        )

    print()  # noqa: T201
    deploys = [r for r in rows if r.verdict == "deploy"]
    keeps = [r for r in rows if r.verdict == "keep_production"]
    tieds = [r for r in rows if r.verdict == "tied"]
    print(  # noqa: T201
        f"Summary: deploy={len(deploys)} | tied={len(tieds)} | "
        f"keep_production={len(keeps)} | missing={len(rows) - len(deploys) - len(tieds) - len(keeps)}"
    )
    if any(r.notes for r in rows):
        print()  # noqa: T201
        print("Notes:")  # noqa: T201
        for r in rows:
            for note in r.notes:
                print(f"  - {r.symbol}: {note}")  # noqa: T201

    # Exit code semantics (used by the pipeline runner to decide
    # what to send to Telegram):
    #   0 = at least one deploy AND no keep_production → safe to deploy
    #   1 = at least one keep_production → keep production (regression)
    #   2 = all rows are 'missing' (nothing to compare)
    missings = [r for r in rows if r.verdict == "missing"]
    if missings and len(missings) == len(rows):
        return 2
    if keeps:
        return 1
    if deploys:
        return 0
    # All tied / mix of tied + missing → no regression but no improvement.
    # Treat as "keep production" (no reason to deploy noise).
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
