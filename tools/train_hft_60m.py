"""Train 1-hour HFT CatBoost models for BTC / ETH / BNB on VPS.

The live HFT trainer (``app.highfreq.trainer``) reads from the Tokyo
Postgres ``highfreq_ofi_1s`` table — L2 order-book seconds — and that
table is only on Tokyo. On a fresh VPS we don't have it.

This script uses ``app.highfreq.pretrain`` instead, which works on a
parquet of Binance Klines (no DB, no L2). That gives us a 60-minute
CatBoost trained on multi-year OHLCV history — exactly what's missing
from Tokyo's weights dir (only ``btcusdt_60m_metrics.json`` exists,
no ``.cbm`` weight file).

Pipeline per symbol
===================

1. **Download** 60-minute Klines for the past 3 years via
   ``tools.binance_klines_download --interval 1h --years 3``. Output:
   ``data/historical/{symbol}_1h_klines.parquet``.
2. **Pretrain** CatBoost via ``app.highfreq.pretrain`` with
   ``--bar-minutes 60 --horizon 60``. Output:
   ``weights/highfreq/{symbol_lower}_60m.cbm``
   ``weights/highfreq/{symbol_lower}_60m_metrics.json``
3. **Validate** the .cbm loads cleanly with ``CatBoostClassifier.load_model``.

After all three symbols complete, ``rsync`` the .cbm + .json files
back to Tokyo's ``/opt/neucast/weights/highfreq/`` and restart
``neucast-highfreq-web.service``. Then the dashboard's
``?horizon=60`` parameter will route to the new model — and the
HorizonPill's "1h скоро" badge can be flipped off.

Run
===

::

    python -m tools.train_hft_60m \\
        --symbols BTCUSDT,ETHUSDT,BNBUSDT \\
        --years 3 \\
        --weights-dir weights/highfreq \\
        --data-dir data/historical

Expected runtime on c4-16-64 VPS: ~30-90 min per symbol (data download
~5 min, pretrain CV ~25-80 min depending on n_folds). Total ~3 hours
for three symbols.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("train_hft_60m")


def _run_subprocess(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a subprocess with stdout/stderr streamed to our logger."""
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=False, text=True,
    )
    return int(proc.returncode)


def _download_klines(
    symbol: str, years: float, out_dir: Path,
) -> Path | None:
    """Run ``tools.binance_klines_download`` for one symbol at 60m bars.

    Returns the parquet path on success, None on failure.
    """
    parquet = out_dir / f"{symbol.lower()}_1h_klines.parquet"
    if parquet.exists() and parquet.stat().st_size > 1024:
        logger.info("%s: parquet already present (%d KB) — skipping download",
                    symbol, parquet.stat().st_size // 1024)
        return parquet

    rc = _run_subprocess([
        sys.executable, "-m", "tools.binance_klines_download",
        "--symbol", symbol,
        "--interval", "1h",
        "--years", str(years),
        "--out-dir", str(out_dir),
    ])
    if rc != 0:
        logger.error("%s: Klines download failed (rc=%d)", symbol, rc)
        return None
    if not parquet.exists():
        logger.error("%s: expected output not found at %s", symbol, parquet)
        return None
    return parquet


def _pretrain_60m(
    symbol: str, parquet: Path, weights_dir: Path,
) -> dict:
    """Run ``app.highfreq.pretrain`` for the 60m horizon.

    Output paths follow the project naming convention:
      ``{symbol_lower}_60m.cbm`` + ``{symbol_lower}_60m_metrics.json``
    so they slot straight into Tokyo's weights/highfreq/ directory.
    """
    sym_l = symbol.lower()
    cbm_out = weights_dir / f"{sym_l}_60m.cbm"
    json_out = weights_dir / f"{sym_l}_60m_metrics.json"

    t0 = time.monotonic()
    rc = _run_subprocess([
        sys.executable, "-m", "app.highfreq.pretrain",
        "--data", str(parquet),
        "--symbol", symbol,
        "--out", str(cbm_out),
        "--report", str(json_out),
        "--bar-minutes", "60",
        "--horizon", "60",
    ])
    elapsed = time.monotonic() - t0

    result = {
        "symbol": symbol,
        "ok": rc == 0 and cbm_out.exists(),
        "rc": int(rc),
        "elapsed_seconds": round(elapsed, 1),
        "cbm_path": str(cbm_out),
        "json_path": str(json_out),
        "cbm_size_kb": (cbm_out.stat().st_size // 1024
                        if cbm_out.exists() else 0),
    }
    if json_out.exists():
        try:
            metrics = json.loads(json_out.read_text())
            result["dir_acc_mean"] = metrics.get("dir_acc_mean")
            result["dir_acc_ci_low"] = metrics.get("dir_acc_ci_low")
            result["dir_acc_ci_high"] = metrics.get("dir_acc_ci_high")
            result["n_folds"] = metrics.get("n_folds")
        except Exception as exc:
            logger.warning("%s: failed to read metrics JSON: %s", symbol, exc)
    return result


def _validate_cbm(cbm_path: Path) -> bool:
    """Smoke-test: can CatBoost load this .cbm without error?"""
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(cbm_path))
        return True
    except Exception as exc:
        logger.error("validate %s: load failed — %s", cbm_path, exc)
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT",
                   help="comma-separated symbol list")
    p.add_argument("--years", type=float, default=3.0,
                   help="years of history to download (default 3)")
    p.add_argument("--weights-dir", default="weights/highfreq",
                   help="where to write .cbm + metrics.json")
    p.add_argument("--data-dir", default="data/historical",
                   help="where to keep Klines parquet files")
    p.add_argument("--skip-download", action="store_true",
                   help="reuse existing parquet files in --data-dir")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    weights_dir = Path(args.weights_dir)
    data_dir = Path(args.data_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(
        "HFT 60m training: symbols=%s, years=%.1f, weights→%s, data→%s",
        symbols, args.years, weights_dir, data_dir,
    )

    results: list[dict] = []
    for sym in symbols:
        logger.info("─" * 60)
        logger.info("Processing %s", sym)

        # ── 1. Download Klines ──
        if args.skip_download:
            parquet = data_dir / f"{sym.lower()}_1h_klines.parquet"
            if not parquet.exists():
                logger.error("%s: --skip-download but no parquet at %s",
                             sym, parquet)
                results.append({"symbol": sym, "ok": False,
                                "reason": "no_parquet"})
                continue
        else:
            parquet = _download_klines(sym, args.years, data_dir)
            if parquet is None:
                results.append({"symbol": sym, "ok": False,
                                "reason": "download_failed"})
                continue

        # ── 2. Pretrain CatBoost ──
        r = _pretrain_60m(sym, parquet, weights_dir)
        results.append(r)
        if not r["ok"]:
            logger.warning("%s: pretrain failed (rc=%d) — skipping validation",
                           sym, r["rc"])
            continue

        # ── 3. Validate ──
        if _validate_cbm(Path(r["cbm_path"])):
            logger.info(
                "%s ✓  CBM %d KB, dir_acc_mean=%s, n_folds=%s, elapsed %.0fs",
                sym, r["cbm_size_kb"],
                r.get("dir_acc_mean"), r.get("n_folds"), r["elapsed_seconds"],
            )
        else:
            r["ok"] = False
            r["reason"] = "validate_failed"
            logger.error("%s ✗  cbm failed to reload", sym)

    # ── Final summary ──
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "symbols": symbols,
        "years_history": args.years,
        "weights_dir": str(weights_dir),
        "results": results,
        "n_ok": sum(1 for r in results if r.get("ok")),
    }
    summary_path = weights_dir / "60m_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("=" * 60)
    logger.info("wrote %s", summary_path)

    print()
    print("Сводка обучения 60-минутных моделей:")
    for r in results:
        if r.get("ok"):
            print(
                f"  {r['symbol']:<8} ✓  "
                f"dir_acc={r.get('dir_acc_mean', 'NA')}  "
                f"n_folds={r.get('n_folds', 'NA')}  "
                f"size={r.get('cbm_size_kb', 0)} KB  "
                f"elapsed={r.get('elapsed_seconds', 0):.0f}s"
            )
        else:
            print(
                f"  {r['symbol']:<8} ✗  reason={r.get('reason', r.get('rc', '?'))}"
            )
    print()
    print(f"Веса лежат в {weights_dir}/. rsync на Tokyo:")
    print(
        "  rsync -avz weights/highfreq/*_60m.* "
        "root@147.45.49.40:/opt/neucast/weights/highfreq/"
    )

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
