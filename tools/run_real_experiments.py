"""Прогонщик реальных экспериментов для §3.6 Таблицы 3.1 курсовой.

Берёт пять активов (золото, серебро, нефть, биткоин, акции Apple),
тянет историю через Yahoo Finance, запускает полный pipeline
``app.prediction.run_prediction`` и собирает метрики MAE / RMSE /
MAPE / R² / directional accuracy в JSON и markdown-таблицу.

Должен запускаться на сервере, где собран полный набор зависимостей
(TensorFlow / Keras + CatBoost + chronos-forecasting). Локально на
Mac пакеты в большинстве случаев не собираются — поэтому стандартный
запуск через SSH на Finland-VPS.

Run from Finland
----------------
::

    set -a; source /etc/neucast/env; set +a
    cd /opt/neucast
    /opt/neucast/venv/bin/python -m tools.run_real_experiments \\
        --since-days 1500 \\
        --days-ahead 30 \\
        --out /tmp/real_experiments.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("real_experiments")


# Пять активов для Таблицы 3.1 курсовой. Покрывают: фьючерс на золото
# (целевой актив темы), фьючерсы на серебро и нефть (другие сырьевые),
# биткоин (волатильная криптовалюта), акции Apple (классический
# fundamental-driven актив).
TICKERS = [
    ("GC=F", "Золото"),
    ("SLV", "Серебро (ETF SLV)"),       # iShares Silver Trust — длинная история без VIX-проблем
    ("USO", "Нефть (ETF USO)"),         # United States Oil Fund — больше истории чем CL=F
    ("BTC-USD", "Биткоин"),
    ("AAPL", "Apple"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since-days", type=int, default=1500,
                   help="hours of history to load (default 1500 ≈ 6 years)")
    p.add_argument("--days-ahead", type=int, default=30,
                   help="forecast horizon in trading days (default 30)")
    p.add_argument("--out", default="/tmp/real_experiments.json",
                   help="path to save metrics JSON")
    p.add_argument("--use-foundation", action="store_true",
                   help="enable Chronos foundation-model component (+20-40s/run)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from app.prediction import fetch_and_preprocess, run_prediction

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=int(args.since_days))
    logger.info(
        "experiment window: %s → %s (%d days)",
        start_date, end_date, args.since_days,
    )
    logger.info(
        "tickers: %s | days_ahead=%d | use_foundation=%s",
        [t for t, _ in TICKERS], args.days_ahead, args.use_foundation,
    )

    results: list[dict] = []
    for ticker, asset_label in TICKERS:
        t0 = time.monotonic()
        logger.info("=" * 60)
        logger.info("running %s (%s)", ticker, asset_label)
        try:
            df = fetch_and_preprocess(
                ticker, str(start_date), str(end_date),
            )
        except Exception as exc:
            logger.warning("fetch failed for %s: %s — skipping", ticker, exc)
            results.append({
                "ticker": ticker, "asset": asset_label,
                "ok": False, "reason": f"fetch_failed: {exc}",
            })
            continue

        if df is None or len(df) < 200:
            logger.warning(
                "%s: insufficient data (rows=%d) — skipping",
                ticker, 0 if df is None else len(df),
            )
            results.append({
                "ticker": ticker, "asset": asset_label,
                "ok": False, "reason": "insufficient_data",
                "n_rows": 0 if df is None else int(len(df)),
            })
            continue

        try:
            res = run_prediction(
                df, days_ahead=int(args.days_ahead),
                sentiment_score=0.0,
                use_foundation=bool(args.use_foundation),
            )
        except Exception as exc:
            logger.exception("predict failed for %s: %s", ticker, exc)
            results.append({
                "ticker": ticker, "asset": asset_label,
                "ok": False, "reason": f"predict_failed: {exc}",
            })
            continue

        elapsed = time.monotonic() - t0
        row = {
            "ticker": ticker,
            "asset": asset_label,
            "ok": True,
            "n_rows_loaded": int(len(df)),
            "mae": float(res.get("mae", 0)),
            "rmse": float(res.get("rmse", 0)),
            "mape": float(res.get("mape", 0)),
            "r2": float(res.get("r2", 0)),
            "dir_acc_pct": float(res.get("dir_acc", 0)),
            "model_name": res.get("model_name", "?"),
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(row)
        logger.info(
            "  done in %.1fs: MAE=%.2f RMSE=%.2f MAPE=%.2f%% R²=%.4f "
            "dir_acc=%.1f%%",
            elapsed, row["mae"], row["rmse"], row["mape"],
            row["r2"], row["dir_acc_pct"],
        )

    # ── Persist ────────────────────────────────────────────────────
    out_path = Path(args.out)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_start": str(start_date),
        "window_end": str(end_date),
        "since_days": int(args.since_days),
        "days_ahead": int(args.days_ahead),
        "use_foundation": bool(args.use_foundation),
        "rows": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out_path)

    # ── Markdown render (для прямой вставки в курсовую) ────────────
    print()
    print("Таблица 3.1 — Точность прогнозов ансамбля моделей по различным активам")
    print()
    print("| Тикер | Актив | MAE | RMSE | MAPE, % | R² | Dir. acc, % |")
    print("|-------|-------|-----|------|---------|-----|-------------|")
    for r in results:
        if not r.get("ok"):
            print(f"| {r['ticker']} | {r['asset']} | — | — | — | — | — |  *(не удалось: {r.get('reason', '?')})*")
            continue
        print(
            f"| {r['ticker']} | {r['asset']} | "
            f"{r['mae']:.2f} | {r['rmse']:.2f} | {r['mape']:.2f} | "
            f"{r['r2']:.4f} | {r['dir_acc_pct']:.1f} |"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
