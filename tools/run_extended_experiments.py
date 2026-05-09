"""Расширенный прогонщик экспериментов для диплома (Гл. 3).

В отличие от ``tools/run_real_experiments.py``, который использует
5 активов и предназначен для иллюстрации в курсовой, этот скрипт
прогоняет полный pipeline на репрезентативной выборке из 30+
финансовых инструментов разных классов:

  * драгоценные металлы (золото, серебро, платина, палладий)
  * энергоносители (нефть, газ)
  * криптовалюты (BTC, ETH, BNB, SOL, ADA)
  * акции крупных компаний из разных секторов (tech, healthcare,
    energy, financials, consumer)
  * биржевые индексы (S&P 500, Nasdaq, Russell 2000)
  * валютные пары (EUR/USD, USD/JPY)

Расчёт занимает приблизительно 30 × 90 секунд = 45 минут на CPU.

Run from Finland
----------------
::

    set -a; source /etc/neucast/env; set +a
    cd /opt/neucast
    /opt/neucast/venv/bin/python -m tools.run_extended_experiments \\
        --since-days 1500 --days-ahead 30 \\
        --out /tmp/extended_experiments_30.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("extended_experiments")


# 30 активов в 6 классах. Тикеры выбраны так, чтобы у Yahoo Finance
# для каждого было >= 5 лет дневной истории и не возникало проблем
# с merge макропризнаков (VIX, DXY).
TICKERS = [
    # Драгметаллы и сырьё (4)
    ("GC=F",  "Золото (GC=F)",         "Сырьё"),
    ("SLV",   "Серебро (SLV)",         "Сырьё"),
    ("PPLT",  "Платина (PPLT)",        "Сырьё"),
    ("PALL",  "Палладий (PALL)",       "Сырьё"),
    # Энергоносители (2)
    ("USO",   "Нефть (USO)",           "Энергия"),
    ("UNG",   "Природный газ (UNG)",   "Энергия"),
    # Криптовалюты (5)
    ("BTC-USD",  "Биткоин",            "Крипто"),
    ("ETH-USD",  "Эфир",               "Крипто"),
    ("BNB-USD",  "Binance Coin",       "Крипто"),
    ("SOL-USD",  "Solana",             "Крипто"),
    ("ADA-USD",  "Cardano",            "Крипто"),
    # US large-cap акции (10)
    ("AAPL",  "Apple",                 "Акции (Tech)"),
    ("MSFT",  "Microsoft",             "Акции (Tech)"),
    ("NVDA",  "NVIDIA",                "Акции (Tech)"),
    ("GOOGL", "Alphabet",              "Акции (Tech)"),
    ("META",  "Meta Platforms",        "Акции (Tech)"),
    ("AMZN",  "Amazon",                "Акции (Consumer)"),
    ("TSLA",  "Tesla",                 "Акции (Consumer)"),
    ("JPM",   "JPMorgan Chase",        "Акции (Finance)"),
    ("XOM",   "ExxonMobil",            "Акции (Energy)"),
    ("JNJ",   "Johnson & Johnson",     "Акции (Healthcare)"),
    # Индексные ETF (3)
    ("SPY",   "S&P 500 (SPY)",         "Индекс"),
    ("QQQ",   "Nasdaq-100 (QQQ)",      "Индекс"),
    ("IWM",   "Russell 2000 (IWM)",    "Индекс"),
    # Валюты (3) — yfinance даёт через FX-тикеры
    ("EURUSD=X", "EUR/USD",            "Валюта"),
    ("USDJPY=X", "USD/JPY",            "Валюта"),
    ("GBPUSD=X", "GBP/USD",            "Валюта"),
    # Облигации / гос.долг (3)
    ("TLT",   "20+ Year T-Bond ETF",   "Облигации"),
    ("HYG",   "High-Yield Bond ETF",   "Облигации"),
    ("LQD",   "Investment-Grade Bond", "Облигации"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since-days", type=int, default=1500)
    p.add_argument("--days-ahead", type=int, default=30)
    p.add_argument("--out", default="/tmp/extended_experiments_30.json")
    p.add_argument("--use-foundation", action="store_true",
                   help="enable Chronos foundation model (+20s/run)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of tickers (for quick smoke runs)")
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
    tickers = TICKERS[:args.limit] if args.limit else TICKERS

    logger.info("=" * 60)
    logger.info(
        "extended experiments: %d tickers, window %s → %s, h=%d",
        len(tickers), start_date, end_date, args.days_ahead,
    )

    results: list[dict] = []
    for i, (ticker, asset, asset_class) in enumerate(tickers, 1):
        t0 = time.monotonic()
        logger.info("[%d/%d] %s (%s, %s)", i, len(tickers), ticker, asset, asset_class)
        try:
            df = fetch_and_preprocess(ticker, str(start_date), str(end_date))
            if df is None or len(df) < 200:
                results.append({
                    "ticker": ticker, "asset": asset, "asset_class": asset_class,
                    "ok": False, "reason": "insufficient_data",
                    "n_rows": 0 if df is None else int(len(df)),
                })
                continue

            res = run_prediction(
                df, days_ahead=int(args.days_ahead),
                sentiment_score=0.0,
                use_foundation=bool(args.use_foundation),
            )
        except Exception as exc:
            logger.warning("%s failed: %s", ticker, exc)
            results.append({
                "ticker": ticker, "asset": asset, "asset_class": asset_class,
                "ok": False, "reason": f"error: {exc}",
            })
            continue

        elapsed = time.monotonic() - t0
        row = {
            "ticker": ticker, "asset": asset, "asset_class": asset_class, "ok": True,
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
            "  done in %.1fs: MAPE=%.2f%% R²=%.4f dir_acc=%.1f%%",
            elapsed, row["mape"], row["r2"], row["dir_acc_pct"],
        )

    # ── Persist ────────────────────────────────────────────────────
    Path(args.out).write_text(json.dumps({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_start": str(start_date),
        "window_end": str(end_date),
        "since_days": int(args.since_days),
        "days_ahead": int(args.days_ahead),
        "use_foundation": bool(args.use_foundation),
        "n_tickers_attempted": len(tickers),
        "n_tickers_ok": sum(1 for r in results if r.get("ok")),
        "rows": results,
    }, indent=2, ensure_ascii=False))
    logger.info("wrote %s", args.out)

    # ── Markdown table ─────────────────────────────────────────────
    print()
    print("Таблица 3.X — Точность прогнозов на расширенной выборке (30 активов)")
    print()
    print("| класс | тикер | актив | MAPE,% | R² | Dir.acc,% |")
    print("|-------|-------|-------|--------|------|-----------|")
    for r in results:
        if not r.get("ok"):
            print(f"| {r['asset_class']} | {r['ticker']} | {r['asset']} | — | — | — |")
            continue
        print(
            f"| {r['asset_class']} | {r['ticker']} | {r['asset']} | "
            f"{r['mape']:.2f} | {r['r2']:.4f} | {r['dir_acc_pct']:.1f} |"
        )

    # ── Aggregate by asset class ──────────────────────────────────
    print()
    print("Сводка по классам активов:")
    by_class: dict[str, list] = {}
    for r in results:
        if not r.get("ok"):
            continue
        by_class.setdefault(r["asset_class"], []).append(r)
    for cls, rows in sorted(by_class.items()):
        if not rows:
            continue
        n = len(rows)
        avg_mape = sum(r["mape"] for r in rows) / n
        avg_r2 = sum(r["r2"] for r in rows) / n
        avg_dir = sum(r["dir_acc_pct"] for r in rows) / n
        print(
            f"  {cls:<22} n={n}  ⌀MAPE={avg_mape:.2f}%  "
            f"⌀R²={avg_r2:.4f}  ⌀dir_acc={avg_dir:.1f}%"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
