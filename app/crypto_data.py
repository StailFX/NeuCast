"""
Crypto OHLCV fetcher with multi-source fallback (A1.1).

Зачем не просто yfinance
────────────────────────
Yahoo Finance для крипты:
  • Hourly ограничен ~720 днями.
  • ~50% баров с volume=0 (Yahoo агрегирует данные с потерей качества).
  • Минимальный шаг — 1h, нет 1m/5m.

Binance Spot — direct exchange data:
  • Public API без ключа (`/api/v3/klines`).
  • Любой период истории (для BTC: с июля 2017, ~70K часовых баров).
  • Реальный volume (никаких нулей).
  • Шаги: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w.
  • Free, rate-limit ≈ 1200 req/min на IP.

Что это даёт hourly-skill probe
───────────────────────────────
yfinance 1h × 720d = 17K баров → CI ±1.6%.
Binance 1h × 5y   = 43K баров → CI ±1.0%.
Binance 5m × 2y   = 210K баров → CI ±0.3% (статистически решающий ответ).

Архитектура
───────────
priority chain:
  1) Binance Spot (`fetch_binance_klines`) — primary
  2) yfinance — fallback (если Binance geo-блокирован / network fail)
graceful: любая ошибка → следующий источник; всё None → (None, "fail").

Symbol mapping yfinance → Binance:
  BTC-USD  → BTCUSDT
  ETH-USD  → ETHUSDT
  SOL-USD  → SOLUSDT
  ...
USDT pegged USD (~1.0 ± 0.05%) — практически идентичная цена для skill-probe.
Override через env CRYPTO_SYMBOL_MAP="BTC-USD:BTCUSDT,ETH-USD:ETHUSDT".

Caching: финальный DataFrame в Redis на CRYPTO_DATA_CACHE_TTL (1 час),
ключ включает (ticker, interval, lookback_days, source).
"""

import os
import time
import logging
import hashlib
import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Binance config ───────────────────────────────────────────────────
BINANCE_BASE = os.getenv("BINANCE_API_BASE", "https://api.binance.com")
BINANCE_TIMEOUT = float(os.getenv("BINANCE_TIMEOUT", "10"))
BINANCE_MAX_LIMIT = 1000  # API максимум на один запрос
BINANCE_MAX_PAGES = int(os.getenv("BINANCE_MAX_PAGES", "100"))  # safety

# ── Cache ────────────────────────────────────────────────────────────
CRYPTO_DATA_CACHE_TTL = int(os.getenv("CRYPTO_DATA_CACHE_TTL", "3600"))  # 1 час

# ── Symbol mapping ───────────────────────────────────────────────────
# Можно перекрыть через env CRYPTO_SYMBOL_MAP="BTC-USD:BTCUSDT,ETH-USD:ETHUSDT"
def _parse_symbol_map_override() -> dict[str, str]:
    raw = os.getenv("CRYPTO_SYMBOL_MAP", "").strip()
    if not raw:
        return {}
    out = {}
    for pair in raw.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            out[k.strip().upper()] = v.strip().upper()
    return out

SYMBOL_MAP_OVERRIDE = _parse_symbol_map_override()


def _yf_to_binance(ticker: str) -> str:
    """Map BTC-USD → BTCUSDT (Binance Spot pair)."""
    t = ticker.upper().strip()
    if t in SYMBOL_MAP_OVERRIDE:
        return SYMBOL_MAP_OVERRIDE[t]
    # Generic: strip -USD/-USDT/-USDC/-EUR suffix, append USDT
    for suffix in ("-USDT", "-USDC", "-USD", "-EUR"):
        if t.endswith(suffix):
            base = t[:-len(suffix)]
            return f"{base}USDT"
    # Уже в Binance-формате (e.g. BTCUSDT)
    return t


def _cache_get(key: str):
    try:
        from app.prediction import _cache_get as _gp
        return _gp(key)
    except Exception:
        return None


def _cache_set(key: str, value, ttl: int):
    try:
        from app.prediction import _cache_set as _sp
        _sp(key, value, ttl)
    except Exception:
        pass


def _binance_klines_page(symbol: str, interval: str, start_ms: int,
                        end_ms: int, limit: int = BINANCE_MAX_LIMIT) -> list:
    """Fetch one page of klines from Binance public API."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=BINANCE_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_binance_klines(ticker: str, lookback_days: int = 1825,
                         interval: str = "1h") -> pd.DataFrame | None:
    """
    Fetch up to lookback_days of OHLCV from Binance Spot.

    Args:
        ticker: yfinance-стиль ("BTC-USD") или binance-стиль ("BTCUSDT").
        lookback_days: сколько дней истории. 1825=5 лет.
        interval: "1m" / "5m" / "15m" / "30m" / "1h" / "4h" / "1d" / ...

    Returns:
        DataFrame с DatetimeIndex (UTC) и колонками [Open, High, Low, Close, Volume],
        либо None если Binance недоступен (geo-block / network fail / no data).
    """
    symbol = _yf_to_binance(ticker)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_days * 24 * 3600 * 1000

    all_rows: list = []
    cursor = start_ms
    pages = 0
    t0 = time.time()

    try:
        while cursor < end_ms and pages < BINANCE_MAX_PAGES:
            data = _binance_klines_page(symbol, interval, cursor, end_ms,
                                        BINANCE_MAX_LIMIT)
            if not data:
                break
            all_rows.extend(data)
            last_open = int(data[-1][0])
            new_cursor = last_open + 1
            if new_cursor <= cursor:
                break  # safety against infinite loop
            cursor = new_cursor
            pages += 1
            # Конечная страница (меньше limit'а вернулось)
            if len(data) < BINANCE_MAX_LIMIT:
                break

        if not all_rows:
            logger.warning("Binance returned no data for %s %s", symbol, interval)
            return None

        # klines schema:
        # [openTime, open, high, low, close, volume, closeTime,
        #  quoteAssetVolume, numTrades, takerBuyBaseVol, takerBuyQuoteVol, ignore]
        cols = ["openTime", "Open", "High", "Low", "Close", "Volume",
                "closeTime", "qav", "trades", "tbbav", "tbqav", "ignore"]
        df = pd.DataFrame(all_rows, columns=cols)
        df["openTime"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
        df = df.set_index("openTime")
        for c in ("Open", "High", "Low", "Close", "Volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Дедуп на стыках страниц (Binance иногда дублирует крайний бар)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        elapsed = time.time() - t0
        logger.info(
            "Binance %s %s: %d rows in %d pages, %.1fs (range %s..%s)",
            symbol, interval, len(df), pages, elapsed,
            df.index[0], df.index[-1],
        )
        return df
    except requests.exceptions.HTTPError as e:
        # 451 = geo-block, 418 = IP banned, 429 = rate limit
        status = e.response.status_code if e.response is not None else None
        logger.warning("Binance HTTP %s for %s: %s", status, symbol, e)
        return None
    except requests.exceptions.RequestException as e:
        logger.warning("Binance network error for %s: %s", symbol, e)
        return None
    except Exception as e:
        logger.warning("Binance unexpected error for %s: %s", symbol, e, exc_info=True)
        return None


def fetch_yfinance_hourly(ticker: str, lookback_days: int = 720,
                          interval: str = "1h") -> pd.DataFrame | None:
    """yfinance fallback. Возвращает тот же формат что и Binance."""
    try:
        import yfinance as yf
        # yfinance hourly limit ≈ 730d
        period_days = min(lookback_days, 720)
        period = f"{period_days}d"
        df_raw = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=False)
        if df_raw is None or df_raw.empty:
            return None
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        keep_cols = [c for c in ("Open", "High", "Low", "Close", "Volume")
                     if c in df_raw.columns]
        if "Close" not in keep_cols:
            return None
        df = df_raw[keep_cols].copy()
        # Гарантируем UTC tzinfo на индексе (Binance отдаёт UTC; для honest
        # сравнения features это важно — иначе hour-of-day будет в TZ юзера).
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return None


def fetch_crypto_ohlcv(
    ticker: str,
    lookback_days: int = 1825,
    interval: str = "1h",
    prefer: str = "binance",
) -> tuple[pd.DataFrame | None, str]:
    """
    Multi-source crypto OHLCV fetcher with cache.

    Args:
        ticker: yfinance-style ("BTC-USD") или Binance ("BTCUSDT").
        lookback_days: сколько дней истории попросить (для yfinance клипуется до 720).
        interval: "1m", "5m", "15m", "1h", "4h", "1d", ...
        prefer: "binance" | "yfinance" | "auto" — порядок попыток.

    Returns:
        (df, source) — DataFrame с UTC-индексом и OHLCV, либо (None, "fail").
        source ∈ {"binance", "yfinance", "fail", "binance_cached", ...}.
    """
    cache_key = "neucast:crypto_ohlcv:" + hashlib.md5(
        f"{ticker}|{interval}|{lookback_days}|{prefer}".encode()
    ).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        df, src = cached
        logger.info("crypto_ohlcv cache HIT %s %s [%s, %d rows]",
                    ticker, interval, src, len(df) if df is not None else 0)
        return df, src

    # Order of attempts based on `prefer`
    order = []
    if prefer == "binance":
        order = ["binance", "yfinance"]
    elif prefer == "yfinance":
        order = ["yfinance", "binance"]
    else:  # "auto" / unknown — same as binance-first
        order = ["binance", "yfinance"]

    for src in order:
        if src == "binance":
            df = fetch_binance_klines(ticker, lookback_days, interval)
        else:
            df = fetch_yfinance_hourly(ticker, lookback_days, interval)
        if df is not None and not df.empty:
            _cache_set(cache_key, (df, src), CRYPTO_DATA_CACHE_TTL)
            return df, src

    return None, "fail"
