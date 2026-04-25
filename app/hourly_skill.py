"""
Hourly directional-skill probe (Tier A, A1).

Зачем это нужно
───────────────
Daily pipeline (TCN + boostings + Foundation) тренируется на 1d барах. На
горизонте 5 дней test-fold = 200 точек, что даёт bootstrap CI ±7%. На таком
шуме разница между dir_acc=49% и dir_acc=55% статистически НЕ значима
(B1 показал, что 95% CI почти всегда пересекает 50%).

Решение: отдельный hourly-probe, который НЕ участвует в основном прогнозе
(не трогаем модели/калибровку/sentiment), а только отвечает на честный
вопрос: «есть ли вообще direction-signal в этом тикере на коротком
горизонте». На hourly за 720 дней получаем ≈ 17K баров → CI ±1.5%, что
позволяет уверенно различать coin-flip от реального skill.

Архитектура
───────────
• Источник данных: yfinance с interval="1h", period="720d" (макс что отдаёт
  yfinance бесплатно). Кэшируем в Redis на HOURLY_CACHE_TTL.
• Фичи: лаги log-returns (1h, 4h, 12h, 24h, 48h), RSI(14), realized vol,
  hour-of-day, day-of-week. Минимум что нужно — без peeking в будущее.
• Модель: LightGBM binary classifier (быстрее TCN, нативно binary,
  устойчив к шуму). Целевая = sign(close[t+1] - close[t]).
• Сплит: 80/20 chronological, без shuffle, embargo 24 часа (выбрасываем
  первые/последние записи буфера от leak'а через MA features).
• Метрика: dir_acc на test-fold + bootstrap 95% CI (тот же подход что в
  основной prediction.py — единообразие интерфейса).
• Last signal: вероятность роста на следующий час, чтобы UI мог показать
  «модель видит up/down/flat сейчас».

Crypto-only (на старте)
───────────────────────
yfinance hourly для крипты даёт 24/7 → много данных и одинаковый процесс.
Для акций hourly есть только в торговые часы (≈ 6.5h × 5d = 32.5h/week
vs 168h/week для крипты), gaps усложняют features. Поэтому на A1 включаем
ТОЛЬКО для крипты (ticker matches CRYPTO_REGEX). Расширим в A2.

Безопасность вызова
───────────────────
• Любая ошибка (no data, lightgbm fail, network) → возвращаем None,
  основной pipeline не падает.
• Кэш по тикеру (TTL = HOURLY_CACHE_TTL, default 1h) — можно переобучать
  не чаще раза в час, hourly данные обновляются раз в час.
• Управляется env-флагом HOURLY_DIAGNOSTIC (default off, чтобы не ломать
  существующий UX до отладки).
"""

import os
import re
import time
import hashlib
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config (env-tunable) ─────────────────────────────────────────────
# Крипто-тикеры из yfinance: BTC-USD, ETH-USD, SOL-USD, ... — `-USD` в конце
# Можно перекрыть whitelist'ом (CSV). Если whitelist не задан — regex по `-USD$`.
CRYPTO_WHITELIST = os.getenv("HOURLY_CRYPTO_WHITELIST", "").strip()
CRYPTO_REGEX = re.compile(r"-(USD|USDT|USDC|EUR)$", re.IGNORECASE)

# Источник данных: Binance (5+ лет) или yfinance (720d). По умолчанию
# пробуем Binance первым (богаче история, чище volume), fallback на yfinance.
HOURLY_SOURCE = os.getenv("HOURLY_SOURCE", "binance")  # "binance" | "yfinance" | "auto"
HOURLY_LOOKBACK_DAYS = int(os.getenv("HOURLY_LOOKBACK_DAYS", "1825"))  # 5 years default
HOURLY_INTERVAL = os.getenv("HOURLY_INTERVAL", "1h")
# Legacy: оставляем для обратной совместимости с тестами, не используется в коде.
HOURLY_PERIOD = os.getenv("HOURLY_PERIOD", f"{HOURLY_LOOKBACK_DAYS}d")

# Cache TTL: 1 час (новый бар каждые 60 минут — нет смысла переобучать чаще)
HOURLY_CACHE_TTL = int(os.getenv("HOURLY_CACHE_TTL", "3600"))

# Bootstrap CI: 1000 ресэмплов хватает для стабильности до 0.1%
HOURLY_BOOTSTRAP = int(os.getenv("HOURLY_BOOTSTRAP", "1000"))

# LightGBM hyperparams (умеренные — не нужно SOTA, нужен честный baseline)
HOURLY_NUM_LEAVES = int(os.getenv("HOURLY_NUM_LEAVES", "31"))
HOURLY_N_ESTIMATORS = int(os.getenv("HOURLY_N_ESTIMATORS", "200"))
HOURLY_LEARNING_RATE = float(os.getenv("HOURLY_LEARNING_RATE", "0.05"))
HOURLY_MIN_DATA_IN_LEAF = int(os.getenv("HOURLY_MIN_DATA_IN_LEAF", "100"))

# Минимум баров для надёжной оценки (иначе CI слишком широкий)
HOURLY_MIN_BARS = int(os.getenv("HOURLY_MIN_BARS", "2000"))


def is_crypto_ticker(ticker: str) -> bool:
    """Решает, применимо ли hourly-probe к этому тикеру."""
    if CRYPTO_WHITELIST:
        wl = {t.strip().upper() for t in CRYPTO_WHITELIST.split(",") if t.strip()}
        return ticker.upper() in wl
    return bool(CRYPTO_REGEX.search(ticker))


def _engineer_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV hourly bars.

    Strict no-leak rules:
      • Все features считаются по close[:t] (включительно), target = sign(close[t+1]-close[t]).
      • Таргет shiftнут на -1 (predict next bar) и затем NaN отброшен.
      • Lag-features считаются как log-returns (стационарность) и
        нормализованы pct_change-based, чтобы XGB не запоминал абсолютные
        уровни цены.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(np.float64)

    # Log returns (стационарны → лучше для tree-models)
    log_ret = np.log(close / close.shift(1))
    out["lr_1"] = log_ret
    out["lr_4"] = log_ret.rolling(4).sum()       # 4h cumulative
    out["lr_12"] = log_ret.rolling(12).sum()     # half-day
    out["lr_24"] = log_ret.rolling(24).sum()     # 1d
    out["lr_48"] = log_ret.rolling(48).sum()     # 2d

    # Realized volatility (rolling std of log-returns)
    out["vol_24"] = log_ret.rolling(24).std()
    out["vol_72"] = log_ret.rolling(72).std()

    # RSI(14) — классический momentum
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Volume features (если есть). Yahoo для крипты часто отдаёт 0 на старых
    # барах (~50% от 720d) → если делать `replace(0, NaN)` → dropna убьёт
    # большую часть выборки. Поэтому: log1p(0)=0 (без NaN), а ratio считаем
    # с защитой — где знаменатель ≤ 0 или нечислов, кладём нейтральное 1.0.
    if "Volume" in df.columns:
        vol = df["Volume"].astype(np.float64).clip(lower=0).fillna(0)
        out["log_vol"] = np.log1p(vol)
        rolling_mean = vol.rolling(24).mean()
        # Безопасное деление: NaN/inf/0-знаменатель → 1.0 (neutral signal)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                rolling_mean.values > 0,
                vol.values / np.where(rolling_mean.values > 0, rolling_mean.values, 1.0),
                1.0,
            )
        out["vol_ratio_24"] = pd.Series(ratio, index=df.index).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    # Calendar features (cyclical encoding — хуже для tree-models, поэтому raw int)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek

    # Target: знак изменения следующего бара (1 = up, 0 = down/flat)
    out["target"] = (close.shift(-1) > close).astype(np.int8)

    return out


def _bootstrap_ci(hits: np.ndarray, n_iter: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap percentile 95% CI на пропорции hits. Тот же стиль что в prediction.py."""
    if len(hits) < 30:
        p = float(hits.mean() * 100)
        return p, p
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(n_iter, len(hits)))
    boot = hits[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)


def _cache_get(key: str):
    """Lightweight Redis cache wrapper (parallels prediction.py._cache_get).
    Импортируем lazy, чтобы модуль грузился даже если redis недоступен
    (тогда просто без кэша)."""
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


def compute_hourly_skill(ticker: str) -> dict | None:
    """
    Compute honest hourly directional-skill estimate for a crypto ticker.

    Returns:
        dict with:
          • skill: float — dir_acc % on hourly test-fold
          • ci_low / ci_high: float — bootstrap 95% CI bounds
          • n_test: int — sample size of test-fold (для понимания ширины CI)
          • last_signal: float in [0,1] — model probability of next-bar UP
          • last_signal_label: "up" | "down" | "flat" (на основе deadband 0.45..0.55)
          • last_bar_ts: ISO timestamp последнего бара (для UI freshness check)
        либо None если тикер не крипто, мало данных, или ошибка.
    """
    t0 = time.time()
    if not is_crypto_ticker(ticker):
        logger.debug("hourly_skill: %s is not crypto, skip", ticker)
        return None

    cache_key = "neucast:hourly_skill:" + hashlib.md5(
        f"{ticker}|{HOURLY_SOURCE}|{HOURLY_LOOKBACK_DAYS}|{HOURLY_INTERVAL}".encode()
    ).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("hourly_skill cache HIT: %s", ticker)
        return cached

    # Multi-source fetcher: Binance (primary, 5y) → yfinance (fallback, 720d).
    # crypto_data.fetch_crypto_ohlcv возвращает (df, source_name) с UTC-индексом
    # и единой схемой OHLCV — никакой MultiIndex headache.
    try:
        from app.crypto_data import fetch_crypto_ohlcv
        df_raw, source = fetch_crypto_ohlcv(
            ticker,
            lookback_days=HOURLY_LOOKBACK_DAYS,
            interval=HOURLY_INTERVAL,
            prefer=HOURLY_SOURCE,
        )
    except Exception as e:
        logger.warning("hourly_skill: crypto_data import/fetch failed %s: %s", ticker, e)
        return None

    if df_raw is None or df_raw.empty:
        logger.warning("hourly_skill: no hourly data for %s (source=%s)", ticker, "fail")
        return None
    if "Close" not in df_raw.columns:
        logger.warning("hourly_skill: no Close column for %s", ticker)
        return None
    if len(df_raw) < HOURLY_MIN_BARS:
        logger.info("hourly_skill: only %d bars for %s (source=%s), need %d — skip",
                    len(df_raw), ticker, source, HOURLY_MIN_BARS)
        return None

    feat = _engineer_hourly_features(df_raw)
    feat = feat.dropna()
    if len(feat) < HOURLY_MIN_BARS:
        logger.info("hourly_skill: %d rows after dropna for %s — skip", len(feat), ticker)
        return None

    feature_cols = [c for c in feat.columns if c != "target"]
    # Используем pd.DataFrame (а не numpy), чтобы LightGBM/sklearn видел имена
    # фич — иначе warning "X does not have valid feature names" при predict.
    X = feat[feature_cols].astype(np.float32)
    y = feat["target"].values.astype(np.int8)

    # 80/20 chronological split (keep last 20% for test-fold)
    split = int(len(feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_test) < 200 or len(np.unique(y_train)) < 2:
        logger.info("hourly_skill: degenerate split for %s (test=%d, classes=%d)",
                    ticker, len(X_test), len(np.unique(y_train)))
        return None

    try:
        import lightgbm as lgbm
        clf = lgbm.LGBMClassifier(
            num_leaves=HOURLY_NUM_LEAVES,
            n_estimators=HOURLY_N_ESTIMATORS,
            learning_rate=HOURLY_LEARNING_RATE,
            min_data_in_leaf=HOURLY_MIN_DATA_IN_LEAF,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(np.int8)
    except Exception as e:
        logger.warning("hourly_skill lightgbm fail for %s: %s", ticker, e, exc_info=True)
        return None

    hits = (pred == y_test).astype(np.int8)
    skill = float(hits.mean() * 100)
    ci_low, ci_high = _bootstrap_ci(hits.astype(np.float64),
                                    n_iter=HOURLY_BOOTSTRAP, seed=42)

    # Last-bar live signal: предсказание на самый свежий бар
    try:
        last_proba = float(clf.predict_proba(X.iloc[-1:])[0, 1])
    except Exception:
        last_proba = 0.5
    if last_proba >= 0.55:
        last_label = "up"
    elif last_proba <= 0.45:
        last_label = "down"
    else:
        last_label = "flat"

    elapsed = time.time() - t0
    result = {
        "skill": round(skill, 2),
        "ci_low": round(ci_low, 2),
        "ci_high": round(ci_high, 2),
        "n_test": int(len(X_test)),
        "n_train": int(len(X_train)),
        "last_signal": round(last_proba, 4),
        "last_signal_label": last_label,
        "last_bar_ts": str(df_raw.index[-1]),
        "interval": HOURLY_INTERVAL,
        "source": source,  # "binance" / "yfinance" / "fail"
        "elapsed_sec": round(elapsed, 2),
        # honest skill flag — параллель с low_directional_skill из prediction.py:
        # если bootstrap-CI пересекает 50%, hourly-probe тоже не отличает coin-flip.
        "low_skill": bool(ci_low <= 50.0),
    }
    logger.info(
        "hourly_skill %s: %.2f%% [%.2f%%–%.2f%%] n_test=%d signal=%.3f (%s) source=%s in %.1fs",
        ticker, skill, ci_low, ci_high, len(X_test), last_proba, last_label, source, elapsed,
    )
    _cache_set(cache_key, result, HOURLY_CACHE_TTL)
    return result
