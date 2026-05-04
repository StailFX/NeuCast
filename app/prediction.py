import os
import json
import hashlib
import pickle
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.layers import CUSTOM_OBJECTS

logger = logging.getLogger(__name__)

# ── Performance tuning (P2) ─────────────────────────────────────────
# Все пороги можно переопределить через env без ребилда.
TCN_EPOCHS = int(os.getenv("TCN_FINE_TUNE_EPOCHS", "10"))            # было 20; 10 почти всегда достаточно с early stopping
TCN_PATIENCE = int(os.getenv("TCN_FINE_TUNE_PATIENCE", "3"))         # было 7
BOOST_ITERS = int(os.getenv("BOOST_ITERS", "500"))
BOOST_EARLY_STOP = int(os.getenv("BOOST_EARLY_STOP", "80"))
MC_SIMS = int(os.getenv("MC_SIMS", "1000"))
YF_CACHE_TTL = int(os.getenv("YF_CACHE_TTL", "1800"))                # 30 мин кэша YFinance
BOOST_CACHE_TTL = int(os.getenv("BOOST_CACHE_TTL", "3600"))          # 1 час кэша обученных бустингов
BOOST_CACHE_ENABLED = os.getenv("BOOST_CACHE_ENABLED", "1") == "1"

# ── Backtest hysteresis ─────────────────────────────────────────────
# Гистерезис на Z-score: вход требует strong signal (>= ENTER), выход в FLAT
# срабатывает при ослаблении до EXIT. Это критично против whipsaw на боковике.
# Без гистерезиса (ENTER == EXIT) стратегия флипает позицию каждый раз, когда
# Z-score случайно пересекает порог → платит 0.14% × N комиссий → сливает B&H.
BACKTEST_ENTER_Z = float(os.getenv("BACKTEST_ENTER_Z", "1.0"))       # вход в LONG/SHORT
BACKTEST_EXIT_Z = float(os.getenv("BACKTEST_EXIT_Z", "0.3"))         # выход в FLAT

# ── Risk management ─────────────────────────────────────────────────
# Stop-loss + cooldown: предотвращает классический "потерял на LONG → флипнул
# в SHORT → потерял на отскоке" V-shape disaster. Если дневной PnL <= STOP_LOSS,
# форсим FLAT и блокируем входы на COOLDOWN дней.
# Vol-adjusted Kelly: режем размер позиции когда realized vol в верхнем квартиле
# (на крипто-крах входим маленьким, чтобы не словить максимум удара).
BACKTEST_STOP_LOSS_PCT = float(os.getenv("BACKTEST_STOP_LOSS_PCT", "-3.0"))   # %, дневной убыток
BACKTEST_COOLDOWN_DAYS = int(os.getenv("BACKTEST_COOLDOWN_DAYS", "3"))         # дни тишины после stop
BACKTEST_VOL_CAP_QUANTILE = float(os.getenv("BACKTEST_VOL_CAP_QUANTILE", "0.85"))  # только верхние 15%
BACKTEST_VOL_CAP_SCALE = float(os.getenv("BACKTEST_VOL_CAP_SCALE", "0.6"))      # коэф. снижения позиции

# ── Trailing stop ────────────────────────────────────────────────────
# Ratchet-стоп: пока позиция в прибыли, двигаем "точку тревоги" за ценой.
# Активируется только после ACTIVATE_PCT прибыли (чтобы не выбивать на шуме входа).
# Срабатывает при просадке TRAIL_PCT от достигнутого high (LONG) / low (SHORT).
# Это превращает unrealized profit в realized — без него тренд может развернуться
# и съесть всю накопленную прибыль до тех пор, пока stop-loss/гистерезис не сработают.
BACKTEST_TRAIL_ACTIVATE_PCT = float(os.getenv("BACKTEST_TRAIL_ACTIVATE_PCT", "1.5"))  # %, минимум прибыли для активации
BACKTEST_TRAIL_STOP_PCT = float(os.getenv("BACKTEST_TRAIL_STOP_PCT", "2.0"))           # %, просадка от high/low

# ── ATR-adaptive risk ─────────────────────────────────────────────────
# Stop-loss/trailing адаптируются к ATR% актива. BTC (~4% ATR/Close) и
# AAPL (~1.5% ATR/Close) получают разные пороги — фикс -3% слишком узок для
# крипто и слишком широк для blue-chip.
# Если выключено → используем фиксированные BACKTEST_STOP_LOSS_PCT и др.
ATR_ADAPTIVE_ENABLED = os.getenv("ATR_ADAPTIVE_ENABLED", "1") == "1"
ATR_MULT_STOP = float(os.getenv("ATR_MULT_STOP", "1.8"))               # stop = 1.8 × ATR%
ATR_MULT_TRAIL_STOP = float(os.getenv("ATR_MULT_TRAIL_STOP", "1.2"))    # trail = 1.2 × ATR%
ATR_MULT_TRAIL_ACTIVATE = float(os.getenv("ATR_MULT_TRAIL_ACTIVATE", "0.8"))  # activate = 0.8 × ATR%
ATR_MIN_STOP_PCT = float(os.getenv("ATR_MIN_STOP_PCT", "1.5"))          # нижняя граница stop
ATR_MAX_STOP_PCT = float(os.getenv("ATR_MAX_STOP_PCT", "8.0"))          # верхняя граница stop

# ── Embargo period для train/val split ──────────────────────────────
# Между train и val ставим gap из EMBARGO_DAYS точек. Это устраняет утечку
# через rolling-window features (RSI, MACD, MA_20 на момент val_split_idx
# знают данные из train_seq_end - 19 дней train). Стандартная практика в
# financial ML (Lopez de Prado).
EMBARGO_DAYS = int(os.getenv("EMBARGO_DAYS", "5"))

# ── Multi-horizon target для бустингов ───────────────────────────────
# Вместо чистого one-step target log_return[t+1] обучаем бустинги на смесь
# горизонтов: y_mh[t] = mean_h( sum(returns[t+1..t+h])/h ) для h in [1,2,3].
# Эффект: даём модели "сглаженный" target — меньше шума одного дня → меньше
# overfitting на отдельных выбросах. Direction-аккурасия на тесте обычно растёт,
# magnitude слегка падает (модель усреднена по горизонтам), но это компенсируется
# z-score нормализацией в backtest.
MULTIHORIZON_ENABLED = os.getenv("MULTIHORIZON_ENABLED", "1") == "1"
MULTIHORIZON_HORIZONS = tuple(
    int(x) for x in os.getenv("MULTIHORIZON_HORIZONS", "1,2,3").split(",") if x.strip()
)

# ── Snapshot ensemble для TCN ────────────────────────────────────────
# Вместо одной финальной точки training (или EarlyStopping best weights) — берём
# веса на нескольких эпохах ближе к концу обучения и усредняем их предсказания.
# Экономика: одна fit-сессия даёт K моделей по цене одной. Эффект: подавление
# noise одной траектории SGD (модель "качается" в локальном минимуме на каждой
# эпохе → разные снимки видят данные чуть по-разному). Это дешёвый proxy
# bagging без переобучения K моделей.
TCN_SNAPSHOT_ENABLED = os.getenv("TCN_SNAPSHOT_ENABLED", "1") == "1"
# Количество последних эпох, чьи веса попадают в ансамбль. По умолчанию 3 — это
# baseline epoch + 2 предыдущих. Если EarlyStopping сработал раньше — берём что есть.
TCN_SNAPSHOT_LAST_N = int(os.getenv("TCN_SNAPSHOT_LAST_N", "3"))

# ── Regime-conditional NNLS ──────────────────────────────────────────
# Стандартный NNLS даёт ОДНИ веса ансамбля для всех режимов. Но на calm-боковике
# хорошо работают бустинги (классические patterns), на crisis — TCN (latent
# momentum). Поэтому считаем NNLS отдельно для каждого регима внутри val-фолда
# и применяем per-regime веса на инференсе.
# Если режим имеет <REGIME_NNLS_MIN_POINTS точек val → fallback к глобальному NNLS.
REGIME_NNLS_ENABLED = os.getenv("REGIME_NNLS_ENABLED", "1") == "1"
REGIME_NNLS_MIN_POINTS = int(os.getenv("REGIME_NNLS_MIN_POINTS", "20"))

# ── Inverse-variance weighting (на инференсе) ────────────────────────
# Если на конкретном шаге модели сильно расходятся (pred_std высок) → ансамбль
# с NNLS-весами рискован: NNLS могла отдать 70% одной модели, которая именно
# здесь промахивается. Soft-mix к uniform весам тогда — страхуем "в пол".
# Weight = (1 - alpha) × nnls_w + alpha × uniform_w, где alpha = sigmoid(z_std).
# При низком расхождении alpha→0 (доверяем NNLS), при высоком alpha→1 (равномерно).
INV_VAR_WEIGHTING_ENABLED = os.getenv("INV_VAR_WEIGHTING_ENABLED", "1") == "1"
# Порог в std-единицах относительно медианного pred_std, после которого alpha → 1.
INV_VAR_WEIGHTING_THRESHOLD = float(os.getenv("INV_VAR_WEIGHTING_THRESHOLD", "1.5"))

# TF: ограничиваем inter-op параллелизм, чтобы не конкурировать с ThreadPoolExecutor на бустингах.
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 2) // 2))
except RuntimeError:
    pass  # TF уже инициализирован — настройки нельзя изменить

# ── Redis cache (опционально) ───────────────────────────────────────
# Используем тот же Redis, что и Celery. Graceful fallback → no-op если недоступен.
_redis_client = None
_REDIS_URL = os.getenv("REDIS_URL", "")


def _get_redis():
    global _redis_client
    if _redis_client is None and _REDIS_URL:
        try:
            import redis  # type: ignore
            _redis_client = redis.from_url(_REDIS_URL, socket_timeout=1, socket_connect_timeout=1)
            _redis_client.ping()
        except Exception as e:
            logger.warning("Redis cache disabled: %s", e)
            _redis_client = False
    return _redis_client if _redis_client not in (None, False) else None


# Code-review C-1 (2026-05-04): JSON cache (was pickle).
#
# Rationale: ``pickle.loads`` on Redis-fetched bytes is RCE if the broker
# is ever compromised (network leak, sibling-tenant, accidental exposure).
# Cached values here include numpy arrays + DataFrames — not natively JSON
# serialisable — so we ship a small numpy-aware encoder below.
#
# Backward-compat with legacy pickle entries: cache_get tries JSON first,
# treats decode errors as "stale entry" and deletes them. Cache TTLs are
# minutes/hours, so the legacy keyspace rolls over naturally within hours
# of deploy. **Never** falls back to pickle.loads — that would re-open
# the very vector this fix closes.


def _np_aware_json_encode(value) -> str:
    """JSON-encode ``value`` with numpy arrays → lists, NaN/Inf → None."""
    import json as _json
    import math as _math

    def _convert(o):
        # numpy scalar / array
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, float) and not _math.isfinite(o):
            return None
        if hasattr(o, "isoformat"):  # datetime / Timestamp
            return o.isoformat()
        # pandas DataFrame / Series — caller should pre-serialise; fall
        # back to to_json which is lossy but at least parseable.
        if hasattr(o, "to_json"):
            return _json.loads(o.to_json(date_format="iso"))
        if hasattr(o, "to_dict"):
            return o.to_dict()
        raise TypeError(f"unserialisable for JSON cache: {type(o)!r}")

    return _json.dumps(value, default=_convert)


def _cache_get(key: str):
    r = _get_redis()
    if not r:
        return None
    try:
        data = r.get(key)
        if not data:
            return None
        try:
            import json as _json
            return _json.loads(data)
        except Exception:
            # Legacy pickle entry from before C-1 deploy. Drop it.
            # We deliberately do NOT call pickle.loads here — that's the
            # exact vector being closed.
            try:
                r.delete(key)
            except Exception:
                pass
            return None
    except Exception as e:
        logger.debug("cache_get fail %s: %s", key, e)
        return None


def _cache_set(key: str, value, ttl: int):
    r = _get_redis()
    if not r:
        return
    try:
        encoded = _np_aware_json_encode(value)
        r.setex(key, ttl, encoded)
    except TypeError as exc:
        # Value contains a type we can't JSON-encode (e.g. a TF model
        # accidentally bundled in). Cache write skipped — caller must
        # tolerate a recompute. Log at debug; this is a perf hint only.
        logger.debug("cache_set skip (unserialisable) %s: %s", key, exc)
    except Exception as e:
        logger.debug("cache_set fail %s: %s", key, e)

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

SEQ_LEN = 60
MODEL_COLS = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "MA_20", "MA_50"]
# Extra features for boosting (TCN input shape is fixed by pre-trained weights)
BOOSTING_EXTRA_COLS = [
    "RV_1d", "RV_5d", "RV_22d",
    "VIX", "DXY", "SP500_ret", "TNX",
    "VIX_term", "HY_OAS", "Yield_Curve",
    "Kalman_trend", "Kalman_dev", "Regime",
]
BOOSTING_COLS = MODEL_COLS + BOOSTING_EXTRA_COLS

# Cross-asset tickers for macro features
_MACRO_TICKERS = {
    "VIX": "^VIX",       # CBOE Volatility Index (fear gauge)
    "VIX3M": "^VIX3M",   # 3-month VIX for term structure
    "DXY": "DX-Y.NYB",   # US Dollar Index
    "SP500": "^GSPC",     # S&P 500
    "TNX": "^TNX",        # 10-Year Treasury Yield
}
TRAIN_RATIO = 0.8

# ── Lazy model loading (critical for Celery prefork) ──
_base_model = None
_MODEL_TYPE = None
_IS_MULTITARGET = None


def _get_model():
    global _base_model, _MODEL_TYPE, _IS_MULTITARGET
    if _base_model is None:
        _base_model = load_model(
            os.path.join(WEIGHTS_DIR, "best_model.h5"),
            compile=False, custom_objects=CUSTOM_OBJECTS,
        )
        _MODEL_TYPE = "returns"
        _IS_MULTITARGET = False
        config_path = os.path.join(WEIGHTS_DIR, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
                _MODEL_TYPE = cfg.get("type", "returns")
                _IS_MULTITARGET = "direction" in cfg.get("outputs", [])
    return _base_model, _MODEL_TYPE, _IS_MULTITARGET


def _fetch_macro_features(start_date, end_date) -> pd.DataFrame:
    """Fetch cross-asset macro features: VIX, DXY, S&P500, TNX, VIX term structure, FRED data.

    P2+: все yfinance-загрузки выполняются параллельно через ThreadPoolExecutor
    (yfinance использует requests → GIL релизится на I/O). Экономит ~2-3 сек при
    холодном кэше, когда надо стянуть 5 тикеров подряд.
    """
    macro = pd.DataFrame()

    def _fetch_one(item):
        name, ticker = item
        try:
            data = yf.download(ticker, start=start_date, end=end_date,
                               interval="1d", progress=False)
            if data.empty:
                return name, None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return name, data["Close"]
        except Exception as e:
            logger.warning(f"Failed to fetch macro {name} ({ticker}): {e}")
            return name, None

    with ThreadPoolExecutor(max_workers=len(_MACRO_TICKERS)) as pool:
        for name, series in pool.map(_fetch_one, _MACRO_TICKERS.items()):
            if series is None:
                continue
            if name == "SP500":
                macro["SP500_ret"] = series.pct_change()
            else:
                macro[name] = series

    # VIX term structure (backwardation = stress signal)
    if "VIX" in macro.columns and "VIX3M" in macro.columns:
        macro["VIX_term"] = macro["VIX"] - macro["VIX3M"]
        macro.drop(columns=["VIX3M"], inplace=True)

    # FRED API: HY OAS credit spreads + yield curve
    try:
        from fredapi import Fred
        api_key = os.environ.get("FRED_API_KEY", "")
        if api_key:
            fred = Fred(api_key=api_key)
            try:
                hy = fred.get_series('BAMLH0A0HYM2', observation_start=start_date, observation_end=end_date)
                if hy is not None and len(hy) > 0:
                    macro['HY_OAS'] = hy
            except Exception:
                pass
            try:
                yc = fred.get_series('T10Y2Y', observation_start=start_date, observation_end=end_date)
                if yc is not None and len(yc) > 0:
                    macro['Yield_Curve'] = yc
            except Exception:
                pass
        else:
            logger.info("FRED_API_KEY not set — skipping FRED features")
    except ImportError:
        logger.info("fredapi not installed — skipping FRED features")
    except Exception as e:
        logger.warning(f"FRED API error: {e}")

    return macro


# ── Kalman Filter (zero-lag adaptive trend estimator) ──
def _kalman_filter(prices: np.ndarray) -> tuple:
    """
    1D Kalman Filter: state = [level, velocity].
    Unlike MAs, Kalman has zero lag and adapts bandwidth via innovation-based R tuning.
    Returns (filtered_levels, velocities).
    """
    n = len(prices)
    x = np.array([float(prices[0]), 0.0])
    P = np.diag([1.0, 0.1])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])   # state transition
    H = np.array([[1.0, 0.0]])                 # observation
    Q = np.diag([1e-4, 1e-5])                  # process noise
    R = np.array([[1.0]])                       # observation noise (adaptive)

    levels = np.zeros(n)
    velocities = np.zeros(n)
    innovations = []

    for i in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        innov = float(prices[i]) - float(H @ x_pred)
        innovations.append(innov)
        # Adaptive R: estimate from recent innovations
        if i > 20:
            R[0, 0] = max(np.var(innovations[-20:]), 1e-6)
        S = float(H @ P_pred @ H.T + R)
        K = (P_pred @ H.T) / S
        x = x_pred + K.flatten() * innov
        P = (np.eye(2) - K @ H) @ P_pred
        levels[i] = x[0]
        velocities[i] = x[1]

    return levels, velocities


# ── Regime Detection (HMM with quantile fallback) ──
def _detect_regime(returns: np.ndarray, volatility: np.ndarray, n_regimes: int = 3) -> np.ndarray:
    """
    Market regime detection. Tries HMM (hmmlearn) first, falls back to quantile-based.
    Regimes sorted by volatility: 0 = calm, 1 = normal, 2 = crisis.
    CatBoost handles discrete features natively — regime is a powerful categorical signal.
    """
    n = len(returns)
    regimes = np.ones(n, dtype=float)

    try:
        from hmmlearn.hmm import GaussianHMM
        X_hmm = np.column_stack([returns, volatility])
        mask = np.isfinite(X_hmm).all(axis=1)
        if mask.sum() < 50:
            raise ValueError("Insufficient data for HMM")
        X_clean = X_hmm[mask]
        model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                            n_iter=100, random_state=42)
        model.fit(X_clean)
        pred = model.predict(X_clean)
        # Sort regimes by mean volatility (0=lowest, 2=highest)
        regime_vols = [X_clean[pred == r, 1].mean() if (pred == r).any() else 0
                       for r in range(n_regimes)]
        sort_map = {old: new for new, old in enumerate(np.argsort(regime_vols))}
        clean_idx = np.where(mask)[0]
        for j, idx in enumerate(clean_idx):
            regimes[idx] = float(sort_map.get(pred[j], 1))
    except Exception:
        # Fallback: quantile-based regime from rolling volatility
        vol = pd.Series(volatility)
        vol_filled = vol.fillna(vol.expanding().mean()).fillna(0)
        q33 = vol_filled.quantile(0.33)
        q66 = vol_filled.quantile(0.66)
        regimes = np.where(vol_filled.values <= q33, 0.0,
                  np.where(vol_filled.values <= q66, 1.0, 2.0))

    return regimes


# ── Temporal Sample Weights ──
def _temporal_weights(n: int, half_life: int = 120,
                      vol_pct: np.ndarray = None) -> np.ndarray:
    """
    Exponential decay: recent samples weighted more heavily (half_life=120 ≈ 6 мес).
    Опционально: vol-aware downweight — точки в high-vol periods получают меньший
    вес, потому что signal/noise там хуже (price-action driven by macro shocks,
    не предсказуемой паттерн). Если vol_pct передан (длина n), множим веса на
    1/(1 + 5×clip(vol, 0, 0.2)). Эффект: точка с vol 5% получает ×0.8, с vol 10%
    получает ×0.67. На calm днях — без штрафа.
    Normalized so mean weight = 1.0.
    """
    decay = np.log(2) / half_life
    weights = np.exp(decay * np.arange(n))
    if vol_pct is not None and len(vol_pct) == n:
        downweight = 1.0 / (1.0 + 5.0 * np.clip(vol_pct, 0.0, 0.2))
        weights = weights * downweight
    return weights / weights.mean()


# ── Snapshot Ensemble Callback ──
class _WeightSnapshotCallback(tf.keras.callbacks.Callback):
    """
    Сохраняем копии весов модели на последних N эпохах. После .fit() имеем список
    snapshot weights (по одному на каждую эпоху, начиная с (total - last_n + 1)).
    Если EarlyStopping отрубил рано — сохраняется столько, сколько было фактически
    эпох, до restore_best_weights.

    Использование:
        snap_cb = _WeightSnapshotCallback(last_n=3)
        model.fit(..., callbacks=[snap_cb, EarlyStopping(...)])
        for w in snap_cb.snapshots:
            model.set_weights(w)
            preds.append(model.predict(X))
        avg_pred = np.mean(preds, axis=0)

    NB: храним deep-copy через [a.copy() for a in get_weights()] — иначе keras
    переписывает массивы in-place.
    """
    def __init__(self, last_n: int = 3):
        super().__init__()
        self.last_n = max(1, int(last_n))
        self.snapshots: list = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            w = [a.copy() for a in self.model.get_weights()]
            self.snapshots.append(w)
            if len(self.snapshots) > self.last_n:
                self.snapshots.pop(0)  # держим только последние last_n
        except Exception as e:
            logger.debug(f"snapshot capture failed at epoch {epoch}: {e}")


# ── Preprocessing ──
def preprocess(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].interpolate(method="time")

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    for w in (5, 10, 20, 50):
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()

    df["BB_mid"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    df["ROC_5"] = df["Close"].pct_change(5)
    df["ROC_10"] = df["Close"].pct_change(10)
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["Volatility_20"] = df["Close"].pct_change().rolling(20).std()

    # HAR-RV: Heterogeneous Autoregressive Realized Volatility (Corsi, 2009)
    daily_ret = df["Close"].pct_change()
    df["RV_1d"] = daily_ret.abs()                    # daily realized vol proxy
    df["RV_5d"] = daily_ret.rolling(5).std()          # weekly realized vol
    df["RV_22d"] = daily_ret.rolling(22).std()        # monthly realized vol

    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Kalman Filter trend (zero-lag adaptive — replaces lagging MAs for boosting)
    try:
        kalman_level, kalman_vel = _kalman_filter(df["Close"].values)
        df["Kalman_trend"] = kalman_vel / np.maximum(df["Close"].values, 1e-8)
        df["Kalman_dev"] = (df["Close"].values - kalman_level) / np.maximum(df["Close"].values, 1e-8)
    except Exception:
        df["Kalman_trend"] = 0.0
        df["Kalman_dev"] = 0.0

    # Regime detection (HMM-based or quantile fallback)
    try:
        df["Regime"] = _detect_regime(
            df["Close"].pct_change().fillna(0).values,
            df["Volatility_20"].fillna(0).values,
        )
    except Exception:
        df["Regime"] = 1.0

    # Merge cross-asset macro features (VIX, DXY, S&P500, 10Y yield)
    if macro_df is not None and not macro_df.empty:
        for col in macro_df.columns:
            if col not in df.columns:
                df[col] = macro_df[col]
    # Fill missing macro columns with 0 so boosting doesn't break
    for col in ["VIX", "DXY", "SP500_ret", "TNX", "VIX_term", "HY_OAS", "Yield_Curve"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.ffill().dropna()
    if len(df) < SEQ_LEN + 1:
        raise ValueError("Недостаточно данных. Нужно минимум ~120 торговых дней.")
    return df


def make_sequences_X(arr: np.ndarray) -> np.ndarray:
    X = []
    for i in range(SEQ_LEN, len(arr)):
        X.append(arr[i - SEQ_LEN: i])
    return np.array(X)


def make_sequences_Xy(scaled, returns):
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN: i])
        y.append(returns[i])
    return np.array(X), np.array(y)


def _apply_multihorizon_target(y: np.ndarray, horizons=(1, 2, 3)) -> np.ndarray:
    """
    Multi-horizon усреднённый daily-equivalent target.
    y[t] обычно = log_return at step t. На выходе:
        y_mh[t] = mean_h( sum(y[t : t+h]) / h ),  h ∈ horizons
    Длина результата = len(y) - max(horizons) + 1 (последние max_h-1 точек
    отбрасываются — для них нет полных будущих данных). Проверки на крайние
    случаи: если данных меньше max_h+1, возвращаем оригинал y[:1] (caller сам
    решит, что делать).
    """
    if not horizons or max(horizons) == 1:
        return np.asarray(y, dtype=np.float64).copy()
    n = len(y)
    max_h = max(horizons)
    if n < max_h:
        return np.asarray(y, dtype=np.float64).copy()
    out_len = n - max_h + 1
    # cumret[i] = sum(y[:i]); cumret[i+h] - cumret[i] = sum(y[i:i+h])
    cumret = np.concatenate([[0.0], np.cumsum(np.asarray(y, dtype=np.float64))])
    target = np.zeros(out_len, dtype=np.float64)
    for h in horizons:
        # h-step return at position i for i ∈ [0, out_len): cumret[i+h] - cumret[i]
        h_ret = (cumret[h: h + out_len] - cumret[: out_len]) / h
        target += h_ret
    target /= len(horizons)
    return target


def make_boosting_features(data: np.ndarray) -> np.ndarray:
    X = []
    for i in range(SEQ_LEN, len(data)):
        window = data[i - SEQ_LEN: i]
        feats = []
        for col in range(window.shape[1]):
            col_data = window[:, col]
            feats.extend([
                col_data[-1], col_data.mean(), col_data.std(),
                col_data[-1] - col_data[0], col_data[-5:].mean(),
                np.min(col_data), np.max(col_data),
            ])
        X.append(feats)
    return np.array(X)


def make_boosting_features_extended(data_tcn: np.ndarray, data_extra: np.ndarray) -> np.ndarray:
    """
    Build boosting features from TCN columns + extra columns.
    Each 60-day window -> 3 stats per column (last, momentum, volatility).

    Note: sentiment был раньше константным feature — бесполезен для gradient boost
    (при training и inference одно и то же число). Теперь sentiment применяется
    post-process через apply_sentiment_bias (см. Tier 5.1).
    """
    n_tcn_cols = data_tcn.shape[1]
    n_extra_cols = data_extra.shape[1] if data_extra is not None else 0
    X = []
    for i in range(SEQ_LEN, len(data_tcn)):
        feats = []
        # TCN columns: 3 stats each (last_value, momentum, volatility)
        window = data_tcn[i - SEQ_LEN: i]
        for col in range(n_tcn_cols):
            col_data = window[:, col]
            feats.extend([col_data[-1], col_data[-1] - col_data[0], col_data.std()])
        # Extra columns: 3 stats each
        if data_extra is not None:
            window_ex = data_extra[i - SEQ_LEN: i]
            for col in range(n_extra_cols):
                col_data = window_ex[:, col]
                feats.extend([col_data[-1], col_data[-1] - col_data[0], col_data.std()])
        X.append(feats)
    return np.array(X)


def _kelly_fraction(pred_returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Quarter-Kelly position sizing: f* = (mu / sigma^2) * 0.25
    More conservative than half-Kelly — reduces variance while keeping ~56% growth rate.
    Clamped to [0.05, 0.5] — max 50% of capital per trade.
    """
    n = len(pred_returns)
    fractions = np.ones(n) * 0.25  # default quarter-position

    for i in range(lookback, n):
        window = pred_returns[i - lookback: i]
        mu = np.mean(window)
        var = np.var(window)
        if var > 1e-10:
            # Quarter Kelly: more conservative, less blowup risk
            f = (mu / var) * 0.25
            fractions[i] = np.clip(abs(f), 0.05, 0.5)

    return fractions


def _run_backtest(
    dates, actual_prices, pred_returns, prev_prices,
    initial_capital=10000.0,
    commission_pct=0.05,     # 0.05% per trade (5 bps)
    slippage_pct=0.02,       # 0.02% slippage (2 bps)
    use_kelly=True,
    pred_std=None,           # per-step std across active models (confidence)
    atr_pct=None,            # ATR/Close ratio per step → adaptive stop/trail
):
    """
    Realistic long/short backtest with:
    - Signal threshold (dead zone): don't trade if |signal| too weak
    - Position holding: only pay costs when position CHANGES direction
    - Kelly sizing: quarter-Kelly, max 50% per trade
    - Transaction costs: only on actual trades (not daily churn)
    - Confidence filter: skip new entries when model disagreement (pred_std) is high
    - Stop-loss + cooldown: force FLAT after big daily loss, block entries N days
    - Vol-adjusted Kelly: halve position size when realized vol is in top quartile
    - ATR-adaptive stop/trail: пороги масштабируются от atr_pct (если передан)
    """
    n = len(dates)
    if n < 2:
        return None

    one_way_cost = (commission_pct + slippage_pct) / 100.0  # entry OR exit cost

    if use_kelly:
        kelly_sizes = _kelly_fraction(pred_returns)
    else:
        kelly_sizes = np.ones(n) * 0.5

    # ── Vol-adjusted Kelly ──────────────────────────────────────────
    # Считаем 20-дневную realized volatility из ФАКТИЧЕСКИХ возвратов.
    # Только когда текущая vol в верхних 15% истории (default Q=0.85) → режем
    # позицию × VOL_CAP_SCALE (default 0.6). Консервативный порог чтобы не
    # резать потенциал в обычной шумной истории — кикает только на реальные
    # extreme events (BTC флэш-крэш, gold spike). На таких событиях direction
    # часто непредсказуемо → лучше быть в позе вдвое меньше.
    actual_daily_returns = np.diff(actual_prices) / actual_prices[:-1]
    vol_window = 20
    realized_vol = pd.Series(actual_daily_returns).rolling(vol_window, min_periods=5).std().fillna(0).values
    pos_vol = realized_vol[realized_vol > 0]
    if len(pos_vol) > 10:
        vol_threshold = float(np.quantile(pos_vol, BACKTEST_VOL_CAP_QUANTILE))
        # Pad realized_vol to length n (it has length n-1 from np.diff)
        vol_padded = np.concatenate([[0.0], realized_vol])
        vol_scale = np.where(vol_padded > vol_threshold, BACKTEST_VOL_CAP_SCALE, 1.0)
        kelly_sizes = kelly_sizes * vol_scale

    # Convert raw pred_returns to Z-score signal (removes systematic bias)
    # Raw returns may have persistent positive/negative bias → all LONG or all SHORT
    # Z-score measures "stronger/weaker than recent average" → balanced signals
    z_lookback = 20
    z_signals = np.zeros(n)
    for i in range(n):
        start = max(0, i - z_lookback)
        window = pred_returns[start:i + 1]
        mu = np.mean(window)
        std = np.std(window)
        if std > 1e-10 and len(window) > 3:
            z_signals[i] = (pred_returns[i] - mu) / std
        else:
            z_signals[i] = 0.0

    # ── Confidence filter ──────────────────────────────────────────
    # Если std предсказаний моделей > медианы (модели расходятся) →
    # confidence_low → не входим в новые позиции (но из существующей не выгоняем
    # принудительно — гистерезис сам выведет). Это срезает тильт-моменты, когда
    # разные модели тянут в разные стороны и ансамбль "посередине" с непредсказуемым
    # направлением.
    if pred_std is not None and len(pred_std) == n and np.std(pred_std) > 0:
        confidence_threshold = float(np.median(pred_std))
        confidence_ok = pred_std <= confidence_threshold
    else:
        confidence_ok = np.ones(n, dtype=bool)

    # ── Hysteresis thresholds ──────────────────────────────────────
    # Старая версия: dead_zone = 0.5 → флип LONG↔SHORT каждый раз, когда
    # |Z| > 0.5; в FLAT не выходили никогда (всегда 100% invested) → на
    # боковике страшный whipsaw + аwerage 0.14% × N комиссий.
    # Новая логика: вход требует |Z| >= ENTER (по умолчанию 1.0σ), выход
    # в FLAT при ослаблении до EXIT (0.3σ). Режим может оставаться FLAT,
    # если сигнал слаб → не торгуем при шуме.
    enter_z = BACKTEST_ENTER_Z
    exit_z = BACKTEST_EXIT_Z

    capital = initial_capital
    equity = [capital]
    positions = []
    daily_returns = []
    wins = 0
    losses = 0
    total_costs = 0.0
    num_trades = 0
    stop_outs = 0   # счётчик срабатываний stop-loss (для UI)
    trail_outs = 0  # счётчик срабатываний trailing stop (для UI)

    # actual_daily_returns уже посчитан выше для vol-adjusted Kelly

    current_position = "FLAT"  # FLAT / LONG / SHORT
    cooldown_remaining = 0     # > 0 → запрет на новые входы (после stop-loss)

    # ── Per-step risk thresholds: фиксированные или ATR-adaptive ────
    # Если ATR_ADAPTIVE_ENABLED и atr_pct передан → пороги растут вместе с
    # волатильностью актива (BTC vs AAPL получают разные значения). Фолбэк
    # к константным BACKTEST_*_PCT если ATR недоступен.
    if (ATR_ADAPTIVE_ENABLED and atr_pct is not None and len(atr_pct) == n
            and np.isfinite(atr_pct).all() and (atr_pct > 0).any()):
        atr_safe = np.clip(atr_pct, 0.005, 0.15)  # safety: 0.5%..15% per day
        stop_loss_arr = -np.clip(
            ATR_MULT_STOP * atr_safe,
            ATR_MIN_STOP_PCT / 100.0, ATR_MAX_STOP_PCT / 100.0,
        )  # negative (loss thresholds)
        trail_stop_arr = ATR_MULT_TRAIL_STOP * atr_safe
        trail_activate_arr = ATR_MULT_TRAIL_ACTIVATE * atr_safe
        atr_adaptive_used = True
    else:
        stop_loss_arr = np.full(n, BACKTEST_STOP_LOSS_PCT / 100.0)
        trail_stop_arr = np.full(n, BACKTEST_TRAIL_STOP_PCT / 100.0)
        trail_activate_arr = np.full(n, BACKTEST_TRAIL_ACTIVATE_PCT / 100.0)
        atr_adaptive_used = False

    # Trailing-stop state
    position_entry_price = None  # цена входа в текущую позу
    position_high = None         # max цена с момента входа (для LONG)
    position_low = None          # min цена с момента входа (для SHORT)

    for i in range(n - 1):
        signal = z_signals[i]
        actual_ret = actual_daily_returns[i]
        position_size = kelly_sizes[i]

        # Hysteresis state machine:
        #  FLAT  → LONG  при signal >= +enter_z  AND confidence_ok  AND cooldown==0
        #  FLAT  → SHORT при signal <= -enter_z  AND confidence_ok  AND cooldown==0
        #  LONG  → FLAT  при signal <  +exit_z   (ослабление, cooldown игнорируется)
        #  LONG  → SHORT при signal <= -enter_z  AND confidence_ok  AND cooldown==0
        #  SHORT → FLAT  при signal >  -exit_z
        #  SHORT → LONG  при signal >= +enter_z  AND confidence_ok  AND cooldown==0
        # Cooldown_remaining блокирует ВСЕ входы и flips, но не блокирует выходы.
        conf = bool(confidence_ok[i])
        can_enter = conf and cooldown_remaining == 0
        if current_position == "FLAT":
            if signal >= enter_z and can_enter:
                desired = "LONG"
            elif signal <= -enter_z and can_enter:
                desired = "SHORT"
            else:
                desired = "FLAT"
        elif current_position == "LONG":
            if signal <= -enter_z and can_enter:
                desired = "SHORT"        # сильный разворот → flip
            elif signal < exit_z:
                desired = "FLAT"         # сигнал ослаб → выходим
            else:
                desired = "LONG"         # держим
        else:  # SHORT
            if signal >= enter_z and can_enter:
                desired = "LONG"         # сильный разворот → flip
            elif signal > -exit_z:
                desired = "FLAT"         # сигнал ослаб → выходим
            else:
                desired = "SHORT"        # держим

        # Trade cost only when position changes
        trade_cost = 0.0
        if desired != current_position:
            if current_position != "FLAT":
                trade_cost += capital * position_size * one_way_cost  # exit old
            if desired != "FLAT":
                trade_cost += capital * position_size * one_way_cost  # enter new
            num_trades += 1
            current_position = desired
            # Reset trailing-stop state on entry / flip
            if current_position != "FLAT":
                position_entry_price = float(actual_prices[i])
                position_high = position_entry_price
                position_low = position_entry_price
            else:
                position_entry_price = None
                position_high = None
                position_low = None

        # PnL based on current position
        if current_position == "LONG":
            gross_pnl = capital * position_size * actual_ret
            action = "LONG"
        elif current_position == "SHORT":
            gross_pnl = capital * position_size * (-actual_ret)
            action = "SHORT"
        else:
            gross_pnl = 0.0
            action = "FLAT"

        net_pnl = gross_pnl - trade_cost
        total_costs += trade_cost

        capital += net_pnl
        equity.append(capital)
        daily_pct = net_pnl / equity[-2] if equity[-2] > 0 else 0
        daily_returns.append(daily_pct)

        # ── Trailing-stop check (после применения PnL) ─────────────
        # Цена на конец дня i — это actual_prices[i+1] (мы только что применили
        # actual_daily_returns[i]). Обновляем high/low от entry; если позиция уже
        # в прибыли >= trail_activate и просела от high/low на >= trail_stop —
        # фиксируем профит. Без cooldown — это не катастрофа, а нормальный exit.
        # Пороги per-step из массивов trail_activate_arr/trail_stop_arr (ATR-adaptive).
        if (current_position != "FLAT" and position_entry_price is not None
                and i + 1 < n):
            cur_price = float(actual_prices[i + 1])
            cur_trail_activate = float(trail_activate_arr[i])
            cur_trail_stop = float(trail_stop_arr[i])
            triggered_trail = False
            if current_position == "LONG":
                if position_high is None or cur_price > position_high:
                    position_high = cur_price
                profit_pct = (position_high - position_entry_price) / position_entry_price
                if profit_pct >= cur_trail_activate:
                    drawdown_pct = (position_high - cur_price) / position_high
                    if drawdown_pct >= cur_trail_stop:
                        triggered_trail = True
            else:  # SHORT
                if position_low is None or cur_price < position_low:
                    position_low = cur_price
                profit_pct = (position_entry_price - position_low) / position_entry_price
                if profit_pct >= cur_trail_activate:
                    rebound_pct = (cur_price - position_low) / position_low
                    if rebound_pct >= cur_trail_stop:
                        triggered_trail = True

            if triggered_trail:
                exit_cost = capital * position_size * one_way_cost
                capital -= exit_cost
                equity[-1] = capital
                total_costs += exit_cost
                trail_outs += 1
                action = "TRAIL"
                current_position = "FLAT"
                position_entry_price = None
                position_high = None
                position_low = None

        # ── Stop-loss check (после trailing) ───────────────────────
        # Если дневной убыток превысил порог И мы были не во FLAT → форсим выход
        # на следующем шаге через cooldown. Это не вернёт сегодняшний убыток, но
        # защитит от классики "потерял на LONG → завтра флипнул в SHORT → потерял
        # на отскоке". Cooldown даёт рынку успокоиться перед новым входом.
        # Порог per-step из stop_loss_arr (ATR-adaptive).
        cur_stop_loss = float(stop_loss_arr[i])
        if daily_pct <= cur_stop_loss and current_position != "FLAT":
            current_position = "FLAT"
            cooldown_remaining = BACKTEST_COOLDOWN_DAYS
            # Платим exit cost (вышли на закрытии в стрессе → slippage хуже, но упрощённо
            # уже учтено в commission+slippage)
            exit_cost = capital * position_size * one_way_cost
            capital -= exit_cost
            equity[-1] = capital
            total_costs += exit_cost
            stop_outs += 1
            action = "STOP"
            position_entry_price = None
            position_high = None
            position_low = None
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1

        if net_pnl > 0:
            wins += 1
        elif net_pnl < 0:
            losses += 1

        positions.append({
            "date": dates[i],
            "action": action,
            "price": round(float(actual_prices[i]), 2),
            "signal": round(float(signal), 6),
            "kelly": round(float(position_size) * 100, 1),
            "pnl": round(float(net_pnl), 2),
            "cost": round(float(trade_cost), 2),
            "capital": round(float(capital), 2),
        })

    # Buy & Hold comparison
    bnh_equity = [initial_capital]
    for i in range(n - 1):
        bnh_equity.append(bnh_equity[-1] * (1 + actual_daily_returns[i]))

    total_trades = wins + losses
    total_return = (capital - initial_capital) / initial_capital * 100
    bnh_return = (bnh_equity[-1] - initial_capital) / initial_capital * 100

    dr = np.array(daily_returns)
    sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(252)) if len(dr) > 1 and np.std(dr) > 0 else 0.0

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_drawdown = float(np.min(drawdown) * 100)

    gross_profit = sum(d for d in daily_returns if d > 0)
    gross_loss = abs(sum(d for d in daily_returns if d < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for d in daily_returns:
        if d > 0:
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        elif d < 0:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
        else:
            cur_win = 0
            cur_loss = 0

    win_returns = [d for d in daily_returns if d > 0]
    loss_returns = [d for d in daily_returns if d < 0]
    avg_win = float(np.mean(win_returns) * 100) if win_returns else 0.0
    avg_loss = float(np.mean(loss_returns) * 100) if loss_returns else 0.0

    ann_return = total_return * 252 / max(n - 1, 1)
    calmar = abs(ann_return / max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "equity": [round(e, 2) for e in equity],
        "bnh_equity": [round(e, 2) for e in bnh_equity],
        "dates": dates,
        "total_return": round(total_return, 2),
        "bnh_return": round(bnh_return, 2),
        "alpha": round(total_return - bnh_return, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
        "total_trades": total_trades,
        "actual_trades": num_trades,
        "wins": wins,
        "losses": losses,
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "final_capital": round(capital, 2),
        "initial_capital": initial_capital,
        "total_costs": round(total_costs, 2),
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct,
        "avg_kelly": round(float(np.mean(kelly_sizes)) * 100, 1),
        "stop_outs": stop_outs,
        "trail_outs": trail_outs,
        "atr_adaptive": atr_adaptive_used,
        "avg_stop_pct": round(float(np.mean(np.abs(stop_loss_arr))) * 100, 2),
        "avg_trail_pct": round(float(np.mean(trail_stop_arr)) * 100, 2),
        "trades": positions[-20:],  # last 20 trades for display
    }


def run_prediction(df: pd.DataFrame, days_ahead: int, sentiment_score: float = 0.0,
                   use_foundation: bool = False):
    """
    Args:
        use_foundation: если True, дополнительно прогоняем Chronos foundation
            model для будущего forecast и блендим её median с нашим. Bands
            заменяются на родные Chronos quantiles (они probabilistically
            калиброваны на 10B time-series точках). +20-40s к времени запроса.
            Если chronos-forecasting не установлен — graceful fallback к
            обычному прогнозу (как будто use_foundation=False).
    """
    base_model, MODEL_TYPE, IS_MULTITARGET = _get_model()

    n_total = len(df)
    split_idx = int(n_total * TRAIN_RATIO)

    # ── TCN path: uses MODEL_COLS (fixed input shape from pre-trained weights) ──
    scaler = MinMaxScaler((0, 1))
    scaler.fit(df.iloc[:split_idx][MODEL_COLS])
    scaled_all = scaler.transform(df[MODEL_COLS])

    log_returns = df["log_return"].values
    close_prices = df["Close"].values

    X_all, y_all = make_sequences_Xy(scaled_all, log_returns)
    train_seq_end = split_idx - SEQ_LEN

    X_train = X_all[:train_seq_end]
    y_train = y_all[:train_seq_end]

    # ── P1: TCN-predictions cache ─────────────────────────────────────
    # Кэшируем массивы предсказаний (не веса модели). Fine-tune TCN — самый дорогой
    # шаг после бустингов. Ключ — хеш X_train/y_train + hyperparams. Промах = обычное
    # обучение и потом store в Redis.
    tcn_cache_key = None
    tcn_pred_returns = None
    tcn_direction_pred = None
    if BOOST_CACHE_ENABLED:
        try:
            sig = hashlib.md5()
            sig.update(X_train.tobytes())
            sig.update(y_train.tobytes())
            sig.update(X_all.tobytes())
            sig.update(
                f"{TCN_EPOCHS}|{TCN_PATIENCE}|{IS_MULTITARGET}|"
                f"snap{int(TCN_SNAPSHOT_ENABLED)}-{TCN_SNAPSHOT_LAST_N}".encode()
            )
            tcn_cache_key = "neucast:tcn:" + sig.hexdigest()
            cached_tcn = _cache_get(tcn_cache_key)
            if cached_tcn is not None:
                logger.info("TCN cache HIT: %s", tcn_cache_key[-12:])
                tcn_pred_returns = cached_tcn.get("returns")
                tcn_direction_pred = cached_tcn.get("direction")
        except Exception as e:
            logger.debug("tcn cache lookup failed: %s", e)
            tcn_cache_key = None

    if tcn_pred_returns is None:
        # Fine-tune TCN with plain MSE loss (directional loss was hurting)
        fine_model = clone_model(base_model)
        fine_model.set_weights(base_model.get_weights())

        if IS_MULTITARGET:
            fine_model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy'},
                loss_weights={'return_output': 1.0, 'direction_output': 0.5},
            )
        else:
            fine_model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='mse')

        # Snapshot ensemble: захватываем веса последних N эпох; после fit
        # усредним предсказания. Если выключено — snap_cb.snapshots пуст и
        # используем только финальные веса (best, благодаря restore_best_weights).
        snap_cb = _WeightSnapshotCallback(last_n=TCN_SNAPSHOT_LAST_N) if TCN_SNAPSHOT_ENABLED else None

        if len(X_train) > SEQ_LEN:
            val_size = max(0.1, SEQ_LEN / len(X_train))
            direction_all = (log_returns[SEQ_LEN:] > 0).astype(np.float32)
            dir_train = direction_all[:train_seq_end]
            cbs = [EarlyStopping(monitor="val_loss", patience=TCN_PATIENCE, restore_best_weights=True, verbose=0)]
            if snap_cb is not None:
                cbs.append(snap_cb)
            if IS_MULTITARGET:
                fine_model.fit(
                    X_train,
                    {'return_output': y_train, 'direction_output': dir_train},
                    epochs=TCN_EPOCHS, batch_size=32,
                    validation_split=val_size,
                    callbacks=cbs,
                    verbose=0,
                )
            else:
                fine_model.fit(
                    X_train, y_train,
                    epochs=TCN_EPOCHS, batch_size=32,
                    validation_split=val_size,
                    callbacks=cbs,
                    verbose=0,
                )

        # ── Snapshot ensemble inference ──
        # Если есть N снимков → каждый предсказывает X_all, усредняем. Финальные
        # веса (best) уже выставлены EarlyStopping → они идут как N+1-й голос.
        # Это даёт K+1 моделей по цене одной обучающей сессии.
        snapshots = list(snap_cb.snapshots) if snap_cb is not None else []
        # Сохраняем финальные best-веса отдельно — restore_best_weights уже их выставил
        best_weights = [a.copy() for a in fine_model.get_weights()]

        all_returns_preds = []
        all_direction_preds = []

        # Сначала best-модель
        raw_pred = fine_model.predict(X_all, verbose=0)
        if IS_MULTITARGET:
            all_returns_preds.append(np.asarray(raw_pred[0]).flatten())
            all_direction_preds.append(np.asarray(raw_pred[1]).flatten())
        else:
            all_returns_preds.append(np.asarray(raw_pred).flatten())

        # Затем snapshot-модели (если есть и снимки отличаются от best)
        snapshot_count_used = 0
        for snap_w in snapshots:
            try:
                fine_model.set_weights(snap_w)
                raw_pred_s = fine_model.predict(X_all, verbose=0)
                if IS_MULTITARGET:
                    all_returns_preds.append(np.asarray(raw_pred_s[0]).flatten())
                    all_direction_preds.append(np.asarray(raw_pred_s[1]).flatten())
                else:
                    all_returns_preds.append(np.asarray(raw_pred_s).flatten())
                snapshot_count_used += 1
            except Exception as e:
                logger.debug(f"snapshot inference failed: {e}")

        # Возвращаем best-веса (на случай если кто-то ещё дёргает fine_model)
        try:
            fine_model.set_weights(best_weights)
        except Exception:
            pass

        # Усредняем предсказания
        tcn_pred_returns = np.mean(np.stack(all_returns_preds, axis=0), axis=0)
        if IS_MULTITARGET and all_direction_preds:
            tcn_direction_pred = np.mean(np.stack(all_direction_preds, axis=0), axis=0)
        else:
            tcn_direction_pred = None

        if snapshot_count_used > 0:
            logger.info(
                f"TCN snapshot ensemble: {snapshot_count_used} snapshots + 1 best "
                f"= {snapshot_count_used + 1} models averaged"
            )

        # Сохраняем в кэш (только предсказания — TF-модель не pickl'ится надёжно).
        if BOOST_CACHE_ENABLED and tcn_cache_key:
            try:
                _cache_set(tcn_cache_key, {
                    "returns": tcn_pred_returns,
                    "direction": tcn_direction_pred,
                }, BOOST_CACHE_TTL)
                logger.info("TCN cache STORE: %s", tcn_cache_key[-12:])
            except Exception as e:
                logger.debug("tcn cache store failed: %s", e)

    # ── Boosting path: extended features + sentiment ──
    scaler_extra = MinMaxScaler((0, 1))
    scaler_extra.fit(df.iloc[:split_idx][BOOSTING_EXTRA_COLS])
    scaled_extra = scaler_extra.transform(df[BOOSTING_EXTRA_COLS])

    X_bst_all = make_boosting_features_extended(scaled_all, scaled_extra)

    # All models predict log_returns (same target as TCN — no triple barrier)
    y_bst_train = y_all[:train_seq_end]

    # Time-based validation split for early stopping (last 20% of train).
    # EMBARGO_DAYS ставит gap между train и val чтобы устранить утечку через
    # rolling-window features. train: [0:val_split_idx - embargo], val: [val_split_idx:]
    val_split_idx = int(train_seq_end * 0.8)
    embargo = max(0, EMBARGO_DAYS) if val_split_idx > 50 + EMBARGO_DAYS else 0
    train_end_with_embargo = max(50, val_split_idx - embargo)
    X_bst_train_sel = X_bst_all[:train_end_with_embargo]
    X_bst_val_sel = X_bst_all[val_split_idx:train_seq_end]
    y_bst_val = y_bst_train[val_split_idx:]
    X_bst_test_sel = X_bst_all[train_seq_end:]
    X_bst_all_sel = X_bst_all

    # ── Vol-aware sample weights ────────────────────────────────────
    # Считаем 20-дневную rolling vol log_returns aligned с обучающими позициями.
    # X_bst_train_sel[i] предсказывает позицию SEQ_LEN + i в df → vol на этой
    # позиции = std(log_returns[SEQ_LEN+i-20 : SEQ_LEN+i]).
    train_n = len(X_bst_train_sel)
    vol_pct_train = None
    try:
        # Полная rolling-vol по log_returns; дальше срез под train.
        vol_full = pd.Series(log_returns).rolling(20, min_periods=5).std().bfill().values
        # Aligned slice: позиция SEQ_LEN + i для i ∈ [0, train_n)
        vol_pct_train = vol_full[SEQ_LEN: SEQ_LEN + train_n]
        if len(vol_pct_train) != train_n or not np.isfinite(vol_pct_train).all():
            vol_pct_train = None
    except Exception:
        vol_pct_train = None

    full_weights = _temporal_weights(train_n, vol_pct=vol_pct_train)

    y_bst_train_sel = y_bst_train[:train_end_with_embargo]

    # ── Multi-horizon target transform (если включён) ──
    # Сглаживаем target по 2-3 горизонтам → меньше overfitting на шуме одного дня.
    # Транформ применяем только к labels для train+val. Inference (X_bst_test_sel
    # / X_bst_all_sel) выдаёт MH-equivalent predictions — далее они идут в NNLS,
    # который оптимизирует под actual one-day returns. Ансамбль автоматически
    # подстроится под новый "stretch" magnitude через coefs.
    # Длина train/val укорачивается на (MAX_H - 1); X тоже урезается с конца.
    if MULTIHORIZON_ENABLED and len(MULTIHORIZON_HORIZONS) > 1:
        max_h = max(MULTIHORIZON_HORIZONS)
        # train slice
        y_train_mh = _apply_multihorizon_target(y_bst_train_sel, MULTIHORIZON_HORIZONS)
        if len(y_train_mh) >= 50:  # минимум разумно для обучения
            y_bst_train_sel = y_train_mh
            X_bst_train_sel = X_bst_train_sel[: len(y_train_mh)]
            full_weights = full_weights[: len(y_train_mh)]
        # val slice
        y_val_mh = _apply_multihorizon_target(y_bst_val, MULTIHORIZON_HORIZONS)
        if len(y_val_mh) >= 10:
            y_bst_val = y_val_mh
            X_bst_val_sel = X_bst_val_sel[: len(y_val_mh)]
        logger.info(f"Multi-horizon target enabled: horizons={MULTIHORIZON_HORIZONS} "
                    f"(train_n={len(y_bst_train_sel)}, val_n={len(y_bst_val)})")

    # ══════════════════════════════════════════════════════
    # ── Train boosting models on train set (parallel P2) ──
    # ══════════════════════════════════════════════════════
    # Все три бустинга — native C++ и релизят GIL.
    # ThreadPoolExecutor(3) даёт ~3× ускорение этого этапа на многоядерной VPS.
    # Делим ядра между моделями, чтобы не было oversubscription.
    cat = None
    cat_pred_test = None
    cat_pred_all = None
    xgb_m = None
    xgb_pred_test = None
    xgb_pred_all = None
    lgb_m = None
    lgb_pred_test = None
    lgb_pred_all = None

    ncpu = max(1, os.cpu_count() or 2)
    per_model_threads = max(1, ncpu // 3)

    # ── P1: Booster-cache ─────────────────────────────────────────────
    # Кэшируем трио (CatBoost, XGBoost, LightGBM) + их предсказания.
    # Хеш строится по содержимому обучающей выборки + hyperparams. Изменился df
    # или ui-параметры — ключ меняется автоматически.
    boost_cache_key = None
    if BOOST_CACHE_ENABLED:
        try:
            sig = hashlib.md5()
            # Используем только тренировочный/val срезы — они полностью определяют веса моделей.
            sig.update(pd.util.hash_pandas_object(pd.DataFrame(X_bst_train_sel)).values.tobytes())
            sig.update(pd.util.hash_pandas_object(pd.DataFrame(X_bst_val_sel)).values.tobytes())
            sig.update(y_bst_train_sel.tobytes())
            sig.update(y_bst_val.tobytes())
            # sentiment_score больше НЕ в кэше (он post-process теперь, Tier 5.1).
            sig.update(
                f"{BOOST_ITERS}|{BOOST_EARLY_STOP}|"
                f"mh{int(MULTIHORIZON_ENABLED)}-{','.join(map(str, MULTIHORIZON_HORIZONS))}".encode()
            )
            boost_cache_key = "neucast:boost:" + sig.hexdigest()
            cached_bst = _cache_get(boost_cache_key)
            if cached_bst is not None:
                logger.info("Boost cache HIT: %s", boost_cache_key[-12:])
                cat = cached_bst.get("cat_model")
                cat_pred_test = cached_bst.get("cat_pred_test")
                cat_pred_all = cached_bst.get("cat_pred_all")
                xgb_m = cached_bst.get("xgb_model")
                xgb_pred_test = cached_bst.get("xgb_pred_test")
                xgb_pred_all = cached_bst.get("xgb_pred_all")
                lgb_m = cached_bst.get("lgb_model")
                lgb_pred_test = cached_bst.get("lgb_pred_test")
                lgb_pred_all = cached_bst.get("lgb_pred_all")
        except Exception as e:
            logger.debug("boost cache lookup failed: %s", e)
            boost_cache_key = None

    # Если все три модели есть в кэше — пропускаем обучение полностью.
    _all_cached = cat is not None and xgb_m is not None and lgb_m is not None
    if not _all_cached:
        def _train_catboost():
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(
                iterations=BOOST_ITERS, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                random_strength=0.5, bagging_temperature=0.3,
                verbose=0, early_stopping_rounds=BOOST_EARLY_STOP,
                thread_count=per_model_threads,
            )
            m.fit(X_bst_train_sel, y_bst_train_sel, sample_weight=full_weights,
                  eval_set=(X_bst_val_sel, y_bst_val), verbose=0)
            return m, m.predict(X_bst_test_sel), m.predict(X_bst_all_sel)

        def _train_xgboost():
            import xgboost as xgb
            m = xgb.XGBRegressor(
                n_estimators=BOOST_ITERS, learning_rate=0.03, max_depth=5,
                subsample=0.8, colsample_bytree=0.6,
                reg_alpha=0.5, reg_lambda=2.0,
                early_stopping_rounds=BOOST_EARLY_STOP, verbosity=0,
                n_jobs=per_model_threads, tree_method="hist",
            )
            m.fit(X_bst_train_sel, y_bst_train_sel, sample_weight=full_weights,
                  eval_set=[(X_bst_val_sel, y_bst_val)], verbose=0)
            return m, m.predict(X_bst_test_sel), m.predict(X_bst_all_sel)

        def _train_lightgbm():
            import lightgbm as lgbm
            m = lgbm.LGBMRegressor(
                n_estimators=BOOST_ITERS, learning_rate=0.03, max_depth=5,
                num_leaves=31, subsample=0.8, colsample_bytree=0.6,
                reg_alpha=0.5, reg_lambda=2.0, verbose=-1,
                n_jobs=per_model_threads,
            )
            m.fit(X_bst_train_sel, y_bst_train_sel, sample_weight=full_weights,
                  eval_set=[(X_bst_val_sel, y_bst_val)],
                  callbacks=[lgbm.early_stopping(BOOST_EARLY_STOP, verbose=False),
                             lgbm.log_evaluation(0)])
            return m, m.predict(X_bst_test_sel), m.predict(X_bst_all_sel)

        # Обучаем только те модели, которых нет в кэше (обычно все три).
        to_train = {}
        if cat is None:
            to_train["CatBoost"] = _train_catboost
        if xgb_m is None:
            to_train["XGBoost"] = _train_xgboost
        if lgb_m is None:
            to_train["LightGBM"] = _train_lightgbm

        with ThreadPoolExecutor(max_workers=max(1, len(to_train))) as pool:
            futs = {name: pool.submit(fn) for name, fn in to_train.items()}
            for name, fut in futs.items():
                try:
                    model_obj, p_test, p_all = fut.result()
                    if name == "CatBoost":
                        cat, cat_pred_test, cat_pred_all = model_obj, p_test, p_all
                    elif name == "XGBoost":
                        xgb_m, xgb_pred_test, xgb_pred_all = model_obj, p_test, p_all
                    elif name == "LightGBM":
                        lgb_m, lgb_pred_test, lgb_pred_all = model_obj, p_test, p_all
                except Exception as e:
                    logger.warning("%s training failed: %s", name, e)

        # Сохраняем трио в кэш, если все обучились (частичный кэш — бесполезен).
        if (BOOST_CACHE_ENABLED and boost_cache_key
                and cat is not None and xgb_m is not None and lgb_m is not None):
            try:
                payload = {
                    "cat_model": cat, "cat_pred_test": cat_pred_test, "cat_pred_all": cat_pred_all,
                    "xgb_model": xgb_m, "xgb_pred_test": xgb_pred_test, "xgb_pred_all": xgb_pred_all,
                    "lgb_model": lgb_m, "lgb_pred_test": lgb_pred_test, "lgb_pred_all": lgb_pred_all,
                }
                _cache_set(boost_cache_key, payload, BOOST_CACHE_TTL)
                logger.info("Boost cache STORE: %s", boost_cache_key[-12:])
            except Exception as e:
                logger.debug("boost cache store failed: %s", e)

    # ══════════════════════════════════════════════════════════
    # ── Smart Ensemble: Rolling-NNLS + Anti-Skill Filter      ──
    # ══════════════════════════════════════════════════════════
    # Эволюция логики:
    #  v1 (leaky):   1/MAPE на test-фолде → утечка → завышенные метрики
    #  v2 (val-MAPE):1/MAPE на val-фолде → честно, но не учитывает направление
    #  v3 (NNLS):    NNLS на одном val-окне → переобучается на режим этого окна
    #  v4 (текущая): (a) anti-skill filter: модели с dir_acc<50% на val → вес 0
    #                (b) Rolling-NNLS: считаем NNLS на нескольких подокнах val
    #                    (30/60/100/all дней) и усредняем coef. Прокси walk-forward
    #                    без K-кратной перетренировки. Веса должны работать на
    #                    разных горизонтах → robust к смене режима (bull→bear etc).
    #                (c) fallback: если все окна дали нулевые coef → (1/MAPE)×dir_edge

    prev_prices_full = close_prices[SEQ_LEN - 1: -1]

    # Validation slice (внутри train, но не использовался для подгонки весов моделей)
    val_actual_prices = close_prices[SEQ_LEN + val_split_idx: SEQ_LEN + train_seq_end]
    val_prev = prev_prices_full[val_split_idx:train_seq_end]

    # Test slice (для метрик в UI; в подсчёте весов НЕ участвует)
    test_actual_prices = close_prices[SEQ_LEN + train_seq_end:]
    test_prev = prev_prices_full[train_seq_end:]

    val_preds_ret = {"TCN": tcn_pred_returns[val_split_idx:train_seq_end]}
    test_preds_ret = {"TCN": tcn_pred_returns[train_seq_end:]}
    all_preds = {"TCN": tcn_pred_returns}

    if cat_pred_test is not None:
        # cat_pred_all покрывает весь датасет (X_bst_all_sel) → можно срезать val
        val_preds_ret["CatBoost"] = cat_pred_all[val_split_idx:train_seq_end]
        test_preds_ret["CatBoost"] = cat_pred_test
        all_preds["CatBoost"] = cat_pred_all
    if xgb_pred_test is not None:
        val_preds_ret["XGBoost"] = xgb_pred_all[val_split_idx:train_seq_end]
        test_preds_ret["XGBoost"] = xgb_pred_test
        all_preds["XGBoost"] = xgb_pred_all
    if lgb_pred_test is not None:
        val_preds_ret["LightGBM"] = lgb_pred_all[val_split_idx:train_seq_end]
        test_preds_ret["LightGBM"] = lgb_pred_test
        all_preds["LightGBM"] = lgb_pred_all

    # ── Per-model val metrics: MAPE + directional accuracy ──
    val_metrics = {}
    val_y_logret = log_returns[SEQ_LEN + val_split_idx: SEQ_LEN + train_seq_end]
    val_actual_dir = (val_actual_prices > val_prev).astype(int)

    if len(val_actual_prices) >= 20:
        for name, pred_ret in val_preds_ret.items():
            pred_price = val_prev * np.exp(pred_ret)
            mape = float(np.mean(np.abs((val_actual_prices - pred_price) / val_actual_prices)) * 100)
            pred_dir = (pred_price > val_prev).astype(int)
            dir_acc = float(np.mean(val_actual_dir == pred_dir))
            val_metrics[name] = {"mape": round(mape, 2), "dir": round(dir_acc * 100, 1)}

    # ── Anti-skill filter: модели с dir_acc <= 50.5% на val исключаются ──
    # Это самый важный фильтр — модель хуже монетки на val почти гарантированно
    # будет хуже монетки на test тоже. Включать её = вредить ансамблю.
    # Threshold 50.5% (было 50.0%): coin-flip baseline = 50%, и при шумном val
    # модель с 50.0-50.5% статистически не отличима от монетки. Берём только
    # те, кто УВЕРЕННО лучше — иначе фильтр пропускает trivial "tomorrow ≈ today"
    # модели (типичный кейс на BTC-USD: все 4 модели дают ровно 50.6%).
    ANTI_SKILL_MIN_DIR = float(os.getenv("ANTI_SKILL_MIN_DIR", "50.5"))
    eligible = [
        n for n, m in val_metrics.items()
        if m["dir"] > ANTI_SKILL_MIN_DIR and 0 < m["mape"] < 15.0
    ]
    if not eligible:
        # Все модели anti-skill — хотя бы оставим TCN с очень низким весом
        eligible = ["TCN"] if "TCN" in val_preds_ret else list(val_preds_ret.keys())[:1]
        logger.warning(
            f"All models <= {ANTI_SKILL_MIN_DIR}%% dir_acc on val "
            f"(metrics={val_metrics}) — fallback to TCN-only"
        )

    base_weights = {n: 0.0 for n in val_preds_ret}

    # ── Primary: Rolling-NNLS multi-window stacking ──
    # X = матрица val-предсказаний (рядов = val_n, колонок = n_models)
    # y = истинные log-returns
    # NNLS на каждом окне даёт неотрицательные w >= 0, минимизирует ||Xw - y||.
    # Усредняем coef по окнам разной длины — модель должна быть стабильна на
    # 30/60/100-дневных горизонтах одновременно.
    stacking_used = False
    if len(val_actual_prices) >= 20 and len(eligible) >= 1:
        try:
            from scipy.optimize import nnls
            stack_X_full = np.column_stack([val_preds_ret[n] for n in eligible])
            stack_y_full = val_y_logret
            # Защита от NaN (могут возникнуть в Kalman/regime фичах на ранней истории)
            mask = np.isfinite(stack_X_full).all(axis=1) & np.isfinite(stack_y_full)
            stack_X = stack_X_full[mask]
            stack_y = stack_y_full[mask]
            n_val = len(stack_X)

            if n_val >= 20:
                # Окна берем с конца (наиболее свежие данные).
                # Минимум 20 точек на окно для надёжного NNLS.
                window_sizes = sorted(set(
                    w for w in [30, 60, 100, n_val] if 20 <= w <= n_val
                ))

                coefs_list = []
                for w in window_sizes:
                    coef_w, _ = nnls(stack_X[-w:], stack_y[-w:])
                    if coef_w.sum() > 1e-9:
                        coefs_list.append(coef_w / coef_w.sum())

                if coefs_list:
                    avg_coef = np.mean(coefs_list, axis=0)
                    s = avg_coef.sum()
                    if s > 1e-9:
                        avg_coef = avg_coef / s
                        for i, n in enumerate(eligible):
                            base_weights[n] = float(avg_coef[i])
                        stacking_used = True
                        logger.info(
                            f"Stacking weights (rolling-NNLS, k={len(coefs_list)} "
                            f"windows={window_sizes}): "
                            f"{ {n: round(base_weights[n], 3) for n in eligible} }"
                        )
        except Exception as e:
            logger.debug(f"Rolling NNLS stacking failed: {e}")

    # ── Fallback: (1/MAPE) × directional edge ──
    # Срабатывает если NNLS не сошёлся или дал нулевые коэф-ты
    if not stacking_used:
        if len(val_actual_prices) >= 20:
            for n in eligible:
                m = val_metrics[n]
                # dir_edge: насколько модель лучше монетки
                # 50%→0.05, 55%→0.10, 60%→0.15, 70%→0.25
                dir_edge = max(0.05, m["dir"] / 100 - 0.45)
                base_weights[n] = (1.0 / m["mape"]) * dir_edge

            total_w = sum(base_weights.values())
            if total_w > 0:
                base_weights = {k: v / total_w for k, v in base_weights.items()}
            else:
                eq = 1.0 / max(len(eligible), 1)
                for n in eligible:
                    base_weights[n] = eq
            logger.info(f"MAPE+dir weights (fallback): "
                        f"{ {n: round(base_weights[n], 3) for n in eligible} }")
        else:
            # Слишком мало данных для надёжных весов — TCN-only
            base_weights = {n: (1.0 if n == "TCN" else 0.0) for n in val_preds_ret}
            logger.info("Val fold too small — TCN-only fallback")

    # ── Direction-aware stacking blend (B2) ──────────────────────────
    # NNLS минимизирует ||X*w - y_returns||² → оптимизирует MAPE. На крипте
    # все модели learn "tomorrow ≈ today" → MAPE одинаковый, NNLS выбирает
    # одну "случайно" (XGBoost) с весом 1.0 — а её dir_acc такой же как у
    # остальных. Direction-aware blend подмешивает веса, пропорциональные
    # `excess_dir = max(0, val_dir - 50%)`. Если все модели на 50.6% →
    # равномерное распределение (лучше чем "1 модель 1.0", т.к. averaging
    # снижает variance). Если одна модель 60% и три 50% → ей дают веса.
    #
    # alpha=0.5 (default): half MAPE-optimal, half direction-optimal.
    # alpha=0 → выключено (старое поведение). alpha=1 → чистый dir-stacking.
    nnls_or_fallback_w = dict(base_weights)  # snapshot для лога
    DIRECTION_STACKING_ALPHA = float(os.getenv("DIRECTION_STACKING_ALPHA", "0.5"))
    if DIRECTION_STACKING_ALPHA > 0 and len(eligible) >= 2:
        excess = {n: max(0.0, val_metrics[n]["dir"] - 50.0) for n in eligible}
        total_skill = sum(excess.values())
        if total_skill > 0:
            dir_w = {n: excess[n] / total_skill for n in eligible}
            # Blend NNLS-веса (отлично для MAPE) и dir-веса (отлично для dir_acc)
            blended = {n: 0.0 for n in val_preds_ret}
            for n in eligible:
                base_w = base_weights.get(n, 0.0)
                d_w = dir_w.get(n, 0.0)
                blended[n] = (1 - DIRECTION_STACKING_ALPHA) * base_w + DIRECTION_STACKING_ALPHA * d_w
            total = sum(blended.values())
            if total > 0:
                base_weights = {k: v / total for k, v in blended.items()}
                logger.info(
                    f"Direction-aware blend (α={DIRECTION_STACKING_ALPHA}): "
                    f"NNLS={ {n: round(nnls_or_fallback_w[n], 2) for n in eligible} } "
                    f"× dir_w={ {n: round(dir_w[n], 2) for n in eligible} } "
                    f"→ final={ {n: round(base_weights[n], 2) for n in eligible} }"
                )

    logger.info(f"Final ensemble weights: {base_weights} | val_metrics: {val_metrics}")

    # Get regime for each data point
    regimes = df["Regime"].values[SEQ_LEN:]  # aligned with sequences

    # ── Regime-conditional NNLS ─────────────────────────────────────
    # Дополнительно к глобальному base_weights считаем NNLS отдельно по
    # каждому режиму внутри val-фолда. На инференсе подменяем base_weights
    # на regime_base_weights[regime] если есть. Idea: в calm-боковике одни
    # модели лучше, в crisis — другие. Pure-mathematical аналог regime_adjustments
    # (там вручную проставлены коэффициенты), но из data, без эвристики.
    regime_base_weights: dict = {}
    if (REGIME_NNLS_ENABLED and len(eligible) >= 2
            and len(val_actual_prices) >= 20 and stacking_used):
        try:
            from scipy.optimize import nnls
            stack_X_full = np.column_stack([val_preds_ret[n] for n in eligible])
            stack_y_full = val_y_logret
            val_regimes = regimes[val_split_idx:train_seq_end]  # выровнено с val
            mask_finite = np.isfinite(stack_X_full).all(axis=1) & np.isfinite(stack_y_full)
            for r in (0, 1, 2):
                if r >= len(val_regimes):
                    continue
                mask_r = (val_regimes == r) & mask_finite
                n_r = int(mask_r.sum())
                if n_r < REGIME_NNLS_MIN_POINTS:
                    continue
                X_r = stack_X_full[mask_r]
                y_r = stack_y_full[mask_r]
                coef_r, _ = nnls(X_r, y_r)
                if coef_r.sum() > 1e-9:
                    coef_r = coef_r / coef_r.sum()
                    w_r = {n: 0.0 for n in val_preds_ret}
                    for i, n in enumerate(eligible):
                        w_r[n] = float(coef_r[i])
                    regime_base_weights[r] = w_r
                    logger.info(
                        f"Regime-NNLS r={r} (n={n_r}): "
                        f"{ {n: round(w_r[n], 3) for n in eligible} }"
                    )
            if regime_base_weights:
                logger.info(
                    f"Regime-NNLS: {len(regime_base_weights)}/3 regimes have own weights "
                    f"(rest fall back to global base_weights)"
                )
        except Exception as e:
            logger.debug(f"Regime-NNLS failed: {e}")

    # Regime-adaptive weight adjustment
    # Calm (0): boost boosting models slightly (they work well in stable markets)
    # Normal (1): use base weights
    # Crisis (2): boost TCN, shrink boosting (less overfitting risk), reduce position sizes
    # NB: regime_adjustments применяется ПОВЕРХ regime_base_weights[regime] (если есть),
    # т.к. это коррекция по экспертной эвристике (TCN ↑ на crisis = anti-overfit boost).
    regime_adjustments = {
        0: {"TCN": 0.8, "CatBoost": 1.3, "XGBoost": 1.3, "LightGBM": 1.3},  # calm
        1: {"TCN": 1.0, "CatBoost": 1.0, "XGBoost": 1.0, "LightGBM": 1.0},  # normal
        2: {"TCN": 1.5, "CatBoost": 0.5, "XGBoost": 0.5, "LightGBM": 0.5},  # crisis
    }

    # ── Per-step confidence: std-dev across active models' predictions ──
    # Считаем РАНЬШЕ, потому что нужно для inverse-variance weighting в цикле ниже.
    # Если все модели согласны (low std) → high confidence.
    # Если расходятся (high std) → low confidence → (a) backtest не торгует, и
    # (b) inv-var weighting подмешивает uniform веса (страхует от over-confident NNLS).
    # active_models = union models с >0 весом в base_weights и regime_base_weights
    _active_set = {k for k, v in base_weights.items() if v > 0}
    for _rw in regime_base_weights.values():
        _active_set.update(k for k, v in _rw.items() if v > 0)
    active_models = sorted(_active_set, key=lambda x: -base_weights.get(x, 0.0))
    if not active_models:
        active_models = [k for k, v in base_weights.items() if v > 0] or list(all_preds.keys())[:1]

    if len(active_models) > 1:
        active_preds_matrix = np.column_stack([all_preds[n] for n in active_models])
        pred_std = np.std(active_preds_matrix, axis=1)
        # Robust z-score через median/MAD (не чувствителен к выбросам в std)
        _med = float(np.median(pred_std))
        _mad = float(np.median(np.abs(pred_std - _med))) + 1e-9
        pred_z = (pred_std - _med) / _mad
    else:
        pred_std = np.zeros(len(tcn_pred_returns))
        pred_z = np.zeros_like(pred_std)

    # alpha[i] ∈ [0,1]: насколько подмешать uniform к NNLS на шаге i.
    # sigmoid(z - threshold): при z=threshold → alpha=0.5; z >> threshold → alpha→1.
    if INV_VAR_WEIGHTING_ENABLED and len(active_models) > 1:
        alpha_per_step = 1.0 / (1.0 + np.exp(-(pred_z - INV_VAR_WEIGHTING_THRESHOLD)))
    else:
        alpha_per_step = np.zeros(len(tcn_pred_returns))

    # Apply regime-aware weights point by point
    pred_returns = np.zeros(len(tcn_pred_returns))
    inv_var_smooth_count = 0
    for i in range(len(tcn_pred_returns)):
        regime = int(regimes[i]) if i < len(regimes) else 1
        regime = max(0, min(2, regime))

        # Шаг 1: выбираем базу — режим-условные NNLS-веса или глобальные.
        bw_for_regime = regime_base_weights.get(regime, base_weights)
        adj = regime_adjustments.get(regime, regime_adjustments[1])

        # Шаг 2: regime_adjustments как мультипликатор поверх базы.
        point_weights = {name: bw_for_regime.get(name, 0.0) * adj.get(name, 1.0)
                         for name in all_preds}

        # Нормализация
        pw_total = sum(point_weights.values())
        if pw_total > 0:
            for name in point_weights:
                point_weights[name] /= pw_total

        # Шаг 3: inv-var soft-mix к uniform по точечному alpha[i].
        # При высоком расхождении моделей (high pred_std) alpha→1: ансамбль
        # склоняется к равномерному — страхуем от over-confident NNLS-ставки
        # на одной модели.
        if INV_VAR_WEIGHTING_ENABLED and len(active_models) > 1:
            a = float(alpha_per_step[i])
            if a > 0.05:  # экономим CPU на noise, ниже 5% эффекта почти нет
                # uniform только по моделям с >0 базой (сохраняем eligibility)
                nonzero_names = [n for n, w in point_weights.items() if w > 0]
                if len(nonzero_names) > 1:
                    uniform_w = 1.0 / len(nonzero_names)
                    for name in nonzero_names:
                        point_weights[name] = (1.0 - a) * point_weights[name] + a * uniform_w
                    inv_var_smooth_count += 1

        # Weighted prediction
        for name, w in point_weights.items():
            if w > 0 and name in all_preds:
                pred_returns[i] += w * all_preds[name][i]

    if INV_VAR_WEIGHTING_ENABLED and inv_var_smooth_count > 0:
        logger.info(
            f"Inv-var weighting: smoothed {inv_var_smooth_count}/{len(tcn_pred_returns)} "
            f"steps (mean alpha={float(np.mean(alpha_per_step)):.3f})"
        )

    model_names = active_models
    model_name = f"Ensemble ({'+'.join(active_models)})" if len(active_models) > 1 else active_models[0]

    prev_prices = close_prices[SEQ_LEN - 1: -1]
    actual_prices = close_prices[SEQ_LEN:]
    pred_prices = prev_prices * np.exp(pred_returns)

    dates = df.index[SEQ_LEN:].strftime("%Y-%m-%d").tolist()

    # Per-model prices (full predictions for visualization)
    model_comparison = {}
    tcn_prices = prev_prices * np.exp(tcn_pred_returns)
    model_comparison["TCN"] = tcn_prices.tolist()
    if cat_pred_all is not None:
        cat_prices = prev_prices * np.exp(cat_pred_all)
        model_comparison["CatBoost"] = cat_prices.tolist()
    if xgb_pred_all is not None:
        xgb_prices = prev_prices * np.exp(xgb_pred_all)
        model_comparison["XGBoost"] = xgb_prices.tolist()
    if lgb_pred_all is not None:
        lgb_prices = prev_prices * np.exp(lgb_pred_all)
        model_comparison["LightGBM"] = lgb_prices.tolist()
    model_comparison["Ensemble"] = pred_prices.tolist()

    # Per-model metrics (test set only)
    test_start = train_seq_end
    y_act_test = actual_prices[test_start:]

    def calc_metrics(y_true, y_pred_arr):
        _mae = mean_absolute_error(y_true, y_pred_arr)
        _rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_arr)))
        _mape = float(np.mean(np.abs((y_true - y_pred_arr) / y_true)) * 100)
        ss_r = np.sum((y_true - y_pred_arr) ** 2)
        ss_t = np.sum((y_true - np.mean(y_true)) ** 2)
        _r2 = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
        return {"mae": round(_mae, 2), "rmse": round(_rmse, 2), "mape": round(_mape, 2), "r2": round(_r2, 4)}

    model_metrics = {}
    model_metrics["TCN"] = calc_metrics(y_act_test, tcn_prices[test_start:])
    if cat_pred_all is not None:
        model_metrics["CatBoost"] = calc_metrics(y_act_test, cat_prices[test_start:])
    if xgb_pred_all is not None:
        model_metrics["XGBoost"] = calc_metrics(y_act_test, xgb_prices[test_start:])
    if lgb_pred_all is not None:
        model_metrics["LightGBM"] = calc_metrics(y_act_test, lgb_prices[test_start:])
    model_metrics["Ensemble"] = calc_metrics(y_act_test, pred_prices[test_start:])

    # ── Foundation augmentation для test-fold (BTC-USD dir_acc fix) ──
    # До этой правки Foundation подмешивался ТОЛЬКО в `future_preds` (будущее).
    # `dir_acc`, который видит пользователь в UI, считается по `pred_prices[test_start:]`
    # (чистый boosting/TCN ensemble) → Foundation вообще не влиял на главную метрику
    # качества. На крипте (BTC-USD) все base-модели learn "tomorrow ≈ today" → MAPE
    # низкий, но dir_acc ≈ 50% (монетка). Foundation (Chronos/TimesFM) обучен на
    # огромных corpora time-series и часто даёт нетривиальный directional signal.
    #
    # Решение: rolling Foundation на test-fold с шагом H=10.
    #  - history = close_prices[:SEQ_LEN + test_start + i] (no leak: только прошлое
    #    относительно прогнозируемого chunk)
    #  - horizon = H (короткий горизонт = лучшая Foundation точность)
    #  - blend 50/50 с pred_prices[test_start+i : test_start+i+H]
    # test_n=147, H=10 → ~15 foundation calls ≈ 50s. Только при use_foundation=True.
    # Data-driven α: считаем Foundation также на val-fold, перебираем
    # α ∈ {0.0, 0.1, ..., 1.0}, берём α* что минимизирует val MAPE, и
    # применяем α* к test. Если α*=0 — Foundation вреден на этом активе и
    # не блендится (Ensemble == чистый local). Это исправляет регрессию,
    # когда хардкод α=0.5 на BTC выпускал MAPE с 1.85% до 3.05%.
    foundation_test_used = False
    foundation_test_models: list[str] = []
    foundation_test_chunks = 0
    foundation_alpha_chosen = 0.0
    foundation_local_val_mape: float | None = None
    foundation_blended_val_mape: float | None = None
    if use_foundation and len(actual_prices) > test_start:
        try:
            from app.foundation import foundation_forecast
            H = int(os.getenv("FOUNDATION_TEST_CHUNK", "10"))
            models_seen: set[str] = set()

            def _rolling_foundation(start_idx: int, end_idx: int) -> np.ndarray:
                """Rolling Foundation forecast по сегменту [start_idx, end_idx).
                Каждый chunk использует только историю до начала chunk'а
                (no leak). Шаг — H баров, длина прогноза — H. Возвращает
                массив длиной (end_idx - start_idx) с NaN там где Foundation
                не вернул прогноз."""
                seg_n = max(0, end_idx - start_idx)
                arr = np.full(seg_n, np.nan, dtype=np.float64)
                if seg_n == 0:
                    return arr
                for i in range(0, seg_n, H):
                    chunk_h = min(H, seg_n - i)
                    history_end = SEQ_LEN + start_idx + i
                    history = close_prices[:history_end]
                    res = foundation_forecast(history, chunk_h)
                    if res is None or "median" not in res:
                        continue
                    med = np.array(res["median"], dtype=np.float64)
                    if len(med) != chunk_h:
                        continue
                    arr[i:i + chunk_h] = med
                    for m in res.get("models_used", []):
                        models_seen.add(m)
                return arr

            # Test-fold rolling (всегда нужно — финальный blend применяется к test)
            test_n = len(actual_prices) - test_start
            fnd_test_arr = _rolling_foundation(test_start, len(actual_prices))
            valid_test = ~np.isnan(fnd_test_arr)
            foundation_test_chunks = int(np.ceil(valid_test.sum() / H)) if valid_test.any() else 0

            # Val-fold rolling (для honest выбора α)
            # val_split_idx определён в boosting-блоке выше; защищаемся на случай
            # когда val-fold пустой (короткая история).
            val_n = train_seq_end - val_split_idx
            if val_n >= H and valid_test.any():
                fnd_val_arr = _rolling_foundation(val_split_idx, train_seq_end)
                valid_val = ~np.isnan(fnd_val_arr)
            else:
                fnd_val_arr = np.full(max(0, val_n), np.nan, dtype=np.float64)
                valid_val = np.zeros_like(fnd_val_arr, dtype=bool)

            if valid_test.any() and valid_val.any():
                local_val = pred_prices[val_split_idx:train_seq_end].astype(np.float64)
                actual_val = actual_prices[val_split_idx:train_seq_end].astype(np.float64)

                def _val_mape(alpha: float) -> float:
                    blended = local_val.copy()
                    blended[valid_val] = (1.0 - alpha) * blended[valid_val] + alpha * fnd_val_arr[valid_val]
                    safe_v = actual_val > 0
                    if not safe_v.any():
                        return float("inf")
                    return float(np.mean(np.abs((blended[safe_v] - actual_val[safe_v]) / actual_val[safe_v])) * 100)

                alphas = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
                mapes = np.array([_val_mape(a) for a in alphas], dtype=np.float64)
                best_idx = int(np.argmin(mapes))
                foundation_alpha_chosen = float(alphas[best_idx])
                foundation_local_val_mape = float(mapes[0])  # α=0 → local only
                foundation_blended_val_mape = float(mapes[best_idx])

                grid_str = ", ".join(f"α={a:.1f}→{m:.2f}%" for a, m in zip(alphas, mapes))
                logger.info(
                    f"Foundation val-fold α grid: {grid_str} → chosen α={foundation_alpha_chosen:.2f} "
                    f"(local_val_mape={foundation_local_val_mape:.2f}%, "
                    f"best_val_mape={foundation_blended_val_mape:.2f}%)"
                )

                if foundation_alpha_chosen > 0:
                    a = foundation_alpha_chosen
                    pred_prices = pred_prices.copy()
                    pred_returns_new = pred_returns.copy()
                    blended_seg = pred_prices[test_start:].astype(np.float64).copy()
                    blended_seg[valid_test] = (1.0 - a) * blended_seg[valid_test] + a * fnd_test_arr[valid_test]
                    pred_prices[test_start:] = blended_seg
                    # Backtest consistency: pred_returns соответствуют blended prices.
                    prev_seg = prev_prices[test_start:].astype(np.float64)
                    safe = prev_seg > 0
                    new_ret = np.zeros_like(prev_seg)
                    new_ret[safe] = np.log(blended_seg[safe] / prev_seg[safe])
                    pred_returns_new[test_start:] = new_ret
                    pred_returns = pred_returns_new
                    model_metrics["Ensemble"] = calc_metrics(
                        y_act_test, pred_prices[test_start:]
                    )
                    foundation_test_used = True
                    foundation_test_models = sorted(models_seen)
                    logger.info(
                        f"Foundation test-fold blend applied: α={a:.2f}, "
                        f"chunks={foundation_test_chunks}×{H}d, "
                        f"{int(valid_test.sum())}/{test_n} pts blended, "
                        f"models={foundation_test_models}, Ensemble→{model_metrics['Ensemble']}"
                    )
                else:
                    logger.info(
                        "Foundation gated OFF: optimal α=0 on val-fold "
                        f"(local val MAPE {foundation_local_val_mape:.2f}% — Foundation hurts on this asset)"
                    )
        except Exception as e:
            logger.warning(
                f"Foundation test-fold blend failed: {e} — using pure ensemble"
            )

    # Feature importance names (3 stats per column: last, momentum, volatility)
    feature_importance = {}
    sel_feat_names = []
    for col_idx in range(len(MODEL_COLS)):
        for stat in ["last", "momentum", "vol"]:
            sel_feat_names.append(f"{MODEL_COLS[col_idx]}_{stat}")
    for col_idx in range(len(BOOSTING_EXTRA_COLS)):
        for stat in ["last", "momentum", "vol"]:
            sel_feat_names.append(f"{BOOSTING_EXTRA_COLS[col_idx]}_{stat}")
    sel_feat_names.append("Sentiment")

    if cat is not None:
        try:
            cat_imp = cat.get_feature_importance()
            top_idx = np.argsort(cat_imp)[::-1][:15]
            feature_importance["CatBoost"] = {
                "names": [sel_feat_names[i] if i < len(sel_feat_names) else f"feat_{i}" for i in top_idx],
                "values": [round(float(cat_imp[i]), 2) for i in top_idx],
            }
        except Exception:
            pass
    if xgb_m is not None:
        try:
            xgb_imp = xgb_m.feature_importances_
            top_idx = np.argsort(xgb_imp)[::-1][:15]
            feature_importance["XGBoost"] = {
                "names": [sel_feat_names[i] if i < len(sel_feat_names) else f"feat_{i}" for i in top_idx],
                "values": [round(float(xgb_imp[i]), 4) for i in top_idx],
            }
        except Exception:
            pass
    if lgb_m is not None:
        try:
            lgb_imp = lgb_m.feature_importances_.astype(float)
            top_idx = np.argsort(lgb_imp)[::-1][:15]
            feature_importance["LightGBM"] = {
                "names": [sel_feat_names[i] if i < len(sel_feat_names) else f"feat_{i}" for i in top_idx],
                "values": [round(float(lgb_imp[i]), 2) for i in top_idx],
            }
        except Exception:
            pass

    # SHAP explanations (TreeExplainer)
    shap_data = {}
    try:
        import shap

        X_shap_test = X_bst_test_sel
        sample_idx = -1  # last prediction

        boosting_models = {}
        if cat is not None:
            boosting_models["CatBoost"] = cat
        if xgb_m is not None:
            boosting_models["XGBoost"] = xgb_m
        if lgb_m is not None:
            boosting_models["LightGBM"] = lgb_m

        for model_label, model_obj in boosting_models.items():
            try:
                explainer = shap.TreeExplainer(model_obj)

                sv_test = explainer.shap_values(X_shap_test)
                mean_abs = np.abs(sv_test).mean(axis=0)
                top_idx = np.argsort(mean_abs)[::-1][:15]

                sv_single = sv_test[sample_idx]
                base_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

                shap_data[model_label] = {
                    "global_names": [sel_feat_names[i] if i < len(sel_feat_names) else f"feat_{i}" for i in top_idx],
                    "global_values": [round(float(mean_abs[i]), 6) for i in top_idx],
                    "waterfall_names": [sel_feat_names[i] if i < len(sel_feat_names) else f"feat_{i}" for i in top_idx],
                    "waterfall_values": [round(float(sv_single[i]), 6) for i in top_idx],
                    "base_value": round(base_value, 6),
                    "output_value": round(float(sv_single.sum() + base_value), 6),
                    "beeswarm_names": [sel_feat_names[i] if i < len(sel_feat_names) else f"feat_{i}" for i in top_idx[:10]],
                    "beeswarm_shap": [[round(float(sv_test[j, i]), 6)
                                       for j in range(0, len(sv_test), max(1, len(sv_test) // 100))]
                                      for i in top_idx[:10]],
                    "beeswarm_feat": [[round(float(X_shap_test[j, i]), 6)
                                       for j in range(0, len(X_shap_test), max(1, len(X_shap_test) // 100))]
                                      for i in top_idx[:10]],
                }
            except Exception:
                pass
    except ImportError:
        pass

    y_pred_test = pred_prices[test_start:]
    residuals = (y_act_test - y_pred_test).tolist()

    mae = model_metrics["Ensemble"]["mae"]
    rmse = model_metrics["Ensemble"]["rmse"]
    mape = model_metrics["Ensemble"]["mape"]
    r2 = model_metrics["Ensemble"]["r2"]

    actual_dir = (actual_prices[test_start:] > prev_prices[test_start:]).astype(int)
    pred_dir = (pred_prices[test_start:] > prev_prices[test_start:]).astype(int)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
    # `final_hits` — массив 0/1 совпадений direction, который пойдёт в bootstrap.
    # Может быть переопределён ниже, если direction-classifier head побеждает.
    final_hits = (actual_dir == pred_dir).astype(int)

    if tcn_direction_pred is not None:
        dir_cls = tcn_direction_pred[test_start:]
        pred_dir_cls = (dir_cls > 0.5).astype(int)
        dir_acc_cls = float(np.mean(actual_dir == pred_dir_cls) * 100)
        if dir_acc_cls > dir_acc:
            dir_acc = dir_acc_cls
            final_hits = (actual_dir == pred_dir_cls).astype(int)

    # ── Bootstrap 95% CI на dir_acc (B1: честная оценка vs noise) ──
    # Точечная оценка dir_acc на N≈200 точках имеет стандартную ошибку
    # ~sqrt(p(1-p)/N) ≈ 3.5% при p=0.5 → 95% CI ≈ [43%, 57%]. Без CI
    # пользователь видит "49%" и думает "плохо", хотя честный ответ
    # "не отличимо от 55%". Bootstrap percentile на исходных hits даёт
    # асимметричный CI (учитывает форму распределения), и считается на
    # тех hits, что используются для финального dir_acc (с учётом
    # direction-classifier override выше).
    n_hits = len(final_hits)
    dir_acc_ci_low = dir_acc
    dir_acc_ci_high = dir_acc
    if n_hits >= 30:
        rng = np.random.default_rng(seed=42)  # детерминизм для стабильности UI
        B = int(os.getenv("DIR_ACC_BOOTSTRAP", "1000"))
        # vectorized bootstrap: одна матрица [B, N] индексов, mean по axis=1.
        idx_matrix = rng.integers(0, n_hits, size=(B, n_hits))
        boot = final_hits[idx_matrix].mean(axis=1)
        dir_acc_ci_low = float(np.percentile(boot, 2.5) * 100)
        dir_acc_ci_high = float(np.percentile(boot, 97.5) * 100)

    corr_cols = ["Close", "Volume", "RSI", "MACD", "MA_5", "MA_20", "MA_50", "ATR", "BB_upper", "BB_lower"]
    corr_data = df[corr_cols].corr().round(3).values.tolist()
    corr_labels = corr_cols

    # ── Future forecast: Conformal-calibrated intervals ───────────────
    # Старая версия: Monte Carlo GBM с σ из последних 60 дней. Это гипотеза
    # "returns log-normal" — на крипто (fat tails) систематически даёт узкие
    # интервалы → реальное покрытие 90% оказывается 60-70%.
    # Новая: Split Conformal Prediction — берём фактические остатки в log-space
    # на val-фолде, считаем signed quantiles. Это ГАРАНТИРУЕТ заявленное
    # покрытие при условии exchangeability (residuals не сильно меняют
    # распределение между val и future). MC оставляем как fallback для случая,
    # когда val слишком короткий (<30 точек).
    # Для multi-step масштабируем quantiles на sqrt(h) (random-walk diffusion).
    future_dates, future_preds, future_upper, future_lower = [], [], [], []
    future_p50, future_p5, future_p95 = [], [], []
    # Local-cal маркеры — обязаны существовать в любом code-path, иначе при
    # days_ahead=0 (BTC-USD dashboard mode) return-dict ниже падает с
    # UnboundLocalError. Внутри `if days_ahead > 0:` блок ниже их перезапишет.
    local_calibration_applied = False
    local_sigma_ratio_val = 1.0

    if days_ahead > 0:
        # Calibration: residuals on val-фолд (not-test) → exchangeability ок
        val_pred_prices_conf = pred_prices[val_split_idx:train_seq_end]
        val_act_prices_conf = actual_prices[val_split_idx:train_seq_end]
        use_conformal = (
            len(val_pred_prices_conf) >= 30
            and bool((val_pred_prices_conf > 0).all())
            and bool((val_act_prices_conf > 0).all())
        )

        N_SIM = MC_SIMS
        last_price = close_prices[-1]
        recent_returns = log_returns[-60:]
        mu_hist = float(np.mean(recent_returns))
        sigma_daily = float(np.std(recent_returns))
        model_signal = float(pred_returns[-1])
        mu = 0.7 * model_signal + 0.3 * mu_hist
        dt = 1.0

        np.random.seed(42)
        Z = np.random.standard_normal((N_SIM, days_ahead))
        mc_returns = (mu - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z
        cum_returns = np.cumsum(mc_returns, axis=1)
        trajectories = last_price * np.exp(cum_returns)

        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(1, "d"), periods=days_ahead
        ).strftime("%Y-%m-%d").tolist()

        # Median trajectory (drift) — берём из MC: это лучшая оценка центра
        future_preds = np.median(trajectories, axis=0).tolist()

        # local_calibration_applied / local_sigma_ratio_val уже инициализированы
        # выше (до `if days_ahead > 0`) — это нужно для случая days_ahead=0,
        # где этот блок целиком не исполняется. Перезапишутся ниже если
        # use_conformal + LOCAL_CALIBRATION=1.

        if use_conformal:
            # Signed log-residuals → asymmetry-aware (bull/bear skew)
            val_log_resid = np.log(val_act_prices_conf / val_pred_prices_conf)
            # Robust quantile через np.quantile (linear interp)
            q05_log = float(np.quantile(val_log_resid, 0.05))
            q25_log = float(np.quantile(val_log_resid, 0.25))
            q75_log = float(np.quantile(val_log_resid, 0.75))
            q95_log = float(np.quantile(val_log_resid, 0.95))

            # ── Tier 5.4: Locally-adaptive calibration (NGBoost) ─────────
            # Conformal выше даёт глобально калиброванные bands (homoskedastic).
            # NGBoost, обученный на (X_cal, val_log_resid), оценивает σ(x) для
            # текущих features — даёт локальный inflation-коэффициент в
            # [SIGMA_RATIO_MIN, SIGMA_RATIO_MAX]. Умножаем все quantiles на него.
            # Включение: LOCAL_CALIBRATION=1. Graceful fallback → ratio=1.0 (no-op).
            #
            # Rolling window (Polish #1): compute_local_inflation() внутри берёт
            # последние NGB_ROLLING_WINDOW=100 строк из (X_cal, residuals). Это
            # держит calibration set близко к X_future по распределению, что
            # резко снижает частоту упирания σ-ratio в clamp.
            try:
                from app.calibration import compute_local_inflation
                # X_cal = те же features что использованы для conformal-residuals.
                # X_future = последняя известная строка features (≈ состояние "сейчас",
                # так как future-features недоступны без data leakage).
                # Alignment: MULTIHORIZON_ENABLED trim'ит X_bst_val_sel на (max_h - 1)
                # строк с конца — передаём общий префикс; rolling window берётся уже
                # внутри calibration.py.
                n_use = min(len(X_bst_val_sel), len(val_log_resid))
                X_cal_local = X_bst_val_sel[:n_use] if n_use > 0 else None
                val_log_resid_local = val_log_resid[:n_use] if n_use > 0 else None
                X_future_local = X_bst_all_sel[-1] if len(X_bst_all_sel) > 0 else None
                if X_cal_local is not None and X_future_local is not None:
                    local_sigma_ratio_val, local_calibration_applied = (
                        compute_local_inflation(
                            X_cal_local, val_log_resid_local, X_future_local
                        )
                    )
                    if local_calibration_applied:
                        q05_log *= local_sigma_ratio_val
                        q25_log *= local_sigma_ratio_val
                        q75_log *= local_sigma_ratio_val
                        q95_log *= local_sigma_ratio_val
                        logger.info(
                            f"Local calibration (NGBoost): σ_ratio="
                            f"{local_sigma_ratio_val:.3f} → bands "
                            f"{'widened' if local_sigma_ratio_val > 1 else 'tightened'}"
                        )
            except Exception as e:
                logger.warning(
                    f"Local calibration hook failed: {e}; using conformal as-is"
                )

            future_p5, future_p95 = [], []
            future_upper, future_lower = [], []
            for h in range(days_ahead):
                scale = np.sqrt(h + 1)  # random-walk diffusion
                base = future_preds[h]
                future_p5.append(round(float(base * np.exp(q05_log * scale)), 4))
                future_p95.append(round(float(base * np.exp(q95_log * scale)), 4))
                future_upper.append(round(float(base * np.exp(q75_log * scale)), 4))
                future_lower.append(round(float(base * np.exp(q25_log * scale)), 4))
            logger.info(
                f"Conformal intervals (n_cal={len(val_log_resid)}): "
                f"q05={q05_log:.4f}, q95={q95_log:.4f}"
            )
        else:
            # Fallback: чистый MC percentiles
            future_p5 = np.percentile(trajectories, 5, axis=0).tolist()
            future_p95 = np.percentile(trajectories, 95, axis=0).tolist()
            future_upper = np.percentile(trajectories, 75, axis=0).tolist()
            future_lower = np.percentile(trajectories, 25, axis=0).tolist()
            logger.info("Conformal calibration skipped (val too short) — using MC fallback")

    # ── Tier 4: Foundation model augmentation (opt-in) ───────────────
    # Если пользователь поставил чекбокс "Foundation-модель" → дополнительно
    # прогоняем Chronos-Bolt zero-shot на close_prices. Используем результат:
    #  (a) median: blend 50/50 с нашим conformal-median (диверсификация source);
    #  (b) bands: заменяем на родные Chronos quantiles (probabilistically
    #      калиброваны на 10B time-series точках, обычно лучше нашего
    #      conformal на коротких val-фолдах).
    # Tier 5.2: теперь это ensemble из Chronos + TimesFM (если обе доступны).
    # foundation_forecast внутри усредняет модели и возвращает объединённый
    # прогноз + список models_used для UI-бейджа. Если ни одна не доступна —
    # silent fallback к conformal.
    foundation_used = False
    foundation_models_used: list[str] = []
    if days_ahead > 0 and use_foundation:
        try:
            from app.foundation import foundation_forecast
            fnd_result = foundation_forecast(close_prices, days_ahead)
            if fnd_result is not None and len(fnd_result.get("median", [])) == days_ahead:
                # Blend: 50/50 наш conformal-median + foundation median.
                blended = []
                for i, h in enumerate(future_preds):
                    blended.append(round((float(h) + float(fnd_result["median"][i])) / 2.0, 4))
                future_preds = blended
                # Bands → foundation quantiles напрямую (они калиброваны на 10-100B точках)
                future_p5 = [round(float(x), 4) for x in fnd_result["p5"]]
                future_p95 = [round(float(x), 4) for x in fnd_result["p95"]]
                future_upper = [round(float(x), 4) for x in fnd_result["p75"]]
                future_lower = [round(float(x), 4) for x in fnd_result["p25"]]
                foundation_used = True
                foundation_models_used = list(fnd_result.get("models_used", []))
                logger.info(
                    f"Foundation ensemble used: models={foundation_models_used}, "
                    f"days={days_ahead}, median_blend=50/50"
                )
            else:
                logger.info("Foundation model requested but unavailable — fallback to conformal")
        except Exception as e:
            logger.warning(f"Foundation augmentation failed: {e} — fallback to conformal")

    # ── Backtesting (test set only) ──
    # ATR/Close ratio per step → adaptive stop/trail thresholds внутри backtest.
    atr_test = df["ATR"].values[SEQ_LEN + test_start: SEQ_LEN + len(actual_prices)]
    close_test = actual_prices[test_start:]
    atr_pct_test = (atr_test / close_test) if len(atr_test) == len(close_test) else None

    backtest = _run_backtest(
        dates[test_start:],
        actual_prices[test_start:],
        pred_returns[test_start:],
        close_prices[SEQ_LEN + test_start - 1: SEQ_LEN + len(actual_prices) - 1],
        pred_std=pred_std[test_start:],
        atr_pct=atr_pct_test,
    )

    return {
        "dates": dates,
        "y_act": actual_prices,
        "y_pred": pred_prices,
        "train_size": train_seq_end,
        "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
        "dir_acc": dir_acc,
        "model_name": model_name,
        "future_dates": future_dates,
        "future_preds": future_preds,
        "future_upper": future_upper,
        "future_lower": future_lower,
        "future_p5": future_p5,
        "future_p95": future_p95,
        "rsi": df["RSI"].values[SEQ_LEN:].round(2).tolist(),
        "macd": df["MACD"].values[SEQ_LEN:].round(2).tolist(),
        "signal": df["Signal"].values[SEQ_LEN:].round(2).tolist(),
        "bb_upper": df["BB_upper"].values[SEQ_LEN:].round(2).tolist(),
        "bb_lower": df["BB_lower"].values[SEQ_LEN:].round(2).tolist(),
        "atr": df["ATR"].values[SEQ_LEN:].round(2).tolist(),
        "date_index": df.index[SEQ_LEN:],
        "model_comparison": model_comparison,
        "model_metrics": model_metrics,
        "feature_importance": feature_importance,
        "shap_data": shap_data,
        "residuals": residuals,
        "corr_data": corr_data,
        "corr_labels": corr_labels,
        "backtest": backtest,
        "foundation_used": foundation_used,
        "foundation_models": foundation_models_used,
        # Foundation-в-test-fold (BTC-USD dir_acc fix): отражает то, что Foundation
        # подмешан и в test-метрики, не только в future_preds.
        "foundation_test_used": foundation_test_used,
        "foundation_test_models": foundation_test_models,
        "foundation_test_chunks": foundation_test_chunks,
        # Data-driven α: какой коэффициент бленда выбран по val-fold.
        # 0.0 → Foundation вреден на этом активе и НЕ применён к test.
        # >0  → Foundation помог на val, применили α к test.
        "foundation_alpha": round(foundation_alpha_chosen, 2),
        "foundation_local_val_mape": (
            round(foundation_local_val_mape, 2)
            if foundation_local_val_mape is not None else None
        ),
        "foundation_blended_val_mape": (
            round(foundation_blended_val_mape, 2)
            if foundation_blended_val_mape is not None else None
        ),
        # ── Honest skill warning ──────────────────────────────────────
        # Bootstrap 95% CI (B1) даёт честный ответ "выше монетки или это
        # просто noise". `low_directional_skill=True` если CI пересекает
        # 50% (нижняя граница ≤ 50%) — direction статистически не
        # отличается от coin-flip даже при point estimate 55%. До B1
        # был грубый порог dir_acc < 52%; теперь принимаем решение по CI.
        # Fallback на старый порог если CI=точечная оценка (n<30).
        "dir_acc_ci_low": dir_acc_ci_low,
        "dir_acc_ci_high": dir_acc_ci_high,
        "low_directional_skill": bool(
            dir_acc_ci_low <= 50.0
            if dir_acc_ci_low != dir_acc
            else dir_acc < float(os.getenv("LOW_DIR_THRESHOLD", "52.0"))
        ),
        "local_calibration_applied": local_calibration_applied,
        "local_sigma_ratio": local_sigma_ratio_val,
    }


# ── Tier 5.1: Sentiment bias as post-processing ────────────────────
# Живой sentiment из FinBERT не должен участвовать в тяжёлом training
# (был бы константой → бесполезен для gradient boost). Вместо этого
# применяем bias ПОСЛЕ run_prediction — это:
#  • сохраняет кэш прогноза (news меняются каждые 30 мин, а train/predict —
#    дорогая операция)
#  • прозрачен для пользователя (видно "+0.5% sentiment bias")
#  • легко откатывается если не нравится
SENTIMENT_STRENGTH = float(os.getenv("SENTIMENT_STRENGTH", "0.008"))  # max bias per |score|=1
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.05"))  # ниже — игнор
SENTIMENT_MIN_ARTICLES = int(os.getenv("SENTIMENT_MIN_ARTICLES", "3"))  # меньше — игнор
SENTIMENT_DECAY_DAYS = float(os.getenv("SENTIMENT_DECAY_DAYS", "3.0"))  # tau для (1 - exp(-i/tau))


def apply_sentiment_bias(
    result: dict,
    sentiment_score: float,
    total_articles: int = 0,
) -> dict:
    """
    Модифицирует future_* в result согласно sentiment score.

    Формула per-step bias (день i, i=1..days_ahead):
        confidence = min(total_articles / 10, 1.0)
        effective_score = clip(sentiment_score * confidence, -1, +1)
        bias[i] = effective_score * SENTIMENT_STRENGTH * (1 - exp(-i / tau))

    Пример: score=+0.5, articles=20, day=5:
        confidence=1.0, effective=0.5
        bias = 0.5 * 0.008 * (1 - exp(-5/3)) = 0.5 * 0.008 * 0.81 ≈ 0.0032 (+0.32%)

    Сдвиг применяется и к median, и к всем bands (ширина сохраняется,
    центр смещается в сторону sentiment).

    Args:
        result: dict из run_prediction (модифицируется in-place)
        sentiment_score: avg_score из SentimentResult ∈ [-1, +1]
        total_articles: сколько новостей учтено (для confidence weighting)

    Returns:
        тот же result с обновлёнными future_* и добавленными:
            sentiment_bias_pct: float — общий средний bias в %
            sentiment_applied: bool
    """
    # Sentinel defaults (чтобы UI всегда имел поля)
    result["sentiment_bias_pct"] = 0.0
    result["sentiment_applied"] = False

    future_preds = result.get("future_preds") or []
    if not future_preds:
        return result  # days_ahead=0

    # Guard rails: слишком слабый signal или мало статей → не применяем
    if abs(sentiment_score) < SENTIMENT_THRESHOLD:
        return result
    if total_articles < SENTIMENT_MIN_ARTICLES:
        return result

    confidence = min(total_articles / 10.0, 1.0)
    effective = float(np.clip(sentiment_score * confidence, -1.0, 1.0))

    def _shift(arr: list, factors: np.ndarray) -> list:
        """Применяет мультипликативный сдвиг (1 + bias) покомпонентно."""
        out = []
        for v, f in zip(arr, factors):
            try:
                out.append(round(float(v) * (1.0 + f), 4))
            except (TypeError, ValueError):
                out.append(v)  # None / нечисло — не трогаем
        return out

    days = len(future_preds)
    # Per-day bias: насыщается к asymptote = effective * SENTIMENT_STRENGTH
    i_arr = np.arange(1, days + 1, dtype=np.float32)
    factors = effective * SENTIMENT_STRENGTH * (1.0 - np.exp(-i_arr / SENTIMENT_DECAY_DAYS))

    result["future_preds"] = _shift(future_preds, factors)
    for key in ("future_upper", "future_lower", "future_p5", "future_p95"):
        if key in result and result[key]:
            result[key] = _shift(result[key], factors)

    total_bias_pct = float(np.mean(factors)) * 100.0
    result["sentiment_bias_pct"] = round(total_bias_pct, 3)
    result["sentiment_applied"] = True
    logger.info(
        f"Sentiment bias applied: score={sentiment_score:+.3f}, articles={total_articles}, "
        f"confidence={confidence:.2f}, effective={effective:+.3f}, "
        f"mean_bias={total_bias_pct:+.3f}%, days={days}"
    )
    return result


def fetch_and_preprocess(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Daily OHLCV + macro features + preprocess pipeline.

    Source priority for the TARGET ticker:
    1. **Crypto tickers** (BTC-USD / ETH-USD / SOL-USD / etc., per
       :func:`app.hourly_skill.is_crypto_ticker`) → Binance Spot via
       :func:`app.crypto_data.fetch_crypto_ohlcv`. Binance has 7+ years
       of daily data per pair, real volume, no zero-volume bars, free
       and rate-limit-friendly. Yahoo Finance is auto-fallback if
       Binance is unreachable (geo-block / network fail).
    2. **Stocks / indices** (AAPL / ^GSPC / etc.) → yfinance directly,
       since Binance doesn't list them.

    Macro features (SP500, VIX, FRED) always come from yfinance/FRED
    regardless of the target — those tickers are stocks/indices and
    don't have crypto equivalents.

    P2: результат кэшируется в Redis на YF_CACHE_TTL секунд (30 мин).
    Сам фичеринг тяжёлый (Kalman, HMM, indicators), поэтому кэшируем финальный
    DataFrame, а не только raw quotes.

    Release T.16 (2026-05-01): swapped target-ticker source from
    pure yfinance to Binance-first for crypto. Same model architecture
    (TCN + LightGBM), but trained on the cleaner Binance OHLCV stream
    (no zero-volume artefacts, longer history when --lookback >5y).
    """
    cache_key = "neucast:fap:" + hashlib.md5(
        f"{ticker}|{start_date}|{end_date}".encode()
    ).hexdigest()

    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("fetch_and_preprocess cache HIT: %s %s..%s",
                    ticker, start_date, end_date)
        return cached

    # Try Binance for crypto tickers; fall through to yfinance otherwise
    # (and as a fallback if Binance returns no data).
    df_raw = None
    source = "yfinance"
    try:
        from app.hourly_skill import is_crypto_ticker
        if is_crypto_ticker(ticker):
            from app.crypto_data import fetch_crypto_ohlcv
            # Convert (start_date, end_date) → lookback_days for the
            # Binance fetcher. We over-shoot by 7 days so the result
            # comfortably brackets the requested range; downstream
            # filter to [start_date, end_date] preserves the original
            # contract.
            days_diff = (
                pd.Timestamp(end_date) - pd.Timestamp(start_date)
            ).days + 7
            df_binance, source = fetch_crypto_ohlcv(
                ticker,
                lookback_days=max(7, days_diff),
                interval="1d",
                prefer="binance",
            )
            if df_binance is not None and not df_binance.empty:
                # Filter to the requested date range. Index is
                # tz-aware UTC; pd.Timestamp(start_date) is tz-naive
                # so localize for comparison.
                start_ts = pd.Timestamp(start_date, tz="UTC")
                end_ts = pd.Timestamp(end_date, tz="UTC")
                df_raw = df_binance.loc[
                    (df_binance.index >= start_ts) & (df_binance.index <= end_ts)
                ].copy()
                # The downstream preprocess() expects tz-naive index
                # (yfinance is naive), strip tz for parity.
                df_raw.index = df_raw.index.tz_localize(None)
                logger.info(
                    "fetch_and_preprocess: %s using %s (%d daily bars)",
                    ticker, source, len(df_raw),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "fetch_and_preprocess: Binance path failed for %s, "
            "falling through to yfinance: %s", ticker, exc,
        )
        df_raw = None

    if df_raw is None or df_raw.empty:
        df_raw = yf.download(
            ticker, start=start_date, end=end_date,
            interval="1d", progress=False,
        )
        source = "yfinance"
        if df_raw.empty:
            raise ValueError(
                f"Нет данных по тикеру {ticker} в диапазоне "
                f"{start_date}..{end_date} (ни на Binance, ни на yfinance)"
            )
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

    # Fetch cross-asset macro features in parallel date range
    macro_df = _fetch_macro_features(start_date, end_date)
    df = preprocess(df_raw, macro_df)
    _cache_set(cache_key, df, YF_CACHE_TTL)
    return df
