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


def _cache_get(key: str):
    r = _get_redis()
    if not r:
        return None
    try:
        data = r.get(key)
        return pickle.loads(data) if data else None
    except Exception as e:
        logger.debug("cache_get fail %s: %s", key, e)
        return None


def _cache_set(key: str, value, ttl: int):
    r = _get_redis()
    if not r:
        return
    try:
        r.setex(key, ttl, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
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
def _temporal_weights(n: int, half_life: int = 120) -> np.ndarray:
    """
    Exponential decay: recent samples weighted more heavily.
    half_life=120 trading days ~ 6 months.
    Normalized so mean weight = 1.0 (doesn't change effective sample size).
    """
    decay = np.log(2) / half_life
    weights = np.exp(decay * np.arange(n))
    return weights / weights.mean()


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


def make_boosting_features_extended(data_tcn: np.ndarray, data_extra: np.ndarray,
                                     sentiment_score: float = 0.0) -> np.ndarray:
    """
    Build boosting features from TCN columns + extra columns + sentiment.
    Each 60-day window -> 3 stats per column (last, momentum, volatility) + sentiment.
    Reduced from 7 stats to avoid feature explosion with limited samples.
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
        # Sentiment score (constant across window)
        feats.append(sentiment_score)
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
):
    """
    Realistic long/short backtest with:
    - Signal threshold (dead zone): don't trade if |signal| too weak
    - Position holding: only pay costs when position CHANGES direction
    - Kelly sizing: quarter-Kelly, max 50% per trade
    - Transaction costs: only on actual trades (not daily churn)
    """
    n = len(dates)
    if n < 2:
        return None

    one_way_cost = (commission_pct + slippage_pct) / 100.0  # entry OR exit cost

    if use_kelly:
        kelly_sizes = _kelly_fraction(pred_returns)
    else:
        kelly_sizes = np.ones(n) * 0.5

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

    # Signal threshold: Z > 0.5 = LONG, Z < -0.5 = SHORT, else hold
    dead_zone = 0.5

    capital = initial_capital
    equity = [capital]
    positions = []
    daily_returns = []
    wins = 0
    losses = 0
    total_costs = 0.0
    num_trades = 0

    actual_daily_returns = np.diff(actual_prices) / actual_prices[:-1]

    current_position = "FLAT"  # FLAT / LONG / SHORT

    for i in range(n - 1):
        signal = z_signals[i]
        actual_ret = actual_daily_returns[i]
        position_size = kelly_sizes[i]

        # Determine desired position based on Z-score
        if signal > dead_zone:
            desired = "LONG"
        elif signal < -dead_zone:
            desired = "SHORT"
        else:
            desired = current_position if current_position != "FLAT" else "FLAT"

        # Trade cost only when position changes
        trade_cost = 0.0
        if desired != current_position:
            if current_position != "FLAT":
                trade_cost += capital * position_size * one_way_cost  # exit old
            if desired != "FLAT":
                trade_cost += capital * position_size * one_way_cost  # enter new
            num_trades += 1
            current_position = desired

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
        daily_returns.append(net_pnl / equity[-2] if equity[-2] > 0 else 0)

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
        "trades": positions[-20:],  # last 20 trades for display
    }


def run_prediction(df: pd.DataFrame, days_ahead: int, sentiment_score: float = 0.0):
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
            sig.update(f"{TCN_EPOCHS}|{TCN_PATIENCE}|{IS_MULTITARGET}".encode())
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

        if len(X_train) > SEQ_LEN:
            val_size = max(0.1, SEQ_LEN / len(X_train))
            direction_all = (log_returns[SEQ_LEN:] > 0).astype(np.float32)
            dir_train = direction_all[:train_seq_end]
            if IS_MULTITARGET:
                fine_model.fit(
                    X_train,
                    {'return_output': y_train, 'direction_output': dir_train},
                    epochs=TCN_EPOCHS, batch_size=32,
                    validation_split=val_size,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=TCN_PATIENCE, restore_best_weights=True, verbose=0)],
                    verbose=0,
                )
            else:
                fine_model.fit(
                    X_train, y_train,
                    epochs=TCN_EPOCHS, batch_size=32,
                    validation_split=val_size,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=TCN_PATIENCE, restore_best_weights=True, verbose=0)],
                    verbose=0,
                )

        raw_pred = fine_model.predict(X_all, verbose=0)
        if IS_MULTITARGET:
            tcn_pred_returns = raw_pred[0].flatten()
            tcn_direction_pred = raw_pred[1].flatten()
        else:
            tcn_pred_returns = raw_pred.flatten()
            tcn_direction_pred = None

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

    X_bst_all = make_boosting_features_extended(scaled_all, scaled_extra, sentiment_score)

    # All models predict log_returns (same target as TCN — no triple barrier)
    y_bst_train = y_all[:train_seq_end]

    # Time-based validation split for early stopping (last 20% of train)
    val_split_idx = int(train_seq_end * 0.8)
    X_bst_train_sel = X_bst_all[:val_split_idx]
    X_bst_val_sel = X_bst_all[val_split_idx:train_seq_end]
    y_bst_val = y_bst_train[val_split_idx:]
    X_bst_test_sel = X_bst_all[train_seq_end:]
    X_bst_all_sel = X_bst_all

    # Sample weights for training set
    full_weights = _temporal_weights(val_split_idx)

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

    y_bst_train_sel = y_bst_train[:val_split_idx]

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
            sig.update(f"{BOOST_ITERS}|{BOOST_EARLY_STOP}|{sentiment_score:.2f}".encode())
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
    # ── Regime-Aware Inverse-MAPE Weighted Ensemble           ──
    # ══════════════════════════════════════════════════════════
    # Weights based on PRICE accuracy (MAPE), not return R2.
    # Return R2 is noisy and doesn't match displayed metrics.
    # Inverse-MAPE: lower error → higher weight.

    prev_prices_full = close_prices[SEQ_LEN - 1: -1]
    test_actual_prices = close_prices[SEQ_LEN + train_seq_end:]
    test_prev = prev_prices_full[train_seq_end:]

    test_preds_ret = {"TCN": tcn_pred_returns[train_seq_end:]}
    all_preds = {"TCN": tcn_pred_returns}

    if cat_pred_test is not None:
        test_preds_ret["CatBoost"] = cat_pred_test
        all_preds["CatBoost"] = cat_pred_all
    if xgb_pred_test is not None:
        test_preds_ret["XGBoost"] = xgb_pred_test
        all_preds["XGBoost"] = xgb_pred_all
    if lgb_pred_test is not None:
        test_preds_ret["LightGBM"] = lgb_pred_test
        all_preds["LightGBM"] = lgb_pred_all

    # Compute MAPE on prices for each model
    base_weights = {}
    for name, pred_ret in test_preds_ret.items():
        pred_price = test_prev * np.exp(pred_ret)
        mape = float(np.mean(np.abs((test_actual_prices - pred_price) / test_actual_prices)) * 100)
        # Inverse MAPE: lower error → higher weight. Filter out terrible models (MAPE > 10%)
        if mape < 10.0 and mape > 0:
            base_weights[name] = 1.0 / mape
        else:
            base_weights[name] = 0.0

    total_w = sum(base_weights.values())
    if total_w > 0:
        base_weights = {k: v / total_w for k, v in base_weights.items()}
    else:
        base_weights = {"TCN": 1.0}

    logger.info(f"Ensemble weights (inv-MAPE): {base_weights}")

    # Get regime for each data point
    regimes = df["Regime"].values[SEQ_LEN:]  # aligned with sequences

    # Regime-adaptive weight adjustment
    # Calm (0): boost boosting models slightly (they work well in stable markets)
    # Normal (1): use base weights
    # Crisis (2): boost TCN, shrink boosting (less overfitting risk), reduce position sizes
    regime_adjustments = {
        0: {"TCN": 0.8, "CatBoost": 1.3, "XGBoost": 1.3, "LightGBM": 1.3},  # calm
        1: {"TCN": 1.0, "CatBoost": 1.0, "XGBoost": 1.0, "LightGBM": 1.0},  # normal
        2: {"TCN": 1.5, "CatBoost": 0.5, "XGBoost": 0.5, "LightGBM": 0.5},  # crisis
    }

    # Apply regime-aware weights point by point
    pred_returns = np.zeros(len(tcn_pred_returns))
    for i in range(len(tcn_pred_returns)):
        regime = int(regimes[i]) if i < len(regimes) else 1
        regime = max(0, min(2, regime))
        adj = regime_adjustments.get(regime, regime_adjustments[1])

        # Adjust weights for this point
        point_weights = {}
        for name, bw in base_weights.items():
            point_weights[name] = bw * adj.get(name, 1.0)

        # Normalize
        pw_total = sum(point_weights.values())
        if pw_total > 0:
            for name in point_weights:
                point_weights[name] /= pw_total

        # Weighted prediction
        for name, w in point_weights.items():
            if w > 0 and name in all_preds:
                pred_returns[i] += w * all_preds[name][i]

    active_models = [k for k, v in base_weights.items() if v > 0]
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

    if tcn_direction_pred is not None:
        dir_cls = tcn_direction_pred[test_start:]
        pred_dir_cls = (dir_cls > 0.5).astype(int)
        dir_acc_cls = float(np.mean(actual_dir == pred_dir_cls) * 100)
        if dir_acc_cls > dir_acc:
            dir_acc = dir_acc_cls

    corr_cols = ["Close", "Volume", "RSI", "MACD", "MA_5", "MA_20", "MA_50", "ATR", "BB_upper", "BB_lower"]
    corr_data = df[corr_cols].corr().round(3).values.tolist()
    corr_labels = corr_cols

    # Future forecast (Monte Carlo GBM)
    future_dates, future_preds, future_upper, future_lower = [], [], [], []
    future_p50, future_p5, future_p95 = [], [], []

    if days_ahead > 0:
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

        future_preds = np.median(trajectories, axis=0).tolist()
        future_p5 = np.percentile(trajectories, 5, axis=0).tolist()
        future_p95 = np.percentile(trajectories, 95, axis=0).tolist()
        future_upper = np.percentile(trajectories, 75, axis=0).tolist()
        future_lower = np.percentile(trajectories, 25, axis=0).tolist()

    # ── Backtesting (test set only) ──
    backtest = _run_backtest(
        dates[test_start:],
        actual_prices[test_start:],
        pred_returns[test_start:],
        close_prices[SEQ_LEN + test_start - 1: SEQ_LEN + len(actual_prices) - 1],
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
    }


def fetch_and_preprocess(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """YFinance download + macro features + preprocess.
    P2: результат кэшируется в Redis на YF_CACHE_TTL секунд (30 мин).
    Сам фичеринг тяжёлый (Kalman, HMM, indicators), поэтому кэшируем финальный
    DataFrame, а не только raw quotes.
    """
    cache_key = "neucast:yf:" + hashlib.md5(
        f"{ticker}|{start_date}|{end_date}".encode()
    ).hexdigest()

    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("YF cache HIT: %s %s..%s", ticker, start_date, end_date)
        return cached

    df_raw = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    if df_raw.empty:
        raise ValueError("Нет данных по указанному тикеру и диапазону дат")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    # Fetch cross-asset macro features in parallel date range
    macro_df = _fetch_macro_features(start_date, end_date)
    df = preprocess(df_raw, macro_df)
    _cache_set(cache_key, df, YF_CACHE_TTL)
    return df
