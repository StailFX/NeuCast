"""
Locally-adaptive conformal calibration (Tier 5.4).

Standard conformal (Tier 2) использует ГЛОБАЛЬНЫЕ квантили residuals на
calibration-сете — bands одинаковой ширины во всех режимах рынка. Это
простой и статистически корректный подход, но теряет sharpness: если
текущее состояние рынка спокойное, bands избыточно широки; если burst
волатильности — недостаточно широки.

NGBoost моделирует σ(x) условно на features → даёт ЛОКАЛЬНЫЙ inflation-
коэффициент. Узкие bands когда регрессор уверен (низкая предсказуемая
volatility), широкие когда не уверен.

Подход:
  1. Fit NGBRegressor(Normal) на (X_cal, val_log_resid) с calibration fold.
     Модель учит: "при каких features residuals имеют высокую σ?"
  2. At inference — predict σ(X_last) и σ(X_cal). Ratio = локальная
     коррекция ширины bands.
  3. Умножаем quantiles q05/q25/q75/q95 на ratio (clamp [0.5, 2.0]).

Env vars:
  LOCAL_CALIBRATION=1  включить (default: 0).
  NGB_N_ESTIMATORS=100
  NGB_LEARNING_RATE=0.05
  NGB_MINIBATCH_FRAC=0.7
  NGB_ROLLING_WINDOW=100  взять last-N строк (X_cal, residuals) из val-фолда.
    Это ключевой фикс "σ-ratio упирается в clamp": X_future — это edge-данных
    ("сейчас"), а наивный X_cal берёт весь val-фолд (хронологически в середине).
    Feature-drift между серединой и сейчас приводит к экстраполяции σ(x).
    Rolling window N=100 держит X_cal близко к X_future по распределению,
    сохраняя exchangeability (val-фолд монотонно во времени, tail ближайший
    к inference). 0 — отключить (использовать весь val-фолд).
  SIGMA_RATIO_MIN=0.7, SIGMA_RATIO_MAX=1.5

Graceful fallback при любой ошибке (import, fit, predict) → возвращаем
ratio=1.0 (= no-op, bands как были).
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

LOCAL_CALIBRATION = os.getenv("LOCAL_CALIBRATION", "0") == "1"
# 100 trees — sweet spot для n_cal~100-200. Больше → overfitting на высокоdim
# calibration set (66 features × 100 samples). Меньше → undertuned σ.
NGB_N_ESTIMATORS = int(os.getenv("NGB_N_ESTIMATORS", "100"))
NGB_LEARNING_RATE = float(os.getenv("NGB_LEARNING_RATE", "0.05"))
# Minibatch subsampling per boost iter — ещё один regularizer (0.5-1.0).
NGB_MINIBATCH_FRAC = float(os.getenv("NGB_MINIBATCH_FRAC", "0.7"))
# Seed для воспроизводимости fit'а. Без него NGBRegressor стохастичен из-за
# minibatch sampling → один и тот же ticker даёт разный σ-ratio (1.33 / 1.50)
# между прогонами, что путает пользователя ("вчера было 1.33×, сегодня 1.50×").
# Фиксируем — пусть выглядит детерминированно. Для дебага можно перезагрузить
# через env (RANDOM_STATE=43, etc).
NGB_RANDOM_STATE = int(os.getenv("NGB_RANDOM_STATE", "42"))
# Clamp ratio. [0.7, 1.5] по дефолту — разумный диапазон: на typical shift
# bands могут ±50% шире/уже. При сильном overfitting clamp спасает UX.
# Можно ослабить до [0.5, 2.0] через env если нужна агрессивность.
SIGMA_RATIO_MIN = float(os.getenv("SIGMA_RATIO_MIN", "0.7"))
SIGMA_RATIO_MAX = float(os.getenv("SIGMA_RATIO_MAX", "1.5"))
# Rolling calibration window — см. module docstring. 0 = весь val-фолд.
NGB_ROLLING_WINDOW = int(os.getenv("NGB_ROLLING_WINDOW", "100"))

# Минимальный размер calibration-фолда для имеет смысл обучать NGBoost.
# Меньше — noise dominated, переобучаемся и ratio становится случайным.
_MIN_CAL_SIZE = 30


def is_enabled() -> bool:
    """Включена ли локальная калибровка (по env var)."""
    return LOCAL_CALIBRATION


def fit_local_variance(X_cal: np.ndarray, residuals: np.ndarray):
    """
    Train NGBoost на calibration residuals.

    Args:
        X_cal: [n_cal, n_features] features в calibration fold.
        residuals: [n_cal,] signed log-residuals (log(actual/pred)) aligned
            с X_cal.

    Returns:
        Fitted NGBRegressor, либо None если: ngboost не установлен,
        недостаточно данных, или fit упал.
    """
    X = np.asarray(X_cal, dtype=np.float32)
    y = np.asarray(residuals, dtype=np.float32)
    if X.ndim != 2 or len(X) != len(y):
        logger.info(f"Local cal skipped: shape mismatch X={X.shape}, y={y.shape}")
        return None
    if len(X) < _MIN_CAL_SIZE:
        logger.info(f"Local cal skipped: too few samples ({len(X)} < {_MIN_CAL_SIZE})")
        return None
    if not np.isfinite(y).all():
        logger.info("Local cal skipped: non-finite residuals")
        return None
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        ngb = NGBRegressor(
            Dist=Normal,
            n_estimators=NGB_N_ESTIMATORS,
            learning_rate=NGB_LEARNING_RATE,
            minibatch_frac=NGB_MINIBATCH_FRAC,
            random_state=NGB_RANDOM_STATE,
            verbose=False,
        )
        ngb.fit(X, y)
        return ngb
    except ImportError as e:
        logger.warning(
            f"ngboost library not installed ({e}). "
            f"Local calibration disabled. Install: pip install ngboost"
        )
        return None
    except Exception as e:
        logger.warning(f"NGBoost fit failed: {e}; local calibration skipped")
        return None


def local_sigma_ratio(model, X_cal: np.ndarray, X_future: np.ndarray) -> float:
    """
    Compute σ(X_future) / median σ(X_cal) — inflation factor для bands.

    Args:
        model: Fitted NGBRegressor (output fit_local_variance).
        X_cal: [n_cal, n_features] — reference set для расчёта typical σ.
        X_future: [n_features] or [1, n_features] — current state features.

    Returns:
        float in [SIGMA_RATIO_MIN, SIGMA_RATIO_MAX]. 1.0 — no-op rescaling.
        Любая ошибка → 1.0 (graceful fallback).
    """
    if model is None:
        return 1.0
    try:
        X_fut = np.asarray(X_future, dtype=np.float32).reshape(1, -1)
        X_c = np.asarray(X_cal, dtype=np.float32)
        dist_future = model.pred_dist(X_fut)
        dist_cal = model.pred_dist(X_c)
        sigma_future = float(dist_future.scale[0])
        sigma_ref = float(np.median(dist_cal.scale))
        if sigma_ref <= 0 or not np.isfinite(sigma_future) or sigma_future <= 0:
            return 1.0
        ratio = sigma_future / sigma_ref
        if not np.isfinite(ratio):
            return 1.0
        return float(np.clip(ratio, SIGMA_RATIO_MIN, SIGMA_RATIO_MAX))
    except Exception as e:
        logger.debug(f"local_sigma_ratio failed: {e}; using 1.0")
        return 1.0


def compute_local_inflation(
    X_cal: np.ndarray,
    residuals: np.ndarray,
    X_future: np.ndarray,
    rolling_window: int | None = None,
) -> tuple[float, bool]:
    """
    Высокоуровневая обёртка: fit + predict за один вызов.

    Args:
        X_cal: [n_cal, n_features] calibration features (val-fold).
        residuals: [n_cal,] signed log-residuals aligned с X_cal.
        X_future: [n_features] or [1, n_features] — "сейчас" features.
        rolling_window: Сколько последних строк X_cal/residuals использовать.
            None → читаем из env NGB_ROLLING_WINDOW (default 100). 0 →
            отключить (весь val-фолд). Ключевой фикс "σ упирается в clamp":
            фичи в начале val-фолда могут быть далеко от X_future, NGBoost
            тогда экстраполирует. Берём хвост → X_cal близко к X_future.

    Returns:
        (ratio, applied) где ratio ∈ [SIGMA_RATIO_MIN, SIGMA_RATIO_MAX] и
        applied=True если NGBoost реально обучен и использован. applied=
        False при любом fallback (ratio будет 1.0).
    """
    if not LOCAL_CALIBRATION:
        logger.info(
            "Local cal skipped: LOCAL_CALIBRATION=0 env var (set to 1 to enable)"
        )
        return 1.0, False
    # Rolling window: оставляем только хвост calibration set. Exchangeability
    # не нарушается, т.к. val-фолд монотонен во времени — tail ближе всего к
    # X_future по distribution shift.
    try:
        X_arr = np.asarray(X_cal)
        r_arr = np.asarray(residuals)
        n_full = min(len(X_arr), len(r_arr))
        X_arr = X_arr[:n_full]
        r_arr = r_arr[:n_full]
        window = NGB_ROLLING_WINDOW if rolling_window is None else rolling_window
        if window and window > 0 and n_full > window:
            X_cal_use = X_arr[-window:]
            residuals_use = r_arr[-window:]
            logger.info(
                f"Local cal: rolling window applied ({n_full} → last {window})"
            )
        else:
            X_cal_use = X_arr
            residuals_use = r_arr
    except Exception as e:
        logger.warning(f"Rolling window slicing failed: {e}; using full set")
        X_cal_use, residuals_use = X_cal, residuals
    model = fit_local_variance(X_cal_use, residuals_use)
    if model is None:
        return 1.0, False
    ratio = local_sigma_ratio(model, X_cal_use, X_future)
    return ratio, True
