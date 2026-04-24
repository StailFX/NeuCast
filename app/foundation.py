"""
Foundation model wrappers для time-series прогнозирования.

Мы ансамблируем несколько pretrained foundation-моделей для снижения variance:
  • Chronos-Bolt (Amazon) — encoder-decoder, обучен на ~10B точках.
  • TimesFM (Google) — decoder-only, обучен на ~100B точках.
  • Moirai (Salesforce) — masked encoder, обучен на ~27B точках, мультивариантный.

Все доступны через opt-in чекбокс. Если несколько установлены — blend median
и bands поэлементно (простое среднее quantiles). Если только одна — fallback
на неё. Если ни одной — graceful fallback к conformal (как без Foundation).

API высокого уровня:
  foundation_forecast(close_prices, days_ahead) →
      dict с {median, p5, p25, p75, p95, models_used: [str]} или None.

Lazy singleton'ы с threading.Lock, graceful fallback при любой ошибке
(ImportError / checkpoint fail / inference error).

Tier 4 (Chronos) + Tier 5.2 (TimesFM) + Tier 5.3 (Moirai) — один чекбокс,
ансамбль управляется env var FOUNDATION_MODELS.

ВНИМАНИЕ: веса Moirai под лицензией CC-BY-NC-4.0 (non-commercial). Поэтому
Moirai по дефолту ВЫКЛЮЧЕН — нужно явно добавить "moirai" в FOUNDATION_MODELS.
"""

import os
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

# ── Chronos-Bolt config ─────────────────────────────────────────────
# Small (48M params, ~250MB) — баланс speed/quality для CPU.
# Для слабых VPS можно "tiny" (9M, 150MB), для GPU — "base" (200M, 1GB).
CHRONOS_MODEL_NAME = os.getenv("CHRONOS_MODEL_NAME", "amazon/chronos-bolt-small")
# Длина истории, которую видит модель. 256 — sweet spot: достаточно контекста
# для тренда + cycle, не слишком долго для inference.
CHRONOS_CONTEXT_LEN = int(os.getenv("CHRONOS_CONTEXT_LEN", "256"))
# Количество sample-trajectories для probabilistic forecast.
# Chronos-Bolt детерминистичен (генерирует все quantiles за один forward),
# но для оригинального Chronos этот параметр критичен. Оставляем для
# совместимости с обоими family.
CHRONOS_NUM_SAMPLES = int(os.getenv("CHRONOS_NUM_SAMPLES", "20"))

# ── TimesFM config ──────────────────────────────────────────────────
# 1.0-200m-pytorch — 200M params, pytorch checkpoint (~800MB на диске).
# Есть ещё 2.0-500m (лучше accuracy, но медленнее на CPU: 5-10s inference).
TIMESFM_MODEL_NAME = os.getenv("TIMESFM_MODEL_NAME", "google/timesfm-1.0-200m-pytorch")
TIMESFM_CONTEXT_LEN = int(os.getenv("TIMESFM_CONTEXT_LEN", "512"))
# horizon_len в модели фиксирован; выставляем >= любого разумного days_ahead.
TIMESFM_HORIZON_LEN = int(os.getenv("TIMESFM_HORIZON_LEN", "128"))

# ── Moirai config (Tier 5.3) ────────────────────────────────────────
# Salesforce Moirai-1.1-R — masked encoder foundation, мультивариантный.
# small (14M params, ~30MB), base (91M, ~180MB), large (311M, ~620MB).
# small — самый быстрый и достаточный для CPU-inference за 1-3s.
# Лицензия чекпоинтов: CC-BY-NC-4.0 (non-commercial) — opt-in по дефолту.
MOIRAI_MODEL_NAME = os.getenv("MOIRAI_MODEL_NAME", "Salesforce/moirai-1.1-R-small")
MOIRAI_CONTEXT_LEN = int(os.getenv("MOIRAI_CONTEXT_LEN", "200"))
# Количество Monte-Carlo sample-путей; Moirai стохастичен, 100 — хороший баланс.
MOIRAI_NUM_SAMPLES = int(os.getenv("MOIRAI_NUM_SAMPLES", "100"))
# patch_size="auto" выбирается моделью; для коротких историй лучше зафиксировать
# (32 даёт стабильные результаты, если len(history) < 300).
MOIRAI_PATCH_SIZE = os.getenv("MOIRAI_PATCH_SIZE", "auto")

# ── Ensembling config ───────────────────────────────────────────────
# Список foundation-моделей через запятую. Можно отключить отдельную:
# FOUNDATION_MODELS="chronos" или "timesfm" или "chronos,timesfm,moirai".
# Moirai по дефолту ВЫКЛЮЧЕН — лицензия CC-BY-NC-4.0. Чтобы включить для
# research/personal use: FOUNDATION_MODELS="chronos,timesfm,moirai".
FOUNDATION_MODELS = [
    m.strip().lower()
    for m in os.getenv("FOUNDATION_MODELS", "chronos,timesfm").split(",")
    if m.strip()
]

# Lazy singletons + locks от race condition при первом запросе.
_chronos_pipeline = None
_chronos_load_failed = False
_chronos_lock = threading.Lock()

_timesfm_pipeline = None
_timesfm_load_failed = False
_timesfm_lock = threading.Lock()

_moirai_module = None
_moirai_load_failed = False
_moirai_lock = threading.Lock()


def _get_chronos():
    """Получаем синглтон ChronosPipeline. Возвращает None при любых ошибках."""
    global _chronos_pipeline, _chronos_load_failed
    if _chronos_load_failed:
        return None
    if _chronos_pipeline is not None:
        return _chronos_pipeline
    with _chronos_lock:
        # Двойная проверка после захвата lock
        if _chronos_pipeline is not None:
            return _chronos_pipeline
        if _chronos_load_failed:
            return None
        try:
            from chronos import BaseChronosPipeline
            import torch
            t0 = __import__("time").time()
            _chronos_pipeline = BaseChronosPipeline.from_pretrained(
                CHRONOS_MODEL_NAME,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info(
                f"Chronos loaded: {CHRONOS_MODEL_NAME} "
                f"({__import__('time').time() - t0:.1f}s)"
            )
        except ImportError as e:
            logger.warning(
                f"Chronos library not installed ({e}). "
                f"Foundation model unavailable. Install: pip install chronos-forecasting"
            )
            _chronos_load_failed = True
            return None
        except Exception as e:
            logger.warning(f"Chronos load failed: {e} — Foundation model unavailable")
            _chronos_load_failed = True
            return None
    return _chronos_pipeline


def chronos_forecast(close_prices: np.ndarray, days_ahead: int) -> dict | None:
    """
    Probabilistic forecast next `days_ahead` price points.

    Args:
        close_prices: 1D array of historical Close prices (most recent last).
        days_ahead: how many future steps to predict.

    Returns:
        dict с ключами {median, p5, p25, p75, p95}, каждый — list длины days_ahead.
        None если Chronos недоступен или прогноз упал.
    """
    if days_ahead <= 0:
        return None
    pipe = _get_chronos()
    if pipe is None:
        return None
    try:
        import torch
        ctx = np.asarray(close_prices, dtype=np.float32)
        # Берём последние CONTEXT_LEN точек (Chronos не нуждается в большем)
        if len(ctx) > CHRONOS_CONTEXT_LEN:
            ctx = ctx[-CHRONOS_CONTEXT_LEN:]
        if len(ctx) < 30:
            logger.warning(f"Chronos: context too short ({len(ctx)} < 30)")
            return None
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32)

        # API chronos-forecasting >= 2.x: inputs= (вместо устаревшего context=).
        # predict_quantiles возвращает (quantiles, mean) tensor пары.
        # Сначала пробуем новый API; если падает (старая версия 1.x с context=) —
        # пробуем legacy.
        # Quantile levels: Chronos-Bolt обучен на [0.1, 0.2, ..., 0.9].
        # Запрашиваем [0.1, 0.25, 0.5, 0.75, 0.9] — крайние совпадают с тренировкой
        # (избегаем warning'а о clipping), средние интерполируются. Возвращаем
        # под ключами p5/p95 для совместимости с downstream — фактически это p10/p90,
        # но даже при запросе [0.05, 0.95] Chronos clamp'ил их к [0.1, 0.9].
        quantiles_tensor = None
        try:
            quantiles_tensor, _mean_tensor = pipe.predict_quantiles(
                inputs=ctx_tensor,
                prediction_length=days_ahead,
                quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            )
        except TypeError:
            # Legacy API (chronos-forecasting <2.0): context= вместо inputs=
            try:
                quantiles_tensor, _mean_tensor = pipe.predict_quantiles(
                    context=ctx_tensor,
                    prediction_length=days_ahead,
                    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
                )
            except Exception as e:
                logger.warning(f"Chronos predict_quantiles failed (both APIs): {e}")
                # Последний fallback: сэмплирующий predict (старый Chronos-T5)
                try:
                    samples_tensor = pipe.predict(
                        ctx_tensor,
                        prediction_length=days_ahead,
                        num_samples=CHRONOS_NUM_SAMPLES,
                    )
                    samples = samples_tensor[0].cpu().numpy()
                    return {
                        "median": np.median(samples, axis=0).tolist(),
                        "p5": np.percentile(samples, 5, axis=0).tolist(),
                        "p25": np.percentile(samples, 25, axis=0).tolist(),
                        "p75": np.percentile(samples, 75, axis=0).tolist(),
                        "p95": np.percentile(samples, 95, axis=0).tolist(),
                    }
                except Exception as e2:
                    logger.warning(f"Chronos predict (sampling) also failed: {e2}")
                    return None

        # quantiles_tensor: [batch=1, prediction_length, num_quantiles]
        q = quantiles_tensor[0].cpu().numpy()  # [prediction_length, 5]
        return {
            "p5": q[:, 0].tolist(),
            "p25": q[:, 1].tolist(),
            "median": q[:, 2].tolist(),
            "p75": q[:, 3].tolist(),
            "p95": q[:, 4].tolist(),
        }
    except Exception as e:
        logger.warning(f"Chronos forecast failed: {e}")
        return None


def _get_timesfm():
    """Синглтон TimesFM. Возвращает None при любых ошибках."""
    global _timesfm_pipeline, _timesfm_load_failed
    if _timesfm_load_failed:
        return None
    if _timesfm_pipeline is not None:
        return _timesfm_pipeline
    with _timesfm_lock:
        if _timesfm_pipeline is not None:
            return _timesfm_pipeline
        if _timesfm_load_failed:
            return None
        try:
            import timesfm  # type: ignore
            t0 = __import__("time").time()
            # 1.0-200m pytorch config: num_layers=20, model_dims=1280, positional_embedding=True.
            # 2.0-500m pytorch config: num_layers=50, use_positional_embedding=False.
            # Определяем по названию чекпоинта (стандартная схема названий от Google).
            is_v2 = "2.0" in TIMESFM_MODEL_NAME
            hparams_kwargs = dict(
                context_len=TIMESFM_CONTEXT_LEN,
                horizon_len=TIMESFM_HORIZON_LEN,
                input_patch_len=32,
                output_patch_len=128,
                per_core_batch_size=32,
                backend="cpu",
                quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                point_forecast_mode="median",
            )
            if is_v2:
                hparams_kwargs.update(num_layers=50, use_positional_embedding=False)
            else:
                hparams_kwargs.update(num_layers=20, use_positional_embedding=True)
            hparams = timesfm.TimesFmHparams(**hparams_kwargs)
            checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=TIMESFM_MODEL_NAME)
            _timesfm_pipeline = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
            logger.info(
                f"TimesFM loaded: {TIMESFM_MODEL_NAME} "
                f"({__import__('time').time() - t0:.1f}s)"
            )
        except ImportError as e:
            logger.warning(
                f"timesfm library not installed ({e}). "
                f"Install: pip install timesfm"
            )
            _timesfm_load_failed = True
            return None
        except Exception as e:
            logger.warning(f"TimesFM load failed: {e} — TimesFM unavailable")
            _timesfm_load_failed = True
            return None
    return _timesfm_pipeline


def timesfm_forecast(close_prices: np.ndarray, days_ahead: int) -> dict | None:
    """
    Probabilistic forecast через Google TimesFM.

    Returns dict {median, p5, p25, p75, p95} — те же ключи что у chronos_forecast
    для совместимости downstream. Фактические quantile levels: [p10, p30, p50,
    p70, p90] (TimesFM обучен на [0.1..0.9] с шагом 0.1). p5/p95 = ближайшие
    доступные p10/p90.

    None если TimesFM недоступен или inference упал.
    """
    if days_ahead <= 0:
        return None
    pipe = _get_timesfm()
    if pipe is None:
        return None
    try:
        ctx = np.asarray(close_prices, dtype=np.float32)
        if len(ctx) > TIMESFM_CONTEXT_LEN:
            ctx = ctx[-TIMESFM_CONTEXT_LEN:]
        if len(ctx) < 30:
            logger.warning(f"TimesFM: context too short ({len(ctx)} < 30)")
            return None
        # freq=0 → high-frequency (daily/intraday). Для weekly=1, monthly=2.
        _point_forecast, quantile_forecast = pipe.forecast(
            [ctx],
            freq=[0],
        )
        # quantile_forecast: [batch=1, horizon, num_quantiles=9+1]
        # У TimesFM 10 выходов: [mean, p10, p20, ..., p90].
        q = quantile_forecast[0]  # [horizon, 10]
        # Берём только days_ahead первых шагов
        q = q[:days_ahead]
        # Индексы в quantile_forecast: [0]=mean, [1]=p10, [2]=p20, ..., [9]=p90
        # Берём [p10, p30, p50, p70, p90] = индексы 1, 3, 5, 7, 9
        # (используем как наши p5/p25/p50/p75/p95 — misnomer, но симметрично
        # тому что делает Chronos на своих trained quantiles).
        p10 = q[:, 1].tolist()
        p30 = q[:, 3].tolist()
        p50 = q[:, 5].tolist()
        p70 = q[:, 7].tolist()
        p90 = q[:, 9].tolist()
        return {
            "p5": [float(x) for x in p10],
            "p25": [float(x) for x in p30],
            "median": [float(x) for x in p50],
            "p75": [float(x) for x in p70],
            "p95": [float(x) for x in p90],
        }
    except Exception as e:
        logger.warning(f"TimesFM forecast failed: {e}")
        return None


def _get_moirai():
    """Синглтон Moirai MoiraiModule (веса). Возвращает None при любых ошибках."""
    global _moirai_module, _moirai_load_failed
    if _moirai_load_failed:
        return None
    if _moirai_module is not None:
        return _moirai_module
    with _moirai_lock:
        if _moirai_module is not None:
            return _moirai_module
        if _moirai_load_failed:
            return None
        try:
            from uni2ts.model.moirai import MoiraiModule  # type: ignore
            t0 = __import__("time").time()
            _moirai_module = MoiraiModule.from_pretrained(MOIRAI_MODEL_NAME)
            logger.info(
                f"Moirai loaded: {MOIRAI_MODEL_NAME} "
                f"({__import__('time').time() - t0:.1f}s)"
            )
        except ImportError as e:
            logger.warning(
                f"uni2ts library not installed ({e}). "
                f"Install: pip install uni2ts"
            )
            _moirai_load_failed = True
            return None
        except Exception as e:
            logger.warning(f"Moirai load failed: {e} — Moirai unavailable")
            _moirai_load_failed = True
            return None
    return _moirai_module


def moirai_forecast(close_prices: np.ndarray, days_ahead: int) -> dict | None:
    """
    Probabilistic forecast через Salesforce Moirai-1.1-R.

    Moirai — masked-encoder foundation model, обучена на 27B точках из
    разных доменов (energy, transport, nature, sales, econ/fin, healthcare,
    CloudOps, web). Ключевое отличие от Chronos/TimesFM: multivariate-ready
    архитектура (хотя в этом wrapper'е пока используем univariate input).

    Returns dict {median, p5, p25, p75, p95}. Квантили честные (не clamp'д),
    т.к. Moirai семплирует num_samples путей и мы перцентилим empirically.

    None если Moirai недоступен или inference упал.
    """
    if days_ahead <= 0:
        return None
    module = _get_moirai()
    if module is None:
        return None
    try:
        from uni2ts.model.moirai import MoiraiForecast  # type: ignore
        import pandas as pd
        from gluonts.dataset.pandas import PandasDataset  # type: ignore

        ctx = np.asarray(close_prices, dtype=np.float32)
        if len(ctx) > MOIRAI_CONTEXT_LEN:
            ctx = ctx[-MOIRAI_CONTEXT_LEN:]
        if len(ctx) < 30:
            logger.warning(f"Moirai: context too short ({len(ctx)} < 30)")
            return None

        # Moirai обёртка принимает gluonts-dataset. Создаём PandasDataset
        # с dummy business-day index (Moirai не использует даты сами по себе,
        # frequency информирует positional encoding).
        df = pd.DataFrame(
            {"target": ctx},
            index=pd.date_range("2000-01-01", periods=len(ctx), freq="B"),
        )
        ds = PandasDataset(df, target="target")

        # patch_size="auto" требует context_len + prediction_length точек
        # в истории. Если история короткая — форсим patch_size=32.
        patch = MOIRAI_PATCH_SIZE
        if patch == "auto" and len(ctx) < MOIRAI_CONTEXT_LEN + days_ahead:
            patch = 32

        context_len_effective = min(MOIRAI_CONTEXT_LEN, len(ctx))

        forecaster = MoiraiForecast(
            module=module,
            prediction_length=days_ahead,
            context_length=context_len_effective,
            patch_size=patch,
            num_samples=MOIRAI_NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = forecaster.create_predictor(batch_size=1, device="cpu")

        # predict возвращает generator; берём первый forecast (batch=1).
        forecast = next(iter(predictor.predict(ds)))
        samples = forecast.samples  # [num_samples, prediction_length]
        return {
            "median": np.median(samples, axis=0).astype(float).tolist(),
            "p5": np.percentile(samples, 5, axis=0).astype(float).tolist(),
            "p25": np.percentile(samples, 25, axis=0).astype(float).tolist(),
            "p75": np.percentile(samples, 75, axis=0).astype(float).tolist(),
            "p95": np.percentile(samples, 95, axis=0).astype(float).tolist(),
        }
    except Exception as e:
        logger.warning(f"Moirai forecast failed: {e}")
        return None


def foundation_forecast(close_prices: np.ndarray, days_ahead: int) -> dict | None:
    """
    Ensemble foundation forecast — усредняет все доступные модели из
    FOUNDATION_MODELS. Возвращает dict с теми же ключами что chronos/timesfm,
    плюс "models_used": [list of str].

    Blending:
      • median / p5 / p25 / p75 / p95 = простое среднее по доступным моделям
      • Если модели разногласят — среднее сглаживает extreme predictions
      • Если только одна модель ответила — используем её напрямую

    Returns None если все модели недоступны или упали.
    """
    if days_ahead <= 0:
        return None
    results: list[tuple[str, dict]] = []
    if "chronos" in FOUNDATION_MODELS:
        r = chronos_forecast(close_prices, days_ahead)
        if r is not None:
            results.append(("chronos", r))
    if "timesfm" in FOUNDATION_MODELS:
        r = timesfm_forecast(close_prices, days_ahead)
        if r is not None:
            results.append(("timesfm", r))
    if "moirai" in FOUNDATION_MODELS:
        r = moirai_forecast(close_prices, days_ahead)
        if r is not None:
            results.append(("moirai", r))

    if not results:
        return None

    if len(results) == 1:
        blended = dict(results[0][1])
        blended["models_used"] = [results[0][0]]
        return blended

    # Среднее по всем моделям покомпонентно
    keys = ("median", "p5", "p25", "p75", "p95")
    blended = {"models_used": [name for name, _ in results]}
    for k in keys:
        vals = np.array([r[k] for _, r in results], dtype=np.float32)  # [N, days]
        blended[k] = vals.mean(axis=0).tolist()
    logger.info(
        f"Foundation ensemble: {'+'.join(blended['models_used'])}, "
        f"horizon={days_ahead}"
    )
    return blended


def is_available() -> bool:
    """Проверка доступности без полной загрузки. Полезно для UI-флага."""
    # Хотя бы одна из моделей должна быть доступна
    if _chronos_load_failed and _timesfm_load_failed and _moirai_load_failed:
        return False
    if (
        _chronos_pipeline is not None
        or _timesfm_pipeline is not None
        or _moirai_module is not None
    ):
        return True
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import timesfm  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import uni2ts  # noqa: F401
        return True
    except ImportError:
        pass
    return False
