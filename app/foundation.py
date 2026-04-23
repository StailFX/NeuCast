"""
Foundation model wrapper для time-series прогнозирования.

Используем Chronos-Bolt от Amazon — pretrained encoder-decoder, обученный на
~10B time-series точках разной природы (equity, crypto, energy, weather, traffic).
В zero-shot выдаёт probabilistic forecast (sample-based) на N шагов вперёд.

API: chronos_forecast(close_prices, days_ahead) → dict с median/quantile bands.
Lazy-load: модель грузится только при первом обращении (~250-400MB).
Graceful fallback: если chronos не установлен или загрузка упала — функция
возвращает None, и run_prediction просто пропускает Foundation-аугментацию.

Tier 4 — opt-in через UI чекбокс "Foundation-модель (медленнее, +5-10% точности)".
По умолчанию выключено.
"""

import os
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
# Chronos-Bolt small (48M params, ~250MB) — баланс speed/quality для CPU.
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

# Lazy singleton + lock от race condition при первом запросе.
_chronos_pipeline = None
_chronos_load_failed = False
_chronos_lock = threading.Lock()


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


def is_available() -> bool:
    """Проверка доступности без полной загрузки. Полезно для UI-флага."""
    if _chronos_load_failed:
        return False
    if _chronos_pipeline is not None:
        return True
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False
