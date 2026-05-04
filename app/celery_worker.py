"""
Celery worker for NeuCast — runs heavy prediction tasks in background.
"""
import os
import json
import hashlib
import pickle
import logging
import numpy as np
import pandas as pd

from celery import Celery
from celery.signals import worker_ready

from app.prediction import run_prediction, fetch_and_preprocess
from app.user_errors import friendly_error

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0")
# Short-TTL cache для повторных запросов того же (ticker, start, end, days).
# Позволяет перезагружать страницу результатов или запускать тот же расчёт
# подряд без повторной тренировки моделей.
PRED_RESULT_CACHE_TTL = int(os.getenv("PRED_RESULT_CACHE_TTL", "300"))  # 5 мин

celery_app = Celery("neucast", broker=REDIS_URL, backend=REDIS_URL)
# Code-review C-1 (2026-05-04): JSON serializer (was pickle). Pickle in the
# broker is an RCE seed — any Redis compromise becomes Python execution on
# every worker. All task signatures use scalar args; result dicts already
# pre-serialise numpy arrays via .tolist() (line ~308-323 below).
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    worker_max_tasks_per_child=50,
    worker_concurrency=2,
    task_soft_time_limit=300,
    task_time_limit=360,
)


@worker_ready.connect
def _warmup_tcn_on_worker_ready(**_kwargs):
    """Preload TCN base model at worker startup so the first prediction
    task doesn't pay the ~3-4s model-load cost on a cold worker.

    The base model is process-global (cached via _get_model in prediction.py)
    so this runs once per forked child when max_tasks_per_child recycles.
    """
    try:
        from app.prediction import _get_model
        import time
        t0 = time.time()
        _get_model()
        logger.info(f"TCN base model warmup completed in {time.time() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"TCN warmup failed: {e}")


def _generate_mini_pdf(ticker: str, result: dict) -> bytes | None:
    """Generate a compact PDF summary for Telegram."""
    try:
        from fpdf import FPDF
        import io

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header
        pdf.set_fill_color(10, 14, 23)
        pdf.rect(0, 0, 210, 40, "F")
        pdf.set_fill_color(16, 185, 129)
        pdf.rect(0, 38, 210, 2, "F")
        pdf.set_xy(15, 8)
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, f"NeuCast — {ticker}")
        pdf.set_xy(15, 22)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 8, f"Model: {result.get('model_name', 'Ensemble')}")

        pdf.set_y(48)
        pdf.set_text_color(30, 41, 59)

        # Metrics
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Model Accuracy (Test Set)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        metrics = [
            ("MAPE", f"{result['mape']:.2f}%"),
            ("MAE", f"${result['mae']:.2f}"),
            ("RMSE", f"${result['rmse']:.2f}"),
            ("R²", f"{result['r2']:.4f}"),
            ("Direction Accuracy", f"{result['dir_acc']:.1f}%"),
        ]
        for name, val in metrics:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(80, 7, f"  {name}:")
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, val, ln=True)

        # Backtest
        bt = result.get("backtest")
        if bt:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Backtesting", ln=True)
            pdf.set_font("Helvetica", "", 10)
            bt_metrics = [
                ("Strategy Return", f"{bt['total_return']:+.2f}%"),
                ("Buy & Hold Return", f"{bt['bnh_return']:+.2f}%"),
                ("Alpha", f"{bt['total_return'] - bt['bnh_return']:+.2f}%"),
                ("Sharpe Ratio", f"{bt['sharpe']}"),
                ("Max Drawdown", f"{bt['max_drawdown']}%"),
                ("Win Rate", f"{bt['win_rate']}%"),
                ("Profit Factor", f"{bt['profit_factor']}"),
            ]
            for name, val in bt_metrics:
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(80, 7, f"  {name}:")
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, val, ln=True)

        # Per-model metrics
        mm = result.get("model_metrics")
        if mm:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Per-Model Metrics", ln=True)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(241, 245, 249)
            pdf.cell(45, 7, "Model", 1, 0, "C", True)
            pdf.cell(30, 7, "MAE ($)", 1, 0, "C", True)
            pdf.cell(30, 7, "RMSE ($)", 1, 0, "C", True)
            pdf.cell(30, 7, "MAPE (%)", 1, 0, "C", True)
            pdf.cell(30, 7, "R²", 1, 1, "C", True)
            pdf.set_font("Helvetica", "", 9)
            for name, m in mm.items():
                pdf.cell(45, 6, name, 1, 0)
                pdf.cell(30, 6, str(m["mae"]), 1, 0, "C")
                pdf.cell(30, 6, str(m["rmse"]), 1, 0, "C")
                pdf.cell(30, 6, f"{m['mape']}%", 1, 0, "C")
                pdf.cell(30, 6, str(m["r2"]), 1, 1, "C")

        # Footer
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 5, "Generated by NeuCast AI — neucast.ru", 0, 0, "C")

        buf = io.BytesIO()
        pdf.output(buf)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to generate PDF: {e}")
        return None


def _send_telegram_notification(user_id: int, ticker: str, result: dict, task_id: str = None):
    """Send Telegram notification with prediction summary + PDF."""
    try:
        from app.telegram_bot import is_configured, is_linked, send_prediction_result

        if not is_configured() or not is_linked(user_id):
            logger.info(f"Telegram skip: configured={is_configured()}, linked={is_linked(user_id)}")
            return

        pdf_bytes = _generate_mini_pdf(ticker, result)
        logger.info(f"PDF generated: {len(pdf_bytes) if pdf_bytes else 0} bytes")

        send_prediction_result(
            user_id=user_id,
            ticker=ticker,
            task_id=task_id,
            pdf_bytes=pdf_bytes,
            mae=f"{result['mae']:.2f}",
            mape=f"{result['mape']:.2f}",
            r2=f"{result['r2']:.4f}",
            dir_acc=f"{result['dir_acc']:.1f}",
            model_name=result.get("model_name", "Ensemble"),
        )
        logger.info(f"Telegram notification sent to user {user_id} for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to send Telegram notification: {e}", exc_info=True)


def _pred_cache_key(ticker: str, start_date: str, end_date: str, days_ahead: int) -> str:
    raw = f"{ticker}|{start_date}|{end_date}|{days_ahead}".encode()
    return "neucast:pred:" + hashlib.md5(raw).hexdigest()


def _pred_cache_get(key: str):
    """Read a cached prediction result.

    Code-review C-1 (2026-05-04): switched from pickle to JSON. Cached values
    are already plain dicts of str/int/float/list (see line ~308-323 where
    numpy arrays and DatetimeIndex are .tolist()'d before caching), so the
    JSON round-trip is lossless.

    Backward compatibility: a fresh deploy will see legacy pickle bytes
    on existing keys. We try JSON first; on failure we treat the entry as
    invalid (returning None forces a recompute). The TTL is short
    (PRED_RESULT_CACHE_TTL=5 min default), so legacy entries roll over
    naturally within minutes of deploy — no manual Redis flush needed.
    """
    try:
        import redis
        r = redis.from_url(REDIS_URL, socket_timeout=1, socket_connect_timeout=1)
        data = r.get(key)
        if not data:
            return None
        try:
            import json as _json
            return _json.loads(data)
        except Exception:
            # Legacy pickle entry from before the C-1 deploy. Drop it —
            # the caller will recompute. Pickle.loads here would re-open
            # the RCE vector we're closing.
            try:
                r.delete(key)
            except Exception:
                pass
            return None
    except Exception as e:
        logger.debug(f"pred cache_get fail: {e}")
        return None


def _pred_cache_set(key: str, value, ttl: int):
    try:
        import redis
        r = redis.from_url(REDIS_URL, socket_timeout=1, socket_connect_timeout=1)
        # Code-review C-1: JSON serialiser (was pickle.HIGHEST_PROTOCOL).
        # Caller is responsible for ensuring ``value`` is JSON-serialisable
        # — current call site at line ~325 passes ``serializable`` which has
        # numpy → list conversions baked in.
        import json as _json
        r.setex(key, ttl, _json.dumps(value, default=str))
    except Exception as e:
        logger.debug(f"pred cache_set fail: {e}")


@celery_app.task(bind=True, name="neucast.predict")
def run_prediction_task(self, ticker: str, start_date: str, end_date: str,
                        days_ahead: int, user_id: int = None,
                        use_foundation: bool = False):
    """Heavy prediction task — runs in Celery worker process.

    Args:
        use_foundation: opt-in флаг для Foundation model (Chronos). Влияет на
            cache_key, чтобы прогнозы с/без Foundation не смешивались.
    """

    # ── Полный result-cache (PRED_RESULT_CACHE_TTL=5 мин по умолчанию) ──
    # Ключ: (ticker, start, end, days, use_foundation). user_id НЕ в ключе.
    cache_key = _pred_cache_key(ticker, start_date, end_date, days_ahead) + f":fnd{int(bool(use_foundation))}"
    cached = _pred_cache_get(cache_key)
    if cached is not None:
        logger.info(f"Prediction CACHE HIT: {ticker} {start_date}..{end_date} +{days_ahead}d (fnd={int(bool(use_foundation))})")
        self.update_state(state="PREDICTING", meta={"status": "Результат из кэша..."})
        # Telegram нотификация всё равно уходит — чтобы каждый пользователь получил своё.
        if user_id:
            # Восстановим result-dict из serializable для notification
            try:
                result_for_tg = {
                    k: v for k, v in cached.items()
                    if k in ("mape", "mae", "rmse", "r2", "dir_acc", "model_name",
                             "backtest", "model_metrics", "foundation_used")
                }
                _send_telegram_notification(user_id, ticker, result_for_tg, task_id=self.request.id)
            except Exception as e:
                logger.warning(f"TG notify from cache failed: {e}")
        return cached

    self.update_state(state="FETCHING", meta={"status": "Загрузка данных с Yahoo Finance..."})

    try:
        df = fetch_and_preprocess(ticker, start_date, end_date)
    except ValueError as e:
        # Известный класс ошибок (плохой тикер, пустой период) — friendly текст
        # уже формирует prediction.fetch_and_preprocess. Прогоняем через
        # mapper на случай если message по-английски пришёл от yfinance.
        logger.warning(f"fetch_and_preprocess ValueError for {ticker}: {e}")
        return {"error": friendly_error(e)}
    except Exception as e:
        logger.exception(f"fetch_and_preprocess unexpected error for {ticker}")
        return {"error": friendly_error(e)}

    pred_status = "Расчёт прогноза (TCN + Ensemble"
    if use_foundation:
        pred_status += " + Foundation"
    pred_status += ")..."
    self.update_state(state="PREDICTING", meta={"status": pred_status})

    # Wrap прогноза в try/except чтобы любая внутренняя ошибка (например,
    # bug типа UnboundLocalError) НЕ улетала в celery FAILURE state с raw
    # tracebackом в UI. Возвращаем friendly dict, full traceback — в логи.
    try:
        result = run_prediction(df, days_ahead, use_foundation=use_foundation)
    except Exception as e:
        logger.exception(
            f"run_prediction failed for {ticker} (days_ahead={days_ahead}, "
            f"use_foundation={use_foundation})"
        )
        return {"error": friendly_error(e)}

    # ── A1: Hourly directional-skill probe (crypto-only diagnostic) ──
    # Honestly answers "is there ANY directional signal at all" via 17K
    # hourly bars (vs 200 daily test points → CI ±1.5% vs ±7%). Не трогает
    # основной прогноз: hourly-probe = отдельная LightGBM-classifier
    # на short-horizon, результат — в дополнительные поля результата.
    # Off by default (HOURLY_DIAGNOSTIC), потому что добавляет ~10-30s
    # на yfinance hourly fetch + LightGBM train. Crypto-only: для акций
    # hourly-данные имеют gaps (только торговые часы) → отдельная задача.
    if os.getenv("HOURLY_DIAGNOSTIC", "0") == "1":
        try:
            from app.hourly_skill import compute_hourly_skill, is_crypto_ticker
            if is_crypto_ticker(ticker):
                self.update_state(
                    state="PREDICTING",
                    meta={"status": "Hourly skill probe..."},
                )
                hourly = compute_hourly_skill(ticker)
                if hourly is not None:
                    result["hourly_skill"] = hourly
                    logger.info(
                        "hourly_skill attached to %s: %s%% [%s%%–%s%%]",
                        ticker, hourly.get("skill"),
                        hourly.get("ci_low"), hourly.get("ci_high"),
                    )
        except Exception:
            # Любая ошибка hourly-probe не должна валить основной прогноз.
            logger.exception("hourly_skill probe failed for %s — continuing", ticker)

    # Send Telegram notification
    if user_id:
        _send_telegram_notification(user_id, ticker, result, task_id=self.request.id)

    # Convert numpy arrays to lists for serialization
    serializable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, pd.DatetimeIndex):
            serializable[k] = v.strftime("%Y-%m-%d").tolist()
        else:
            serializable[k] = v

    serializable["_df_json"] = df.to_json(date_format="iso")
    serializable["ticker"] = ticker
    serializable["start_date"] = start_date
    serializable["end_date"] = end_date
    serializable["days_ahead"] = days_ahead
    serializable["use_foundation"] = bool(use_foundation)

    # Кэшируем финальный сериализуемый результат — повторные запросы = мгновенно.
    _pred_cache_set(cache_key, serializable, PRED_RESULT_CACHE_TTL)

    return serializable
