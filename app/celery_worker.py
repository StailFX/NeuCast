"""
Celery worker for NeuCast — runs heavy prediction tasks in background.
"""
import os
import json
import logging
import numpy as np
import pandas as pd

from celery import Celery

from app.prediction import run_prediction, fetch_and_preprocess

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0")

celery_app = Celery("neucast", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle", "json"],
    task_track_started=True,
    result_expires=3600,
    worker_max_tasks_per_child=50,
    worker_concurrency=2,
    task_soft_time_limit=300,
    task_time_limit=360,
)


def _send_telegram_notification(user_id: int, ticker: str, result: dict):
    """Send Telegram notification with prediction summary."""
    try:
        from app.telegram_bot import is_configured, is_linked, send_prediction_result

        if not is_configured() or not is_linked(user_id):
            return

        send_prediction_result(
            user_id=user_id,
            ticker=ticker,
            prediction_id=None,  # not saved to DB yet at this point
            mae=f"{result['mae']:.2f}",
            mape=f"{result['mape']:.2f}",
            r2=f"{result['r2']:.4f}",
            dir_acc=f"{result['dir_acc']:.1f}",
            model_name=result.get("model_name", "Ensemble"),
        )
        logger.info(f"Telegram notification sent to user {user_id} for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to send Telegram notification: {e}")


@celery_app.task(bind=True, name="neucast.predict")
def run_prediction_task(self, ticker: str, start_date: str, end_date: str,
                        days_ahead: int, user_id: int = None):
    """Heavy prediction task — runs in Celery worker process."""
    self.update_state(state="FETCHING", meta={"status": "Загрузка данных с Yahoo Finance..."})

    try:
        df = fetch_and_preprocess(ticker, start_date, end_date)
    except ValueError as e:
        return {"error": str(e)}

    self.update_state(state="PREDICTING", meta={"status": "Расчёт прогноза (TCN + Ensemble)..."})

    result = run_prediction(df, days_ahead)

    # Send Telegram notification
    if user_id:
        _send_telegram_notification(user_id, ticker, result)

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

    return serializable
