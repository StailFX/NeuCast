"""
Celery worker for NeuCast — runs heavy prediction tasks in background.
"""
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0")

celery_app = Celery("neucast", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle", "json"],
    task_track_started=True,
    result_expires=3600,
    worker_max_tasks_per_child=50,   # restart worker after 50 tasks to free memory
    worker_concurrency=2,            # max 2 parallel predictions
    task_soft_time_limit=300,        # 5 min soft limit
    task_time_limit=360,             # 6 min hard limit
)

# ── Load model once per worker process ──
_model = None
_model_config = None


def _get_model():
    global _model, _model_config
    if _model is None:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.layers import (
            Layer, Conv1D, Dense, Dropout, GlobalAveragePooling1D,
            LayerNormalization,
        )

        class AttentionLayer(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            def build(self, input_shape):
                self.W = self.add_weight(name='att_W', shape=(input_shape[-1], input_shape[-1]),
                                          initializer='glorot_uniform', trainable=True)
                self.b = self.add_weight(name='att_b', shape=(input_shape[-1],),
                                          initializer='zeros', trainable=True)
                self.u = self.add_weight(name='att_u', shape=(input_shape[-1],),
                                          initializer='glorot_uniform', trainable=True)
            def call(self, x):
                score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
                weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
                return tf.reduce_sum(x * tf.expand_dims(weights, -1), axis=1)
            def get_config(self):
                return super().get_config()

        class SEBlock(Layer):
            def __init__(self, ratio=4, **kwargs):
                super().__init__(**kwargs)
                self.ratio = ratio
            def build(self, input_shape):
                ch = input_shape[-1]
                self.squeeze = GlobalAveragePooling1D()
                self.fc1 = Dense(ch // self.ratio, activation='relu')
                self.fc2 = Dense(ch, activation='sigmoid')
            def call(self, x):
                se = self.squeeze(x)
                se = self.fc1(se)
                se = self.fc2(se)
                se = tf.expand_dims(se, 1)
                return x * se
            def get_config(self):
                config = super().get_config()
                config["ratio"] = self.ratio
                return config

        class TCNBlock(Layer):
            def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
                super().__init__(**kwargs)
                self.filters = filters
                self.kernel_size = kernel_size
                self.dilation_rate = dilation_rate
                self.dropout_rate = dropout_rate
            def build(self, input_shape):
                self.conv1 = Conv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate,
                                    padding='causal', activation=None)
                self.bn1 = LayerNormalization()
                self.conv2 = Conv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate,
                                    padding='causal', activation=None)
                self.bn2 = LayerNormalization()
                self.dropout = Dropout(self.dropout_rate)
                self.se = SEBlock(ratio=4)
                if input_shape[-1] != self.filters:
                    self.residual_conv = Conv1D(self.filters, 1, padding='same')
                else:
                    self.residual_conv = None
            def call(self, x, training=None):
                res = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = tf.nn.gelu(out)
                out = self.dropout(out, training=training)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.se(out)
                if self.residual_conv is not None:
                    res = self.residual_conv(res)
                out = tf.nn.gelu(out + res)
                return out
            def get_config(self):
                config = super().get_config()
                config.update({
                    "filters": self.filters, "kernel_size": self.kernel_size,
                    "dilation_rate": self.dilation_rate, "dropout_rate": self.dropout_rate,
                })
                return config

        custom_objects = {
            "AttentionLayer": AttentionLayer,
            "SEBlock": SEBlock,
            "TCNBlock": TCNBlock,
        }
        _model = load_model("best_model.h5", compile=False, custom_objects=custom_objects)

        if os.path.exists("model_config.json"):
            with open("model_config.json") as f:
                _model_config = json.load(f)
        else:
            _model_config = {}

    return _model, _model_config


@celery_app.task(bind=True, name="neucast.predict")
def run_prediction_task(self, ticker: str, start_date: str, end_date: str, days_ahead: int):
    """
    Heavy prediction task — runs in Celery worker process.
    Returns the full result dict.
    """
    # Import here to avoid circular imports
    import app as app_module

    self.update_state(state="FETCHING", meta={"status": "Загрузка данных с Yahoo Finance..."})

    try:
        df = app_module.fetch_and_preprocess(ticker, start_date, end_date)
    except ValueError as e:
        return {"error": str(e)}

    self.update_state(state="PREDICTING", meta={"status": "Расчёт прогноза (TCN + Ensemble)..."})

    result = app_module.run_prediction(df, days_ahead)

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
