"""
NeuCast — Скрипт обучения ансамблевой модели.

Архитектура (Stacking Ensemble):
  Level-0 (базовые модели):
    1. LSTM с Attention-механизмом — временные зависимости
    2. CatBoost — градиентный бустинг, хорош на табличных данных
    3. XGBoost — ещё один бустинг для диверсификации

  Level-1 (мета-модель):
    Ridge Regression — объединяет предсказания L0

  Препроцессинг:
    - Wavelet Denoising (очистка сигнала от шума)
    - 12 фич: OHLCV + RSI + MACD + Signal + MA(5,10,20,50)
    - MinMaxScaler(0, 1)

Запуск:
  python train_model.py
  python train_model.py --ticker GC=F --start 2015-01-01 --end 2025-12-31 --epochs 100
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
    Input, Layer, Multiply, Permute, RepeatVector, Flatten, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

SEQ_LEN = 60

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI", "MACD", "Signal",
    "MA_5", "MA_10", "MA_20", "MA_50",
]


# ============================================================
# Wavelet Denoising
# ============================================================
def wavelet_denoise(signal: np.ndarray, wavelet='db4', level=3) -> np.ndarray:
    """Denoise a signal using Discrete Wavelet Transform."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Universal threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    # Soft thresholding on detail coefficients
    denoised_coeffs = [coeffs[0]]  # keep approximation
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
    return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]


def preprocess(df: pd.DataFrame, denoise: bool = True) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].interpolate(method="time")

    # Wavelet denoising on Close price
    if denoise and len(df) > 50:
        df["Close_raw"] = df["Close"].copy()
        df["Close"] = wavelet_denoise(df["Close"].values)

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    # MACD (12/26/9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Moving Averages
    for w in (5, 10, 20, 50):
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()

    df = df.dropna()
    return df


# ============================================================
# Sequences for LSTM
# ============================================================
def make_sequences(data: np.ndarray):
    X, y = [], []
    close_idx = FEATURES.index("Close")
    for i in range(SEQ_LEN, len(data)):
        X.append(data[i - SEQ_LEN: i])
        y.append(data[i, close_idx])
    return np.array(X), np.array(y)


# Flat features for boosting: last SEQ_LEN rows flattened + rolling stats
def make_boosting_features(data: np.ndarray):
    X, y = [], []
    close_idx = FEATURES.index("Close")
    for i in range(SEQ_LEN, len(data)):
        window = data[i - SEQ_LEN: i]
        # Flattened window would be too big, use summary statistics
        feats = []
        for col in range(window.shape[1]):
            col_data = window[:, col]
            feats.extend([
                col_data[-1],           # last value
                col_data.mean(),        # mean
                col_data.std(),         # std
                col_data[-1] - col_data[0],  # trend (last - first)
                col_data[-5:].mean(),   # short-term mean (5 days)
                np.min(col_data),       # min
                np.max(col_data),       # max
            ])
        X.append(feats)
        y.append(data[i, close_idx])
    return np.array(X), np.array(y)


# ============================================================
# Attention Layer
# ============================================================
class AttentionLayer(Layer):
    """Bahdanau-style attention for time series."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform', trainable=True,
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros', trainable=True,
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(input_shape[-1],),
            initializer='glorot_uniform', trainable=True,
        )

    def call(self, x):
        # x shape: (batch, timesteps, features)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context = tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
        return context

    def get_config(self):
        return super().get_config()


# ============================================================
# Build LSTM + Attention model
# ============================================================
def build_lstm_attention(n_features: int) -> Model:
    inputs = Input(shape=(SEQ_LEN, n_features))

    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Attention
    x = AttentionLayer()(x)
    x = Dropout(0.2)(x)

    # Dense head
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse', metrics=['mae'],
    )
    return model


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Ensemble model for NeuCast")
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="best_model.h5")
    args = parser.parse_args()

    # ---- Load data ----
    print(f"\n{'='*60}")
    print(f"  NeuCast — Ensemble Training Pipeline")
    print(f"{'='*60}")
    print(f"\nТикер: {args.ticker}")
    print(f"Период: {args.start} — {args.end}")

    df_raw = yf.download(args.ticker, start=args.start, end=args.end, interval="1d", progress=True)
    if df_raw.empty:
        print("Ошибка: нет данных!")
        return
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df = preprocess(df_raw, denoise=True)
    print(f"Данных: {len(df)} записей")
    print(f"Фичи ({len(FEATURES)}): {FEATURES}")

    # ---- Scale ----
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(df[FEATURES])

    # ---- Split ----
    n = len(scaled)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # LSTM sequences
    X_seq, y_seq = make_sequences(scaled)
    X_tr_s, y_tr_s = X_seq[:train_end - SEQ_LEN], y_seq[:train_end - SEQ_LEN]
    X_va_s, y_va_s = X_seq[train_end - SEQ_LEN:val_end - SEQ_LEN], y_seq[train_end - SEQ_LEN:val_end - SEQ_LEN]
    X_te_s, y_te_s = X_seq[val_end - SEQ_LEN:], y_seq[val_end - SEQ_LEN:]

    # Boosting features
    X_bst, y_bst = make_boosting_features(scaled)
    X_tr_b, y_tr_b = X_bst[:train_end - SEQ_LEN], y_bst[:train_end - SEQ_LEN]
    X_va_b, y_va_b = X_bst[train_end - SEQ_LEN:val_end - SEQ_LEN], y_bst[train_end - SEQ_LEN:val_end - SEQ_LEN]
    X_te_b, y_te_b = X_bst[val_end - SEQ_LEN:], y_bst[val_end - SEQ_LEN:]

    print(f"\nSplit: Train={len(X_tr_s)}, Val={len(X_va_s)}, Test={len(X_te_s)}")

    # ---- Inverse scaling helper ----
    n_feat = len(FEATURES)
    def invert(y_scaled):
        filler = np.zeros((len(y_scaled), n_feat - 1))
        arr = np.hstack([filler, y_scaled.reshape(-1, 1)])
        return scaler.inverse_transform(arr)[:, -1]

    # ============================================================
    # Model 1: LSTM + Attention
    # ============================================================
    print(f"\n{'─'*60}")
    print("  [1/3] LSTM + Attention")
    print(f"{'─'*60}")

    lstm_model = build_lstm_attention(len(FEATURES))
    lstm_model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    lstm_model.fit(
        X_tr_s, y_tr_s,
        validation_data=(X_va_s, y_va_s),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    lstm_pred_val = lstm_model.predict(X_va_s, verbose=0).flatten()
    lstm_pred_test = lstm_model.predict(X_te_s, verbose=0).flatten()

    # ============================================================
    # Model 2: CatBoost
    # ============================================================
    print(f"\n{'─'*60}")
    print("  [2/3] CatBoost")
    print(f"{'─'*60}")

    try:
        from catboost import CatBoostRegressor

        cat_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            random_strength=0.5,
            bagging_temperature=0.3,
            verbose=200,
            early_stopping_rounds=100,
            task_type='CPU',
        )

        cat_model.fit(
            X_tr_b, y_tr_b,
            eval_set=(X_va_b, y_va_b),
            verbose=200,
        )

        cat_pred_val = cat_model.predict(X_va_b)
        cat_pred_test = cat_model.predict(X_te_b)
        cat_model.save_model("catboost_model.cbm")
        print("CatBoost модель сохранена: catboost_model.cbm")
        has_catboost = True

    except ImportError:
        print("CatBoost не установлен (pip install catboost). Пропускаем.")
        cat_pred_val = np.zeros_like(lstm_pred_val)
        cat_pred_test = np.zeros_like(lstm_pred_test)
        has_catboost = False

    # ============================================================
    # Model 3: XGBoost
    # ============================================================
    print(f"\n{'─'*60}")
    print("  [3/3] XGBoost")
    print(f"{'─'*60}")

    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=100,
            verbosity=1,
        )

        xgb_model.fit(
            X_tr_b, y_tr_b,
            eval_set=[(X_va_b, y_va_b)],
            verbose=200,
        )

        xgb_pred_val = xgb_model.predict(X_va_b)
        xgb_pred_test = xgb_model.predict(X_te_b)
        xgb_model.save_model("xgboost_model.json")
        print("XGBoost модель сохранена: xgboost_model.json")
        has_xgboost = True

    except ImportError:
        print("XGBoost не установлен (pip install xgboost). Пропускаем.")
        xgb_pred_val = np.zeros_like(lstm_pred_val)
        xgb_pred_test = np.zeros_like(lstm_pred_test)
        has_xgboost = False

    # ============================================================
    # Meta-model: Stacking with Ridge
    # ============================================================
    print(f"\n{'─'*60}")
    print("  Stacking Ensemble (Meta-learner: Ridge)")
    print(f"{'─'*60}")

    # Build stacking features
    models_used = ['lstm']
    stack_val = [lstm_pred_val]
    stack_test = [lstm_pred_test]

    if has_catboost:
        models_used.append('catboost')
        stack_val.append(cat_pred_val)
        stack_test.append(cat_pred_test)

    if has_xgboost:
        models_used.append('xgboost')
        stack_val.append(xgb_pred_val)
        stack_test.append(xgb_pred_test)

    print(f"Модели в ансамбле: {models_used}")

    X_meta_val = np.column_stack(stack_val)
    X_meta_test = np.column_stack(stack_test)

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_val, y_va_s)

    # Save meta model
    with open("meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)

    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save ensemble config
    ensemble_config = {
        "models": models_used,
        "features": FEATURES,
        "seq_len": SEQ_LEN,
        "n_features": len(FEATURES),
        "meta_weights": meta_model.coef_.tolist(),
        "meta_intercept": float(meta_model.intercept_),
    }
    with open("ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    # ============================================================
    # Final evaluation
    # ============================================================
    print(f"\n{'='*60}")
    print("  Результаты на тестовой выборке")
    print(f"{'='*60}")

    # Ensemble prediction
    ensemble_pred_test = meta_model.predict(X_meta_test)

    # Invert all predictions
    y_test_real = invert(y_te_s)
    lstm_real = invert(lstm_pred_test)
    ensemble_real = invert(ensemble_pred_test)

    def print_metrics(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"  {name:25s}  MAE=${mae:8.2f}  RMSE=${rmse:8.2f}  MAPE={mape:5.2f}%")

    print()
    print_metrics("LSTM + Attention", y_test_real, lstm_real)

    if has_catboost:
        cat_real = invert(cat_pred_test)
        print_metrics("CatBoost", y_test_real, cat_real)

    if has_xgboost:
        xgb_real = invert(xgb_pred_test)
        print_metrics("XGBoost", y_test_real, xgb_real)

    print_metrics("ENSEMBLE (Stacking)", y_test_real, ensemble_real)

    print(f"\n{'='*60}")
    print(f"  Файлы:")
    print(f"    LSTM модель:      {args.output}")
    if has_catboost:
        print(f"    CatBoost модель:  catboost_model.cbm")
    if has_xgboost:
        print(f"    XGBoost модель:   xgboost_model.json")
    print(f"    Мета-модель:      meta_model.pkl")
    print(f"    Скейлер:          scaler.pkl")
    print(f"    Конфиг ансамбля:  ensemble_config.json")
    print(f"{'='*60}")
    print(f"\nДля использования перезапустите app.py")


if __name__ == "__main__":
    main()
