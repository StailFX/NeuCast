"""
NeuCast — Retrain LSTM to predict log returns.
Matches app.py pipeline exactly:
  - 9 features: OHLCV + MA_5, MA_10, MA_20, MA_50
  - Target: log_return = ln(Close_t / Close_{t-1})
  - MinMaxScaler on train only
  - SEQ_LEN = 60

Usage:
  python retrain_returns.py
  python retrain_returns.py --ticker GC=F --start 2015-01-01 --epochs 80
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
    Input, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEQ_LEN = 60
MODEL_COLS = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "MA_20", "MA_50"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Same preprocessing as app.py"""
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

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    return df


class AttentionLayer(Layer):
    """Bahdanau-style attention for temporal data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_W', shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform', trainable=True,
        )
        self.b = self.add_weight(
            name='att_b', shape=(input_shape[-1],),
            initializer='zeros', trainable=True,
        )
        self.u = self.add_weight(
            name='att_u', shape=(input_shape[-1],),
            initializer='glorot_uniform', trainable=True,
        )

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        return tf.reduce_sum(x * tf.expand_dims(weights, -1), axis=1)

    def get_config(self):
        return super().get_config()


def build_model(n_features: int) -> Model:
    """LSTM + Attention model for log return prediction."""
    inputs = Input(shape=(SEQ_LEN, n_features))

    # Bidirectional LSTM
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

    model = Model(inputs, outputs, name="NeuCast_LSTM_Returns")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-03-21")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="best_model.h5")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NeuCast — LSTM Returns Retraining")
    print(f"{'='*60}")
    print(f"  Ticker: {args.ticker}")
    print(f"  Period: {args.start} — {args.end}")
    print(f"  Target: log returns")
    print(f"  Features: {MODEL_COLS}")

    # Download data
    df_raw = yf.download(args.ticker, start=args.start, end=args.end, interval="1d", progress=True)
    if df_raw.empty:
        print("ERROR: No data!")
        return
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df = preprocess(df_raw)
    print(f"  Data points: {len(df)}")

    # Split 70/15/15
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # Scale features on train only (same as app.py)
    scaler = MinMaxScaler((0, 1))
    scaler.fit(df.iloc[:train_end][MODEL_COLS])
    scaled_all = scaler.transform(df[MODEL_COLS])

    log_returns = df["log_return"].values

    # Build sequences: X = scaled features, y = log return
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled_all)):
        X.append(scaled_all[i - SEQ_LEN: i])
        y.append(log_returns[i])
    X = np.array(X)
    y = np.array(y)

    train_seq_end = train_end - SEQ_LEN
    val_seq_end = val_end - SEQ_LEN

    X_train, y_train = X[:train_seq_end], y[:train_seq_end]
    X_val, y_val = X[train_seq_end:val_seq_end], y[train_seq_end:val_seq_end]
    X_test, y_test = X[val_seq_end:], y[val_seq_end:]

    print(f"\n  Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Build and train model
    model = build_model(len(MODEL_COLS))
    model.summary()

    # Custom loss: MSE + direction penalty
    # This encourages the model to get the sign of the return correct
    def directional_mse(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        # Direction component: penalize wrong sign predictions
        sign_true = tf.sign(y_true)
        sign_pred = tf.sign(y_pred)
        direction_penalty = tf.reduce_mean(tf.maximum(0.0, -sign_true * sign_pred))
        return mse + 0.1 * direction_penalty

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=directional_mse,
        metrics=['mae'],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    print(f"\n{'─'*60}")
    print("  Training LSTM + Attention on log returns...")
    print(f"{'─'*60}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    pred_returns_test = model.predict(X_test, verbose=0).flatten()
    close_prices = df["Close"].values
    prev_prices_test = close_prices[SEQ_LEN + val_seq_end - 1: -1]
    actual_prices_test = close_prices[SEQ_LEN + val_seq_end:]
    pred_prices_test = prev_prices_test * np.exp(pred_returns_test)

    mae = mean_absolute_error(actual_prices_test, pred_prices_test)
    rmse = np.sqrt(mean_squared_error(actual_prices_test, pred_prices_test))
    mape = np.mean(np.abs((actual_prices_test - pred_prices_test) / actual_prices_test)) * 100

    # Direction accuracy
    actual_dir = (actual_prices_test > prev_prices_test).astype(int)
    pred_dir = (pred_prices_test > prev_prices_test).astype(int)
    dir_acc = np.mean(actual_dir == pred_dir) * 100

    # R²
    ss_r = np.sum((actual_prices_test - pred_prices_test) ** 2)
    ss_t = np.sum((actual_prices_test - np.mean(actual_prices_test)) ** 2)
    r2 = 1 - ss_r / ss_t

    print(f"\n{'='*60}")
    print(f"  Test Results (LSTM alone)")
    print(f"{'='*60}")
    print(f"  MAE:           ${mae:.2f}")
    print(f"  RMSE:          ${rmse:.2f}")
    print(f"  MAPE:          {mape:.2f}%")
    print(f"  R²:            {r2:.4f}")
    print(f"  Direction Acc: {dir_acc:.1f}%")
    print(f"{'='*60}")

    # Save config
    config = {
        "type": "returns",
        "features": MODEL_COLS,
        "seq_len": SEQ_LEN,
        "n_features": len(MODEL_COLS),
    }
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Model saved: {args.output}")
    print(f"  Config saved: model_config.json")
    print(f"\n  Restart app.py to use the new model.")


if __name__ == "__main__":
    main()
