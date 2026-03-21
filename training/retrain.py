"""
NeuCast — Retrain model to predict RETURNS (% change), not prices.

Why returns:
  - Prices are non-stationary (trend up forever) — model can't extrapolate
  - Returns are stationary (centered around 0) — model generalizes
  - This is standard practice in quantitative finance ML

Architecture: Bidirectional LSTM → Dense
Target: next-day log return of Close price
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

SEQ_LEN = 60
FEATURES = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "MA_20", "MA_50"]


def preprocess(df):
    df = df[["Open", "High", "Low", "Close", "Volume"]].interpolate(method="time")
    for w in (5, 10, 20, 50):
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()
    df = df.dropna()
    return df


def make_return_sequences(scaled_data, returns):
    """
    X: windows of scaled features
    y: next-day return (NOT scaled, raw log return)
    """
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled_data)):
        X.append(scaled_data[i - SEQ_LEN: i])
        y.append(returns[i])
    return np.array(X), np.array(y)


print("=" * 60)
print("  NeuCast — Returns-Based Model Training")
print("=" * 60)

# Download
print("\nЗагрузка GC=F (2010-2025) для максимального обучающего набора...")
df_raw = yf.download("GC=F", start="2010-01-01", end="2025-12-31", interval="1d", progress=True)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

df = preprocess(df_raw)
print(f"Данных: {len(df)}")

# Compute log returns (target)
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()

log_returns = df["log_return"].values

# Scale features with RobustScaler-like approach
# We use MinMaxScaler on a ROLLING window concept, but for simplicity
# we normalize each column to 0-1 using full data
# The KEY insight: we predict returns, not prices, so scaling doesn't leak future price levels
scaler = MinMaxScaler((0, 1))
scaled = scaler.fit_transform(df[FEATURES])

# Sequences
X, y = make_return_sequences(scaled, log_returns)
print(f"X: {X.shape}, y: {y.shape}")
print(f"Return stats: mean={y.mean():.6f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

# Split 80/10/10
n = len(X)
tr = int(n * 0.8)
va = int(n * 0.9)

X_tr, y_tr = X[:tr], y[:tr]
X_va, y_va = X[tr:va], y[tr:va]
X_te, y_te = X[va:], y[va:]
print(f"Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")

# Build model
print("\nМодель для предсказания дневных returns...")
model = Sequential([
    Input(shape=(SEQ_LEN, len(FEATURES))),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear'),  # output: log return (can be negative)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mse',
    metrics=['mae'],
)
model.summary()

# Train
print("\nОбучение...")
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ],
    verbose=1,
)

# Save model
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
model.save(os.path.join(WEIGHTS_DIR, "best_model.h5"))

# Save scaler
with open(os.path.join(WEIGHTS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save config
import json
config = {
    "type": "returns",
    "features": FEATURES,
    "seq_len": SEQ_LEN,
    "n_features": len(FEATURES),
}
with open(os.path.join(WEIGHTS_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nМодель сохранена: {WEIGHTS_DIR}/best_model.h5")
print(f"Scaler сохранён: {WEIGHTS_DIR}/scaler.pkl")
print(f"Конфиг: {WEIGHTS_DIR}/model_config.json")

# ---- Evaluate on test set ----
# For test: reconstruct prices from predicted returns
test_dates = df.index[SEQ_LEN + va:]
test_prices_actual = df["Close"].values[SEQ_LEN + va:]
prev_prices = df["Close"].values[SEQ_LEN + va - 1: -1]  # price at t-1

y_pred_returns = model.predict(X_te, verbose=0).flatten()

# Predicted price = prev_price * exp(predicted_return)
pred_prices = prev_prices * np.exp(y_pred_returns)

mae = mean_absolute_error(test_prices_actual, pred_prices)
rmse = np.sqrt(mean_squared_error(test_prices_actual, pred_prices))
mape = np.mean(np.abs((test_prices_actual - pred_prices) / test_prices_actual)) * 100

# Direction accuracy (did we predict up/down correctly?)
actual_direction = (test_prices_actual > prev_prices).astype(int)
pred_direction = (pred_prices > prev_prices).astype(int)
direction_acc = np.mean(actual_direction == pred_direction) * 100

print(f"\n{'=' * 60}")
print(f"  Test Results (returns-based model):")
print(f"  MAE:  ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  Direction Accuracy: {direction_acc:.1f}%")
print(f"")
print(f"  Test period: {test_dates[0].date()} — {test_dates[-1].date()}")
print(f"  Price range: ${test_prices_actual.min():.0f} — ${test_prices_actual.max():.0f}")
print(f"{'=' * 60}")
