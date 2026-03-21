"""
NeuCast — Train TCN with Multi-Target output.

The model predicts TWO things simultaneously:
  1. log_return (continuous) — for price prediction
  2. direction (binary 0/1) — for direction accuracy

This forces the model to learn both magnitude AND direction.

Usage:
  python retrain_tcn.py
  python retrain_tcn.py --epochs 120
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, BatchNormalization,
    Layer, Add, Activation, GlobalAveragePooling1D, Multiply,
    LayerNormalization, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEQ_LEN = 60
MODEL_COLS = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "MA_20", "MA_50"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
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

    # Momentum features
    df["ROC_5"] = df["Close"].pct_change(5)
    df["ROC_10"] = df["Close"].pct_change(10)
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["Volatility_20"] = df["Close"].pct_change().rolling(20).std()
    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    return df


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
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate,
        })
        return config


def build_tcn_multitarget(n_features: int, seq_len: int = SEQ_LEN) -> Model:
    """
    TCN with dual output:
      - return_output: predicts log return (regression)
      - direction_output: predicts up/down (classification)

    Shared backbone learns features useful for BOTH tasks.
    The direction head forces the model to pay attention to sign.
    """
    inputs = Input(shape=(seq_len, n_features))

    # Multi-scale inception
    branch3 = Conv1D(48, 3, padding='causal', activation='gelu')(inputs)
    branch5 = Conv1D(48, 5, padding='causal', activation='gelu')(inputs)
    branch7 = Conv1D(48, 7, padding='causal', activation='gelu')(inputs)
    x = Concatenate()([branch3, branch5, branch7])  # 144 channels
    x = LayerNormalization()(x)

    # Deeper TCN stack
    for dilation in [1, 2, 4, 8, 16, 32]:
        x = TCNBlock(filters=144, kernel_size=3, dilation_rate=dilation, dropout_rate=0.15)(x)

    # Global context
    gap = GlobalAveragePooling1D()(x)
    last = x[:, -1, :]
    shared = Concatenate()([gap, last])

    # Shared dense backbone
    shared = Dense(192, activation='gelu')(shared)
    shared = Dropout(0.25)(shared)
    shared = Dense(96, activation='gelu')(shared)
    shared = Dropout(0.2)(shared)

    # Head 1: Return prediction (regression)
    ret_x = Dense(48, activation='gelu')(shared)
    return_output = Dense(1, activation='linear', name='return_output')(ret_x)

    # Head 2: Direction prediction (classification)
    dir_x = Dense(48, activation='gelu')(shared)
    direction_output = Dense(1, activation='sigmoid', name='direction_output')(dir_x)

    model = Model(inputs, [return_output, direction_output], name="NeuCast_TCN_MultiTarget")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-03-21")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "weights", "best_model.h5"))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NeuCast — TCN Multi-Target Training")
    print(f"{'='*60}")
    print(f"  Ticker: {args.ticker}")
    print(f"  Period: {args.start} — {args.end}")
    print(f"  Targets: log_return + direction (multi-task)")

    df_raw = yf.download(args.ticker, start=args.start, end=args.end, interval="1d", progress=True)
    if df_raw.empty:
        print("ERROR: No data!")
        return
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df = preprocess(df_raw)
    print(f"  Data points: {len(df)}")

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    scaler = MinMaxScaler((0, 1))
    scaler.fit(df.iloc[:train_end][MODEL_COLS])
    scaled_all = scaler.transform(df[MODEL_COLS])

    log_returns = df["log_return"].values
    close_prices = df["Close"].values

    # Direction target: 1 if price went up, 0 if down
    direction = (log_returns > 0).astype(np.float32)

    # Build sequences with dual targets
    X, y_ret, y_dir = [], [], []
    for i in range(SEQ_LEN, len(scaled_all)):
        X.append(scaled_all[i - SEQ_LEN: i])
        y_ret.append(log_returns[i])
        y_dir.append(direction[i])
    X = np.array(X)
    y_ret = np.array(y_ret)
    y_dir = np.array(y_dir)

    train_seq_end = train_end - SEQ_LEN
    val_seq_end = val_end - SEQ_LEN

    X_train = X[:train_seq_end]
    y_ret_train, y_dir_train = y_ret[:train_seq_end], y_dir[:train_seq_end]

    X_val = X[train_seq_end:val_seq_end]
    y_ret_val, y_dir_val = y_ret[train_seq_end:val_seq_end], y_dir[train_seq_end:val_seq_end]

    X_test = X[val_seq_end:]
    y_ret_test, y_dir_test = y_ret[val_seq_end:], y_dir[val_seq_end:]

    print(f"  Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"  Direction balance (train): {y_dir_train.mean():.1%} up")

    # Build model
    model = build_tcn_multitarget(len(MODEL_COLS))
    model.summary()

    # Compile with dual loss
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss={
            'return_output': 'mse',
            'direction_output': 'binary_crossentropy',
        },
        loss_weights={
            'return_output': 1.0,
            'direction_output': 0.5,  # direction is auxiliary task
        },
        metrics={
            'return_output': 'mae',
            'direction_output': 'accuracy',
        },
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    print(f"\n{'─'*60}")
    print("  Training TCN Multi-Target...")
    print(f"{'─'*60}\n")

    model.fit(
        X_train,
        {'return_output': y_ret_train, 'direction_output': y_dir_train},
        validation_data=(
            X_val,
            {'return_output': y_ret_val, 'direction_output': y_dir_val},
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──
    pred_ret, pred_dir = model.predict(X_test, verbose=0)
    pred_ret = pred_ret.flatten()
    pred_dir = pred_dir.flatten()

    prev_prices = close_prices[SEQ_LEN + val_seq_end - 1: -1]
    actual_prices = close_prices[SEQ_LEN + val_seq_end:]
    pred_prices = prev_prices * np.exp(pred_ret)

    mae = mean_absolute_error(actual_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100

    # Direction accuracy from return predictions
    actual_dir = (actual_prices > prev_prices).astype(int)
    pred_dir_from_ret = (pred_prices > prev_prices).astype(int)
    dir_acc_ret = np.mean(actual_dir == pred_dir_from_ret) * 100

    # Direction accuracy from direction head
    pred_dir_cls = (pred_dir > 0.5).astype(int)
    dir_acc_cls = np.mean(actual_dir == pred_dir_cls) * 100

    # Combined: use direction head to adjust return sign
    # If direction head says down but return says up (or vice versa), trust direction head
    adjusted_ret = pred_ret.copy()
    for i in range(len(adjusted_ret)):
        if pred_dir[i] > 0.6 and adjusted_ret[i] < 0:
            adjusted_ret[i] = abs(adjusted_ret[i]) * 0.5  # flip to positive, dampened
        elif pred_dir[i] < 0.4 and adjusted_ret[i] > 0:
            adjusted_ret[i] = -abs(adjusted_ret[i]) * 0.5  # flip to negative, dampened

    adjusted_prices = prev_prices * np.exp(adjusted_ret)
    adj_dir = (adjusted_prices > prev_prices).astype(int)
    dir_acc_adj = np.mean(actual_dir == adj_dir) * 100

    ss_r = np.sum((actual_prices - pred_prices) ** 2)
    ss_t = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
    r2 = 1 - ss_r / ss_t

    print(f"\n{'='*60}")
    print(f"  Test Results")
    print(f"{'='*60}")
    print(f"  MAE:                    ${mae:.2f}")
    print(f"  RMSE:                   ${rmse:.2f}")
    print(f"  MAPE:                   {mape:.2f}%")
    print(f"  R²:                     {r2:.4f}")
    print(f"  Direction (from return): {dir_acc_ret:.1f}%")
    print(f"  Direction (cls head):    {dir_acc_cls:.1f}%")
    print(f"  Direction (combined):    {dir_acc_adj:.1f}%")
    print(f"{'='*60}")

    config = {
        "type": "returns",
        "features": MODEL_COLS,
        "seq_len": SEQ_LEN,
        "n_features": len(MODEL_COLS),
        "architecture": "TCN_MultiTarget",
        "outputs": ["return", "direction"],
    }
    config_path = os.path.join(os.path.dirname(__file__), "..", "weights", "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Model saved: {args.output}")
    print(f"  Config saved: {config_path}")


if __name__ == "__main__":
    main()
