import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.layers import CUSTOM_OBJECTS

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

SEQ_LEN = 60
MODEL_COLS = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "MA_20", "MA_50"]
TRAIN_RATIO = 0.8

# ── Lazy model loading (critical for Celery prefork) ──
_base_model = None
_MODEL_TYPE = None
_IS_MULTITARGET = None


def _get_model():
    global _base_model, _MODEL_TYPE, _IS_MULTITARGET
    if _base_model is None:
        _base_model = load_model(
            os.path.join(WEIGHTS_DIR, "best_model.h5"),
            compile=False, custom_objects=CUSTOM_OBJECTS,
        )
        _MODEL_TYPE = "returns"
        _IS_MULTITARGET = False
        config_path = os.path.join(WEIGHTS_DIR, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
                _MODEL_TYPE = cfg.get("type", "returns")
                _IS_MULTITARGET = "direction" in cfg.get("outputs", [])
    return _base_model, _MODEL_TYPE, _IS_MULTITARGET


# ── Preprocessing ──
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

    df["ROC_5"] = df["Close"].pct_change(5)
    df["ROC_10"] = df["Close"].pct_change(10)
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["Volatility_20"] = df["Close"].pct_change().rolling(20).std()
    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df = df.dropna()
    if len(df) < SEQ_LEN + 1:
        raise ValueError("Недостаточно данных. Нужно минимум ~120 торговых дней.")
    return df


def make_sequences_X(arr: np.ndarray) -> np.ndarray:
    X = []
    for i in range(SEQ_LEN, len(arr)):
        X.append(arr[i - SEQ_LEN: i])
    return np.array(X)


def make_sequences_Xy(scaled, returns):
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN: i])
        y.append(returns[i])
    return np.array(X), np.array(y)


def make_boosting_features(data: np.ndarray) -> np.ndarray:
    X = []
    for i in range(SEQ_LEN, len(data)):
        window = data[i - SEQ_LEN: i]
        feats = []
        for col in range(window.shape[1]):
            col_data = window[:, col]
            feats.extend([
                col_data[-1], col_data.mean(), col_data.std(),
                col_data[-1] - col_data[0], col_data[-5:].mean(),
                np.min(col_data), np.max(col_data),
            ])
        X.append(feats)
    return np.array(X)


def run_prediction(df: pd.DataFrame, days_ahead: int):
    base_model, MODEL_TYPE, IS_MULTITARGET = _get_model()

    n_total = len(df)
    split_idx = int(n_total * TRAIN_RATIO)

    scaler = MinMaxScaler((0, 1))
    scaler.fit(df.iloc[:split_idx][MODEL_COLS])
    scaled_all = scaler.transform(df[MODEL_COLS])

    log_returns = df["log_return"].values
    close_prices = df["Close"].values

    X_all, y_all = make_sequences_Xy(scaled_all, log_returns)
    train_seq_end = split_idx - SEQ_LEN

    X_train = X_all[:train_seq_end]
    y_train = y_all[:train_seq_end]

    # Fine-tune TCN on train data
    fine_model = clone_model(base_model)
    fine_model.set_weights(base_model.get_weights())

    direction_all = (log_returns[SEQ_LEN:] > 0).astype(np.float32)
    dir_train = direction_all[:train_seq_end]

    if IS_MULTITARGET:
        fine_model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy'},
            loss_weights={'return_output': 1.0, 'direction_output': 0.5},
        )
    else:
        fine_model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss="mse")

    if len(X_train) > SEQ_LEN:
        val_size = max(0.1, SEQ_LEN / len(X_train))
        if IS_MULTITARGET:
            fine_model.fit(
                X_train,
                {'return_output': y_train, 'direction_output': dir_train},
                epochs=10, batch_size=32,
                validation_split=val_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)],
                verbose=0,
            )
        else:
            fine_model.fit(
                X_train, y_train,
                epochs=10, batch_size=32,
                validation_split=val_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)],
                verbose=0,
            )

    raw_pred = fine_model.predict(X_all, verbose=0)
    if IS_MULTITARGET:
        tcn_pred_returns = raw_pred[0].flatten()
        tcn_direction_pred = raw_pred[1].flatten()
    else:
        tcn_pred_returns = raw_pred.flatten()
        tcn_direction_pred = None

    # Boosting ensemble
    X_bst_all = make_boosting_features(scaled_all)
    X_bst_train = X_bst_all[:train_seq_end]
    y_bst_train = y_all[:train_seq_end]

    cat_pred = None
    xgb_pred = None
    lgb_pred = None
    model_names = ["TCN"]

    try:
        from catboost import CatBoostRegressor
        cat = CatBoostRegressor(
            iterations=1000, learning_rate=0.03, depth=7, l2_leaf_reg=3,
            random_strength=0.5, bagging_temperature=0.3,
            verbose=0, early_stopping_rounds=80,
        )
        cat.fit(X_bst_train, y_bst_train,
                eval_set=(X_bst_all[train_seq_end:], y_all[train_seq_end:]), verbose=0)
        cat_pred = cat.predict(X_bst_all)
        model_names.append("CatBoost")
    except Exception:
        pass

    try:
        import xgboost as xgb
        xgb_m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=80, verbosity=0,
        )
        xgb_m.fit(X_bst_train, y_bst_train,
                  eval_set=[(X_bst_all[train_seq_end:], y_all[train_seq_end:])], verbose=0)
        xgb_pred = xgb_m.predict(X_bst_all)
        model_names.append("XGBoost")
    except Exception:
        pass

    try:
        import lightgbm as lgbm
        lgb_m = lgbm.LGBMRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=7,
            num_leaves=63, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
        )
        lgb_m.fit(X_bst_train, y_bst_train,
                  eval_set=[(X_bst_all[train_seq_end:], y_all[train_seq_end:])],
                  callbacks=[lgbm.early_stopping(80, verbose=False), lgbm.log_evaluation(0)])
        lgb_pred = lgb_m.predict(X_bst_all)
        model_names.append("LightGBM")
    except Exception:
        pass

    # Stacking
    from sklearn.linear_model import Ridge

    stack_train = [tcn_pred_returns[:train_seq_end]]
    stack_all = [tcn_pred_returns]

    if cat_pred is not None:
        stack_train.append(cat_pred[:train_seq_end])
        stack_all.append(cat_pred)
    if xgb_pred is not None:
        stack_train.append(xgb_pred[:train_seq_end])
        stack_all.append(xgb_pred)
    if lgb_pred is not None:
        stack_train.append(lgb_pred[:train_seq_end])
        stack_all.append(lgb_pred)

    if len(stack_train) > 1:
        X_meta_train = np.column_stack(stack_train)
        meta = Ridge(alpha=1.0)
        meta.fit(X_meta_train, y_bst_train)
        X_meta_all = np.column_stack(stack_all)
        pred_returns = meta.predict(X_meta_all)
        model_name = f"Ensemble ({'+'.join(model_names)})"
    else:
        pred_returns = tcn_pred_returns
        model_name = "TCN"

    # Direction head adjustment
    if tcn_direction_pred is not None:
        adjusted_returns = pred_returns.copy()
        for i in range(len(adjusted_returns)):
            if tcn_direction_pred[i] > 0.6 and adjusted_returns[i] < 0:
                adjusted_returns[i] = abs(adjusted_returns[i]) * 0.5
            elif tcn_direction_pred[i] < 0.4 and adjusted_returns[i] > 0:
                adjusted_returns[i] = -abs(adjusted_returns[i]) * 0.5
        pred_returns = adjusted_returns

    prev_prices = close_prices[SEQ_LEN - 1: -1]
    actual_prices = close_prices[SEQ_LEN:]
    pred_prices = prev_prices * np.exp(pred_returns)

    dates = df.index[SEQ_LEN:].strftime("%Y-%m-%d").tolist()

    # Per-model prices
    model_comparison = {}
    tcn_prices = prev_prices * np.exp(tcn_pred_returns)
    model_comparison["TCN"] = tcn_prices.tolist()
    if cat_pred is not None:
        cat_prices = prev_prices * np.exp(cat_pred)
        model_comparison["CatBoost"] = cat_prices.tolist()
    if xgb_pred is not None:
        xgb_prices = prev_prices * np.exp(xgb_pred)
        model_comparison["XGBoost"] = xgb_prices.tolist()
    if lgb_pred is not None:
        lgb_prices = prev_prices * np.exp(lgb_pred)
        model_comparison["LightGBM"] = lgb_prices.tolist()
    model_comparison["Ensemble"] = pred_prices.tolist()

    # Per-model metrics
    test_start = train_seq_end
    y_act_test = actual_prices[test_start:]

    def calc_metrics(y_true, y_pred_arr):
        _mae = mean_absolute_error(y_true, y_pred_arr)
        _rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_arr)))
        _mape = float(np.mean(np.abs((y_true - y_pred_arr) / y_true)) * 100)
        ss_r = np.sum((y_true - y_pred_arr) ** 2)
        ss_t = np.sum((y_true - np.mean(y_true)) ** 2)
        _r2 = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
        return {"mae": round(_mae, 2), "rmse": round(_rmse, 2), "mape": round(_mape, 2), "r2": round(_r2, 4)}

    model_metrics = {}
    model_metrics["TCN"] = calc_metrics(y_act_test, tcn_prices[test_start:])
    if cat_pred is not None:
        model_metrics["CatBoost"] = calc_metrics(y_act_test, cat_prices[test_start:])
    if xgb_pred is not None:
        model_metrics["XGBoost"] = calc_metrics(y_act_test, xgb_prices[test_start:])
    if lgb_pred is not None:
        model_metrics["LightGBM"] = calc_metrics(y_act_test, lgb_prices[test_start:])
    model_metrics["Ensemble"] = calc_metrics(y_act_test, pred_prices[test_start:])

    # Feature importance
    feature_importance = {}
    feat_names = []
    for col_idx in range(len(MODEL_COLS)):
        for stat in ["last", "mean", "std", "diff", "ma5", "min", "max"]:
            feat_names.append(f"{MODEL_COLS[col_idx]}_{stat}")

    if cat_pred is not None:
        try:
            cat_imp = cat.get_feature_importance()
            top_idx = np.argsort(cat_imp)[::-1][:15]
            feature_importance["CatBoost"] = {
                "names": [feat_names[i] for i in top_idx],
                "values": [round(float(cat_imp[i]), 2) for i in top_idx],
            }
        except Exception:
            pass
    if xgb_pred is not None:
        try:
            xgb_imp = xgb_m.feature_importances_
            top_idx = np.argsort(xgb_imp)[::-1][:15]
            feature_importance["XGBoost"] = {
                "names": [feat_names[i] for i in top_idx],
                "values": [round(float(xgb_imp[i]), 4) for i in top_idx],
            }
        except Exception:
            pass
    if lgb_pred is not None:
        try:
            lgb_imp = lgb_m.feature_importances_.astype(float)
            top_idx = np.argsort(lgb_imp)[::-1][:15]
            feature_importance["LightGBM"] = {
                "names": [feat_names[i] for i in top_idx],
                "values": [round(float(lgb_imp[i]), 2) for i in top_idx],
            }
        except Exception:
            pass

    # SHAP explanations (TreeExplainer — fast for boosting models)
    shap_data = {}
    try:
        import shap

        # Use last test sample for waterfall, test set for beeswarm summary
        X_bst_test = X_bst_all[train_seq_end:]
        sample_idx = -1  # last prediction

        boosting_models = {}
        if cat_pred is not None:
            boosting_models["CatBoost"] = cat
        if xgb_pred is not None:
            boosting_models["XGBoost"] = xgb_m
        if lgb_pred is not None:
            boosting_models["LightGBM"] = lgb_m

        for model_label, model_obj in boosting_models.items():
            try:
                explainer = shap.TreeExplainer(model_obj)

                # Global: mean |SHAP| across test set (top 15 features)
                sv_test = explainer.shap_values(X_bst_test)
                mean_abs = np.abs(sv_test).mean(axis=0)
                top_idx = np.argsort(mean_abs)[::-1][:15]

                # Local: SHAP for last prediction (waterfall)
                sv_single = sv_test[sample_idx]
                base_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

                shap_data[model_label] = {
                    # Global bar chart (mean |SHAP|)
                    "global_names": [feat_names[i] for i in top_idx],
                    "global_values": [round(float(mean_abs[i]), 6) for i in top_idx],
                    # Waterfall for last prediction
                    "waterfall_names": [feat_names[i] for i in top_idx],
                    "waterfall_values": [round(float(sv_single[i]), 6) for i in top_idx],
                    "base_value": round(base_value, 6),
                    "output_value": round(float(sv_single.sum() + base_value), 6),
                    # Beeswarm data (top 10 features, sampled for performance)
                    "beeswarm_names": [feat_names[i] for i in top_idx[:10]],
                    "beeswarm_shap": [[round(float(sv_test[j, i]), 6)
                                       for j in range(0, len(sv_test), max(1, len(sv_test) // 100))]
                                      for i in top_idx[:10]],
                    "beeswarm_feat": [[round(float(X_bst_test[j, i]), 6)
                                       for j in range(0, len(X_bst_test), max(1, len(X_bst_test) // 100))]
                                      for i in top_idx[:10]],
                }
            except Exception:
                pass
    except ImportError:
        pass

    y_pred_test = pred_prices[test_start:]
    residuals = (y_act_test - y_pred_test).tolist()

    mae = model_metrics["Ensemble"]["mae"]
    rmse = model_metrics["Ensemble"]["rmse"]
    mape = model_metrics["Ensemble"]["mape"]
    r2 = model_metrics["Ensemble"]["r2"]

    actual_dir = (actual_prices[test_start:] > prev_prices[test_start:]).astype(int)
    pred_dir = (pred_prices[test_start:] > prev_prices[test_start:]).astype(int)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)

    if tcn_direction_pred is not None:
        dir_cls = tcn_direction_pred[test_start:]
        pred_dir_cls = (dir_cls > 0.5).astype(int)
        dir_acc_cls = float(np.mean(actual_dir == pred_dir_cls) * 100)
        if dir_acc_cls > dir_acc:
            dir_acc = dir_acc_cls

    corr_cols = ["Close", "Volume", "RSI", "MACD", "MA_5", "MA_20", "MA_50", "ATR", "BB_upper", "BB_lower"]
    corr_data = df[corr_cols].corr().round(3).values.tolist()
    corr_labels = corr_cols

    # Future forecast (Monte Carlo GBM)
    future_dates, future_preds, future_upper, future_lower = [], [], [], []
    future_p50, future_p5, future_p95 = [], [], []

    if days_ahead > 0:
        N_SIM = 1000
        last_price = close_prices[-1]
        recent_returns = log_returns[-60:]
        mu_hist = float(np.mean(recent_returns))
        sigma_daily = float(np.std(recent_returns))
        model_signal = float(pred_returns[-1])
        mu = 0.7 * model_signal + 0.3 * mu_hist
        dt = 1.0

        np.random.seed(42)
        Z = np.random.standard_normal((N_SIM, days_ahead))
        daily_returns = (mu - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z
        cum_returns = np.cumsum(daily_returns, axis=1)
        trajectories = last_price * np.exp(cum_returns)

        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(1, "d"), periods=days_ahead
        ).strftime("%Y-%m-%d").tolist()

        future_preds = np.median(trajectories, axis=0).tolist()
        future_p5 = np.percentile(trajectories, 5, axis=0).tolist()
        future_p95 = np.percentile(trajectories, 95, axis=0).tolist()
        future_upper = np.percentile(trajectories, 75, axis=0).tolist()
        future_lower = np.percentile(trajectories, 25, axis=0).tolist()

    return {
        "dates": dates,
        "y_act": actual_prices,
        "y_pred": pred_prices,
        "train_size": train_seq_end,
        "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
        "dir_acc": dir_acc,
        "model_name": model_name,
        "future_dates": future_dates,
        "future_preds": future_preds,
        "future_upper": future_upper,
        "future_lower": future_lower,
        "future_p5": future_p5,
        "future_p95": future_p95,
        "rsi": df["RSI"].values[SEQ_LEN:].round(2).tolist(),
        "macd": df["MACD"].values[SEQ_LEN:].round(2).tolist(),
        "signal": df["Signal"].values[SEQ_LEN:].round(2).tolist(),
        "bb_upper": df["BB_upper"].values[SEQ_LEN:].round(2).tolist(),
        "bb_lower": df["BB_lower"].values[SEQ_LEN:].round(2).tolist(),
        "atr": df["ATR"].values[SEQ_LEN:].round(2).tolist(),
        "date_index": df.index[SEQ_LEN:],
        "model_comparison": model_comparison,
        "model_metrics": model_metrics,
        "feature_importance": feature_importance,
        "shap_data": shap_data,
        "residuals": residuals,
        "corr_data": corr_data,
        "corr_labels": corr_labels,
    }


def fetch_and_preprocess(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df_raw = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    if df_raw.empty:
        raise ValueError("Нет данных по указанному тикеру и диапазону дат")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    return preprocess(df_raw)
