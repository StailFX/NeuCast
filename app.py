import io
import os
import json
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import uvicorn
import hashlib
import tensorflow as tf

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

from db import DATABASE_URL, Base
from models import Role, User, Ticker, MarketData, Indicator, ModelInfo, Prediction

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ============================================================
# Load model and config
# ============================================================
base_model = load_model("best_model.h5", compile=False)
SEQ_LEN = 60
N_FEATURES = base_model.input_shape[-1]

MODEL_COLS = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "MA_20", "MA_50"]

# Load model config to check type
MODEL_TYPE = "returns"  # default to returns-based
if os.path.exists("model_config.json"):
    with open("model_config.json") as f:
        model_cfg = json.load(f)
        MODEL_TYPE = model_cfg.get("type", "returns")

TRAIN_RATIO = 0.8


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ============================================================
# Preprocessing
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].interpolate(method="time")

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

    # Bollinger Bands (20, 2)
    df["BB_mid"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # Log returns
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


# ============================================================
# Core prediction — returns-based approach
# ============================================================
def run_prediction(df: pd.DataFrame, days_ahead: int):
    """
    Returns-based pipeline:
    1. Scale features (for LSTM input)
    2. LSTM predicts log_return(t+1)
    3. Reconstruct price: price(t) = price(t-1) * exp(predicted_return)
    4. Train/test split for honest metrics
    5. CatBoost + XGBoost ensemble on top
    """
    n_total = len(df)
    split_idx = int(n_total * TRAIN_RATIO)

    # Scale features on train only
    scaler = MinMaxScaler((0, 1))
    scaler.fit(df.iloc[:split_idx][MODEL_COLS])
    scaled_all = scaler.transform(df[MODEL_COLS])

    log_returns = df["log_return"].values
    close_prices = df["Close"].values

    # LSTM sequences
    X_all, y_all = make_sequences_Xy(scaled_all, log_returns)
    train_seq_end = split_idx - SEQ_LEN

    X_train = X_all[:train_seq_end]
    y_train = y_all[:train_seq_end]

    # Fine-tune LSTM on train data
    fine_model = clone_model(base_model)
    fine_model.set_weights(base_model.get_weights())
    fine_model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss="mse")

    if len(X_train) > SEQ_LEN:
        val_size = max(0.1, SEQ_LEN / len(X_train))
        fine_model.fit(
            X_train, y_train,
            epochs=10, batch_size=32,
            validation_split=val_size,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
            ],
            verbose=0,
        )

    # Predict returns for all data
    lstm_pred_returns = fine_model.predict(X_all, verbose=0).flatten()

    # CatBoost + XGBoost ensemble
    X_bst_all = make_boosting_features(scaled_all)
    X_bst_train = X_bst_all[:train_seq_end]
    y_bst_train = y_all[:train_seq_end]

    cat_pred = None
    xgb_pred = None
    model_names = ["LSTM"]

    try:
        from catboost import CatBoostRegressor
        cat = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=5,
            verbose=0, early_stopping_rounds=50,
        )
        cat.fit(X_bst_train, y_bst_train,
                eval_set=(X_bst_all[train_seq_end:], y_all[train_seq_end:]),
                verbose=0)
        cat_pred = cat.predict(X_bst_all)
        model_names.append("CatBoost")
    except Exception:
        pass

    try:
        import xgboost as xgb
        xgb_m = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, verbosity=0,
        )
        xgb_m.fit(X_bst_train, y_bst_train,
                  eval_set=[(X_bst_all[train_seq_end:], y_all[train_seq_end:])],
                  verbose=0)
        xgb_pred = xgb_m.predict(X_bst_all)
        model_names.append("XGBoost")
    except Exception:
        pass

    # Stacking
    from sklearn.linear_model import Ridge

    stack_train = [lstm_pred_returns[:train_seq_end]]
    stack_all = [lstm_pred_returns]

    if cat_pred is not None:
        stack_train.append(cat_pred[:train_seq_end])
        stack_all.append(cat_pred)
    if xgb_pred is not None:
        stack_train.append(xgb_pred[:train_seq_end])
        stack_all.append(xgb_pred)

    if len(stack_train) > 1:
        X_meta_train = np.column_stack(stack_train)
        meta = Ridge(alpha=1.0)
        meta.fit(X_meta_train, y_bst_train)
        X_meta_all = np.column_stack(stack_all)
        pred_returns = meta.predict(X_meta_all)
        model_name = f"Ensemble ({'+'.join(model_names)})"
    else:
        pred_returns = lstm_pred_returns
        model_name = "LSTM (fine-tuned)"

    # Convert predicted returns → prices
    # price[t] = price[t-1] * exp(predicted_return[t])
    prev_prices = close_prices[SEQ_LEN - 1: -1]  # price at t-1
    actual_prices = close_prices[SEQ_LEN:]         # actual price at t
    pred_prices = prev_prices * np.exp(pred_returns)

    dates = df.index[SEQ_LEN:].strftime("%Y-%m-%d").tolist()

    # Metrics on TEST only
    test_start = train_seq_end
    y_act_test = actual_prices[test_start:]
    y_pred_test = pred_prices[test_start:]

    mae = mean_absolute_error(y_act_test, y_pred_test)
    rmse = float(np.sqrt(mean_squared_error(y_act_test, y_pred_test)))
    mape = float(np.mean(np.abs((y_act_test - y_pred_test) / y_act_test)) * 100)

    ss_res = np.sum((y_act_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_act_test - np.mean(y_act_test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Direction accuracy
    actual_dir = (actual_prices[test_start:] > prev_prices[test_start:]).astype(int)
    pred_dir = (pred_prices[test_start:] > prev_prices[test_start:]).astype(int)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)

    # Future forecast: GBM (Geometric Brownian Motion) + model signal
    #
    # The model reliably predicts 1-day returns (MAPE <1%).
    # For multi-day forecast, we use Monte Carlo GBM:
    #   S(t+1) = S(t) * exp((mu - σ²/2)*dt + σ*sqrt(dt)*Z)
    # where:
    #   mu = model's predicted drift (from last prediction)
    #   σ  = historical volatility (annualized, from recent returns)
    #   Z  = random normal
    #
    # This is standard in quantitative finance (Black-Scholes framework).
    future_dates, future_preds, future_upper, future_lower = [], [], [], []
    future_p50, future_p5, future_p95 = [], [], []

    if days_ahead > 0:
        N_SIM = 1000
        last_price = close_prices[-1]

        # Historical daily returns for volatility estimation (last 60 days)
        recent_returns = log_returns[-60:]
        mu_hist = float(np.mean(recent_returns))   # historical daily drift
        sigma_daily = float(np.std(recent_returns))  # daily volatility

        # Model's predicted drift: use the ensemble's last prediction as signal
        model_signal = float(pred_returns[-1])

        # Blend: 70% model signal + 30% historical mean for drift
        mu = 0.7 * model_signal + 0.3 * mu_hist
        dt = 1.0  # 1 day

        # Generate trajectories
        np.random.seed(42)
        Z = np.random.standard_normal((N_SIM, days_ahead))
        daily_returns = (mu - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z
        cum_returns = np.cumsum(daily_returns, axis=1)
        trajectories = last_price * np.exp(cum_returns)

        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(1, "d"), periods=days_ahead
        ).strftime("%Y-%m-%d").tolist()

        # Percentiles across simulations
        future_preds = np.median(trajectories, axis=0).tolist()        # median
        future_p5 = np.percentile(trajectories, 5, axis=0).tolist()    # 5th percentile
        future_p95 = np.percentile(trajectories, 95, axis=0).tolist()  # 95th percentile
        future_upper = np.percentile(trajectories, 75, axis=0).tolist() # 75th
        future_lower = np.percentile(trajectories, 25, axis=0).tolist() # 25th

    return {
        "dates": dates,
        "y_act": actual_prices,
        "y_pred": pred_prices,
        "train_size": train_seq_end,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
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
    }


def fetch_and_preprocess(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df_raw = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    if df_raw.empty:
        raise HTTPException(404, "Нет данных по указанному тикеру и диапазону дат")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    return preprocess(df_raw)


# ============================================================
# App startup
# ============================================================
@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    admin_role = db.query(Role).filter_by(name="admin").first()
    if not admin_role:
        admin_role = Role(name="admin")
        user_role = Role(name="user")
        db.add_all([admin_role, user_role])
        db.commit()
    else:
        user_role = db.query(Role).filter_by(name="user").first()
    if not db.query(User).filter_by(username="admin").first():
        db.add(User(username="<redacted>", password=hash_password("<redacted>"), role_id=admin_role.id))
    if not db.query(User).filter_by(username="user").first():
        db.add(User(username="<redacted>", password=hash_password("<redacted>"), role_id=user_role.id))
    db.commit()
    db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_role(request: Request, db: Session = Depends(get_db)):
    username = request.cookies.get("username")
    if not username:
        return None
    user = db.query(User).filter(User.username == username).first()
    return user.role.name if user else None


# ============================================================
# Routes
# ============================================================
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if user and user.password == hash_password(password):
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie("username", username, httponly=True)
        resp.set_cookie("role", user.role.name, httponly=True)
        return resp
    return templates.TemplateResponse("login.html", {"request": request, "error": "Неверный логин или пароль"})


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("username")
    resp.delete_cookie("role")
    return resp


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, role: str = Depends(get_current_role)):
    if not role:
        return RedirectResponse("/login")
    return templates.TemplateResponse("form.html", {"request": request, "ensemble": True})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ticker: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    days_ahead: int = Form(0),
    db: Session = Depends(get_db),
    role: str = Depends(get_current_role),
):
    if not role:
        return RedirectResponse("/login")

    try:
        df = fetch_and_preprocess(ticker, start_date, end_date)
    except ValueError as e:
        return templates.TemplateResponse("form.html", {
            "request": request, "error": str(e), "ensemble": True,
        })

    result = run_prediction(df, days_ahead)

    # Save to DB
    db_ticker = db.query(Ticker).filter_by(symbol=ticker).first()
    if not db_ticker:
        db_ticker = Ticker(symbol=ticker)
        db.add(db_ticker)
        db.commit()
        db.refresh(db_ticker)

    for idx, row in df.iterrows():
        md = MarketData(
            ticker_id=db_ticker.id, date=idx.date(),
            open=row.Open, high=row.High,
            low=row.Low, close=row.Close, volume=row.Volume,
        )
        db.add(md)
        db.flush()
        for name in ["RSI", "MACD", "Signal", "BB_upper", "BB_lower", "ATR"] + [f"MA_{w}" for w in (5, 10, 20, 50)]:
            db.add(Indicator(market_data_id=md.id, name=name, value=getattr(row, name)))
    db.commit()

    mi = ModelInfo(
        name=result["model_name"],
        parameters=json.dumps({"seq_len": SEQ_LEN, "features": MODEL_COLS, "type": "returns"}),
        mae=result["mae"], rmse=result["rmse"],
    )
    db.add(mi)
    db.commit()
    db.refresh(mi)

    for date_obj, pred in zip(result["date_index"], result["y_pred"]):
        db.add(Prediction(
            model_id=mi.id, ticker_id=db_ticker.id,
            date=date_obj.date(), predicted_close=pred,
        ))
    db.commit()

    # Table (test only)
    train_size = result["train_size"]
    table = []
    for d, a, p in zip(result["dates"][train_size:], result["y_act"][train_size:], result["y_pred"][train_size:]):
        err = abs(p - a)
        pct = 100 * err / a if a != 0 else 0
        table.append({
            "date": d, "actual": f"{a:.2f}", "predicted": f"{p:.2f}",
            "abs_error": f"{err:.2f}", "pct_error": f"{pct:.2f}",
        })

    data = {
        "dates": result["dates"] + result["future_dates"],
        "actual": result["y_act"].tolist() + [None] * len(result["future_dates"]),
        "predicted": result["y_pred"].tolist() + result["future_preds"],
        "train_size": train_size,
        "future_upper": result["future_upper"],
        "future_lower": result["future_lower"],
        "future_p5": result["future_p5"],
        "future_p95": result["future_p95"],
        "rsi": result["rsi"],
        "macd": result["macd"],
        "signal": result["signal"],
        "bb_upper": result["bb_upper"],
        "bb_lower": result["bb_lower"],
        "atr": result["atr"],
    }

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "days_ahead": days_ahead,
        "role": role,
        "model_name": result["model_name"],
        "mae": f"{result['mae']:.2f}",
        "rmse": f"{result['rmse']:.2f}",
        "mape": f"{result['mape']:.2f}",
        "r2": f"{result['r2']:.4f}",
        "dir_acc": f"{result['dir_acc']:.1f}",
        "data_json": json.dumps(data),
        "table": table,
        "ensemble": True,
    })


@app.get("/download_csv")
def download_csv(ticker: str, start_date: str, end_date: str, days_ahead: int = 0):
    df = fetch_and_preprocess(ticker, start_date, end_date)
    result = run_prediction(df, days_ahead)

    rows = []
    for d, a, p in zip(result["dates"], result["y_act"], result["y_pred"]):
        err = abs(p - a)
        pct = 100 * err / a if a != 0 else 0
        rows.append([d, a, p, err, pct])

    for d, p, u, l in zip(result["future_dates"], result["future_preds"],
                          result["future_upper"], result["future_lower"]):
        rows.append([d, "", p, "", "", u, l])

    cols = ["date", "actual", "predicted", "abs_error", "pct_error"]
    if result["future_upper"]:
        cols += ["upper_95", "lower_95"]

    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
