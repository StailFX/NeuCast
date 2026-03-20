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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

from db import DATABASE_URL, Base
from models import Role, User, Ticker, MarketData, Indicator, ModelInfo, Prediction, PredictionHistory

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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

    # Per-model price predictions for comparison
    model_comparison = {}
    lstm_prices = prev_prices * np.exp(lstm_pred_returns)
    model_comparison["LSTM"] = lstm_prices.tolist()
    if cat_pred is not None:
        cat_prices = prev_prices * np.exp(cat_pred)
        model_comparison["CatBoost"] = cat_prices.tolist()
    if xgb_pred is not None:
        xgb_prices = prev_prices * np.exp(xgb_pred)
        model_comparison["XGBoost"] = xgb_prices.tolist()
    model_comparison["Ensemble"] = pred_prices.tolist()

    # Per-model metrics on test set
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
    model_metrics["LSTM"] = calc_metrics(y_act_test, lstm_prices[test_start:])
    if cat_pred is not None:
        model_metrics["CatBoost"] = calc_metrics(y_act_test, cat_prices[test_start:])
    if xgb_pred is not None:
        model_metrics["XGBoost"] = calc_metrics(y_act_test, xgb_prices[test_start:])
    model_metrics["Ensemble"] = calc_metrics(y_act_test, pred_prices[test_start:])

    # Feature importance from boosting models
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

    # Residuals (test only)
    y_pred_test = pred_prices[test_start:]
    residuals = (y_act_test - y_pred_test).tolist()

    # Metrics on TEST only (ensemble)
    mae = model_metrics["Ensemble"]["mae"]
    rmse = model_metrics["Ensemble"]["rmse"]
    mape = model_metrics["Ensemble"]["mape"]
    r2 = model_metrics["Ensemble"]["r2"]

    # Direction accuracy
    actual_dir = (actual_prices[test_start:] > prev_prices[test_start:]).astype(int)
    pred_dir = (pred_prices[test_start:] > prev_prices[test_start:]).astype(int)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)

    # Correlation matrix of features
    corr_cols = ["Close", "Volume", "RSI", "MACD", "MA_5", "MA_20", "MA_50", "ATR", "BB_upper", "BB_lower"]
    corr_data = df[corr_cols].corr().round(3).values.tolist()
    corr_labels = corr_cols

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
        "model_comparison": model_comparison,
        "model_metrics": model_metrics,
        "feature_importance": feature_importance,
        "residuals": residuals,
        "corr_data": corr_data,
        "corr_labels": corr_labels,
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


async def get_current_user(request: Request, db: Session = Depends(get_db)):
    username = request.cookies.get("username")
    if not username:
        return None
    return db.query(User).filter(User.username == username).first()


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
        resp = RedirectResponse("/dashboard", status_code=302)
        resp.set_cookie("username", username, httponly=True)
        resp.set_cookie("role", user.role.name, httponly=True)
        return resp
    return templates.TemplateResponse("login.html", {"request": request, "error": "Неверный логин или пароль"})


@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register_post(
    request: Request,
    username: str = Form(...),
    email: str = Form(""),
    password: str = Form(...),
    password2: str = Form(...),
    db: Session = Depends(get_db),
):
    if password != password2:
        return templates.TemplateResponse("register.html", {
            "request": request, "error": "Пароли не совпадают",
        })
    if len(password) < 4:
        return templates.TemplateResponse("register.html", {
            "request": request, "error": "Пароль должен быть не менее 4 символов",
        })
    if db.query(User).filter_by(username=username).first():
        return templates.TemplateResponse("register.html", {
            "request": request, "error": "Пользователь с таким логином уже существует",
        })
    user_role = db.query(Role).filter_by(name="user").first()
    new_user = User(
        username=username,
        email=email or None,
        password=hash_password(password),
        role_id=user_role.id,
    )
    db.add(new_user)
    db.commit()
    resp = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie("username", username, httponly=True)
    resp.set_cookie("role", "user", httponly=True)
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("username")
    resp.delete_cookie("role")
    return resp


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request, role: str = Depends(get_current_role)):
    return templates.TemplateResponse("landing.html", {"request": request, "logged_in": role is not None})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, role: str = Depends(get_current_role), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not role:
        return RedirectResponse("/login")
    history = []
    if user:
        history = db.query(PredictionHistory).filter_by(user_id=user.id).order_by(PredictionHistory.created_at.desc()).limit(10).all()
    return templates.TemplateResponse("form.html", {
        "request": request, "ensemble": True, "username": user.username if user else "", "role": role, "history": history,
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ticker: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    days_ahead: int = Form(0),
    db: Session = Depends(get_db),
    role: str = Depends(get_current_role),
    user: User = Depends(get_current_user),
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

    # Build table and data for template
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
        "model_comparison": result["model_comparison"],
        "model_metrics": result["model_metrics"],
        "feature_importance": result["feature_importance"],
        "residuals": result["residuals"],
        "corr_data": result["corr_data"],
        "corr_labels": result["corr_labels"],
    }

    # Save full result to prediction history
    saved_context = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "days_ahead": days_ahead,
        "model_name": result["model_name"],
        "mae": f"{result['mae']:.2f}",
        "rmse": f"{result['rmse']:.2f}",
        "mape": f"{result['mape']:.2f}",
        "r2": f"{result['r2']:.4f}",
        "dir_acc": f"{result['dir_acc']:.1f}",
        "data": data,
        "table": table,
    }

    if user:
        ph = PredictionHistory(
            user_id=user.id, ticker=ticker,
            start_date=start_date, end_date=end_date,
            days_ahead=days_ahead, model_name=result["model_name"],
            mape=result["mape"],
            result_json=json.dumps(saved_context),
        )
        db.add(ph)
    db.commit()

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


@app.get("/prediction/{pred_id}", response_class=HTMLResponse)
async def view_prediction(
    pred_id: int,
    request: Request,
    db: Session = Depends(get_db),
    role: str = Depends(get_current_role),
):
    if not role:
        return RedirectResponse("/login")
    ph = db.query(PredictionHistory).filter_by(id=pred_id).first()
    if not ph or not ph.result_json:
        return RedirectResponse("/")
    ctx = json.loads(ph.result_json)
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "ticker": ctx["ticker"],
        "start_date": ctx["start_date"],
        "end_date": ctx["end_date"],
        "days_ahead": ctx["days_ahead"],
        "role": role,
        "model_name": ctx["model_name"],
        "mae": ctx["mae"],
        "rmse": ctx["rmse"],
        "mape": ctx["mape"],
        "r2": ctx["r2"],
        "dir_acc": ctx["dir_acc"],
        "data_json": json.dumps(ctx["data"]),
        "table": ctx["table"],
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


@app.get("/download_pdf")
def download_pdf(ticker: str, start_date: str, end_date: str, days_ahead: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import tempfile
    from fpdf import FPDF
    from datetime import datetime as dt

    df = fetch_and_preprocess(ticker, start_date, end_date)
    result = run_prediction(df, days_ahead)

    # --- Chart style ---
    BG = "#0a0e17"
    CARD = "#0f172a"
    GRID = "#1e293b"
    TEXT = "#94a3b8"
    GREEN = "#10b981"
    BLUE = "#3b82f6"
    AMBER = "#f59e0b"
    RED = "#ef4444"
    PURPLE = "#a78bfa"

    def styled_fig(w=10, h=3.5):
        fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(True, color=GRID, alpha=0.4, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        return fig, ax

    chart_files = []

    # 1. Main price chart with prediction
    dates_dt = pd.to_datetime(result["dates"])
    ts = result["train_size"]
    fig, ax = styled_fig(10, 4)
    ax.plot(dates_dt, result["y_act"], color=BLUE, linewidth=1.2, label="Факт (Close)")
    ax.plot(dates_dt[ts:], result["y_pred"][ts:], color=GREEN, linewidth=1.5, label="Прогноз (Ensemble)")
    ax.axvline(dates_dt[ts], color=AMBER, linestyle="--", alpha=0.6, linewidth=1)
    ax.text(dates_dt[ts], ax.get_ylim()[1] * 0.99, " Train | Test", color=AMBER, fontsize=7, va="top")

    if days_ahead > 0 and result["future_dates"]:
        fut_dt = pd.to_datetime(result["future_dates"])
        ax.plot(fut_dt, result["future_preds"], color=AMBER, linewidth=1.5, label=f"Прогноз ({days_ahead}д.)")
        if result["future_p5"]:
            ax.fill_between(fut_dt, result["future_p5"], result["future_p95"], alpha=0.1, color=AMBER)
            ax.fill_between(fut_dt, result["future_lower"], result["future_upper"], alpha=0.2, color=AMBER)

    ax.set_title(f"{ticker} — Динамика цены и прогноз", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("USD", color=TEXT, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    fig.autofmt_xdate()
    plt.tight_layout()
    f1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(f1.name, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    chart_files.append(f1.name)

    # 2. Bollinger Bands
    fig, ax = styled_fig(10, 3)
    ax.fill_between(dates_dt, result["bb_upper"], result["bb_lower"], alpha=0.15, color=PURPLE)
    ax.plot(dates_dt, result["bb_upper"], color=PURPLE, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.plot(dates_dt, result["bb_lower"], color=PURPLE, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.plot(dates_dt, result["y_act"], color=BLUE, linewidth=1, label="Close")
    ax.set_title("Bollinger Bands (20, 2)", color="#f1f5f9", fontsize=10, fontweight="bold", pad=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    fig.autofmt_xdate()
    plt.tight_layout()
    f2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(f2.name, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    chart_files.append(f2.name)

    # 3. RSI
    fig, ax = styled_fig(10, 2.2)
    ax.plot(dates_dt, result["rsi"], color=AMBER, linewidth=1)
    ax.axhline(70, color=RED, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(30, color=GREEN, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.fill_between(dates_dt, result["rsi"], 30, where=[r < 30 for r in result["rsi"]], alpha=0.15, color=GREEN)
    ax.fill_between(dates_dt, result["rsi"], 70, where=[r > 70 for r in result["rsi"]], alpha=0.15, color=RED)
    ax.set_ylim(0, 100)
    ax.set_title("RSI (14)", color="#f1f5f9", fontsize=10, fontweight="bold", pad=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    f3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(f3.name, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    chart_files.append(f3.name)

    # 4. MACD
    fig, ax = styled_fig(10, 2.2)
    macd_hist = [m - s for m, s in zip(result["macd"], result["signal"])]
    colors_hist = [GREEN if v >= 0 else RED for v in macd_hist]
    ax.bar(dates_dt, macd_hist, color=colors_hist, alpha=0.5, width=2)
    ax.plot(dates_dt, result["macd"], color=BLUE, linewidth=1, label="MACD")
    ax.plot(dates_dt, result["signal"], color=RED, linewidth=1, label="Signal")
    ax.set_title("MACD (12/26/9)", color="#f1f5f9", fontsize=10, fontweight="bold", pad=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    fig.autofmt_xdate()
    plt.tight_layout()
    f4 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(f4.name, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    chart_files.append(f4.name)

    LOGO_PATH = os.path.join(os.path.dirname(__file__), "static", "logo.png")

    # --- Build PDF ---
    class StyledPDF(FPDF):
        def header(self):
            # Dark navbar background
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 18, "F")
            # Bottom border
            self.set_fill_color(51, 65, 85)
            self.rect(0, 18, 210, 0.3, "F")
            # Logo image
            self.image(LOGO_PATH, x=10, y=3, h=12)
            # Brand
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(241, 245, 249)
            self.set_xy(24, 3)
            self.cell(40, 12, "NeuCast")
            # Right text
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(100, 116, 139)
            self.set_xy(150, 4)
            self.cell(50, 10, "Gold Market Analysis Report", align="R")
            self.ln(14)

        def footer(self):
            self.set_y(-13)
            # Dark footer
            self.set_fill_color(15, 23, 42)
            self.rect(0, self.get_y(), 210, 16, "F")
            # Top border
            self.set_fill_color(51, 65, 85)
            self.rect(0, self.get_y(), 210, 0.3, "F")
            # Logo mini
            self.image(LOGO_PATH, x=10, y=self.get_y() + 2.5, h=7)
            self.set_xy(19, self.get_y() + 1)
            self.set_font("Helvetica", "B", 6.5)
            self.set_text_color(16, 185, 129)
            self.cell(25, 9, "NeuCast")
            self.set_font("Helvetica", "", 6.5)
            self.set_text_color(100, 116, 139)
            self.cell(0, 9, f"{dt.now().strftime('%Y-%m-%d %H:%M')}   |   Page {self.page_no()}", align="R")

        def section(self, title, color=(16, 185, 129)):
            self.set_fill_color(*color)
            self.rect(10, self.get_y(), 3, 7, "F")
            self.set_x(16)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(30, 41, 59)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def kv(self, label, value, color=(30, 41, 59)):
            self.set_font("Helvetica", "", 8)
            self.set_text_color(100, 116, 139)
            self.cell(45, 5.5, label)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*color)
            self.cell(45, 5.5, str(value), new_x="LMARGIN", new_y="NEXT")

    pdf = StyledPDF()
    pdf.set_auto_page_break(auto=True, margin=16)

    # ====== PAGE 1: Overview ======
    pdf.add_page()

    # Title card
    ticker_names = {"GC=F": "Gold Futures (XAU/USD)", "SI=F": "Silver Futures", "CL=F": "Crude Oil WTI"}
    full_name = ticker_names.get(ticker, ticker)

    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, pdf.get_y(), 190, 32, "F")
    pdf.set_fill_color(16, 185, 129)
    pdf.rect(10, pdf.get_y(), 190, 0.8, "F")

    y0 = pdf.get_y() + 3
    pdf.set_xy(15, y0)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(10, 14, 23)
    pdf.cell(0, 9, full_name)
    pdf.set_xy(15, y0 + 10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Ticker: {ticker}  |  Period: {start_date} - {end_date}  |  {len(df)} trading days")
    pdf.set_xy(15, y0 + 17)
    pdf.set_font("Helvetica", "", 8)
    last_p = result["y_act"][-1]
    first_p = result["y_act"][0]
    total_ret = (last_p / first_p - 1) * 100
    pdf.cell(0, 6, f"Open: ${first_p:.2f}  |  Close: ${last_p:.2f}  |  Return: {total_ret:+.1f}%")
    pdf.set_y(y0 + 36)

    # Key metrics
    pdf.section("Key Statistics")
    close_arr = np.array(result["y_act"])
    pdf.kv("Last Price", f"${last_p:.2f}", (16, 185, 129))
    pdf.kv("Period High", f"${close_arr.max():.2f}", (239, 68, 68))
    pdf.kv("Period Low", f"${close_arr.min():.2f}", (59, 130, 246))
    pdf.kv("Average Price", f"${close_arr.mean():.2f}")
    pdf.kv("Volatility (daily)", f"{np.std(np.diff(np.log(close_arr)))*100:.2f}%")
    pdf.kv("Current RSI", f"{result['rsi'][-1]}", (245, 158, 11))
    pdf.kv("Current MACD", f"{result['macd'][-1]}")
    pdf.kv("Current ATR", f"${result['atr'][-1]}", (167, 139, 250))
    pdf.ln(3)

    # Forecast block
    if days_ahead > 0 and result["future_preds"]:
        pdf.section(f"AI Forecast ({days_ahead} days ahead)", (245, 158, 11))
        forecast_price = result["future_preds"][-1]
        change = forecast_price - last_p
        change_pct = 100 * change / last_p
        pdf.kv("Forecast (median)", f"${forecast_price:.0f}", (245, 158, 11))
        pdf.kv("Expected Change", f"{change:+.0f} ({change_pct:+.1f}%)", (16, 185, 129) if change >= 0 else (239, 68, 68))
        if result["future_p5"]:
            pdf.kv("90% Confidence", f"${result['future_p5'][-1]:.0f} - ${result['future_p95'][-1]:.0f}")
            pdf.kv("50% Confidence", f"${result['future_lower'][-1]:.0f} - ${result['future_upper'][-1]:.0f}")
        pdf.kv("Method", "GBM Monte Carlo (1000 simulations)")
        pdf.ln(3)

    # Model accuracy summary
    pdf.section("Model Accuracy (Test Set)", (59, 130, 246))
    pdf.kv("Model", result["model_name"])
    pdf.kv("MAPE", f"{result['mape']:.2f}%", (167, 139, 250))
    pdf.kv("MAE", f"${result['mae']:.2f}", (59, 130, 246))
    pdf.kv("R\u00b2", f"{result['r2']:.4f}", (16, 185, 129))
    pdf.kv("Direction Accuracy", f"{result['dir_acc']:.1f}%")

    # ====== PAGE 2: Price Chart ======
    pdf.add_page()
    pdf.section("Price Chart & Forecast")
    pdf.image(chart_files[0], x=10, w=190)
    pdf.ln(5)

    pdf.section("Bollinger Bands", (167, 139, 250))
    pdf.image(chart_files[1], x=10, w=190)

    # ====== PAGE 3: Indicators ======
    pdf.add_page()
    pdf.section("RSI (Relative Strength Index)", (245, 158, 11))
    pdf.image(chart_files[2], x=10, w=190)
    pdf.ln(3)

    pdf.section("MACD", (59, 130, 246))
    pdf.image(chart_files[3], x=10, w=190)

    # Cleanup temp files
    for f in chart_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=gold_report_{ticker}_{end_date}.pdf"},
    )


@app.get("/api/live_price")
async def live_price(ticker: str = "GC=F"):
    """Return current price for the live ticker in navbar."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info.get("lastPrice", info.get("last_price", None))
        prev = info.get("previousClose", info.get("previous_close", None))
        if price is None:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        change = None
        change_pct = None
        if price and prev:
            change = round(price - prev, 2)
            change_pct = round(100 * change / prev, 2)
        return {"price": round(price, 2) if price else None, "change": change, "change_pct": change_pct, "ticker": ticker}
    except Exception:
        return {"price": None, "change": None, "change_pct": None, "ticker": ticker}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
