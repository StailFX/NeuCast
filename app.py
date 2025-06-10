import io
import json
import pandas as pd
import numpy as np
import yfinance as yf
import uvicorn
import tensorflow as tf

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from db import DATABASE_URL, Base
from models import Role, User, Ticker, MarketData, Indicator, ModelInfo, Prediction

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = load_model("best_model.h5", compile=False)

model.compile(optimizer="adam", loss="mse")
SEQ_LEN = 60

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open","High","Low","Close","Volume"]].interpolate(method="time")
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["RSI"] = 100 - (100/(1 + avg_gain/avg_loss))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    for w in (5,10,20,50):
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()
    df = df.dropna()
    if len(df) < SEQ_LEN + 1:
        raise ValueError("Недостаточно данных после предобработки")
    return df

def make_sequences(arr: np.ndarray) -> np.ndarray:
    X = []
    for i in range(SEQ_LEN, len(arr)):
        X.append(arr[i-SEQ_LEN : i])
    return np.array(X)

def invert_scale(y_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    filler = np.zeros((len(y_scaled), scaler.scale_.shape[0]-1))
    arr = np.hstack([filler, y_scaled.reshape(-1,1)])
    return scaler.inverse_transform(arr)[:,-1]

def forecast_future(window: np.ndarray, days: int, scaler: MinMaxScaler, idx_close: int) -> np.ndarray:
    fut = []
    win = window.copy()
    for _ in range(days):
        p = model.predict(win[np.newaxis,:,:])[0,0]
        fut.append(p)
        row = win[-1].copy()
        row[idx_close] = p
        win = np.vstack([win[1:], row])
    return np.array(fut)


@app.on_event("startup")
def startup_event():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    # create roles
    admin_role = db.query(Role).filter_by(name="admin").first()
    if not admin_role:
        admin_role = Role(name="admin")
        user_role = Role(name="user")
        db.add_all([admin_role, user_role])
        db.commit()
    else:
        user_role = db.query(Role).filter_by(name="user").first()
    # create users
    if not db.query(User).filter_by(username="admin").first():
        db.add(User(username="admin", password="admin", role_id=admin_role.id))
    if not db.query(User).filter_by(username="user").first():
        db.add(User(username="user", password="user", role_id=user_role.id))
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


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_post(request: Request,
                     username: str = Form(...),
                     password: str = Form(...),
                     db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if user and user.password == password:
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie("username", username)
        resp.set_cookie("role", user.role.name)
        return resp
    return templates.TemplateResponse("login.html", {"request": request, "error": "Неверные данные"})

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
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  ticker: str = Form(...),
                  start_date: str = Form(...),
                  end_date: str = Form(...),
                  days_ahead: int = Form(0),
                  db: Session = Depends(get_db),
                  role: str = Depends(get_current_role)):
    if not role:
        return RedirectResponse("/login")

    df_raw = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    if df_raw.empty:
        raise HTTPException(404, "Нет данных")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df = preprocess(df_raw)

    db_ticker = db.query(Ticker).filter_by(symbol=ticker).first()
    if not db_ticker:
        db_ticker = Ticker(symbol=ticker)
        db.add(db_ticker); db.commit(); db.refresh(db_ticker)
    for idx, row in df.iterrows():
        md = MarketData(ticker_id=db_ticker.id,
                        date=idx.date(),
                        open=row.Open, high=row.High,
                        low=row.Low, close=row.Close,
                        volume=row.Volume)
        db.add(md); db.commit(); db.refresh(md)
        for name in ["RSI","MACD","Signal"] + [f"MA_{w}" for w in (5,10,20,50)]:
            db.add(Indicator(market_data_id=md.id, name=name, value=getattr(row, name)))
    db.commit()


    model_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_5", "MA_10", "MA_20", "MA_50",
]
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(df[model_cols])
    X = make_sequences(scaled)
    y_act = df["Close"].values[SEQ_LEN:]
    y_pred = invert_scale(model.predict(X).flatten(), scaler)
    dates = df.index[SEQ_LEN:].strftime("%Y-%m-%d").tolist()

    mae = mean_absolute_error(y_act, y_pred)
    rmse = np.sqrt(mean_squared_error(y_act, y_pred))

    mi = ModelInfo(name="LSTM_Gold", parameters=json.dumps({"seq_len":SEQ_LEN}), mae=mae, rmse=rmse)
    db.add(mi); db.commit(); db.refresh(mi)
    date_objs = df.index[SEQ_LEN:]  # это DatetimeIndex
    for date_obj, pred in zip(date_objs, y_pred):
        db.add(Prediction(
            model_id=mi.id,
            ticker_id=db_ticker.id,
            date=date_obj.date(),
            predicted_close=pred
        ))
    db.commit()

    idx_close = model_cols.index("Close")
    future_dates, future_preds = [], []
    if days_ahead > 0:
        fut_norm = forecast_future(scaled[-SEQ_LEN:], days_ahead, scaler, idx_close)
        future_dates = pd.bdate_range(start=df.index[-1]+pd.Timedelta(1, "d"),
                                      periods=days_ahead).strftime("%Y-%m-%d").tolist()
        future_preds = invert_scale(fut_norm, scaler).tolist()

    table = []
    for d,a,p in zip(dates, y_act, y_pred):
        err = abs(p - a)
        pct = 100 * err / a
        table.append({
            "date": d, "actual": f"{a:.2f}", "predicted": f"{p:.2f}",
            "abs_error": f"{err:.2f}", "pct_error": f"{pct:.2f}"
        })
    data = {
        "dates": dates + future_dates,
        "actual": y_act.tolist() + [None]*len(future_dates),
        "predicted": y_pred.tolist() + future_preds,
        "rsi": df["RSI"].values[SEQ_LEN:].round(2).tolist(),
        "macd": df["MACD"].values[SEQ_LEN:].round(2).tolist(),
        "signal": df["Signal"].values[SEQ_LEN:].round(2).tolist()
    }

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "days_ahead": days_ahead,
        "role": role,
        "mae": f"{mae:.6f}",
        "rmse": f"{rmse:.6f}",
        "data_json": json.dumps(data),
        "table": table
    })

@app.get("/download_csv")
def download_csv(ticker: str, start_date: str, end_date: str, days_ahead: int = 0):
    df_raw = yf.download(ticker, start=start_date, end=end_date,
                         interval="1d", progress=False)
    if df_raw.empty:
        raise HTTPException(404)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df = preprocess(df_raw)

    model_cols = ["Open","High","Low","Close","Volume"] + [f"MA_{w}" for w in (5,10,20,50)]
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(df[model_cols])
    X = make_sequences(scaled)
    y_act = df["Close"].values[SEQ_LEN:]
    y_pred = invert_scale(model.predict(X).flatten(), scaler)
    dates = df.index[SEQ_LEN:]

    rows = []
    for d,a,p in zip(dates, y_act, y_pred):
        err = abs(p - a)
        pct = 100 * err / a
        rows.append([d.strftime("%Y-%m-%d"), a, p, err, pct])

    if days_ahead > 0:
        fut_norm = forecast_future(scaled[-SEQ_LEN:], days_ahead, scaler,
                                   model_cols.index("Close"))
        fut_preds = invert_scale(fut_norm, scaler)
        fut_dates = pd.bdate_range(start=dates[-1]+pd.Timedelta(1, "d"),
                                   periods=days_ahead)
        for d,p in zip(fut_dates, fut_preds):
            rows.append([d.strftime("%Y-%m-%d"), "", p, "", ""])

    buf = io.StringIO()
    pd.DataFrame(rows, columns=["date","actual","predicted","abs_error","pct_error"])\
      .to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(buf,
                             media_type="text/csv",
                             headers={"Content-Disposition":"attachment; filename=predictions.csv"})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
