import io
import os
import json
import asyncio
import time
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf
import uvicorn

from fastapi import FastAPI, Request, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db import Base, engine, SessionLocal
from app.models import Role, User, Ticker, MarketData, Indicator, ModelInfo, Prediction, PredictionHistory
from app.prediction import (
    run_prediction, fetch_and_preprocess, SEQ_LEN, MODEL_COLS,
)
from app.sentiment import analyze_sentiment
from app.portfolio import optimize_portfolio
from app import telegram_bot

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Limit concurrent predictions to avoid OOM on VPS
MAX_CONCURRENT_PREDICTIONS = 2
_prediction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)
_prediction_queue_count = 0

# Simple TTL cache for predictions (key -> (result, timestamp))
_prediction_cache: dict[str, tuple] = {}
CACHE_TTL = 3600  # 1 hour

# Celery integration
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0")
USE_CELERY = os.getenv("USE_CELERY", "0") == "1"
celery_app = None
if USE_CELERY:
    try:
        from celery import Celery
        celery_app = Celery("neucast", broker=REDIS_URL, backend=REDIS_URL)
        celery_app.conf.update(
            task_serializer="pickle",
            result_serializer="pickle",
            accept_content=["pickle", "json"],
        )
    except ImportError:
        USE_CELERY = False


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


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

    # Celery mode: dispatch task and redirect to waiting page
    if USE_CELERY and celery_app:
        task = celery_app.send_task(
            "neucast.predict",
            args=[ticker, start_date, end_date, days_ahead],
            kwargs={"user_id": user.id if user else None},
            serializer="pickle",
        )
        return RedirectResponse(
            f"/predict/status/{task.id}?ticker={ticker}&start_date={start_date}&end_date={end_date}&days_ahead={days_ahead}",
            status_code=303,
        )

    # Fallback: synchronous mode with semaphore + cache
    cache_key = f"{ticker}:{start_date}:{end_date}:{days_ahead}"
    cached = _prediction_cache.get(cache_key)
    use_cache = False
    if cached:
        result, df, cached_at = cached
        if time.time() - cached_at < CACHE_TTL:
            use_cache = True

    if not use_cache:
        global _prediction_queue_count
        _prediction_queue_count += 1
        try:
            async with _prediction_semaphore:
                try:
                    loop = asyncio.get_event_loop()
                    df = await loop.run_in_executor(None, fetch_and_preprocess, ticker, start_date, end_date)
                except ValueError as e:
                    return templates.TemplateResponse("form.html", {
                        "request": request, "error": str(e), "ensemble": True,
                    })

                result = await loop.run_in_executor(None, run_prediction, df, days_ahead)
                _prediction_cache[cache_key] = (result, df, time.time())
                if len(_prediction_cache) > 20:
                    oldest = min(_prediction_cache, key=lambda k: _prediction_cache[k][2])
                    del _prediction_cache[oldest]
        finally:
            _prediction_queue_count -= 1

    # Sentiment analysis (run in executor, non-blocking)
    loop = asyncio.get_event_loop()
    try:
        sentiment = await loop.run_in_executor(None, analyze_sentiment, ticker)
        sentiment_data = {
            "avg_score": sentiment.avg_score,
            "positive_pct": sentiment.positive_pct,
            "negative_pct": sentiment.negative_pct,
            "neutral_pct": sentiment.neutral_pct,
            "total_articles": sentiment.total_articles,
            "signal": sentiment.signal,
            "signal_strength": sentiment.signal_strength,
            "news": [
                {
                    "title": n.title,
                    "source": n.source,
                    "published": n.published,
                    "url": n.url,
                    "sentiment": n.sentiment,
                    "score": n.score,
                    "sentiment_value": n.sentiment_value,
                }
                for n in sentiment.news
            ],
        }
    except Exception:
        sentiment_data = None

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
        "shap_data": result.get("shap_data", {}),
        "backtest": result.get("backtest"),
        "residuals": result["residuals"],
        "corr_data": result["corr_data"],
        "corr_labels": result["corr_labels"],
        "sentiment": sentiment_data,
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


# ── Celery task status endpoints ──

@app.get("/predict/status/{task_id}", response_class=HTMLResponse)
async def predict_status_page(
    task_id: str,
    request: Request,
    ticker: str = "",
    start_date: str = "",
    end_date: str = "",
    days_ahead: int = 0,
    role: str = Depends(get_current_role),
):
    if not role:
        return RedirectResponse("/login")
    return templates.TemplateResponse("waiting.html", {
        "request": request,
        "task_id": task_id,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "days_ahead": days_ahead,
    })


@app.get("/api/task/{task_id}")
async def task_status(task_id: str):
    if not USE_CELERY or not celery_app:
        return {"state": "FAILURE", "error": "Celery not configured"}

    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return {"state": "PENDING", "status": "В очереди..."}
    elif result.state == "FETCHING":
        return {"state": "FETCHING", "status": "Загрузка данных..."}
    elif result.state == "PREDICTING":
        return {"state": "PREDICTING", "status": "Расчёт прогноза..."}
    elif result.state == "SUCCESS":
        res = result.result
        if isinstance(res, dict) and "error" in res:
            return {"state": "FAILURE", "error": res["error"]}
        return {"state": "SUCCESS", "task_id": task_id}
    elif result.state == "FAILURE":
        return {"state": "FAILURE", "error": str(result.info)}
    else:
        return {"state": result.state, "status": "Обработка..."}


@app.websocket("/ws/task/{task_id}")
async def ws_task_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time task progress updates."""
    await websocket.accept()

    if not USE_CELERY or not celery_app:
        await websocket.send_json({"state": "FAILURE", "error": "Celery not configured"})
        await websocket.close()
        return

    from celery.result import AsyncResult

    prev_state = None
    try:
        while True:
            result = AsyncResult(task_id, app=celery_app)
            state = result.state

            # Only send update when state changes
            if state != prev_state:
                prev_state = state

                if state == "PENDING":
                    await websocket.send_json({"state": "PENDING", "status": "В очереди..."})
                elif state == "FETCHING":
                    await websocket.send_json({"state": "FETCHING", "status": "Загрузка данных..."})
                elif state == "PREDICTING":
                    await websocket.send_json({"state": "PREDICTING", "status": "Расчёт прогноза..."})
                elif state == "SUCCESS":
                    res = result.result
                    if isinstance(res, dict) and "error" in res:
                        await websocket.send_json({"state": "FAILURE", "error": res["error"]})
                    else:
                        await websocket.send_json({"state": "SUCCESS", "task_id": task_id})
                    await websocket.close()
                    return
                elif state == "FAILURE":
                    await websocket.send_json({"state": "FAILURE", "error": str(result.info)})
                    await websocket.close()
                    return
                else:
                    await websocket.send_json({"state": state, "status": "Обработка..."})

            await asyncio.sleep(0.5)  # Check every 500ms (much faster than polling)
    except WebSocketDisconnect:
        pass  # Client closed tab — that's fine
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/predict/result/{task_id}", response_class=HTMLResponse)
async def predict_result(
    task_id: str,
    request: Request,
    db: Session = Depends(get_db),
    role: str = Depends(get_current_role),
    user: User = Depends(get_current_user),
):
    if not role:
        return RedirectResponse("/login")
    if not USE_CELERY or not celery_app:
        return RedirectResponse("/dashboard")

    from celery.result import AsyncResult
    async_result = AsyncResult(task_id, app=celery_app)

    if async_result.state != "SUCCESS":
        return RedirectResponse("/dashboard")

    result = async_result.result
    if isinstance(result, dict) and "error" in result:
        return templates.TemplateResponse("form.html", {
            "request": request, "error": result["error"], "ensemble": True,
        })

    # Reconstruct df from JSON
    df = pd.read_json(io.StringIO(result.pop("_df_json")))

    # Reconstruct numpy arrays
    for key in ["y_act", "y_pred"]:
        if key in result and isinstance(result[key], list):
            result[key] = np.array(result[key])

    ticker = result.get("ticker", "")
    start_date = result.get("start_date", "")
    end_date = result.get("end_date", "")
    days_ahead = result.get("days_ahead", 0)

    # Sentiment analysis for celery result
    try:
        sentiment = analyze_sentiment(ticker)
        sentiment_data = {
            "avg_score": sentiment.avg_score,
            "positive_pct": sentiment.positive_pct,
            "negative_pct": sentiment.negative_pct,
            "neutral_pct": sentiment.neutral_pct,
            "total_articles": sentiment.total_articles,
            "signal": sentiment.signal,
            "signal_strength": sentiment.signal_strength,
            "news": [
                {
                    "title": n.title, "source": n.source,
                    "published": n.published, "url": n.url,
                    "sentiment": n.sentiment, "score": n.score,
                    "sentiment_value": n.sentiment_value,
                }
                for n in sentiment.news
            ],
        }
    except Exception:
        sentiment_data = None

    # Save to DB
    db_ticker = db.query(Ticker).filter_by(symbol=ticker).first()
    if not db_ticker:
        db_ticker = Ticker(symbol=ticker)
        db.add(db_ticker)
        db.commit()
        db.refresh(db_ticker)

    mi = ModelInfo(
        name=result["model_name"],
        parameters=json.dumps({"seq_len": SEQ_LEN, "features": MODEL_COLS, "type": "returns"}),
        mae=result["mae"], rmse=result["rmse"],
    )
    db.add(mi)
    db.commit()
    db.refresh(mi)

    train_size = result["train_size"]
    table = []
    dates = result["dates"]
    y_act = result["y_act"] if isinstance(result["y_act"], list) else result["y_act"].tolist()
    y_pred = result["y_pred"] if isinstance(result["y_pred"], list) else result["y_pred"].tolist()
    for d, a, p in zip(dates[train_size:], y_act[train_size:], y_pred[train_size:]):
        err = abs(p - a)
        pct = 100 * err / a if a != 0 else 0
        table.append({
            "date": d, "actual": f"{a:.2f}", "predicted": f"{p:.2f}",
            "abs_error": f"{err:.2f}", "pct_error": f"{pct:.2f}",
        })

    data = {
        "dates": dates + result["future_dates"],
        "actual": y_act + [None] * len(result["future_dates"]),
        "predicted": y_pred + result["future_preds"],
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
        "shap_data": result.get("shap_data", {}),
        "backtest": result.get("backtest"),
        "residuals": result["residuals"],
        "corr_data": result["corr_data"],
        "corr_labels": result["corr_labels"],
        "sentiment": sentiment_data,
    }

    saved_context = {
        "ticker": ticker, "start_date": start_date, "end_date": end_date,
        "days_ahead": days_ahead, "model_name": result["model_name"],
        "mae": f"{result['mae']:.2f}", "rmse": f"{result['rmse']:.2f}",
        "mape": f"{result['mape']:.2f}", "r2": f"{result['r2']:.4f}",
        "dir_acc": f"{result['dir_acc']:.1f}",
        "data": data, "table": table,
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
        "ticker": ticker, "start_date": start_date,
        "end_date": end_date, "days_ahead": days_ahead,
        "role": role, "model_name": result["model_name"],
        "mae": f"{result['mae']:.2f}", "rmse": f"{result['rmse']:.2f}",
        "mape": f"{result['mape']:.2f}", "r2": f"{result['r2']:.4f}",
        "dir_acc": f"{result['dir_acc']:.1f}",
        "data_json": json.dumps(data),
        "table": table, "ensemble": True,
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

    LOGO_PATH = os.path.join(BASE_DIR, "static", "logo.png")

    # --- Build PDF ---
    class StyledPDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 18, "F")
            self.set_fill_color(51, 65, 85)
            self.rect(0, 18, 210, 0.3, "F")
            self.image(LOGO_PATH, x=10, y=3, h=12)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(241, 245, 249)
            self.set_xy(24, 3)
            self.cell(40, 12, "NeuCast")
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(100, 116, 139)
            self.set_xy(150, 4)
            self.cell(50, 10, "Gold Market Analysis Report", align="R")
            self.ln(14)

        def footer(self):
            self.set_y(-13)
            self.set_fill_color(15, 23, 42)
            self.rect(0, self.get_y(), 210, 16, "F")
            self.set_fill_color(51, 65, 85)
            self.rect(0, self.get_y(), 210, 0.3, "F")
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


@app.get("/api/queue_status")
async def queue_status():
    return {
        "active": MAX_CONCURRENT_PREDICTIONS - _prediction_semaphore._value,
        "waiting": max(0, _prediction_queue_count - MAX_CONCURRENT_PREDICTIONS),
        "max_concurrent": MAX_CONCURRENT_PREDICTIONS,
    }


@app.get("/api/live_price")
async def live_price(ticker: str = "GC=F"):
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


# ── Portfolio optimization ──

@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_form(request: Request, role: str = Depends(get_current_role)):
    if not role:
        return RedirectResponse("/login")
    return templates.TemplateResponse("portfolio.html", {"request": request})


@app.post("/portfolio", response_class=HTMLResponse)
async def portfolio_optimize(
    request: Request,
    tickers: str = Form(...),
    budget: float = Form(10000),
    period: str = Form("1y"),
    role: str = Depends(get_current_role),
):
    if not role:
        return RedirectResponse("/login")

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, optimize_portfolio, ticker_list, budget, period
        )
    except ValueError as e:
        return templates.TemplateResponse("portfolio.html", {
            "request": request, "error": str(e),
            "tickers_str": tickers, "budget": budget, "period": period,
        })
    except Exception as e:
        return templates.TemplateResponse("portfolio.html", {
            "request": request, "error": f"Ошибка: {e}",
            "tickers_str": tickers, "budget": budget, "period": period,
        })

    import dataclasses
    result_dict = dataclasses.asdict(result)

    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "result": result,
        "result_json": json.dumps(result_dict),
        "tickers_str": tickers,
        "budget": budget,
        "period": period,
    })


# ── Telegram integration ──

@app.get("/api/telegram/link")
async def telegram_link(request: Request, user: User = Depends(get_current_user)):
    """Generate a Telegram link token for the current user."""
    if not user:
        return {"ok": False, "error": "Not authenticated"}
    if not telegram_bot.is_configured():
        return {"ok": False, "error": "Telegram bot not configured"}

    token = telegram_bot.generate_link_token(user.id)
    bot_username = os.getenv("TELEGRAM_BOT_USERNAME", "NeuCastBot")
    deep_link = f"https://t.me/{bot_username}?start={token}"
    return {"ok": True, "url": deep_link, "linked": telegram_bot.is_linked(user.id)}


@app.get("/api/telegram/status")
async def telegram_status(user: User = Depends(get_current_user)):
    if not user:
        return {"configured": False, "linked": False}
    return {
        "configured": telegram_bot.is_configured(),
        "linked": telegram_bot.is_linked(user.id),
    }


@app.post("/api/telegram/unlink")
async def telegram_unlink(user: User = Depends(get_current_user)):
    if user:
        telegram_bot.unlink(user.id)
    return {"ok": True}


@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request):
    """Incoming updates from Telegram Bot API webhook."""
    try:
        update = await request.json()
        telegram_bot.process_update(update)
    except Exception:
        pass
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
