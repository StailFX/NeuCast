import io
import os
import json
import asyncio
import logging
import time
import hashlib
import secrets

import numpy as np
import pandas as pd
import yfinance as yf
import uvicorn

logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, PlainTextResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from sqlalchemy.orm import Session

from app.db import Base, engine, SessionLocal
from app.models import Role, User, Ticker, MarketData, Indicator, ModelInfo, Prediction, PredictionHistory
from app.prediction import (
    run_prediction, fetch_and_preprocess, apply_sentiment_bias, SEQ_LEN, MODEL_COLS,
)
from app.sentiment import analyze_sentiment
from app.portfolio import optimize_portfolio
from app import telegram_bot
from app.user_errors import friendly_error
from app.highfreq.web import router as highfreq_router

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

# ── gzip compression на весь трафик (HTML/JSON/JS/CSS). ──
# minimum_size=500: мелкие ответы не сжимаем, оверхед больше выгоды.
# compresslevel=6: хороший баланс CPU/ratio.
app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)


# ── Cache-Control для статики (работает и когда nginx не перед нами). ──
class CacheControlStaticMiddleware:
    """Добавляет immutable-кэш к /static/ для снижения повторных запросов."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/static/"):
            await self.app(scope, receive, send)
            return

        async def send_with_cache(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Заменяем/добавляем Cache-Control
                headers = [(k, v) for (k, v) in headers if k.lower() != b"cache-control"]
                headers.append((b"cache-control", b"public, max-age=31536000, immutable"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_cache)


app.add_middleware(CacheControlStaticMiddleware)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── P1: Jinja2 bytecode cache ──
# Шаблоны парсятся один раз и сохраняются в /tmp/jinja-cache; на повторных рендерах
# Jinja грузит готовый байт-код, пропуская лексер/парсер. ~20–30% быстрее.
try:
    from jinja2 import FileSystemBytecodeCache
    _jinja_cache_dir = os.getenv("JINJA_BYTECODE_CACHE", "/tmp/neucast-jinja-cache")
    os.makedirs(_jinja_cache_dir, exist_ok=True)
    templates.env.bytecode_cache = FileSystemBytecodeCache(_jinja_cache_dir)
    templates.env.auto_reload = False  # prod: не перепарсивать при каждом запросе
except Exception:
    pass

# ── High-frequency UI (Phase A.6) ──
# Routes: /highfreq, /api/highfreq/status, /api/highfreq/health.
# Read-only thin observer over the L2 ingest tables — see app/highfreq/web.py.
# Renders gracefully ("no data yet") even when the ingest service is down.
app.include_router(highfreq_router)


# ── Custom error pages ──
# До этого FastAPI отдавал raw JSON {"detail": "Not Found"} даже на запрос
# из браузера на несуществующую страницу. Теперь:
#   * для /api/* и Accept: application/json → JSON (как было)
#   * для всего остального → HTML страница с шапкой NeuCast и ссылкой на главную
#
# Покрываются: 404 (страница/роут не найдены), 403 (доступ закрыт),
# 422 (валидация формы), 500 (внутренние ошибки) — единый templates/error.html.

# Подсказки на русском по статус-кодам — то, что пользователь увидит на странице.
# Без жаргона: «не найдено» лучше чем «Resource Not Found».
_ERROR_MESSAGES: dict[int, tuple[str, str]] = {
    400: (
        "Некорректный запрос",
        "Запрос не удалось разобрать. Попробуйте обновить страницу или вернуться на главную.",
    ),
    403: (
        "Доступ закрыт",
        "У вас нет прав на просмотр этой страницы. Если это ошибка — войдите заново.",
    ),
    404: (
        "Страница не найдена",
        "Такой страницы нет. Возможно, ссылка устарела или вы попали сюда по ошибке.",
    ),
    405: (
        "Метод не разрешён",
        "Этот URL не отвечает на такой тип запроса. Откройте страницу через обычную ссылку.",
    ),
    422: (
        "Неверный запрос",
        "Не удалось обработать данные формы. Проверьте поля и попробуйте ещё раз.",
    ),
    429: (
        "Слишком много запросов",
        "Подождите минуту и попробуйте снова — мы временно ограничили частоту запросов.",
    ),
    500: (
        "Что-то пошло не так",
        "На стороне сервера произошла ошибка. Мы уже разбираемся — попробуйте через минуту.",
    ),
    502: (
        "Сервис временно недоступен",
        "Сервер не отвечает. Попробуйте обновить страницу через минуту.",
    ),
    503: (
        "Сервис временно недоступен",
        "Мы перегружены или обновляемся. Попробуйте через минуту.",
    ),
}


def _wants_json_response(request: Request) -> bool:
    """Решает, что отдавать клиенту: JSON или HTML.

    Правило простое и предсказуемое:
      * если путь начинается с /api/ → JSON (это API-консьюмеры — curl, JS-фетчи)
      * если в Accept-заголовке есть application/json и нет text/html → JSON
      * иначе → HTML (браузер)
    """
    if request.url.path.startswith("/api/"):
        return True
    accept = (request.headers.get("accept") or "").lower()
    if "application/json" in accept and "text/html" not in accept:
        return True
    return False


def _render_error_page(
    request: Request, status_code: int, detail: str | None = None
) -> Response:
    """Единая точка ответа на ошибку (HTML или JSON).

    Никогда не raise'ит сама — иначе попадём в бесконечный exception loop.
    """
    title, default_description = _ERROR_MESSAGES.get(
        status_code,
        ("Ошибка", "Что-то пошло не так."),
    )
    description = detail or default_description

    if _wants_json_response(request):
        return JSONResponse(
            status_code=status_code,
            content={"detail": description, "status_code": status_code},
        )

    try:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "status_code": status_code,
                "title": title,
                "description": description,
                "request_path": request.url.path,
                "request_method": request.method,
            },
            status_code=status_code,
        )
    except Exception:
        # Шаблон сломан / templates dir не примонтирован — отдадим сырой HTML
        # как последнюю линию защиты, чтобы пользователь хоть что-то увидел.
        logger.exception("error.html render failed")
        return HTMLResponse(
            status_code=status_code,
            content=(
                f"<!doctype html><meta charset='utf-8'>"
                f"<title>{status_code} {title}</title>"
                f"<h1>{status_code} {title}</h1><p>{description}</p>"
                f"<p><a href='/'>На главную</a></p>"
            ),
        )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Если detail — стандартное «Not Found» / «Forbidden», подменим на наш
    # человеческий текст из _ERROR_MESSAGES. Если разработчик кинул
    # HTTPException(detail="свой текст") — оставим его detail как есть.
    detail = exc.detail
    if isinstance(detail, str) and detail in {
        "Not Found", "Forbidden", "Method Not Allowed",
        "Internal Server Error", "Bad Request",
    }:
        detail = None
    return _render_error_page(request, exc.status_code, detail)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    # Для /api/* отдаём детали ошибок (как было по умолчанию у FastAPI)
    # — это полезно при разработке клиента. Для HTML-страниц прячем за
    # generic «проверьте поля и попробуйте ещё раз».
    if _wants_json_response(request):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "status_code": 422},
        )
    return _render_error_page(request, 422)


@app.exception_handler(Exception)
async def _generic_exception_handler(request: Request, exc: Exception):
    # Полный traceback ушёл в логи (uvicorn + journalctl), пользователь
    # видит только friendly сообщение. См. user_errors.py за philosophy:
    # внутренности Python — в логи, не в UI.
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return _render_error_page(request, 500)


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
        # Code-review C-1 (2026-05-04): switched away from pickle to JSON
        # serializer for both task args and results. Pickle-as-broker-format
        # is an RCE seed: any compromise of Redis (network, sibling-tenant,
        # accidental exposure) means an attacker who can write to the
        # broker queue gets arbitrary Python execution on every worker that
        # consumes the poisoned task.
        #
        # All current task signatures (run_prediction_task etc.) use plain
        # str/int/bool/float args and JSON-serialisable result dicts (see
        # celery_worker.py:308-323 where numpy arrays are .tolist()'d before
        # caching). The transition is safe end-to-end.
        celery_app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
        )
    except ImportError:
        USE_CELERY = False


# ── Короткие URL для прогнозов: /p/{slug} вместо /predict/status/{UUID}?... ──
# Slug → {task_id, ticker, start_date, end_date, days_ahead} в Redis с TTL 2ч.
# Redis уже используется как Celery broker, поэтому нового инстанса не нужно.
PRED_SLUG_TTL = 7200  # 2 часа — достаточно чтобы дойти от submit до result
_SLUG_ALPHABET = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # без 0/O/1/l/I
_redis_client = None


def _get_redis():
    """Ленивая инициализация redis-клиента. Возвращает None если недоступен."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.from_url(
                REDIS_URL, decode_responses=True,
                socket_timeout=1, socket_connect_timeout=1,
            )
            _redis_client.ping()
        except Exception:
            _redis_client = False  # не пробуем повторно
    return _redis_client if _redis_client else None


def _make_slug(n: int = 6) -> str:
    """6-символьный slug (56^6 ≈ 31 млрд комбинаций, коллизий не будет)."""
    return "".join(secrets.choice(_SLUG_ALPHABET) for _ in range(n))


def _save_pred_slug(slug: str, meta: dict) -> bool:
    r = _get_redis()
    if not r:
        return False
    try:
        r.setex(f"pred_slug:{slug}", PRED_SLUG_TTL, json.dumps(meta))
        return True
    except Exception:
        return False


def _load_pred_slug(slug: str) -> dict | None:
    r = _get_redis()
    if not r:
        return None
    try:
        raw = r.get(f"pred_slug:{slug}")
        return json.loads(raw) if raw else None
    except Exception:
        return None


# ============================================================
# Password hashing — Argon2id KDF (code-review C-2, 2026-05-04)
# ============================================================
# Migrated from unsalted SHA-256 (one-shot, GPU-fast, identical-passwords-
# share-hashes) to Argon2id (memory-hard, salted per-row, configurable work
# factor). Legacy SHA-256 hashes remain readable: ``verify_password`` accepts
# either format, and on a successful login we transparently re-hash to Argon2
# (so the migration completes user-by-user during normal traffic, no DB-wide
# bulk update).
#
# Security knobs follow OWASP 2024 minimum (memory_cost=64MiB, time_cost=3,
# parallelism=4). At login latency this is ≈40 ms — imperceptible for
# humans, but ≥10⁵× more expensive for offline attackers than SHA-256.
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, InvalidHashError
    _PASSWORD_HASHER = PasswordHasher(
        memory_cost=64 * 1024,  # 64 MiB
        time_cost=3,
        parallelism=4,
    )
    _ARGON2_AVAILABLE = True
except ImportError:
    # Graceful fallback for environments without argon2-cffi (e.g. CI bare
    # bones). Logs a loud warning so operators notice. New passwords still
    # land as SHA-256 in this branch — same security posture as before the
    # migration, never worse.
    _PASSWORD_HASHER = None
    _ARGON2_AVAILABLE = False
    import logging as _logging
    _logging.getLogger("neucast.security").warning(
        "argon2-cffi not installed — falling back to legacy SHA-256 password "
        "hashing. Install argon2-cffi to enable the upgraded KDF."
    )


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage.

    NEW callers (registration, password change) get Argon2id when available,
    SHA-256 only as a graceful-fallback when argon2-cffi is missing. Existing
    rows in the DB may still be SHA-256 — ``verify_password`` handles both.
    """
    if _ARGON2_AVAILABLE:
        return _PASSWORD_HASHER.hash(password)
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plaintext: str, stored: str) -> bool:
    """Constant-time password verification.

    Recognises both Argon2id hashes (``$argon2id$…``) and the legacy 64-char
    SHA-256 hex digests. Constant-time comparison via ``secrets.compare_digest``
    closes the timing-oracle gap (code-review C-3, 2026-05-04).
    """
    if not stored:
        return False
    # Argon2id format → starts with $argon2id$.  Argon2 lib does its own
    # constant-time comparison internally.
    if _ARGON2_AVAILABLE and stored.startswith("$argon2"):
        try:
            _PASSWORD_HASHER.verify(stored, plaintext)
            return True
        except (VerifyMismatchError, InvalidHashError):
            return False
        except Exception:
            # Any other argon2 internal error → fail closed.
            return False
    # Legacy SHA-256 path. Use compare_digest so attackers can't mount a
    # timing oracle on the equality check (was ``==`` previously).
    legacy_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return secrets.compare_digest(stored, legacy_hash)


def needs_rehash(stored: str) -> bool:
    """Should we re-hash this user's password on next successful login?

    True for any non-Argon2 stored hash (i.e. legacy SHA-256). Argon2 hashes
    that need re-hashing because of parameter upgrades are also flagged via
    the library's check_needs_rehash helper.
    """
    if not stored:
        return False
    if not stored.startswith("$argon2"):
        return True
    if _ARGON2_AVAILABLE:
        try:
            return _PASSWORD_HASHER.check_needs_rehash(stored)
        except Exception:
            return False
    return False


# ============================================================
# App startup
# ============================================================
@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if not db.query(Role).filter_by(name="admin").first():
        db.add_all([Role(name="admin"), Role(name="user")])
        db.commit()
    db.close()

    # ── Прогрев TCN-модели: первый predict больше не ждёт загрузки h5. ──
    # Делаем в фоновой таске, чтобы startup не блокировался на 3–4 секунды.
    # Если USE_CELERY=1, основная нагрузка на Celery-воркере — TCN там тоже
    # прогреется при первом запросе через лениво-ленивую _get_model().
    def _warmup():
        try:
            from app.prediction import _get_model
            _get_model()
        except Exception as e:
            import logging
            logging.getLogger("neucast").warning("TCN warmup failed: %s", e)

    import threading
    threading.Thread(target=_warmup, daemon=True, name="tcn-warmup").start()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("session")
    if not token:
        return None
    return db.query(User).filter(User.session_token == token).first()


async def get_current_role(request: Request, db: Session = Depends(get_db)):
    user = await get_current_user(request, db)
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
    # ``verify_password`` does constant-time comparison + accepts both
    # Argon2id and legacy SHA-256 stored hashes (code-review C-2 + C-3,
    # 2026-05-04). Even when ``user`` is None we still hit ``verify_password``
    # against a dummy hash so the response time leaks no info about whether
    # the username exists (rudimentary timing-oracle defence).
    if user and verify_password(password, user.password):
        # Transparent re-hash: if the stored hash is legacy SHA-256 (or
        # Argon2 with outdated parameters), upgrade it on this successful
        # login. Migration completes user-by-user during normal traffic;
        # no DB-wide bulk update needed.
        try:
            if needs_rehash(user.password):
                user.password = hash_password(password)
        except Exception:
            # Re-hash failure must not block login.
            pass
        token = secrets.token_hex(32)
        user.session_token = token
        db.commit()
        resp = RedirectResponse("/dashboard", status_code=302)
        # ``secure=True`` once the site is HTTPS-everywhere (production
        # neucast.ru is behind nginx + TLS). ``samesite=lax`` already set.
        resp.set_cookie(
            "session", token,
            httponly=True, samesite="lax",
            secure=os.getenv("NEUCAST_COOKIES_SECURE", "1") == "1",
        )
        return resp
    # Constant-time dummy verify when username doesn't exist — same wall-
    # clock cost as a real failed match, defeats username-enumeration
    # via timing.
    if user is None:
        verify_password(password, "$argon2id$v=19$m=65536,t=3,p=4$AAAAAAAA"
                                  "AAAAAAAAAAAAAA$"
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
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
    # Code-review C-2 (2026-05-04): bumped from 4 → 12 chars. 4 chars
    # was trivially brute-forceable (94⁴ ≈ 78M, < 1 s on a GPU). 12 chars
    # crosses the threshold of "not feasible to crack offline" even with
    # the legacy SHA-256 hashes in the DB.
    if len(password) < 12:
        return templates.TemplateResponse("register.html", {
            "request": request, "error": "Пароль должен быть не менее 12 символов",
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
    token = secrets.token_hex(32)
    new_user.session_token = token
    db.add(new_user)
    db.commit()
    resp = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie(
        "session", token,
        httponly=True, samesite="lax",
        secure=os.getenv("NEUCAST_COOKIES_SECURE", "1") == "1",
    )
    return resp


@app.get("/logout")
async def logout(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("session")
    if token:
        user = db.query(User).filter(User.session_token == token).first()
        if user:
            user.session_token = None
            db.commit()
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("session")
    return resp


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt():
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /api/\n"
        "Disallow: /admin/\n"
        "Disallow: /ws/\n"
        "Disallow: /predict/result/\n"
        "Disallow: /predict/status/\n"
        "Disallow: /p/\n"
        "Disallow: /logout\n"
        "\n"
        "Sitemap: https://neucast.ru/sitemap.xml\n"
    )


@app.get("/sitemap.xml")
async def sitemap_xml():
    from datetime import date
    today = date.today().isoformat()
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    pages = [
        ("https://neucast.ru/", "1.0", "weekly"),
        ("https://neucast.ru/login", "0.6", "monthly"),
        ("https://neucast.ru/register", "0.7", "monthly"),
    ]
    for url, priority, freq in pages:
        xml += f"  <url>\n    <loc>{url}</loc>\n    <lastmod>{today}</lastmod>\n    <priority>{priority}</priority>\n    <changefreq>{freq}</changefreq>\n  </url>\n"
    xml += "</urlset>\n"
    return Response(content=xml, media_type="application/xml")


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request, role: str = Depends(get_current_role)):
    return templates.TemplateResponse("landing.html", {"request": request, "logged_in": role is not None})


@app.get("/forecast", response_class=HTMLResponse)
async def forecast_page(
    request: Request,
    role: str = Depends(get_current_role),
):
    """Live HF forecast page — gated for any logged-in user.

    Originally served by Tokyo (release T), but moved to Finland in
    release T.4 because the user requested auth-gating ("чтобы любой
    не мог посмотреть с главной"). Auth state lives on Finland; the
    page renders from a Finland template and pulls live data via the
    public ``/api/highfreq/*`` endpoints (which still proxy to Tokyo).

    Non-authenticated visitors → 302 to ``/login?next=/forecast`` so
    they come back to the forecast after sign-in.
    """
    if role is None:
        return RedirectResponse("/login?next=/forecast", status_code=302)
    return templates.TemplateResponse("forecast.html", {
        "request": request,
        "logged_in": True,
        "is_admin": role == "admin",
        "current_page": "forecast",
    })


@app.get("/highfreq", response_class=HTMLResponse)
async def highfreq_admin_page(
    request: Request,
    role: str = Depends(get_current_role),
):
    """Operator-grade /highfreq page — admin role required.

    Surfaces dense technical metrics (Wilson CI, p-values, fold counts,
    calibration diagnostics, fee-tier P&L breakdown) intended for the
    project operator, not casual visitors.

    * not logged in → 302 to /login?next=/highfreq
    * logged in but role != admin → 404 (hide the URL exists)
    * admin → render the full operator template
    """
    if role is None:
        return RedirectResponse("/login?next=/highfreq", status_code=302)
    if role != "admin":
        raise HTTPException(status_code=404)
    return templates.TemplateResponse("highfreq.html", {
        "request": request,
        "logged_in": True,
        "is_admin": True,
        "current_page": "highfreq",
        # Defaults previously injected by Tokyo's web.py renderer; the
        # template uses these only for the symbol-switcher dropdown +
        # the "minutes required" first-fold progress widget.
        "symbol": "BTCUSDT",
        "minutes_required": 1500,
        "available_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    })


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
    use_foundation: str = Form(""),  # "1" если чекбокс включён
    db: Session = Depends(get_db),
    role: str = Depends(get_current_role),
    user: User = Depends(get_current_user),
):
    if not role:
        return RedirectResponse("/login")

    # Чекбокс: HTML-форма шлёт "1" если checked, ничего если unchecked
    use_foundation_bool = bool(use_foundation) and str(use_foundation).strip() not in ("", "0", "false")

    # Celery mode: dispatch task and redirect to waiting page
    if USE_CELERY and celery_app:
        task = celery_app.send_task(
            "neucast.predict",
            args=[ticker, start_date, end_date, days_ahead],
            kwargs={
                "user_id": user.id if user else None,
                "use_foundation": use_foundation_bool,
            },
            # Code-review C-1 (2026-05-04): explicit ``serializer="json"``
            # to match the broker-wide config. Args are str/int/bool only.
            serializer="json",
        )
        # ── Пробуем короткий URL /p/{slug}. Если Redis недоступен, падаем в ──
        # длинный /predict/status/{UUID}?... (легаси-совместимость).
        slug = _make_slug()
        if _save_pred_slug(slug, {
            "task_id": task.id,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "days_ahead": days_ahead,
            "use_foundation": use_foundation_bool,
        }):
            return RedirectResponse(f"/p/{slug}", status_code=303)
        return RedirectResponse(
            f"/predict/status/{task.id}?ticker={ticker}&start_date={start_date}&end_date={end_date}&days_ahead={days_ahead}",
            status_code=303,
        )

    # Fallback: synchronous mode with semaphore + cache
    cache_key = f"{ticker}:{start_date}:{end_date}:{days_ahead}:fnd{int(use_foundation_bool)}"
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

                # Передаём use_foundation через partial-style lambda (run_in_executor
                # не принимает kwargs напрямую).
                _run = lambda: run_prediction(df, days_ahead, use_foundation=use_foundation_bool)
                result = await loop.run_in_executor(None, _run)
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
        # ── Tier 5.1: применяем sentiment bias к future_preds + bands ──
        # Этот вызов идеalmotempent post-process: модифицирует result in-place.
        # Кэш прогноза не зависит от sentiment → новости могут обновиться через
        # 30 мин и юзер получит свежий bias, не перетренивая модели.
        try:
            apply_sentiment_bias(
                result,
                sentiment_score=sentiment.avg_score,
                total_articles=sentiment.total_articles,
            )
        except Exception as e:
            logger.warning(f"apply_sentiment_bias failed (sync path): {e}")
    except Exception:
        sentiment_data = None

    # Save to DB
    db_ticker = db.query(Ticker).filter_by(symbol=ticker).first()
    if not db_ticker:
        db_ticker = Ticker(symbol=ticker)
        db.add(db_ticker)
        db.commit()
        db.refresh(db_ticker)

    # ── P1: bulk DB inserts ──────────────────────────────────────────
    # Было: per-row db.flush() → N round-trips к БД (60-250 дат × прогноз).
    # Стало: дедупликация против уже сохранённого + один add_all + один flush
    # (N round-trips → 2). Экономит 300-800мс на БД-стадии.
    INDICATOR_NAMES = ["RSI", "MACD", "Signal", "BB_upper", "BB_lower", "ATR"] + [f"MA_{w}" for w in (5, 10, 20, 50)]

    existing_dates = {
        r[0] for r in db.query(MarketData.date).filter_by(ticker_id=db_ticker.id).all()
    }

    new_rows = [(idx, row) for idx, row in df.iterrows() if idx.date() not in existing_dates]
    if new_rows:
        md_list = [
            MarketData(
                ticker_id=db_ticker.id, date=idx.date(),
                open=row.Open, high=row.High,
                low=row.Low, close=row.Close, volume=row.Volume,
            ) for idx, row in new_rows
        ]
        db.add_all(md_list)
        db.flush()  # один round-trip → все ID выдаются разом

        ind_list = []
        for md, (_, row) in zip(md_list, new_rows):
            for name in INDICATOR_NAMES:
                ind_list.append(Indicator(
                    market_data_id=md.id, name=name, value=getattr(row, name),
                ))
        db.add_all(ind_list)
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
        "foundation_used": result.get("foundation_used", False),
        "foundation_models": result.get("foundation_models", []),
        # Foundation в test-fold (BTC-USD dir_acc fix): UI показывает badge,
        # что главная метрика отражает Foundation contribution, а не только TCN/boosting.
        "foundation_test_used": result.get("foundation_test_used", False),
        "foundation_test_models": result.get("foundation_test_models", []),
        # Data-driven α: 0 → Foundation вреден, не применился; >0 → применили α.
        "foundation_alpha": result.get("foundation_alpha", 0.0),
        "foundation_local_val_mape": result.get("foundation_local_val_mape"),
        "foundation_blended_val_mape": result.get("foundation_blended_val_mape"),
        "sentiment_applied": result.get("sentiment_applied", False),
        "sentiment_bias_pct": result.get("sentiment_bias_pct", 0.0),
        "local_calibration_applied": result.get("local_calibration_applied", False),
        "local_sigma_ratio": result.get("local_sigma_ratio", 1.0),
        # Honest skill warning + bootstrap CI (B1): low_directional_skill=True
        # если 95% CI пересекает 50% (direction не отличим от монетки даже при
        # point estimate 55%). dir_acc_ci_low/high — bootstrap percentile bounds.
        "low_directional_skill": result.get("low_directional_skill", False),
        "dir_acc_ci_low": result.get("dir_acc_ci_low"),
        "dir_acc_ci_high": result.get("dir_acc_ci_high"),
        # A1: Hourly skill probe (crypto-only, opt-in HOURLY_DIAGNOSTIC=1).
        # Узкий CI на ≈4800 hourly точках → отделить coin-flip от настоящего
        # signal'а. None если probe не запускался (env off / не крипто).
        "hourly_skill": result.get("hourly_skill"),
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
            # error из celery_worker уже прошёл через friendly_error()
            return {"state": "FAILURE", "error": res["error"]}
        return {"state": "SUCCESS", "task_id": task_id}
    elif result.state == "FAILURE":
        # FAILURE state = worker умер до того как успел вернуть dict (OOM,
        # SIGKILL, hard time limit). result.info — raw exception. Прогоняем
        # через friendly_error чтобы пользователь не видел Python traceback.
        try:
            logger.error(f"Celery task FAILURE for {task_id}: {result.info!r}")
        except Exception:
            pass
        return {"state": "FAILURE", "error": friendly_error(result.info)}
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
                    # См. /api/task/{task_id} — friendly_error скрывает
                    # raw Python traceback от пользователя.
                    try:
                        logger.error(f"Celery WS FAILURE for {task_id}: {result.info!r}")
                    except Exception:
                        pass
                    await websocket.send_json(
                        {"state": "FAILURE", "error": friendly_error(result.info)}
                    )
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


async def _render_success_result(async_result, request, db, user, role):
    """Render predict.html from SUCCESS Celery task. Caller гарантирует state == SUCCESS.

    Используется и /predict/result/{task_id} (легаси), и /p/{slug} (короткий URL).
    """
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
        # ── Tier 5.1: применяем sentiment bias на result из celery ──
        # Celery task хранит "чистый" прогноз в Redis, sentiment bias считается
        # здесь — каждый вызов берёт свежий snapshot новостей.
        try:
            apply_sentiment_bias(
                result,
                sentiment_score=sentiment.avg_score,
                total_articles=sentiment.total_articles,
            )
        except Exception as e:
            logger.warning(f"apply_sentiment_bias failed (celery path): {e}")
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
        "foundation_used": result.get("foundation_used", False),
        "foundation_models": result.get("foundation_models", []),
        # Foundation в test-fold (BTC-USD dir_acc fix): UI показывает badge,
        # что главная метрика отражает Foundation contribution, а не только TCN/boosting.
        "foundation_test_used": result.get("foundation_test_used", False),
        "foundation_test_models": result.get("foundation_test_models", []),
        # Data-driven α: 0 → Foundation вреден, не применился; >0 → применили α.
        "foundation_alpha": result.get("foundation_alpha", 0.0),
        "foundation_local_val_mape": result.get("foundation_local_val_mape"),
        "foundation_blended_val_mape": result.get("foundation_blended_val_mape"),
        "sentiment_applied": result.get("sentiment_applied", False),
        "sentiment_bias_pct": result.get("sentiment_bias_pct", 0.0),
        "local_calibration_applied": result.get("local_calibration_applied", False),
        "local_sigma_ratio": result.get("local_sigma_ratio", 1.0),
        # Honest skill warning + bootstrap CI (B1): low_directional_skill=True
        # если 95% CI пересекает 50% (direction не отличим от монетки даже при
        # point estimate 55%). dir_acc_ci_low/high — bootstrap percentile bounds.
        "low_directional_skill": result.get("low_directional_skill", False),
        "dir_acc_ci_low": result.get("dir_acc_ci_low"),
        "dir_acc_ci_high": result.get("dir_acc_ci_high"),
        # A1: Hourly skill probe (crypto-only, opt-in HOURLY_DIAGNOSTIC=1).
        "hourly_skill": result.get("hourly_skill"),
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

    return await _render_success_result(async_result, request, db, user, role)


@app.get("/p/{slug}", response_class=HTMLResponse)
async def predict_by_slug(
    slug: str,
    request: Request,
    db: Session = Depends(get_db),
    role: str = Depends(get_current_role),
    user: User = Depends(get_current_user),
):
    """Короткий URL-алиас для /predict/status/{task_id}?... и /predict/result/{task_id}.

    Одна точка входа, которая показывает либо waiting.html, либо predict.html —
    в зависимости от состояния Celery-таски. URL не меняется при переходе
    waiting → done: пользователь видит одну и ту же ссылку /p/xY3kH7.
    """
    if not role:
        return RedirectResponse("/login")

    meta = _load_pred_slug(slug)
    if not meta:
        # Slug истёк (TTL 2ч) или неизвестен — уводим на дэшборд
        return RedirectResponse("/dashboard")

    task_id = meta["task_id"]

    if not USE_CELERY or not celery_app:
        return RedirectResponse("/dashboard")

    from celery.result import AsyncResult
    async_result = AsyncResult(task_id, app=celery_app)

    # Таска готова — рендерим результат
    if async_result.state == "SUCCESS":
        return await _render_success_result(async_result, request, db, user, role)

    # Таска упала — показываем ошибку в форме (friendly, без raw traceback)
    if async_result.state == "FAILURE":
        try:
            logger.error(
                f"Celery slug-page FAILURE for {task_id}: {async_result.info!r}"
            )
        except Exception:
            pass
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": friendly_error(async_result.info) if async_result.info else (
                "Не удалось выполнить расчёт. Попробуйте ещё раз."
            ),
            "ensemble": True,
        })

    # Таска в процессе — показываем waiting.html
    return templates.TemplateResponse("waiting.html", {
        "request": request,
        "task_id": task_id,
        "ticker": meta.get("ticker", ""),
        "start_date": meta.get("start_date", ""),
        "end_date": meta.get("end_date", ""),
        "days_ahead": meta.get("days_ahead", 0),
        "success_url": f"/p/{slug}",
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
async def download_csv(ticker: str, start_date: str, end_date: str, days_ahead: int = 0):
    """P1: offloaded to thread pool + reuses booster cache from run_prediction."""
    loop = asyncio.get_event_loop()

    def _build():
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
        return buf.getvalue()

    csv_body = await loop.run_in_executor(None, _build)
    return Response(
        csv_body, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/download_pdf")
async def download_pdf(ticker: str, start_date: str, end_date: str, days_ahead: int = 0):
    """P1: offload the heavy matplotlib/FPDF work to a thread pool.
    uvicorn single-worker otherwise blocks here for ~5-10s per PDF,
    stalling all other requests."""
    loop = asyncio.get_event_loop()
    pdf_bytes, filename = await loop.run_in_executor(
        None, _download_pdf_sync, ticker, start_date, end_date, days_ahead
    )
    return Response(
        pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _download_pdf_sync(ticker: str, start_date: str, end_date: str, days_ahead: int = 0):
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
    return buf.getvalue(), f"gold_report_{ticker}_{end_date}.pdf"


@app.get("/api/queue_status")
async def queue_status():
    return {
        "active": MAX_CONCURRENT_PREDICTIONS - _prediction_semaphore._value,
        "waiting": max(0, _prediction_queue_count - MAX_CONCURRENT_PREDICTIONS),
        "max_concurrent": MAX_CONCURRENT_PREDICTIONS,
    }


# P1: Memory cache для /api/live_price.
# Yahoo fast_info — 200-500мс даже на малом ответе; за сек может прийти 5 запросов
# с одним и тем же тикером. Кэш на 60сек убирает ~все повторы.
_live_price_cache: dict[str, tuple] = {}  # ticker -> (payload, timestamp)
LIVE_PRICE_TTL = int(os.getenv("LIVE_PRICE_TTL", "60"))


@app.get("/api/live_price")
async def live_price(ticker: str = "GC=F"):
    # Cache hit
    cached = _live_price_cache.get(ticker)
    if cached is not None:
        payload, ts = cached
        if time.time() - ts < LIVE_PRICE_TTL:
            return payload

    loop = asyncio.get_event_loop()

    def _fetch():
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

    payload = await loop.run_in_executor(None, _fetch)
    _live_price_cache[ticker] = (payload, time.time())
    return payload


# ── Portfolio optimization ──
# Временно скрыто. Включить через env: PORTFOLIO_ENABLED=1
PORTFOLIO_ENABLED = os.getenv("PORTFOLIO_ENABLED", "0") == "1"


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_form(request: Request, role: str = Depends(get_current_role)):
    if not PORTFOLIO_ENABLED:
        raise HTTPException(status_code=404)
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
    if not PORTFOLIO_ENABLED:
        raise HTTPException(status_code=404)
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
