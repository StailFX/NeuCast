"""Tests for friendly HTML error pages with JSON content negotiation.

Phase A.7.1: до этого FastAPI отдавал raw `{"detail": "Not Found"}`
даже когда пользователь приходил из браузера. Это:
  * выглядело как баг (а не как «страница не найдена»)
  * показывало внутренний жаргон (`detail` ключ)
  * ломало доверие — главная страница красивая, а 404 голая.

Здесь мы проверяем три инварианта:

1. **Content negotiation работает.** `/api/*` и Accept: application/json
   получают JSON; всё остальное — HTML с шапкой NeuCast.
2. **HTML страница рендерится.** Шаблон `templates/error.html`
   принимает status_code, title, description и не падает.
3. **422 (валидация формы) тоже HTML.** Это самый частый источник
   raw-JSON в проде — POST формы без обязательного поля.

Все тесты строят **изолированный** FastAPI app, в который копируют
обработчики из ``app.main``. Так мы проверяем именно их код, не
рискуя поднять `/predict` или `/login` (которые требуют Postgres /
ML-моделей).
"""
from __future__ import annotations

import os
import sys
from types import ModuleType

# `app.db` читает DATABASE_URL на import — sqlite:///:memory: безопасен,
# create_engine не коннектится сразу. Делаем это ДО импорта app.main.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# ``app.main`` wires ML services into production routes at import time.  The
# error-page tests never call those routes, so loading TensorFlow and the rest
# of the training stack here would make this lightweight unit suite needlessly
# slow.  Minimal module stubs keep the test focused on the HTTP handlers.
_prediction = ModuleType("app.prediction")
_prediction.run_prediction = lambda *args, **kwargs: None
_prediction.fetch_and_preprocess = lambda *args, **kwargs: None
_prediction.apply_sentiment_bias = lambda *args, **kwargs: None
_prediction.SEQ_LEN = 1
_prediction.MODEL_COLS = []
sys.modules["app.prediction"] = _prediction

_sentiment = ModuleType("app.sentiment")
_sentiment.analyze_sentiment = lambda *args, **kwargs: None
sys.modules["app.sentiment"] = _sentiment

_portfolio = ModuleType("app.portfolio")
_portfolio.optimize_portfolio = lambda *args, **kwargs: None
sys.modules["app.portfolio"] = _portfolio

import pytest
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.testclient import TestClient

from app.main import (  # noqa: E402  (env-setup must come first)
    _ERROR_MESSAGES,
    _generic_exception_handler,
    _http_exception_handler,
    _render_error_page,
    _validation_exception_handler,
    _wants_json_response,
)
import app as _app_package

# Do not leak the lightweight stubs into the rest of the test session: several
# other modules exercise the real prediction implementation.
for _module_name, _attribute, _stub in (
    ("app.prediction", "prediction", _prediction),
    ("app.sentiment", "sentiment", _sentiment),
    ("app.portfolio", "portfolio", _portfolio),
):
    sys.modules.pop(_module_name, None)
    if getattr(_app_package, _attribute, None) is _stub:
        delattr(_app_package, _attribute)

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# ─────────────────────────── test app ───────────────────────────


def _build_test_app() -> FastAPI:
    """Минимальное FastAPI приложение с теми же обработчиками,
    что и production. Добавляет несколько тестовых роутов: 403, 500,
    POST с обязательным полем (для 422) и API-вариант, отдающий 500.
    """
    test_app = FastAPI()
    test_app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    test_app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    test_app.add_exception_handler(Exception, _generic_exception_handler)

    @test_app.get("/raise-403")
    async def _raise_403():
        raise HTTPException(status_code=403)

    @test_app.get("/raise-403-with-detail")
    async def _raise_403_with_detail():
        raise HTTPException(status_code=403, detail="Только для пробной группы")

    @test_app.get("/raise-500")
    async def _raise_500():
        raise RuntimeError("boom")

    @test_app.get("/api/raise-500")
    async def _api_raise_500():
        raise RuntimeError("api boom")

    @test_app.post("/submit")
    async def _submit(name: str = Form(...)):  # требует поле name
        return {"name": name}

    @test_app.post("/api/submit")
    async def _api_submit(name: str = Form(...)):
        return {"name": name}

    return test_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_test_app(), raise_server_exceptions=False)


# ─────────────────── _wants_json_response (pure) ────────────────


def _mock_request(path: str = "/", accept: str | None = None) -> Request:
    """Лёгкий request-объект: только то, что использует
    `_wants_json_response` (path и accept-заголовок).
    """
    headers = []
    if accept is not None:
        headers.append((b"accept", accept.encode("latin-1")))
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 0),
    }
    return Request(scope)


def test_wants_json_for_api_path():
    req = _mock_request("/api/highfreq/status", accept="text/html")
    # /api/* всегда JSON — даже если браузер шлёт Accept: text/html
    # (это бывает: пользователь открывает /api/* напрямую, но мы
    # хотим, чтобы curl/JS-фетчи получали JSON стабильно).
    assert _wants_json_response(req) is True


def test_wants_json_for_explicit_json_accept():
    req = _mock_request("/foo", accept="application/json")
    assert _wants_json_response(req) is True


def test_wants_html_for_browser_accept():
    # Браузеры обычно шлют:
    #   text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
    # — наличие text/html → ожидают HTML.
    req = _mock_request("/foo", accept="text/html,application/xhtml+xml,*/*;q=0.8")
    assert _wants_json_response(req) is False


def test_wants_html_for_mixed_accept_includes_html():
    # Edge case: AJAX-запрос с Accept: application/json, text/html
    # — text/html в наличии, значит браузер тоже принимает HTML.
    # Отдаём HTML, чтобы не порвать визуал в случайных кейсах.
    req = _mock_request("/foo", accept="application/json, text/html")
    assert _wants_json_response(req) is False


def test_wants_html_for_no_accept_header():
    # curl без -H "Accept" → пустой accept → HTML по умолчанию
    # (HTML — безопасный fallback, его и человек, и парсер прочитают).
    req = _mock_request("/foo")
    assert _wants_json_response(req) is False


# ───────────────────── 404 — отсутствующий путь ─────────────────


def test_404_html_for_browser(client: TestClient):
    r = client.get("/nonexistent-page", headers={"Accept": "text/html"})
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]
    body = r.text
    # Главные элементы шаблона:
    assert "404" in body
    assert "Страница не найдена" in body
    assert 'href="/"' in body  # ссылка на главную


def test_404_json_for_api_path(client: TestClient):
    r = client.get("/api/nonexistent")
    assert r.status_code == 404
    assert "application/json" in r.headers["content-type"]
    payload = r.json()
    assert payload["status_code"] == 404
    # detail должен быть непустой строкой на русском (а не "Not Found")
    assert isinstance(payload["detail"], str)
    assert payload["detail"]
    assert "Not Found" not in payload["detail"]


def test_404_json_when_accept_is_json(client: TestClient):
    # Не-API путь, но Accept: application/json — отдаём JSON
    r = client.get("/nonexistent", headers={"Accept": "application/json"})
    assert r.status_code == 404
    assert "application/json" in r.headers["content-type"]
    assert r.json()["status_code"] == 404


def test_404_template_does_not_leak_request_path(client: TestClient):
    # Раньше шаблон показывал «GET /requested/path» внизу — это dev-внутрянка
    # на пользовательской странице. Теперь не показывает: путь не должен
    # утекать в HTML (а в JSON через /api/* — уходит в логи запроса, не в UI).
    r = client.get("/some/deep/path", headers={"Accept": "text/html"})
    assert r.status_code == 404
    assert "/some/deep/path" not in r.text
    # «GET» как слово может встретиться в комментариях/JS лендинга, но
    # точная строка «GET /some/deep/path» не должна.
    assert "GET /some/deep/path" not in r.text


# ───────────────────── 403 — Forbidden ──────────────────────────


def test_403_html_page(client: TestClient):
    r = client.get("/raise-403", headers={"Accept": "text/html"})
    assert r.status_code == 403
    assert "Доступ закрыт" in r.text
    assert "403" in r.text


def test_403_explicit_detail_preserved(client: TestClient):
    # HTTPException(detail="свой текст") → этот текст должен попасть
    # в description страницы (не подменяться на generic).
    r = client.get("/raise-403-with-detail", headers={"Accept": "text/html"})
    assert r.status_code == 403
    assert "Только для пробной группы" in r.text


# ───────────────────── 422 — RequestValidationError ─────────────


def test_422_html_for_form_post(client: TestClient):
    # POST формы без поля name → FastAPI бросает RequestValidationError.
    # Раньше пользователь видел raw [{"loc": ..., "msg": ..., ...}].
    r = client.post("/submit", data={}, headers={"Accept": "text/html"})
    assert r.status_code == 422
    assert "text/html" in r.headers["content-type"]
    assert "Неверный запрос" in r.text
    # Технические детали ошибок НЕ должны утекать в HTML:
    assert "loc" not in r.text or "field required" not in r.text


def test_422_json_for_api_form_post(client: TestClient):
    # /api/* — наоборот, отдаём детали (полезно для JS-консьюмера).
    r = client.post("/api/submit", data={})
    assert r.status_code == 422
    assert "application/json" in r.headers["content-type"]
    payload = r.json()
    assert payload["status_code"] == 422
    # detail должен быть массивом ошибок валидации (как раньше у FastAPI)
    assert isinstance(payload["detail"], list)
    assert len(payload["detail"]) >= 1


# ───────────────────── 500 — необработанные исключения ──────────


def test_500_html_for_browser(client: TestClient):
    r = client.get("/raise-500", headers={"Accept": "text/html"})
    assert r.status_code == 500
    assert "text/html" in r.headers["content-type"]
    assert "Что-то пошло не так" in r.text
    # Внутренние детали (сообщение исключения) НЕ должны утекать:
    assert "boom" not in r.text
    assert "RuntimeError" not in r.text


def test_500_json_for_api(client: TestClient):
    r = client.get("/api/raise-500")
    assert r.status_code == 500
    assert "application/json" in r.headers["content-type"]
    payload = r.json()
    assert payload["status_code"] == 500
    assert "api boom" not in payload["detail"]  # без traceback в API ответе


# ───────────────────── шаблон: render API ───────────────────────


def test_render_error_page_html_includes_status_and_title():
    req = _mock_request("/foo", accept="text/html")
    resp = _render_error_page(req, 404)
    assert resp.status_code == 404
    body = resp.body.decode("utf-8")
    assert "404" in body
    assert "Страница не найдена" in body


def test_render_error_page_json_for_api_path():
    req = _mock_request("/api/foo")
    resp = _render_error_page(req, 503)
    assert resp.status_code == 503
    import json
    payload = json.loads(resp.body)
    assert payload["status_code"] == 503
    assert isinstance(payload["detail"], str)
    assert payload["detail"]
    # должно быть на ru, не «Service Unavailable»
    assert any("\u0400" <= ch <= "\u04ff" for ch in payload["detail"])


def test_render_error_page_unknown_status_uses_fallback():
    # Status code, которого нет в _ERROR_MESSAGES (e.g. 418).
    # Не должно падать — должен отдать "Ошибка" / "Что-то пошло не так."
    req = _mock_request("/api/foo")
    resp = _render_error_page(req, 418)
    assert resp.status_code == 418
    import json
    payload = json.loads(resp.body)
    assert payload["status_code"] == 418
    assert payload["detail"]  # не пусто


# ───────────────────── _ERROR_MESSAGES sanity ───────────────────


def test_error_messages_cover_common_codes():
    # Минимальный набор, который точно должен быть покрыт:
    for code in (400, 403, 404, 422, 500):
        assert code in _ERROR_MESSAGES, f"missing message for {code}"
        title, description = _ERROR_MESSAGES[code]
        assert title and description, f"empty entry for {code}"


def test_error_messages_are_russian():
    # Это пользовательский UX-текст, должен быть на ru — иначе
    # «Not Found» возвращается нашему пользователю и весь смысл
    # этой работы теряется.
    for code, (title, _) in _ERROR_MESSAGES.items():
        # быстрый чек: заголовок должен содержать кириллицу
        assert any("\u0400" <= ch <= "\u04ff" for ch in title), (
            f"title for {code} not in cyrillic: {title!r}"
        )
