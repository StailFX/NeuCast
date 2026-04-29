"""Tests for ``GET /forecast`` — the public-facing forecast page
(release T, 2026-04-29).

Smoke-grade only: the page is a static template shell with all data
fetched client-side from existing endpoints (forecast / paper_trades).
We pin:

* the route is mounted and returns 200 HTML,
* no auth gate accidentally got added (the page is meant to be public),
* the rendered HTML carries the elements the JS expects to bind to
  (data-symbol cards, trades-list container, stats containers),
* the page is index-able by search engines (no robots noindex).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.web import router


def _make_app() -> FastAPI:
    """Minimal FastAPI app with just the highfreq router — sister of
    the helper in test_highfreq_forecast_endpoint."""
    app = FastAPI()
    app.include_router(router)
    return app


def test_forecast_page_returns_200():
    """Page renders without server-side data — the JS does the rest.
    A 500 here would mean the template can't be found / parsed."""
    client = TestClient(_make_app())
    res = client.get("/forecast")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")


def test_forecast_page_has_no_auth_gate():
    """Public-facing by design (the whole point of release T).  A
    redirect to /login or 401/403 here would break the user's request
    that the page be visible to non-operator visitors."""
    client = TestClient(_make_app())
    res = client.get("/forecast", follow_redirects=False)
    assert res.status_code == 200
    assert "location" not in {k.lower() for k in res.headers.keys()}


def test_forecast_page_carries_three_symbol_cards():
    """The JS binds to ``data-symbol="BTCUSDT"`` etc. — pin the
    template carries all three so a refactor that drops one
    accidentally is caught."""
    client = TestClient(_make_app())
    html = client.get("/forecast").text
    assert 'data-symbol="BTCUSDT"' in html
    assert 'data-symbol="ETHUSDT"' in html
    assert 'data-symbol="BNBUSDT"' in html


def test_forecast_page_has_trades_list_and_stats_anchors():
    """The polling loop populates ``#trades-list``, ``#stat-trades``,
    ``#stat-winrate``, ``#stat-pnl``. Pin the IDs so a markup refactor
    can't silently break the JS bindings."""
    client = TestClient(_make_app())
    html = client.get("/forecast").text
    assert 'id="trades-list"' in html
    assert 'id="stat-trades"' in html
    assert 'id="stat-winrate"' in html
    assert 'id="stat-pnl"' in html


def test_forecast_page_is_indexable():
    """The page replaces the noindex'd /highfreq for ordinary visitors
    and IS meant to be in search results — pin no ``robots noindex``
    accidentally crept in."""
    client = TestClient(_make_app())
    html = client.get("/forecast").text
    # Must NOT contain a noindex meta — we explicitly opt INTO
    # search engines for this page.
    assert 'name="robots" content="noindex"' not in html
    assert 'noindex, nofollow' not in html


def test_forecast_page_has_user_friendly_disclaimer():
    """sim-only / not financial advice — pin so a refactor can't
    silently drop the disclaimer that protects us legally."""
    client = TestClient(_make_app())
    html = client.get("/forecast").text.lower()
    # Russian or English variant accepted.
    assert (
        "симул" in html
        or "sim-only" in html
        or "не является" in html
        or "not financial advice" in html
    )


def test_forecast_page_polls_correct_api_endpoints():
    """The JS in the page must hit the existing endpoints —
    ``/api/highfreq/forecast`` and ``/api/highfreq/paper_trades``.
    Pin both string references so a rename of either endpoint
    breaks loudly here, not silently in production."""
    client = TestClient(_make_app())
    html = client.get("/forecast").text
    assert "/api/highfreq/forecast" in html
    assert "/api/highfreq/paper_trades" in html
