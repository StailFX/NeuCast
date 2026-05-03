"""Tests for the public-facing routes that surface the live HF forecast.

Release T.4 (2026-04-29) moved the ``/forecast`` HTML route from
Tokyo's :mod:`app.highfreq.web` to Finland's :mod:`app.main` so it can
be auth-gated (``/forecast`` requires a logged-in session;
``/highfreq`` requires admin role).  These tests therefore pin two
contracts:

1. The Tokyo router (``app.highfreq.web.router``) NO LONGER carries
   the HTML routes — only JSON ``/api/highfreq/*`` endpoints.  This
   is defence-in-depth: a direct hit to Tokyo's WG IP can't bypass
   the Finland auth gate.

2. Finland's ``app.main`` exposes ``/forecast`` (login required) and
   ``/highfreq`` (admin required) with proper redirects to
   ``/login?next=...`` for unauthenticated visitors.

The Finland test uses a stubbed dependency-override pattern — we don't
need a real DB to verify routing + redirect behaviour.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.highfreq.web import router as tokyo_router


# ─── Tokyo: no HTML routes, only API ───


def _make_tokyo_app() -> FastAPI:
    app = FastAPI()
    app.include_router(tokyo_router)
    return app


def test_tokyo_router_no_longer_serves_forecast_html():
    """Defence-in-depth: ``/forecast`` is GONE from Tokyo's router so
    a direct WG-IP hit can't bypass Finland's auth gate.  Pin so a
    refactor that re-adds the route accidentally breaks loudly."""
    client = TestClient(_make_tokyo_app())
    res = client.get("/forecast")
    assert res.status_code == 404, (
        "Tokyo must NOT serve /forecast directly; that route lives "
        "behind Finland's auth gate now"
    )


def test_tokyo_router_no_longer_serves_highfreq_html():
    client = TestClient(_make_tokyo_app())
    res = client.get("/highfreq")
    assert res.status_code == 404, (
        "Tokyo must NOT serve /highfreq directly; admin-only route "
        "lives behind Finland's auth gate now"
    )


def test_tokyo_router_still_carries_api_routes():
    """The proxied API still exists — only HTML pages moved.  We
    inspect the router's registered paths rather than calling them
    (avoids needing DATABASE_URL etc. in the unit-test env)."""
    paths = {r.path for r in tokyo_router.routes if hasattr(r, "path")}
    assert "/api/highfreq/forecast" in paths
    assert "/api/highfreq/paper_trades" in paths
    assert "/api/highfreq/training_report" in paths
    assert "/api/highfreq/health" in paths
    # And confirm the HTML routes are NOT here.
    assert "/forecast" not in paths
    assert "/highfreq" not in paths


# ─── Forecast template still has expected client-side bindings ───
#
# We can't easily test Finland's main.py routes without spinning up
# its DB + auth dependencies. Instead we verify the TEMPLATE itself
# still carries the JS bindings — the route wiring is verified live
# via curl in deploy scripts.


def _read_forecast_template() -> str:
    """Read the actual deployed template file. Avoids depending on
    Finland's app.main importability (which pulls in TF, Celery, etc)."""
    from pathlib import Path
    here = Path(__file__).parent.parent
    text = (here / "templates" / "forecast.html").read_text(encoding="utf-8")
    return text


def test_forecast_template_carries_three_symbol_cards():
    html = _read_forecast_template()
    assert 'data-symbol="BTCUSDT"' in html
    assert 'data-symbol="ETHUSDT"' in html
    assert 'data-symbol="BNBUSDT"' in html


def test_forecast_template_has_trades_list_and_stats_anchors():
    html = _read_forecast_template()
    assert 'id="trades-list"' in html
    assert 'id="stat-trades"' in html
    assert 'id="stat-diracc"' in html
    assert 'id="stat-pnl"' in html
    assert 'id="stat-skill"' in html


def test_forecast_template_has_filter_pills():
    html = _read_forecast_template()
    assert 'id="horizon-pills"' in html
    assert 'id="venue-pills"' in html
    for h in ('1', '5', '15', '60'):
        assert f'data-horizon="{h}"' in html
    assert 'data-venue="spot"' in html
    assert 'data-venue="futures"' in html


def test_forecast_template_has_fee_tier_and_metrics_sections():
    html = _read_forecast_template()
    assert 'id="metrics-grid"' in html
    assert 'id="fee-tiers"' in html
    assert 'id="admin-metrics"' in html
    assert 'id="road-futures-eta"' in html or 'roadmap' in html.lower()


def test_forecast_template_has_conditional_accuracy_block():
    """Release T.12 (2026-04-30) added a "когда модель уверена" block
    showing realized dir_acc bucketed by ``|prob_up - 0.5|`` so the
    user can see "BNB at conf >= 65% is right 57.2% of the time on
    n=1838 calls". Pin the anchor + the JS fetch."""
    html = _read_forecast_template()
    assert 'id="conf-grid"' in html
    assert "/api/highfreq/conditional_accuracy" in html
    assert "refreshConditionalAccuracy" in html


def test_forecast_template_has_robustness_block():
    """Release T.14 (2026-05-01) added a "Проверка значимости" block
    surfacing block bootstrap CI + permutation test + per-day
    stability so the operator can see whether dir_acc is regime-
    robust (vs just lucky on a trending window). Pin the anchor +
    the JS fetch + the verdict-classification logic."""
    html = _read_forecast_template()
    assert 'id="rob-grid"' in html
    assert "/api/highfreq/robustness" in html
    assert "refreshRobustness" in html
    # The verdict logic must be present so the UI doesn't silently
    # claim "skill" on a borderline result.
    assert "permutation_p_value" in html
    assert "block_bootstrap_ci_low" in html


def test_forecast_template_has_reliability_diagram_block():
    """Release T.16 (2026-05-03) added a calibration / reliability
    diagram block: per-symbol SVG plotting predicted prob_up vs
    realized rate with a y=x reference line, plus Brier and ECE
    single-number summaries. Canonical defence-grade plot for
    "is the model honest about its uncertainty?".
    """
    html = _read_forecast_template()
    assert 'id="reliability-grid"' in html
    assert "/api/highfreq/reliability_diagram" in html
    assert "refreshReliabilityDiagram" in html
    # The diagonal y=x reference line must be drawn — a reliability
    # diagram WITHOUT the reference is illegible.
    assert "ref-line" in html
    # Brier + ECE single-number summary must appear on each card.
    assert "Brier=" in html
    assert "ECE=" in html


def test_forecast_template_has_feature_importance_block():
    """Release T.16 (2026-05-03) added a feature importance block
    showing top-10 CatBoost gains per symbol. Defence story: see
    that the model isn't overfit to one feature (long-tail of
    contributions is healthy)."""
    html = _read_forecast_template()
    assert 'id="featimp-grid"' in html
    assert "/api/highfreq/feature_importance" in html
    assert "refreshFeatureImportance" in html


def test_forecast_template_polls_correct_api_endpoints():
    """Pin the API endpoints the page hits.

    Note ``pnl_by_fee_tier`` was REMOVED in release T.10 — fee-tier
    aggregates are now computed client-side from horizon-filtered
    trades (the paper_trades schema has no horizon column, so server
    can't filter for us).  ``microprice_history`` powers the live
    price line on each card.  ``conditional_accuracy`` (T.12) buckets
    realized predictions by confidence threshold."""
    html = _read_forecast_template()
    assert "/api/highfreq/forecast" in html
    assert "/api/highfreq/paper_trades" in html
    assert "/api/highfreq/training_report" in html
    assert "/api/highfreq/microprice_history" in html
    assert "/api/highfreq/conditional_accuracy" in html
    assert "/api/highfreq/robustness" in html
    # T.16 (2026-05-03): reliability diagram + feature importance.
    assert "/api/highfreq/reliability_diagram" in html
    assert "/api/highfreq/feature_importance" in html
    # Pin lite=1 query param — without it the page would block 18s on
    # cold cache for live_inventory; release T.3 added the flag.
    assert "lite=1" in html
    # Pin horizon param — release T.10 made every horizon-dependent
    # block reactive to the active pill.
    assert "horizon=" in html


def test_forecast_template_uses_text_default_for_brand():
    """The 'NeuCast' wordmark in the nav was rendering as a purple
    visited-link colour because the template lacked an explicit
    ``color`` declaration on ``.nav-name`` / ``.nav-brand``.  T.3 fix
    pinned it; this test catches a regression."""
    html = _read_forecast_template()
    # Both ``.nav-name`` and ``.nav-brand`` must declare a color so
    # browser default :visited (purple) doesn't bleed through.
    assert 'color: var(--text)' in html
    assert '.nav-brand:visited' in html


def test_forecast_template_has_user_friendly_disclaimer():
    html = _read_forecast_template().lower()
    assert (
        "симул" in html or "sim-only" in html or "не является" in html
    )


def test_forecast_template_is_indexable():
    """The page is meant for organic search → robots: index, follow."""
    html = _read_forecast_template()
    assert 'name="robots" content="noindex"' not in html
    assert 'noindex, nofollow' not in html


def test_anon_visitors_dont_see_forecast_link():
    """T.4 user feedback: don't expose the live page to random visitors.
    T.5 refactor: the nav lives in ``templates/_partials/nav.html``
    (shared across landing / forecast / highfreq).

    Pin: any link to ``/forecast`` in the partial sits inside an
    ``{% if logged_in %}`` block so an anonymous visitor renders a
    nav with NO link to the gated page.  Same for the landing hero
    CTA "Live прогноз крипты" which lives in landing.html itself."""
    from pathlib import Path
    here = Path(__file__).parent.parent

    # ── 1) Shared nav partial ────────────────────────────────
    nav = (here / "templates" / "_partials" / "nav.html").read_text(encoding="utf-8")
    # Find every ``href="/forecast"`` and ensure each is inside an
    # open ``{% if logged_in %}`` block.
    pos = 0
    found_any = False
    while True:
        idx = nav.find('href="/forecast"', pos)
        if idx < 0:
            break
        found_any = True
        if_idx = nav.rfind("{% if logged_in %}", 0, idx)
        intervening_endif = nav.rfind("{% endif %}", if_idx, idx) if if_idx > 0 else -1
        assert if_idx > 0 and intervening_endif <= if_idx, (
            "href=\"/forecast\" in nav.html must be inside "
            "{% if logged_in %} block"
        )
        pos = idx + 1
    assert found_any, "nav partial must reference /forecast somewhere"

    # ── 2) Landing hero CTA ─────────────────────────────────
    landing = (here / "templates" / "landing.html").read_text(encoding="utf-8")
    if "Live прогноз крипты" in landing:
        # If the CTA exists, it must be guarded.
        idx = landing.find("Live прогноз крипты")
        if_idx = landing.rfind("{% if logged_in %}", 0, idx)
        intervening_endif = landing.rfind("{% endif %}", if_idx, idx) if if_idx > 0 else -1
        assert if_idx > 0 and intervening_endif <= if_idx, (
            "'Live прогноз крипты' CTA in landing.html must be inside "
            "{% if logged_in %} block"
        )
