"""Minimal FastAPI app exposing ONLY the high-frequency routes.

This module exists so the Tokyo HFT VPS can serve ``/highfreq`` and
``/api/highfreq/*`` directly, without pulling in the full ``app.main``
module — which transitively imports TensorFlow, PyTorch, transformers,
chronos, timesfm, and several gigabytes of model wheels we deliberately
keep off the 4 GB ingest box.

Architecture (ADR-007):

    user → neucast.ru (Finland nginx)
           ├── /highfreq, /api/highfreq/*  → reverse-proxy → Tokyo:8000 (THIS APP)
           └── everything else              → Finland uvicorn (app.main)

The reverse-proxy keeps the public URL unchanged while letting Tokyo
remain the single source of truth for HFT data — no replication, no
Tokyo-vs-Finland drift to debug.

Static assets (CSS/JS for the highfreq page) continue to be served by
Finland's nginx from ``/static/`` — only the dynamic endpoints need to
hit the Tokyo Postgres.

Run as::

    DATABASE_URL=postgresql://neucast:...@127.0.0.1:5433/neucast \\
        uvicorn app.highfreq.web_app:app --host 127.0.0.1 --port 8000

Or via the ``neucast-highfreq-web.service`` systemd unit, which sources
``DATABASE_URL`` (and all other env) from ``/etc/neucast/env``.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.highfreq.web import router as highfreq_router

logger = logging.getLogger(__name__)


# ``docs_url=None / redoc_url=None``: this app is internal-only,
# reached only via the Finland nginx reverse-proxy. Closing the docs
# endpoint shrinks the public attack surface to the four routes that
# actually matter (/highfreq + 3 × /api/highfreq/*).
app = FastAPI(
    title="NeuCast HFT (Tokyo)",
    description=(
        "Slim ASGI app serving the HFT routes for reverse-proxy from "
        "the Finland public nginx. Single source of truth lives in this "
        "process's local Postgres."
    ),
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(highfreq_router)


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Liveness ping for the reverse-proxy upstream check.

    Intentionally distinct from ``/api/highfreq/health`` (which checks
    DB freshness too) — this endpoint is for nginx to verify the
    upstream is reachable, even when Postgres is briefly down.
    """
    return JSONResponse(content={"ok": True, "service": "neucast-highfreq-web"})


@app.on_event("startup")
async def _on_startup() -> None:
    logger.info(
        "neucast-highfreq-web started — serving /highfreq + /api/highfreq/* "
        "for reverse-proxy from Finland nginx"
    )
