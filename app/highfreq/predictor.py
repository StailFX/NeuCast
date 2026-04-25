"""Live 1-minute predictor — FastAPI route.

Loads the latest CatBoost model from ``weights/highfreq/`` and exposes:

* ``GET /highfreq/latest`` — most recent 1-minute forecast as JSON
* ``GET /highfreq/history?hours=N`` — recent forecast / outcome pairs for the UI
* ``GET /highfreq/health`` — operator health endpoint (model age, last
  successful inference, WebSocket connection state)

The route is mounted by :mod:`app.main` alongside the existing daily-forecast
routes — see also the ``/highfreq`` UI template at ``templates/highfreq.html``.
"""
from __future__ import annotations

# Phase A.0 skeleton — implementation in Phase A.6.
