"""Live 1-minute predictor — Phase B placeholder.

This module was originally scaffolded in Phase A.0 to host live inference
(loading the CatBoost model from ``weights/highfreq/``, scoring the most
recent feature row, exposing per-minute forecasts via FastAPI). Phase A
shipped a different surface area:

* :mod:`app.highfreq.web`      — UI router + status / health endpoints
* :mod:`app.highfreq.trainer`  — walk-forward CV + bootstrap CI + CLI
* :mod:`app.highfreq.backtest` — sim-backtest with maker/taker P&L

Phase A is sim-only by design (see :doc:`ADR-005 <../../docs/highfreq/architecture>`),
so live inference is deliberately deferred until Phase B has accumulated
2 weeks of data and demonstrated `dir_acc ≥ 53 %` with positive maker P&L.

Until then, "live forecast" on the UI is the **most recent 1-second
feature row** (microprice, OFI, depth-imbalance) — directly observable
truth, not a model output that could be wrong.
"""
from __future__ import annotations

# Intentionally empty — Phase B will populate this module.
