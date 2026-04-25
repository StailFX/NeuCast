"""High-frequency forecasting module — Phase A.

This package provides a 1-minute directional forecasting pipeline for
BTC-USDT (and any other Binance Spot symbol), running alongside the
existing daily NeuCast service on the same VPS.

Architecture, design decisions, and roadmap live in ``docs/highfreq/architecture.md``.

Module layout:

* :mod:`app.highfreq.l2_consumer`  – Binance WebSocket → in-memory ring buffer
* :mod:`app.highfreq.ofi_features` – OFI / microprice / depth-imbalance computation
* :mod:`app.highfreq.aggregator`   – 1-s and 1-m aggregation + Postgres writer
* :mod:`app.highfreq.trainer`      – CatBoost walk-forward training (Celery task)
* :mod:`app.highfreq.predictor`    – Live inference + FastAPI route
* :mod:`app.highfreq.backtest`     – Sim-backtest engine with maker / taker fees

Phase A is **sim-backtest only** — no orders are ever placed. See ADR-006 in
the architecture doc for the resource-constrained coexistence design.
"""

__all__ = [
    "l2_consumer",
    "ofi_features",
    "aggregator",
    "trainer",
    "predictor",
    "backtest",
]
