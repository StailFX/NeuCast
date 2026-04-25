"""Sim-backtest engine for the 1-minute directional model.

Replays historical ``highfreq_features_1m`` rows through the trained model
and produces:

* **Maker P&L curve** — assumes every signal is filled as a post-only limit
  order at the prevailing mid, paying the −0.001 % rebate. Filled at a
  configurable rate (default 50 %).
* **Taker P&L curve** — assumes every signal crosses the spread immediately,
  paying 0.1 % per trade.
* **Fill-rate sweep** — same maker P&L recomputed at 30 / 50 / 70 / 100 %
  fill assumptions, surfaced on the UI as a sensitivity strip.
* **Per-bucket directional accuracy** — confusion matrix and rolling 7-day
  dir_acc, with bootstrap 95 % CI (re-using ``app.honest_skill``).

Design rationale: see ADR-005 — honest, separated maker / taker reporting is
the core value proposition of this module.
"""
from __future__ import annotations

# Phase A.0 skeleton — implementation in Phase A.5.
