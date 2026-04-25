"""CatBoost walk-forward trainer (Celery task).

Trains a binary classifier on ``sign(return_1m)`` using the columns of
``highfreq_features_1m``. Runs as a Celery beat task at 04:00 UTC daily —
the lowest-traffic hour for the other apps on this VPS (see ADR-006).

Design choices
--------------

* **Walk-forward, not random k-fold.** Time-series leakage is the most
  common silent killer in financial ML; we never let a model see a future
  bar during training.
* **Class-weighted log-loss.** BTC return signs are roughly balanced
  (51 % positive, 49 % negative on 1-minute) but we weight inversely to
  empirical frequency to keep the calibration honest.
* **Single-thread training.** ``thread_count=2`` to leave headroom for
  the L2 consumer and other apps on the shared VPS (ADR-006).
* **Train + validation + test split** = 70 / 15 / 15, time-ordered.
  The test fold is also used to populate the sim-backtest's ``maker_pnl``
  and ``taker_pnl`` curves (see :mod:`app.highfreq.backtest`).

Loss alignment with reported metric — see ADR-004.
"""
from __future__ import annotations

# Phase A.0 skeleton — implementation lands in Phase A.4.
