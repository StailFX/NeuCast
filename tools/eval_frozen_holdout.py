"""Evaluate the deployed CatBoost model on the frozen holdout.

Why this exists
===============

The trainer (``app.highfreq.trainer``) does walk-forward CV — that's
out-of-sample for individual folds, but hyperparameters (CatBoost
depth, learning rate, neutral-band threshold) were chosen by looking
at the CV outputs. This is a small but real leak: the "OOS" CV
accuracy is conditioned on choices that saw earlier CV results.

A *frozen* holdout closes this gap. Set ``--frozen-holdout-days 7`` in
the trainer (default since 2026-04-27) and the trainer literally
cannot load the last 7 days. This script then loads exactly those
7 days and evaluates the deployed ``.cbm`` against them. The number
it prints is the **only** dir_acc number that has nothing in common
with hyperparameter tuning — academic-grade defence answer to "how
do I know you didn't overfit your CV?".

What it does
------------

1. Load the deployed CatBoost model from ``weights/highfreq/<symbol>_1m.cbm``.
2. Read the model's metrics.json to discover its training cutoff
   (``holdout_cutoff_iso``). Use that as the **lower** bound of the
   evaluation window — anything before is data the model already
   saw at training.
3. Load 1-second OFI rows from Postgres for the holdout window.
4. Run the canonical feature pipeline (same code paths as the trainer
   uses — that's the whole reason ``feature_pipeline`` is a separate
   module).
5. Predict on every minute in the holdout. Compare against the actual
   ``y`` ( ``sign(return_1m)`` after neutral-band drop, same as
   trainer).
6. Compute & emit:
   * ``dir_acc`` (point estimate)
   * Bootstrap 95 % CI (1000 resamples, seed=42 — matches trainer)
   * One-sided binomial p-value (``p > 0.5``)
   * Sample size, base rate, log-loss
7. Write a JSON report next to the model:
   ``weights/highfreq/<symbol>_1m_holdout.json`` — picked up by the
   web UI (``/api/highfreq/training_report?include_holdout=1``).

Run from Tokyo
--------------

::

    # Default: evaluate yesterday's deployed model on the holdout.
    python -m tools.eval_frozen_holdout --symbol BTCUSDT

    # Override the model path / report path (testing different cbm).
    python -m tools.eval_frozen_holdout \\
        --symbol BTCUSDT \\
        --weights weights/highfreq/btcusdt_1m.cbm \\
        --report  weights/highfreq/btcusdt_1m_holdout.json

Schedule on Tokyo via the systemd timer
``neucast-highfreq-holdout-eval@.timer`` (runs weekly, e.g. Mondays
04:30 UTC — staggered after the daily trainer at 04:00 UTC so it picks
up the freshly retrained model).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HoldoutReport:
    """Match the on-disk JSON shape — readable by the web UI."""
    symbol: str
    evaluated_at: str
    weights_path: str
    weights_mtime_iso: str
    holdout_cutoff_iso: str | None
    n_seconds_loaded: int
    n_minutes_after_aggregation: int
    n_minutes_after_neutral_drop: int
    n_eligible: int
    base_rate: float
    dir_acc: float
    dir_acc_ci_low: float
    dir_acc_ci_high: float
    dir_acc_p_value: float
    log_loss: float
    elapsed_seconds: float

    def to_json(self) -> str:
        def _scrub(o: Any) -> Any:
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: _scrub(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_scrub(v) for v in o]
            return o
        return json.dumps(_scrub(asdict(self)), indent=2, default=str)


def _read_holdout_cutoff_from_metrics(metrics_path: Path) -> Optional[str]:
    """Look up ``holdout_cutoff_iso`` from the trainer's metrics.json.

    None if metrics.json doesn't exist (cold start) or doesn't carry
    the field (older training run before this feature was added).
    Caller falls back to a default window in that case.
    """
    if not metrics_path.exists():
        return None
    try:
        d = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return d.get("holdout_cutoff_iso") or None


def _is_bootstrap_mode_run(metrics_path: Path) -> bool:
    """True when the latest trainer run was in bootstrap mode
    (``frozen_holdout_days=0``) — meaning the model was fit on the
    FULL dataset and there is no honest holdout to evaluate against.

    Crucial guard: without this check, holdout-eval would happily
    score the model on data it was trained on, producing leaked
    near-100 % dir_acc. Defense-grade: refuse to publish a number
    that's known-bogus.
    """
    if not metrics_path.exists():
        return False
    try:
        d = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    holdout_days = d.get("frozen_holdout_days")
    if holdout_days is None:
        # Legacy report (pre-release-A) — no field at all. Treat as
        # bootstrap to avoid the leak in either case.
        return True
    return int(holdout_days) <= 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.eval_frozen_holdout",
                                description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT", help="trading pair (default: BTCUSDT)")
    p.add_argument("--weights", default=None,
                   help="path to .cbm; default = weights/highfreq/<sym>_1m.cbm")
    p.add_argument("--report", default=None,
                   help="path to write JSON report; default = <weights>_holdout.json")
    p.add_argument("--fallback-days", type=int, default=7,
                   help="if metrics.json has no holdout_cutoff_iso, use the last "
                        "N days as the eval window (default 7)")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    symbol = args.symbol.upper()
    weights_path = (
        Path(args.weights) if args.weights
        else Path(f"weights/highfreq/{symbol.lower()}_1m.cbm")
    )
    if not weights_path.exists():
        logger.error("model weights not found: %s — has the trainer run yet?", weights_path)
        return 2

    metrics_path = weights_path.with_suffix("").with_name(weights_path.stem + "_metrics.json")

    # Refuse to run if the trainer was in bootstrap mode (no holdout
    # was reserved) — evaluating the model on the data it was trained
    # on would leak and produce a meaningless ~99 % dir_acc number,
    # which is much WORSE than no number at all (bootstrap regression
    # 2026-04-27).
    if _is_bootstrap_mode_run(metrics_path):
        logger.warning(
            "trainer is in bootstrap mode (frozen_holdout_days=0) — "
            "no honest holdout to evaluate against. Skipping eval. "
            "Once enough data accumulates, raise --frozen-holdout-days "
            "in the trainer's systemd ExecStart."
        )
        # Still write a stub report so the heartbeat fires + the UI
        # surfaces "bootstrap" rather than missing-data.
        report = HoldoutReport(
            symbol=symbol,
            evaluated_at=__import__("pandas").Timestamp.utcnow().isoformat(),
            weights_path=str(weights_path),
            weights_mtime_iso=__import__("pandas").Timestamp(
                weights_path.stat().st_mtime, unit="s", tz="UTC",
            ).isoformat(),
            holdout_cutoff_iso=None,
            n_seconds_loaded=0,
            n_minutes_after_aggregation=0,
            n_minutes_after_neutral_drop=0,
            n_eligible=0,
            base_rate=float("nan"),
            dir_acc=float("nan"),
            dir_acc_ci_low=float("nan"),
            dir_acc_ci_high=float("nan"),
            dir_acc_p_value=float("nan"),
            log_loss=float("nan"),
            elapsed_seconds=0.0,
        )
        report_path = (
            Path(args.report) if args.report
            else weights_path.with_suffix("").with_name(
                weights_path.stem + "_holdout.json"
            )
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.to_json())
        try:
            from app.highfreq.cron_metrics import write_cron_success
            write_cron_success(
                "neucast_hf_holdout_eval_last_success_timestamp_seconds",
                file_stem=f"neucast_hf_holdout_{symbol.lower()}",
                labels={"symbol": symbol, "mode": "bootstrap_skip"},
            )
        except Exception:
            pass
        return 0

    cutoff_iso = _read_holdout_cutoff_from_metrics(metrics_path)
    if cutoff_iso is None:
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        cutoff_iso = (_dt.now(tz=_tz.utc) - _td(days=args.fallback_days)).isoformat()
        logger.warning(
            "metrics.json missing holdout_cutoff_iso — falling back to "
            "last %d days (cutoff=%s)", args.fallback_days, cutoff_iso,
        )
    else:
        logger.info("evaluating holdout: cutoff=%s", cutoff_iso)

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is required")
        return 2

    # Lazy imports — keeps `python -m tools.eval_frozen_holdout --help`
    # snappy even on a fresh checkout without catboost installed.
    import numpy as np
    import pandas as pd
    from app.highfreq.feature_pipeline import (
        FEATURE_COLUMNS, NEUTRAL_BAND_BPS, make_supervised,
    )
    from app.highfreq.trainer import (
        _binary_logloss, binom_test_p_greater_half, bootstrap_dir_acc_ci,
        load_seconds, _make_supervised_for_feature_set,
    )

    # Read the model's feature_set from its metrics.json so this script
    # works for cross_asset / long_horizon models, not just the legacy
    # 18-col microstructure default. Without this dispatch the
    # CatBoost predict_proba blows up with "Feature N is present in
    # model but not in pool" because the X matrix is 18-wide while
    # the model expects 22 (BTC cross_asset) or 27 (ETH/BNB cross_asset).
    feature_set_for_eval = "microstructure"
    metrics_path_for_eval = weights_path.with_name(
        weights_path.stem + "_metrics.json"
    )
    if metrics_path_for_eval.exists():
        try:
            metrics_json = json.loads(metrics_path_for_eval.read_text())
            feature_set_for_eval = str(
                metrics_json.get("feature_set", "microstructure")
            )
        except Exception:
            feature_set_for_eval = "microstructure"
    logger.info(
        "model feature_set=%s (read from metrics.json)",
        feature_set_for_eval,
    )

    started = time.monotonic()

    # Load enough seconds to cover the holdout window — fallback_days
    # is generous; 24-hour padding ensures we don't miss the boundary.
    df_secs = load_seconds(
        dsn, symbol=symbol, since_hours=(args.fallback_days + 1) * 24,
    )
    cutoff_ts = pd.Timestamp(cutoff_iso)
    df_secs = df_secs.loc[df_secs["ts"] >= cutoff_ts].copy()
    n_secs = len(df_secs)
    logger.info("loaded %d seconds in holdout window", n_secs)

    if n_secs == 0:
        logger.warning("no holdout data yet — model is younger than cutoff")
        report = HoldoutReport(
            symbol=symbol,
            evaluated_at=pd.Timestamp.utcnow().isoformat(),
            weights_path=str(weights_path),
            weights_mtime_iso=pd.Timestamp(weights_path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
            holdout_cutoff_iso=cutoff_iso,
            n_seconds_loaded=0,
            n_minutes_after_aggregation=0,
            n_minutes_after_neutral_drop=0,
            n_eligible=0,
            base_rate=float("nan"),
            dir_acc=float("nan"),
            dir_acc_ci_low=float("nan"),
            dir_acc_ci_high=float("nan"),
            dir_acc_p_value=float("nan"),
            log_loss=float("nan"),
            elapsed_seconds=time.monotonic() - started,
        )
    else:
        from app.highfreq.feature_pipeline import aggregate_to_minute
        minute_df = aggregate_to_minute(df_secs)
        n_min = len(minute_df)

        # cross_asset on ETH/BNB needs BTC seconds aligned to the same
        # window as a reference. For BTC itself there's no reference
        # (would be identity); microstructure / long_horizon don't need
        # one either.
        ref_df_secs = None
        if feature_set_for_eval == "cross_asset" and symbol.upper() != "BTCUSDT":
            try:
                ref_df_secs = load_seconds(
                    dsn, symbol="BTCUSDT",
                    since_hours=(args.fallback_days + 1) * 24,
                )
                ref_df_secs = ref_df_secs.loc[
                    ref_df_secs["ts"] >= cutoff_ts
                ].copy()
                logger.info(
                    "loaded %d BTCUSDT reference seconds for cross_asset",
                    len(ref_df_secs),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "failed to load BTC reference for cross_asset; "
                    "cross-asset cols will be zero-filled: %s", exc,
                )
                ref_df_secs = None

        X, y, meta = _make_supervised_for_feature_set(
            df_secs,
            feature_set=feature_set_for_eval,
            bar_minutes=1,
            reference_df_secs=ref_df_secs,
            target_symbol=symbol,
        )
        n_eligible = len(X)

        if n_eligible == 0:
            logger.warning("0 eligible bars in holdout (all neutral-band-dropped?)")
            report = HoldoutReport(
                symbol=symbol,
                evaluated_at=pd.Timestamp.utcnow().isoformat(),
                weights_path=str(weights_path),
                weights_mtime_iso=pd.Timestamp(weights_path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                holdout_cutoff_iso=cutoff_iso,
                n_seconds_loaded=n_secs,
                n_minutes_after_aggregation=n_min,
                n_minutes_after_neutral_drop=0,
                n_eligible=0,
                base_rate=float("nan"),
                dir_acc=float("nan"),
                dir_acc_ci_low=float("nan"),
                dir_acc_ci_high=float("nan"),
                dir_acc_p_value=float("nan"),
                log_loss=float("nan"),
                elapsed_seconds=time.monotonic() - started,
            )
        else:
            from catboost import CatBoostClassifier
            clf = CatBoostClassifier()
            clf.load_model(str(weights_path))

            # X is already in the canonical column order returned by
            # _make_supervised_for_feature_set — pass through directly
            # rather than reindexing on FEATURE_COLUMNS (which would
            # silently truncate cross_asset's 22/27 cols to 18).
            X_arr = X.to_numpy()
            proba = clf.predict_proba(X_arr)
            # Defensive: predict_proba shape (N, 2). Class-1 column is
            # P(up). Mirrors how walk_forward_evaluate consumes it.
            if proba.ndim == 2 and proba.shape[1] >= 2:
                p_up = proba[:, 1]
            else:
                p_up = np.asarray(proba).ravel()
            y_pred = (p_up > 0.5).astype(np.int8)
            y_true = y.to_numpy().astype(np.int8)

            n_correct = int((y_pred == y_true).sum())
            dir_acc = n_correct / max(1, len(y_true))
            ll = _binary_logloss(y_true, p_up)
            point, ci_lo, ci_hi = bootstrap_dir_acc_ci(y_true, y_pred)
            p_value = binom_test_p_greater_half(n_correct, len(y_true))
            base_rate = float(max(y_true.mean(), 1.0 - y_true.mean()))

            logger.info(
                "HOLDOUT EVAL | symbol=%s | n=%d | dir_acc=%.4f [%.4f, %.4f] | "
                "p=%.4f | logloss=%.4f | base_rate=%.4f",
                symbol, len(y_true),
                point, ci_lo, ci_hi, p_value, ll, base_rate,
            )

            report = HoldoutReport(
                symbol=symbol,
                evaluated_at=pd.Timestamp.utcnow().isoformat(),
                weights_path=str(weights_path),
                weights_mtime_iso=pd.Timestamp(weights_path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                holdout_cutoff_iso=cutoff_iso,
                n_seconds_loaded=n_secs,
                n_minutes_after_aggregation=n_min,
                n_minutes_after_neutral_drop=n_eligible,
                n_eligible=n_eligible,
                base_rate=base_rate,
                dir_acc=point,
                dir_acc_ci_low=ci_lo,
                dir_acc_ci_high=ci_hi,
                dir_acc_p_value=p_value,
                log_loss=ll,
                elapsed_seconds=time.monotonic() - started,
            )

    report_path = (
        Path(args.report) if args.report
        else weights_path.with_suffix("").with_name(weights_path.stem + "_holdout.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_json())
    logger.info("holdout report written to %s", report_path)

    # Heartbeat for the cron-stale alert framework.
    try:
        from app.highfreq.cron_metrics import write_cron_success
        write_cron_success(
            "neucast_hf_holdout_eval_last_success_timestamp_seconds",
            file_stem=f"neucast_hf_holdout_{symbol.lower()}",
            labels={"symbol": symbol},
        )
    except Exception:
        # Heartbeat is best-effort.
        logger.warning("holdout heartbeat write failed", exc_info=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
