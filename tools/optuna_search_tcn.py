"""Optuna hyperparameter search for the daily TCN + ensemble pipeline.

Designed to run on a beefy VPS (16+ vCPU, 32GB+ RAM) for ~6-12h.
Stores study state in SQLite so the run is resumable across
restarts / spot-instance preemptions.

What it tunes
=============

Hyperparameters that ``app.prediction`` reads from env vars at module
import time. We mutate ``os.environ`` then ``importlib.reload(app.prediction)``
between trials so each trial sees the new values without subprocess
overhead. The trials run sequentially in-process — TF graph state is
reset via ``tf.keras.backend.clear_session()`` after each trial.

Tunable knobs
-------------

* ``TCN_FINE_TUNE_EPOCHS``     ∈ [5, 25]
* ``TCN_FINE_TUNE_PATIENCE``   ∈ [2, 8]
* ``TCN_SNAPSHOT_LAST_N``      ∈ [1, 5]   (snapshot ensemble depth)
* ``BOOST_ITERS``              ∈ [200, 1500]
* ``BOOST_EARLY_STOP``         ∈ [30, 200]
* ``EMBARGO_DAYS``             ∈ [3, 10]
* ``HF_SAMPLE_WEIGHT_HALF_LIFE`` ∈ [120, 1440]  (off-by-default if 0,
                                                  but we always try > 0)

Validation set
--------------

Three diverse tickers (BTC-USD, GC=F, SPY) — different asset classes,
different volatility regimes, different macro sensitivity. Mean
MAPE × 0.5 + (1 - mean dir_acc) × 0.5 as the joint objective. Lower
is better; objective ≈ 0.01 → strong, ≈ 0.05 → mediocre.

Run
===

::

    python -m tools.optuna_search_tcn \\
        --n-trials 300 \\
        --tickers BTC-USD,GC=F,SPY \\
        --since-days 1500 --days-ahead 30 \\
        --study-name neucast_tcn_v1 \\
        --storage sqlite:///tools/optuna_studies/tcn.db \\
        --out tools/optuna_studies/best_params.json

Resume: just re-run with the same ``--study-name`` and ``--storage``;
Optuna reads completed trials from SQLite and picks up where it left off.

Output
======

* SQLite study at ``--storage`` (full trial history; can be browsed
  via ``optuna-dashboard sqlite:///<path>``).
* JSON at ``--out`` with the best params + final metric.
* Final stdout summary: top-10 trials sorted by objective.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("optuna_tcn_search")


# ─── Tunable knob definitions ──────────────────────────────────────
#
# Each entry: env-var-name -> (suggest_kind, *args)
# suggest_kind is one of optuna's ``suggest_int`` / ``suggest_float`` /
# ``suggest_categorical``. The trial function builds the env-var map
# from these specs.
KNOBS: dict[str, tuple] = {
    "TCN_FINE_TUNE_EPOCHS":      ("int", 5, 25),
    "TCN_FINE_TUNE_PATIENCE":    ("int", 2, 8),
    "TCN_SNAPSHOT_LAST_N":       ("int", 1, 5),
    "BOOST_ITERS":               ("int", 200, 1500, 100),     # step 100
    "BOOST_EARLY_STOP":          ("int", 30, 200, 10),         # step 10
    "EMBARGO_DAYS":              ("int", 3, 10),
    "HF_SAMPLE_WEIGHT_HALF_LIFE": ("int", 120, 1440, 60),     # step 60
}


def _suggest(trial, name: str, spec: tuple):
    """Translate a knob spec → an Optuna trial.suggest_* call."""
    kind = spec[0]
    if kind == "int":
        if len(spec) == 4:
            return trial.suggest_int(name, spec[1], spec[2], step=spec[3])
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"unknown suggest kind: {kind}")


def _reload_prediction_module():
    """Re-import ``app.prediction`` so its module-level env-var reads
    pick up the new ``os.environ`` values."""
    # Clear any cached TF state from previous trial.
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass

    import app.prediction as _p
    importlib.reload(_p)
    return _p


def _evaluate_on_ticker(
    prediction_mod,
    ticker: str,
    start_date: str,
    end_date: str,
    days_ahead: int,
) -> dict:
    """Run one full prediction pipeline on one ticker, return summary.

    On failure (insufficient data, yfinance hiccup, model crash) returns
    ``{"ok": False, "reason": ...}`` so the trial can either prune or
    average over the surviving tickers.
    """
    try:
        df = prediction_mod.fetch_and_preprocess(ticker, start_date, end_date)
        if df is None or len(df) < 200:
            return {"ok": False, "reason": "insufficient_data"}
        res = prediction_mod.run_prediction(
            df, days_ahead=int(days_ahead), sentiment_score=0.0,
            use_foundation=False,
        )
        return {
            "ok": True,
            "mape": float(res.get("mape", 0)),
            "r2": float(res.get("r2", 0)),
            "dir_acc": float(res.get("dir_acc", 0)),
        }
    except Exception as exc:
        return {"ok": False, "reason": f"error: {exc}"}


def make_objective(tickers: list[str], start_date: str, end_date: str,
                   days_ahead: int):
    """Build the Optuna objective closure with the validation set baked in."""

    def objective(trial) -> float:
        # 1. Sample knobs.
        sampled: dict[str, str] = {}
        for env_name, spec in KNOBS.items():
            v = _suggest(trial, env_name, spec)
            sampled[env_name] = str(v)
            os.environ[env_name] = str(v)
        # Snapshot ensemble is always on for tuning runs — we're tuning
        # its depth, not whether to use it.
        os.environ["TCN_SNAPSHOT_ENABLED"] = "1"
        # Disable yfinance caching across trials — we want clean runs.
        os.environ["YF_CACHE_TTL"] = "0"

        # 2. Reload module so the new env vars take effect.
        try:
            prediction_mod = _reload_prediction_module()
        except Exception as exc:
            logger.warning("reload failed: %s — pruning trial", exc)
            raise

        # 3. Evaluate on each validation ticker.
        per_ticker: list[dict] = []
        for sym in tickers:
            t0 = time.monotonic()
            r = _evaluate_on_ticker(
                prediction_mod, sym, start_date, end_date, days_ahead,
            )
            r["ticker"] = sym
            r["elapsed_seconds"] = round(time.monotonic() - t0, 1)
            per_ticker.append(r)
            if r.get("ok"):
                logger.info(
                    "  %s: MAPE=%.2f%% dir_acc=%.1f%% (%.1fs)",
                    sym, r["mape"], r["dir_acc"], r["elapsed_seconds"],
                )
            else:
                logger.warning("  %s failed: %s", sym, r.get("reason"))

        ok = [r for r in per_ticker if r.get("ok")]
        if not ok:
            # All tickers failed — bad params. Return a huge penalty
            # so Optuna learns to avoid this region.
            trial.set_user_attr("per_ticker", per_ticker)
            return 1.0

        mean_mape = sum(r["mape"] for r in ok) / len(ok)
        mean_dir_acc = sum(r["dir_acc"] for r in ok) / len(ok)
        # Note: dir_acc is in PERCENT (0-100). Normalize to [0, 1] to
        # combine with MAPE which is a percentage too.
        # Objective:  0.5 × MAPE  +  0.5 × (1 - dir_acc/100)
        # Lower is better. Roughly bounded ∈ [0, 1].
        # Example: MAPE 1.2% → 0.012; dir_acc 56% → (1-0.56) = 0.44.
        # → objective ≈ 0.5 × 0.012 + 0.5 × 0.44 = 0.226.
        # A ~5% MAPE with 50% dir_acc → 0.5 × 0.05 + 0.5 × 0.5 = 0.275.
        # An ideal model (MAPE 0.5%, dir_acc 65%) → 0.5 × 0.005 + 0.5 × 0.35 = 0.178.
        objective_value = 0.5 * (mean_mape / 100.0) + 0.5 * (1.0 - mean_dir_acc / 100.0)

        # Stash per-ticker metrics on the trial for downstream analysis.
        trial.set_user_attr("per_ticker", per_ticker)
        trial.set_user_attr("mean_mape", round(mean_mape, 4))
        trial.set_user_attr("mean_dir_acc", round(mean_dir_acc, 4))
        trial.set_user_attr("sampled_env", sampled)

        return float(objective_value)

    return objective


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=200,
                   help="number of Optuna trials (default 200)")
    p.add_argument("--tickers", default="BTC-USD,GC=F,SPY",
                   help="comma-separated validation tickers")
    p.add_argument("--since-days", type=int, default=1500)
    p.add_argument("--days-ahead", type=int, default=30)
    p.add_argument("--study-name", default="neucast_tcn_v1")
    p.add_argument("--storage", default="sqlite:///tools/optuna_studies/tcn.db",
                   help="Optuna storage URL (SQLite by default for resumability)")
    p.add_argument("--out", default="tools/optuna_studies/best_params.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Import optuna lazily so the script can be syntax-checked without it.
    try:
        import optuna
    except ImportError:
        logger.error("optuna is not installed. Run: pip install optuna")
        return 2

    # Ensure the storage dir exists (SQLite needs the parent folder).
    storage_path = args.storage.replace("sqlite:///", "")
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=int(args.since_days))
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    logger.info("=" * 60)
    logger.info(
        "Optuna TCN+ensemble search: n_trials=%d, validation=%s, "
        "window %s → %s, h=%d",
        args.n_trials, tickers, start_date, end_date, args.days_ahead,
    )
    logger.info("Study: %s @ %s", args.study_name, args.storage)
    logger.info("Tunable knobs: %s", list(KNOBS.keys()))

    # TPE sampler — Optuna's default Bayesian optimizer. Pruner: median.
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    # Count any already-completed trials (resumability).
    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    logger.info("Resuming from %d completed trials.", n_done)

    objective = make_objective(
        tickers, str(start_date), str(end_date), int(args.days_ahead),
    )

    study.optimize(
        objective,
        n_trials=max(0, int(args.n_trials) - n_done),
        show_progress_bar=False,
    )

    # ─── Persist best result ──────────────────────────────────────
    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "n_trials_total": len(study.trials),
        "n_trials_complete": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "best_trial_number": best.number,
        "best_objective": float(best.value) if best.value is not None else None,
        "best_params": best.params,
        "best_mean_mape": best.user_attrs.get("mean_mape"),
        "best_mean_dir_acc": best.user_attrs.get("mean_dir_acc"),
        "best_per_ticker": best.user_attrs.get("per_ticker"),
        "validation_tickers": tickers,
        "window_start": str(start_date),
        "window_end": str(end_date),
        "days_ahead": int(args.days_ahead),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("wrote best params to %s", args.out)

    # ─── stdout: top-10 trials ────────────────────────────────────
    print()
    print("=" * 80)
    print(f"Top-10 trials (objective = 0.5×MAPE + 0.5×(1-dir_acc))")
    print("=" * 80)
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: t.value)
    for i, t in enumerate(completed[:10], 1):
        attrs = t.user_attrs
        print(
            f"#{i:2d} trial {t.number:3d}  obj={t.value:.4f}  "
            f"MAPE={attrs.get('mean_mape', 'NA')}%  "
            f"dir_acc={attrs.get('mean_dir_acc', 'NA')}%  "
            f"params={t.params}"
        )
    print()
    print(f"Best params written to {args.out}")
    print(f"Study DB: {args.storage}")
    print(
        "Browse interactively:  "
        f"optuna-dashboard {args.storage}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
