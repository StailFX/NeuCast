"""Hourly drift check — compares last 1h of feature distribution
to the trainer's reference window and ships a Telegram alert if
KS exceeds threshold (T.18.b).

Pulls feature data from the live OFI table for both windows:
* Reference: bars from ``trainer_reference_hours`` ago, ending at
  the trainer's holdout cutoff (so we compare against EXACTLY what
  the trainer fit on, not contaminated by the held-out window).
* Recent: last ``recent_hours`` of bars.

Writes a JSON artifact to ``weights/highfreq/<sym>_drift.json`` so
the dashboard endpoint can surface drift status without re-running
the test.

Run from a systemd timer (suggested cadence: every 1h)::

    python -m tools.drift_check --symbol BTCUSDT
    python -m tools.drift_check --symbol BTCUSDT --threshold 0.20

Telegram is optional — alert only fires when severity ∈ {warn, high}
AND ``HF_TELEGRAM_SIGNAL_ENABLED=1``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_minute_features(
    df_secs, *, feature_set: str, bar_minutes: int,
    target_symbol: str, reference_df_secs=None,
    futures_df_secs=None,
):
    """Run the supervised pipeline (no target) just to get the
    feature DataFrame matching what the trainer fit on.

    For drift, we only need the feature columns — the target
    isn't relevant. We piggyback on ``_make_supervised_for_feature_set``
    and ignore the (y, meta) outputs.

    ``futures_df_secs`` (T.23.b extension): when feature_set is
    ``microstructure_v3``, the supervised builder needs futures
    seconds at the same wall-clock window so the 5 futures-basis
    cols can be computed. Without it those cols zero-fill and the
    drift KS on them is uniformly 0 (uninformative).
    """
    from app.highfreq.trainer import _make_supervised_for_feature_set

    X, _, meta = _make_supervised_for_feature_set(
        df_secs,
        feature_set=feature_set,
        bar_minutes=bar_minutes,
        reference_df_secs=reference_df_secs,
        target_symbol=target_symbol,
        futures_df_secs=futures_df_secs,
    )
    return X, meta


def _fetch_metrics_json(weights_dir: Path, symbol: str) -> dict | None:
    p = weights_dir / f"{symbol.lower()}_1m_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics.json read failed for %s: %s", symbol, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True)
    p.add_argument("--threshold", type=float, default=0.15,
                   help="alert when max KS statistic ≥ threshold")
    p.add_argument("--reference-hours", type=float, default=24.0,
                   help="size of trainer-reference window in hours")
    p.add_argument("--recent-hours", type=float, default=1.0,
                   help="size of recent serve window in hours")
    p.add_argument(
        "--reference-mode", choices=("rolling", "training_cutoff", "auto"),
        default="auto",
        help="how to position the reference window's right edge:\n"
             "  rolling — always [now - recent_hours - reference_hours, "
             "now - recent_hours] (the LAST 24h before this hour, always "
             "fresh). Use this when you want to detect intra-day regime "
             "variation.\n"
             "  training_cutoff — pin to the model's metrics.json "
             "holdout_cutoff_iso. Use this to detect drift WRT the data "
             "the model was trained on (slow-moving, may be stale).\n"
             "  auto (default) — training_cutoff if metrics has it, else "
             "rolling. (T.22-extension fix, 2026-05-04: prior behaviour "
             "was 'training_cutoff with rolling fallback' but the silent "
             "fallback hid stale-reference bugs — now mode is explicit.)",
    )
    p.add_argument("--out", default=None,
                   help="JSON output path; default "
                        "weights/highfreq/<sym>_drift.json")
    p.add_argument("--root", default=os.getenv("NEUCAST_ROOT", "/opt/neucast"))
    p.add_argument("--no-telegram", action="store_true",
                   help="skip Telegram alert even if drift detected")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is required")
        return 2

    from app.highfreq.drift_detector import (
        compute_per_feature_drift,
        summarise_drift,
        alert_payload,
    )
    from app.highfreq.trainer import load_seconds

    sym = args.symbol.upper()
    root = Path(args.root)
    weights_dir = root / "weights" / "highfreq"

    metrics = _fetch_metrics_json(weights_dir, sym) or {}
    feature_set = metrics.get("feature_set", "microstructure")
    bar_minutes = int(metrics.get("bar_minutes", 1))
    cutoff_iso = metrics.get("holdout_cutoff_iso")

    # Reference window: explicit mode selection (T.22-extension fix
    # 2026-05-04). Earlier code did ``training_cutoff with silent
    # rolling fallback`` — but when an old model's metrics.json carried
    # a 3-day-stale holdout_cutoff_iso, the reference window was
    # 3-days-old vs 1-hour-recent, producing KS=0.95+ false positives
    # on stable features like ``spread_bps_mean``. Now the operator
    # picks the policy explicitly via --reference-mode.
    mode = args.reference_mode
    if mode == "auto":
        mode = "training_cutoff" if cutoff_iso else "rolling"
    if mode == "training_cutoff":
        if not cutoff_iso:
            logger.error(
                "--reference-mode=training_cutoff requested but metrics.json "
                "lacks holdout_cutoff_iso. Use --reference-mode=rolling or "
                "retrain with --frozen-holdout-days >= 1."
            )
            return 2
        ref_end = datetime.fromisoformat(cutoff_iso)
    else:  # rolling
        ref_end = datetime.now(tz=timezone.utc) - timedelta(hours=args.recent_hours)
    ref_start = ref_end - timedelta(hours=args.reference_hours)

    # Recent window: now back ``recent_hours``.
    rec_end = datetime.now(tz=timezone.utc)
    rec_start = rec_end - timedelta(hours=args.recent_hours)

    logger.info(
        "drift check: %s feature_set=%s bar_minutes=%d "
        "reference=[%s..%s] recent=[%s..%s] threshold=%.3f",
        sym, feature_set, bar_minutes,
        ref_start.isoformat(), ref_end.isoformat(),
        rec_start.isoformat(), rec_end.isoformat(),
        args.threshold,
    )

    # Pull seconds for both windows. Cheapest: pull the union, slice.
    union_hours = (rec_end - ref_start).total_seconds() / 3600.0 + 1
    df_all = load_seconds(dsn, symbol=sym, since_hours=union_hours)
    df_all["ts"] = __import__("pandas").to_datetime(df_all["ts"], utc=True)

    df_ref_secs = df_all[(df_all["ts"] >= ref_start) & (df_all["ts"] < ref_end)].copy()
    df_rec_secs = df_all[(df_all["ts"] >= rec_start) & (df_all["ts"] <= rec_end)].copy()
    logger.info(
        "loaded %d ref seconds, %d recent seconds",
        len(df_ref_secs), len(df_rec_secs),
    )

    # cross_asset needs BTC reference at the same windows.
    ref_btc, rec_btc = None, None
    if feature_set == "cross_asset" and sym != "BTCUSDT":
        df_btc = load_seconds(dsn, symbol="BTCUSDT", since_hours=union_hours)
        df_btc["ts"] = __import__("pandas").to_datetime(df_btc["ts"], utc=True)
        ref_btc = df_btc[(df_btc["ts"] >= ref_start) & (df_btc["ts"] < ref_end)].copy()
        rec_btc = df_btc[(df_btc["ts"] >= rec_start) & (df_btc["ts"] <= rec_end)].copy()

    # microstructure_v3 needs futures seconds at the same windows so
    # the basis_bps / mark_premium / funding_bps cols can be computed.
    # Without this, the 5 futures cols zero-fill in BOTH ref and recent
    # → KS=0 on those cols → drift only detects the 18 base cols.
    ref_fut, rec_fut = None, None
    if feature_set == "microstructure_v3":
        try:
            df_fut = load_seconds(
                dsn, symbol=sym, since_hours=union_hours, venue="futures",
            )
            df_fut["ts"] = __import__("pandas").to_datetime(
                df_fut["ts"], utc=True,
            )
            ref_fut = df_fut[
                (df_fut["ts"] >= ref_start) & (df_fut["ts"] < ref_end)
            ].copy()
            rec_fut = df_fut[
                (df_fut["ts"] >= rec_start) & (df_fut["ts"] <= rec_end)
            ].copy()
            logger.info(
                "loaded %d ref futures seconds, %d recent futures seconds (v3)",
                len(ref_fut), len(rec_fut),
            )
        except Exception:
            logger.warning(
                "failed to load %s futures seconds for v3 drift; "
                "5 futures cols will zero-fill (KS will be 0 on them)",
                sym, exc_info=True,
            )
            ref_fut, rec_fut = None, None

    # Build features for both windows.
    X_ref, _ = _build_minute_features(
        df_ref_secs, feature_set=feature_set, bar_minutes=bar_minutes,
        target_symbol=sym, reference_df_secs=ref_btc,
        futures_df_secs=ref_fut,
    )
    X_rec, _ = _build_minute_features(
        df_rec_secs, feature_set=feature_set, bar_minutes=bar_minutes,
        target_symbol=sym, reference_df_secs=rec_btc,
        futures_df_secs=rec_fut,
    )

    if X_ref.empty or X_rec.empty:
        logger.warning(
            "drift check: insufficient data (ref=%d, recent=%d bars) — skipping",
            len(X_ref), len(X_rec),
        )
        return 1

    drifts = compute_per_feature_drift(X_ref, X_rec)
    summary = summarise_drift(drifts, threshold=args.threshold)
    payload = alert_payload(summary)
    payload["symbol"] = sym
    payload["feature_set"] = feature_set
    payload["bar_minutes"] = bar_minutes
    payload["reference_window"] = {
        "start": ref_start.isoformat(),
        "end": ref_end.isoformat(),
        "n_bars": int(len(X_ref)),
        "mode": mode,  # rolling | training_cutoff
    }
    payload["recent_window"] = {
        "start": rec_start.isoformat(),
        "end": rec_end.isoformat(),
        "n_bars": int(len(X_rec)),
    }
    payload["evaluated_at"] = datetime.now(tz=timezone.utc).isoformat()

    out_path = Path(args.out) if args.out else (
        weights_dir / f"{sym.lower()}_drift.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s (severity=%s max_ks=%.4f)",
                out_path, payload["severity"], payload["max_ks"])

    # Telegram alert when drifted.
    if not args.no_telegram and summary.is_drifted():
        try:
            from tools.telegram_notify import send_telegram

            bot_token = os.environ.get("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "").strip()
            chat_id = os.environ.get("HF_TELEGRAM_SIGNAL_CHAT_ID", "").strip()
            enabled = os.environ.get("HF_TELEGRAM_SIGNAL_ENABLED", "").strip().lower()
            if bot_token and chat_id and enabled in ("1", "true", "yes", "on"):
                emoji = "🚨" if payload["severity"] == "high" else "⚠️"
                top_lines = "\n".join(
                    f"  • {f['feature']}: KS={f['ks_stat']:.3f} "
                    f"(ref_mean={f['reference_mean']:.3g} → "
                    f"recent={f['recent_mean']:.3g})"
                    for f in payload["top_features"]
                )
                body = (
                    f"{emoji} <b>NeuCast HF feature drift detected — {sym}</b>\n"
                    f"\n"
                    f"severity: <b>{payload['severity'].upper()}</b>\n"
                    f"max KS: <b>{payload['max_ks']:.3f}</b> "
                    f"(threshold {payload['threshold']:.2f}) "
                    f"on feature <code>{payload['max_ks_feature']}</code>\n"
                    f"alarming features: {payload['n_features_alarming']} "
                    f"of {payload['n_features']}\n"
                    f"\n"
                    f"<b>Top 5 by KS:</b>\n"
                    f"<pre>{top_lines}</pre>\n"
                    f"\n"
                    f"reference: n={payload['reference_window']['n_bars']} bars\n"
                    f"recent: n={payload['recent_window']['n_bars']} bars\n"
                    f"feature_set: {feature_set}\n"
                )
                code, resp = send_telegram(
                    bot_token=bot_token, chat_id=chat_id,
                    body_html=body,
                )
                logger.info("telegram alert: code=%d", code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram alert failed (non-fatal): %s", exc)

    return 0 if not summary.is_drifted() else 1


if __name__ == "__main__":
    raise SystemExit(main())
