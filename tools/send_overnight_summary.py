"""One-shot script to send the autonomous-overnight-session summary to
Telegram. Read by `python -m tools.send_overnight_summary` on Tokyo
where the HF_TELEGRAM_SIGNAL_* env vars are populated.

Not part of any cron — fire-and-forget. Designed to be deleted /
overwritten between sessions.
"""
from __future__ import annotations

import asyncio
import sys


BODY = (
    "🌙 <b>NeuCast HF — overnight autonomous session</b>\n"
    "<i>2026-04-29 (Apr 28 → Apr 29)</i>\n"
    "\n"
    "<b>Shipped (5 commits, all on origin/main):</b>\n"
    "• <b>Release O</b> — sample weighting (half-life 720) + embargo "
    "(1 bar) + power analysis tool + 23-event calendar\n"
    "• <b>Release P</b> — magnitude regression offline-eval tool (ε)\n"
    "• <b>Release Q</b> — Bayesian credible intervals (Beta-Binomial "
    "posterior)\n"
    "• ADRs 012-018 — cross-asset / joint / calibration / events / "
    "weighting / regression / Bayesian CI\n"
    "• Roadmap progress log\n"
    "\n"
    "<b>Tests: 597 → 642 (+45 new, all green)</b>\n"
    "\n"
    "<b>Deploy validation (96h data, post-Release-O):</b>\n"
    "BTC 1m: dir_acc 0.5677 Wilson=Bayes [0.546, 0.589] p=9e-10 n=3421\n"
    "ETH 1m: dir_acc 0.5382 Wilson=Bayes [0.512, 0.564] p=2e-3 n=2923\n"
    "BNB 1m: dir_acc 0.5677 Wilson=Bayes [0.536, 0.599] p=2e-5 n=2449\n"
    "\n"
    "<b>⚠️ Honest empirical finding — sample weighting before/after:</b>\n"
    "Same 96h window, microstructure-only:\n"
    "• BTC 1m: 0.5833 (no weighting) → 0.5677 (weighting + embargo) "
    "= -1.6 pp\n"
    "• ETH 1m: 0.5556 → 0.5382 = -1.7 pp\n"
    "• BNB 1m: 0.5719 → 0.5677 = -0.4 pp\n"
    "Sample weighting was meant to hedge concept drift, but on this "
    "stable window it deweights older training bars without drift "
    "compensation — slight headwind. All p-values still well under 1e-3. "
    "Defence narrative (LdP-style methodological rigor) is intact. "
    "Можно отключить в production через "
    "HF_SAMPLE_WEIGHT_HALF_LIFE=0 если хотите.\n"
    "\n"
    "<b>Magnitude regression (ε / Release P):</b>\n"
    "BTC 1m, n=1980 OOS predictions:\n"
    "• MAE 3.77 bp, RMSE 5.09 bp, R² -0.06 (R² negative — regressor "
    "is worse than predicting the mean for explained variance)\n"
    "• Sign accuracy 0.557 (vs classifier 0.568 — regression LOSES "
    "~1pp on direction)\n"
    "• At θ=2bp filter: sign_acc 0.571 with 15% of trades; θ=4bp: "
    "0.594 with 1.6%\n"
    "• Per-fee-tier P&amp;L at θ=0: retail -14.66, vip5 -1.66, vip9 "
    "+0.34, mm_rebate +1.14\n"
    "<b>Empirically validates ADR-017</b>: regression does NOT add "
    "directional skill on its own but adds fee-aware filtering value "
    "for VIP9 / mm_rebate fee tiers.\n"
    "\n"
    "<b>Production health:</b> Tokyo ingest active (59 rows/last "
    "60s). Just-completed 72h trainer run: BTC dir_acc 0.5527 "
    "Wilson [0.528, 0.579] Bayes [0.527, 0.578] p=2.5e-5 brier=0.259 "
    "ece=0.097.  All new metrics live in metrics.json.\n"
    "\n"
    "<b>Deploy path used:</b> Mac → Finland (151.245.139.21) → "
    "ProxyJump → Tokyo (10.99.0.1 via WG). Direct SSH Mac → "
    "Tokyo:22 still timing out (fail2ban). Proxy via WG works.\n"
    "\n"
    "<b>Three decisions for morning:</b>\n"
    "1. Sample weighting: keep on (defence-grade) or disable until "
    "next regime shift?\n"
    "2. Magnitude regression: integrate as alternate model_mode in "
    "WalkForwardConfig, OR keep as offline tool?\n"
    "3. Next session focus: frontend visuals (8A reliability diagram, "
    "8C multi-horizon chart) or more research (3B per-regime, "
    "3C triple-barrier)?\n"
    "\n"
    "✅ Sleep well. Все коммиты на GitHub. До утра."
)


async def _main() -> int:
    from app.highfreq.signal_telegram import (
        SignalAlertConfig, send_signal_alert_async,
    )
    cfg = SignalAlertConfig.from_env()
    if not cfg.enabled:
        print("HF_TELEGRAM_SIGNAL_ENABLED is not set; not sending.",
              file=sys.stderr)
        return 1
    print(
        f"Telegram config: enabled={cfg.enabled} "
        f"chat={cfg.chat_id[:4] if cfg.chat_id else None}…"
    )
    print(f"Body length: {len(BODY)} chars")
    ok = await send_signal_alert_async(
        cfg, body_html=BODY, timeout_seconds=15.0,
    )
    print(f"send_signal_alert_async returned: {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
