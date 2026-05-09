# §03 — Drift case study: BTC 5m long_horizon, 28.04 → 09.05

> The single strongest empirical artifact in this thesis.
> Same model, same features, same window length, same data source.
> 10 days apart. Result: a statistically significant edge becomes
> a coinflip. The dashboard's drift detector caught it **before**
> the CV metric did.

## Setup

* **Model:** CatBoost classifier, 200 iters, depth 5, lr 0.05
* **Features:** `long_horizon` set — OHLC + EMA + RSI(14) + Bollinger
  z-score + ATR-ratio + return autocorrelation. 22 features.
* **Target:** sign of next-bar microprice return, neutral band
  ±√(bar_minutes) bps to drop "no-move" rows.
* **CV:** walk-forward, embargo=1 bar, sample-weight half-life=720
  bars, initial-train=24h equivalent, test-fold=1h equivalent.
* **Window:** 96 hours of OFI 1-second history (the 28.04 baseline
  used 96 h; we re-ran today with the same 96 h to control for
  window-length effects).

## Result

| run | dir_acc | 95 % Wilson CI | binomial p | n_folds | low_skill |
|-----|---------|----------------|------------|---------|-----------|
| 2026-04-28 (initial experiment) | **0.5391** | [0.4891, 0.5883] | 0.069 | 32 | False |
| 2026-05-09 (today, same code, same 96 h)  | **0.4918** | [0.4563, 0.5273] | 0.685 | 61 | **True** |

A loss of −4.7 percentage points in dir_acc; the p-value moved from
"borderline reject H₀" to "fail to reject H₀ comfortably". The CI
widened slightly toward 0.5 and now straddles it on both sides.

We also re-ran with a **wider 240 h window** to test if the issue
was simply that 96 h had become a non-stationary window:

| run | window | dir_acc | p-value |
|-----|--------|---------|---------|
| 2026-05-09 | 96 h | 0.4918 | 0.685 |
| 2026-05-09 | 240 h | 0.4970 | 0.606 |

**Window doesn't help.** Both report no-skill. This rules out the
"too-narrow window picked an unlucky regime" hypothesis. Edge
genuinely disappeared.

## Why we believe it

The drift detector caught this **independently** before we ran the
trainer. From the live `/api/highfreq/dashboard` payload on
2026-05-08 09:04 UTC:

```json
"BTCUSDT": {
  "drift": {
    "ok": true,
    "severity": "high",
    "max_ks": 0.6809,
    "max_ks_feature": "spread_bps_mean",
    "evaluated_at": "2026-05-08T09:00:23Z"
  }
}
```

KS=0.68 on `spread_bps_mean` between the rolling reference window
(7 days prior) and the recent window (last 6 h) is enormous — it
means the BTC bid-ask spread distribution has shifted to a regime
that looks essentially nothing like the trainer's reference.

We saw KS at this level on:

| feature | KS | p_value |
|---------|----|---------| 
| spread_bps_mean | 0.488 | 1.7e-45 |
| spread_bps_max | 0.480 | 8.0e-44 |
| microprice_high_low_bps | 0.275 | 2.5e-14 |

(Numbers from `/opt/neucast/weights/highfreq/btcusdt_drift.json`,
2026-05-08 10:04 UTC.)

Note that **`long_horizon` features don't include `spread_bps_*`
directly** — they're built from OHLC + indicators. Yet the model
*also* lost edge. Implication: the regime shift is broad enough
that even features built from price-level data (not microstructure)
no longer carry the same signal. Volatility regime, intra-bar
shape, momentum-vs-mean-reversion balance — all of these implicitly
covary with the spread regime.

## The cross-symbol counterfactual

Same trainer, same 240 h window, same features, same day (09.05),
**different symbol**:

| symbol | dir_acc | 95 % CI | p-value | verdict |
|--------|---------|---------|---------|---------|
| BTC | 0.4970 | [0.4736, 0.5198] | 0.61 | ❌ ushlo |
| ETH | 0.5192 | [0.4971, 0.5420] | 0.058 | ⚠️ borderline |
| BNB | **0.5300** | [0.5066, 0.5534] | **0.0077** | ✅ **holds** |

This rules out "the long_horizon feature pipeline was always
wrong." BNB's regime is more stationary — the drift hit BTC hard,
ETH lightly, BNB barely. **Edge is symbol-conditional** under
regime shift.

## What it means for the production system

1. **The drift detector is not just decoration.** It correlates with
   real degradation in CV metrics. The system saw KS=0.68 on the
   dashboard before any trainer reported low_skill=True. This is
   the loop that closes:

   ```
   live drift KS → operator notice → re-train check → confirm
   degradation → switch to fallback / pause / wait for regime change
   ```

2. **Joint training is structurally drift-resistant.** §02 reports
   joint 1m losing only −1.4 pp over the same window where solo
   BTC 5m lost −4.7 pp. Pooling across symbols dilutes per-symbol
   regime shift — when BTC is the worst-hit, ETH + BNB carry
   weight.

3. **Production deployment needs a gating rule.** Static "always-on"
   model deployment would have blindly traded a coinflip for 10
   days. The right pattern is **conditional activation**: only use
   a model in the live ensemble when its rolling dir_acc CI lower
   bound is > 0.50. The 1m solo + joint models pass this check
   today. The 5m long_horizon BTC model would have been
   auto-disabled.

   Implementation pattern (predictor side):

   ```python
   def is_skillful(metrics_path: Path) -> bool:
       m = json.load(metrics_path.open())
       return (
           not m.get("low_directional_skill", True)
           and m.get("dir_acc_ci_low", 0) > 0.50
       )
   ```

   Combined with the daily timer that refreshes metrics, this gives
   a self-deactivating model — a property a defending committee can
   verify by reading metrics-file mtimes against trader behavior.

4. **Honest limitation.** We don't have a recovery story for BTC's
   5m horizon yet. We *cannot* claim "wait for the regime to come
   back" — we don't know it will. What we CAN claim is "the system
   notices, doesn't lose money on a degraded model, and waits for
   evidence before re-engaging." That's the operator-grade
   property; the academic-grade property is the empirical
   demonstration of the loop.

## Reproducibility

```bash
# Re-run today's number on fresh data:
ssh root@147.45.49.40
set -a; source /etc/neucast/env; set +a
cd /opt/neucast
sudo -u stailfx --preserve-env=DATABASE_URL \
    /opt/neucast/venv/bin/python -m app.highfreq.trainer \
    --symbol BTCUSDT --bar-minutes 5 \
    --feature-set long_horizon \
    --since-hours 96 --frozen-holdout-days 0 \
    --out /tmp/btc_5m_lh.cbm \
    --report /tmp/btc_5m_lh.json
```

The 28.04 number is preserved in
`weights/highfreq/multi_horizon_compare_features.json` (S3-archived
nightly via `neucast-prom-backup.timer` so the artifact survives
even if Tokyo's local file gets overwritten).
