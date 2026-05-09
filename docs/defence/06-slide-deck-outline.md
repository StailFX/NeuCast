# §06 — Slide deck outline

> Mapping from the markdown thesis sections to a 20–25 minute defence
> presentation. Each slide gets: a 1-line title, the single SVG that
> visualises it, and 2–3 talking-point bullets. Use this as the
> directly-importable scaffold; format is your choice (Beamer /
> Keynote / Pitch / Google Slides).

## Target arc — 22 slides, ~60 sec/slide

```
┌───────────────────────┐
│ 0. Cover              │
├───────────────────────┤
│ 1. The question        │ Why this thesis exists
│ 2. The system          │ Two-VPS topology figure
│ 3. Data plane          │ OFI ingest → feature pipeline → predictor
├───────────────────────┤
│ 4. Solo results       │ fig-01 LEFT half (per-symbol bars)
│ 5. Statistical power  │ fig-05 (folds + CI tightness)
│ 6. Joint results      │ fig-01 FULL (with joint bar)
│ 7. Frozen holdout     │ fig-01 hatched overlay
├───────────────────────┤
│ 8. Calibration        │ fig-02 conditional accuracy
│ 9. Fee-tier economics │ fig-04
├───────────────────────┤
│10. Multi-horizon eval │ table (5m/15m/60m verdicts)
│11. Drift case study   │ fig-03 + side-by-side dir_acc collapse
│12. Drift detection    │ live dashboard screenshot
│13. Why joint helps    │ cross-symbol counterfactual numbers
├───────────────────────┤
│14. Anti-skill detector│ winrate-CI logic + halt policy
│15. Calibration gate   │ require_calibrated invariant
│16. Conformal intervals│ split-conformal coverage guarantee
│17. Engineering depth  │ 14 heartbeats + 5 alerts table
├───────────────────────┤
│18. What didn't work   │ 5m BTC long_horizon, cross_asset 1m, 60m solo
│19. Limitations        │ 240h non-stationarity, regime-conditional edge
│20. Future work        │ Phase 2.3 ensemble flip, longer-horizon TA
│21. Q&A                │ pointer to docs/defence/00-narrative.md table
└───────────────────────┘
```

## Slide-by-slide content

### Slide 0 — Cover

```
NeuCast HF: Cross-Asset CatBoost Forecasting
on 1-Second Microstructure Data

[Author], [Advisor], [Date]
github.com/StailFX/NeuCast
```

### Slide 1 — The question

> "Can we extract directional skill from L2 order-flow microstructure
> at the 1-minute horizon, and does it survive in production?"

* Microstructure features (OFI, spread_bps, depth_imb) have second-
  to-minute half-life — skill is theoretically detectable here
  (Easley-O'Hara) but **only if you can actually trade fast enough**.
* Two practical questions: (1) does the model have honest skill?
  (2) when does the edge become economically tradable?

### Slide 2 — The system

> "Real infrastructure, not a notebook."

* **Image:** two-VPS ASCII art from `01-architecture.md` § 1.1.
* Tokyo (HFT) ↔ WireGuard ↔ Finland (public). 19 ms RTT to Binance.
* 14 systemd units, 5 Grafana alerts, S3 cold archive, dual-VPS
  deploy. Source-of-truth on GitHub; Tokyo is rsync deploy target.

### Slide 3 — Data plane

> "From WebSocket bytes to model output in ~50 ms."

* L2 ingest → highfreq_l2_snapshots (Postgres JSONB) →
  OFI 1s aggregation → highfreq_ofi_1s → feature_pipeline →
  CatBoost 1m model → calibrator → /api/highfreq/forecast
* Production-running for 2+ weeks; ~640k rows/symbol of OFI history.

### Slide 4 — Solo results

> "Per-symbol models, real walk-forward CV, real CI."

* **Image:** fig-01 cropped to first 3 bars + holdout hatch.
* BTC 0.529 [0.506, 0.551] p=5.5e-3. ETH 0.534 [0.513, 0.554]
  p=5.8e-4. BNB 0.555 [0.533, 0.577] p=7e-7.
* Edge over base rate: +2.2 / +2.7 / +4.7 percentage points.
* All three reject H₀ at α=0.01.

### Slide 5 — Statistical power

> "Why we believe these numbers."

* **Image:** fig-05 (n_folds + CI half-width side-by-side).
* Solo n_folds = 32-39 per model. Joint pooling = 353. **10× more
  OOS sample size.**
* CI half-width drops from ±2.2 pp (solo) to ±0.67 pp (joint).
* The model isn't just "slightly better point estimate" — it's
  *much more confident* point estimate.

### Slide 6 — Joint results

> "One model, three symbols, structural drift resistance."

* **Image:** fig-01 full (4 bars, including JOINT in green).
* dir_acc=0.541 [0.534, 0.548] p=4.97e-33 (n_folds=353).
* Constructed via `feature_pipeline_joint.py` — symbol identity
  one-hots (`is_btc`/`is_eth`/`is_bnb`) appended to base features.
* Production-deployed 2026-05-09: `joint_1m.cbm` + `_calibrator.pkl`,
  daily timer at 04:50 UTC, exposed via
  `/api/highfreq/forecast_joint`.

### Slide 7 — Frozen holdout

> "Walk-forward CV is honest because the holdout confirms it."

* **Image:** fig-01 with hatched overlay.
* Frozen-holdout dir_acc ≥ CV dir_acc for all 3 symbols
  (BTC: 0.529 → 0.540, ETH: 0.534 → 0.548, BNB: 0.555 → 0.564).
* Strongest single piece of "no overfit" evidence: model
  *outperforms* on data it never indexed.

### Slide 8 — Calibration

> "Does the model know what it doesn't know?"

* **Image:** fig-02 conditional accuracy curve.
* dir_acc rises monotonically with confidence threshold.
* BTC: 0.5516 → 0.5645 → 0.5704 across conf_55/60/65.
* Calibrated probabilities (Niculescu-Mizil & Caruana 2005,
  isotonic regression at n_oos ≥ 1000, Platt below).

### Slide 9 — Fee-tier economics

> "Edge ≠ profit. Where does it pay?"

* **Image:** fig-04 fee-tier P&L bars.
* BTC 1m solo: retail −14.7 / vip5 −1.7 / **vip9 +0.33** /
  mm_rebate +1.13 bps per trade.
* The threshold question is itself a result: *the model's
  edge becomes profitable above ≈ vip9 fees.*

### Slide 10 — Multi-horizon eval

> "At what horizon does microstructure stop carrying signal?"

* Table of dir_acc by (symbol, bar_minutes, feature_set):
  - 1m microstructure: 0.55-0.57 ✅
  - 5m long_horizon: 0.49-0.55 ⚠️ regime-conditional
  - 15m long_horizon: 0.45-0.58 ❌ mostly noise
  - 60m: data-starved (90 bars in 96 h)
* Microstructure half-life is short — by 5m the OFI imbalance
  has played out. Long-horizon TA features help in *some*
  regimes (BNB) but not others (BTC after 09.05).

### Slide 11 — Drift case study

> "Same model. Same window length. 10 days apart. Result flipped."

* **Image:** fig-03 KS by feature.
* BTC 5m long_horizon: dir_acc 0.539 (28.04, p=0.07) →
  0.492 (09.05, p=0.69). Same code, same data source, fresh data.
* spread_bps_mean: KS=0.488 vs alarm threshold 0.15. **Regime
  shift detected before CV metrics confirmed it.**

### Slide 12 — Drift detection

> "The dashboard told us first."

* Live screenshot: `/api/highfreq/dashboard` payload showing
  drift severity=high, max_ks=0.68 on spread_bps_mean (2026-05-08).
* Closes the loop: live drift signal → operator notice → re-train
  check → confirm degradation → wait for regime change. The
  alert framework caught it 42 hours into the outage.

### Slide 13 — Why joint helps

> "Pooling diversifies regime risk."

* Same 10-day window where solo BTC 5m collapsed −4.7 pp:
  joint 1m lost only −1.4 pp.
* When BTC's regime shifts, ETH+BNB carry weight. The
  cross-symbol gradient is more stable than the per-symbol
  marginal distribution.
* Empirical, not just theoretical.

### Slide 14 — Anti-skill detector

> "Don't trade an inverted model."

* `app/highfreq/anti_skill_detector.py` — bootstrap 95% CI on
  rolling winrate of last 50 closed trades.
* If CI upper bound < 0.5 → policy ∈ {alert, halt, invert}.
* Checked every paper-trader tick. Telegram + Grafana
  notifications.

### Slide 15 — Calibration gate

> "An uncalibrated model can't trade."

* `paper_trader.PaperTraderConfig.require_calibrated=True`.
* Predictor must report `is_calibrated=True`:
  `_calibrator.pkl` exists AND `dir_acc_ci_low > 0.5`.
* Forces model to *prove* it has honest skill before any capital
  is allocated. Combined with anti-skill detector this is a
  defence-in-depth pair.

### Slide 16 — Conformal prediction intervals

> "How confident should the trader be?"

* Split-conformal (Vovk-Gammerman-Shafer 2005,
  Angelopoulos-Bates 2023).
* Compute non-conformity quantile at α=0.10 over walk-forward
  OOS. Live ``conformal_90 = [max(0, p-q), min(1, p+q)]``.
* Coverage guarantee: P(true outcome ∈ interval) ≥ 0.90 under
  exchangeability (which walk-forward CV approximately satisfies).

### Slide 17 — Engineering depth

> "Defence-grade isn't slideware — every claim has a file."

* 27 backend Python tests (paper_trader, predictor, calibration,
  drift, anti-skill, multi_horizon, joint, …).
* 55 frontend Vitest tests (RTL + jsdom + MSW).
* 14 Prometheus heartbeat metrics + 5 Grafana alert rules.
* systemd hardening on every unit — `NoNewPrivileges`,
  `ProtectSystem=strict`, `ReadWritePaths=` whitelists.

### Slide 18 — What didn't work

> "Honest negatives matter for thesis credibility."

* **5m BTC long_horizon** — was 0.539 at 28.04, became coinflip
  10 days later. Regime-conditional, not a permanent property.
* **Cross-asset features at 1m** — 0.564 vs 0.579 microstructure-only.
  *Pointing at yourself yields no extra signal* (BTC is the reference).
* **60m solo training** — n_minutes_after_neutral_drop=90,
  n_folds=0. Insufficient data; only joint+long_horizon at 60m
  showed any signal (and only with n=154 over 96h).

### Slide 19 — Limitations

> "What I cannot claim."

* **Stationarity** — 240 h windows already include regime variation.
  Edge is conditional on regime continuity.
* **Recovery** — when 5m BTC edge collapsed, we don't have a
  prediction for when (if ever) it returns. The system pauses
  trading on degraded models, doesn't promise recovery.
* **Fees out of operator control** — academic-tier user can't
  reach VIP9 without $10M+ monthly volume.

### Slide 20 — Future work

> "Phase 2.3 onwards."

* **Phase 2.3** — flip live ensemble to use joint after collecting
  ~1-2 weeks of shadow agreement data.
  ``/api/highfreq/forecast_joint`` is already serving in shadow.
* **Joint long_horizon at 60m** — accumulate more data (currently
  154 bars), validate sustained skill.
* **Conditional model activation** — only use a model in the live
  ensemble when its rolling dir_acc CI lower bound > 0.50.
  Self-deactivating during regime shift.

### Slide 21 — Q&A

> "Reading list for unanticipated questions."

* `docs/defence/00-narrative.md` § "Reviewer Q&A" — table of 7 likely
  reviewer questions with file pointers.
* `docs/defence/05-engineering-depth.md` § 5.6 — "Show me where
  you handle X" with file/line for every safety property.

## Talking-time budget

| section | slides | minutes | cumulative |
|---------|--------|---------|------------|
| Setup (1-3) | 3 | 3 | 3 |
| Solo + joint results (4-7) | 4 | 5 | 8 |
| Calibration + economics (8-9) | 2 | 3 | 11 |
| Multi-horizon + drift (10-13) | 4 | 5 | 16 |
| Engineering safety (14-17) | 4 | 4 | 20 |
| Limits + Q&A (18-21) | 4 | 5 | 25 |

22 minutes of talking + 3 minutes buffer = 25 min defence.
Adjust by collapsing 14-17 into one slide if your defence is
shorter, or expanding the "drift" slides if reviewers care about
operations.
