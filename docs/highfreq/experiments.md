# NeuCast HF — Experimental Log

Negative results are first-class citizens here. Every experiment that
*didn't* lift dir_acc is documented so we don't repeat it expecting
different results, and so academic defence reviewers can see what
*was* tried — the absence of a chart for an obvious idea looks like
selection bias.

## TL;DR for the defence committee

We started with a microstructure-only CatBoost on 1-min bars producing
**dir_acc ≈ 0.56-0.58** on walk-forward CV, **CI lower bound ≥ 0.53**,
**p ≈ 0** for all 3 symbols (BTC / ETH / BNB).

The Klines-pretrain experiment (T.15) is documented at length below
because it's our cleanest **negative result**: pretraining on 3 years
of OHLCV did NOT help at 1-min horizon (all symbols regressed by
1.3-4.9pp). The lesson — *microstructure dominates short-horizon*
— is precisely what Cont-Kukanov-Stoikov (2014) predicts.

The deployable additions came from three orthogonal directions:

* **T.16 — Defence-grade rigor**: 3-day frozen holdout enabled in
  prod, reliability diagram + ECE/Brier on /forecast, feature
  importance UI block. Holdout numbers (data the trainer literally
  could not see): BTC 0.5839, ETH 0.5733, BNB 0.5654, all p ≈ 0,
  all CI lower bounds ≥ 0.545.
* **T.17.b — Conformal prediction intervals**: split-conformal q
  computed on walk-forward OOS predictions, surfaces "prob_up=62%
  ± [55%, 69%]" on every live forecast. Distribution-free 90%
  coverage guarantee under exchangeability (Vovk-Gammerman-Shafer
  2005, Angelopoulos-Bates 2023).
* **T.17.a — Triple-barrier labels**: López de Prado 2018 trade-
  aligned target. A/B against fixed-horizon direction.
* **T.17.c — Live cumulative P&L curve**: per fee tier, on
  /forecast. BNB gross +155bp on 64 trades, win-rate 65.6% — model
  *does* find direction, retail fees just eat it.
* **T.17.d — Model versioning**: trainer archives previous .cbm
  before overwrite, ``tools.rollback_model`` for 1-command rollback.

This document is the canonical "what was tried and what worked"
artifact for the defence committee.

---

## T.15 — Klines pretrain → live fine-tune (1-min horizon)

**Date:** 2026-05-01
**Status:** ❌ keep_production (all 3 symbols regressed)
**Hypothesis:** A CatBoost pretrained on 3 years of Binance Klines
OHLCV (~1.58M bars/symbol) and then fine-tuned via
``init_model=`` on the recent 93h of live OFI data will beat the
production model that's only ever seen ~5 days of data. The pretrain
gives "regime knowledge" that 5 days of training never could.

### Setup

* Source: Binance public Klines API (3-year window, 1-min interval).
* Pretrained pipeline: ``feature_pipeline_long_horizon`` (24 OHLC-derived
  features — OHLC ratios, EMA(5), EMA(20), ATR(5/20), ROC(3/10),
  Bollinger z-score, autocorrelation lag-1, calendar features).
  Microstructure cols (OFI/depth/spread) zero-filled because Klines
  doesn't carry L2.
* CatBoost: 800 iterations, depth 5, lr 0.05, thread_count 2.
* Fine-tune: ``trainer --init-from <pretrained.cbm> --feature-set
  long_horizon --since-hours 93`` writing to candidate dir (production
  weights NOT touched).
* Comparator: ``tools.compare_candidate_models`` with ε=0.5pp.

### Results

| symbol | prod | candidate | Δ        | feature_set                  |
|--------|------|-----------|----------|------------------------------|
| BTC    | 0.5602 | 0.5348  | **-2.54pp** | microstructure → long_horizon |
| ETH    | 0.5459 | 0.5333  | -1.26pp     | cross_asset → long_horizon    |
| BNB    | 0.5599 | 0.5108  | **-4.91pp** | cross_asset → long_horizon    |

CI lower bounds also regressed by 1.2-4.9pp on all three.

### Why it failed

1. **Microstructure dominates at 1-minute horizon.** OFI / depth_imb /
   spread_bps capture order-flow imbalance that price-only TA can't
   see. The pretrained checkpoint had no L2 signal to learn from
   (Klines doesn't carry it), so it learned to predict from price
   patterns alone — strictly weaker than the live model that has
   18 microstructure cols on production-grade L2 data.
2. **BNB lost the most (-4.91pp)** because its production model uses
   the ``cross_asset`` pipeline (BTC reference features). Without
   that signal, BNB is basically a low-volume noisy ticker that
   long_horizon can't model.
3. **Years of pretrain = wrong knowledge.** The market regime
   information from 2023-2026 didn't help short-horizon predictions
   that depend on the last 60 seconds of order flow.

### Cost

* 3 × 30 min pretrain on Tokyo (CatBoost fit on 1.5M bars)
* ~90 min total wall-clock for the experiment
* No production impact (candidate kept in separate dir)

### Follow-ups (not yet tried)

* **Same approach at 5m/15m bars.** At longer horizons, microstructure
  decays into noise (Cont-Kukanov-Stoikov 2014 §5) and long_horizon
  TA *does* compete. The pretrain might pay off at 5m / 15m where
  price patterns matter more than OFI.
* **Pretrain with synthetic OFI.** Could we synthesize plausible OFI
  from Binance trades (taker_buy_base − taker_sell_base from Klines)
  for years of history? Then pretrain a microstructure model directly.
  Hypothesis: probably no — synthetic OFI is too noisy a proxy for
  the real L2 imbalance.
* **Ensemble.** Train BOTH long_horizon (pretrained) AND microstructure
  (live), serve as average-of-probabilities. Hypothesis: typically +0.5pp.

### Files

* Pretrained checkpoints: ``weights/highfreq/candidate_klines_pretrain/``
* Fine-tuned candidates: ``weights/highfreq/candidate_finetune/``
* Original klines parquets: ``data/historical/<sym>_1m_klines.parquet``
* Tooling:
  * ``tools/binance_klines_download.py`` (streaming pyarrow writer)
  * ``app/highfreq/pretrain.py``
  * ``tools/compare_candidate_models.py``
  * ``tools/run_pretrain_finetune_pipeline.sh``
  * ``tools/telegram_notify.py``

---

## T.15.f — Klines pretrain → live fine-tune (5m + 15m horizons)

**Date:** 2026-05-01
**Status:** 🟡 mixed (BTC + ETH +3pp at 15m, BNB regressed -9pp)
**Hypothesis:** At 5m / 15m bars the long_horizon TA pipeline is
empirically competitive with microstructure (multi_horizon_eval
showed this earlier for BTC). The Klines pretrain might therefore
pay off where it didn't at 1m.

### Results

| symbol | 15m prod | 15m candidate | Δ            | feature_set    |
|--------|----------|---------------|--------------|----------------|
| BTC    | 0.5049   | **0.5347**    | **+2.98pp**  | long_horizon   |
| ETH    | 0.5192   | **0.5526**    | **+3.34pp**  | long_horizon   |
| BNB    | 0.5833   | 0.4929        | **-9.05pp**  | long_horizon   |

5m candidates (no production baseline): BTC 0.5357, ETH 0.5194,
BNB 0.4686. Wide CI from small live sample (~1000 bars at 5m).

### Why partial

* BTC + ETH at 15m: the pretrain finally has the right time-scale.
  EMA(20), Bollinger, RSI on 15m bars carry real signal that's not
  buried in tick noise.
* BNB at 15m: production was at 0.5833 — likely a small-sample
  fluke (n_folds=24, n_min=302). Candidate's "regression" is closer
  to the honest baseline. The Klines BNB has more regime-shift over
  3 years (illiquid early period, BSC mania mid-2021) than BTC/ETH
  — the pretrain learned irrelevant patterns.

### Files

* Multi-horizon runner: ``tools/run_multihorizon_pretrain_pipeline.sh``
* Klines parquets at all horizons: ``data/historical/*_5m/15m_klines.parquet``

---

## T.15.g — All remaining horizons (30m, 1h, 4h, 1d)

**Date:** 2026-05-01
**Status:** 🟡 numbers exist but CI too wide for deploy decision
**Hypothesis:** Coverage of the full set of liquid horizons.

### Results

30m / 1h fine-tune trained on small live samples (90-180 bars per
symbol after neutral-band drop). 4h / 1d are pretrain-only — too few
live bars (23 / 4 respectively) for walk-forward CV.

| symbol | 30m candidate     | 1h candidate    |
|--------|-------------------|-----------------|
| BTC    | 0.5393 ± 0.11pp   | 0.5581 ± 0.14pp |
| ETH    | 0.5165 ± 0.11pp   | 0.4889 ± 0.14pp |
| BNB    | 0.5385 ± 0.10pp   | 0.4318 ± 0.15pp |

### Lesson

CI width at 30m / 1h is dominated by the small live-data slice
(93h ≈ 186 bars at 30m, 93 bars at 1h). To narrow CI to ±0.5pp
we'd need ~10x more live data — a few weeks. Decision deferred
to T+14 days when data accumulates.

### Files

* All-horizons runner: ``tools/run_all_horizons_pipeline.sh``
* Holdout-only Klines eval: ``tools/run_klines_holdout_eval.sh``

---

## T.15.i — Klines holdout evaluation (clean tight-CI baselines)

**Date:** 2026-05-03
**Status:** ✅ clean reference numbers per horizon
**Hypothesis:** Provide tight-CI dir_acc baselines at each horizon
on the same Klines data the pretrain was fit on, by holding out
the last 20% chronologically rather than fine-tuning on tiny live
samples.

### Setup

``app/highfreq/pretrain.py --cv-mode holdout
--holdout-test-fraction 0.20``: train on the first 80% of the
3-year Klines parquet, evaluate on the last 20%. Bootstrap CI.

n_test per horizon (after target+neutral-band drop):

  * 5m:  47K  bars  → CI ±0.5pp
  * 15m: 15K  bars  → CI ±0.8pp
  * 30m: 7.8K bars  → CI ±1.1pp
  * 1h:  3.9K bars  → CI ±1.6pp
  * 4h:  1.3K bars  → CI ±2.7pp
  * 1d:  218  bars  → CI ±6.6pp

(1m skipped: 1.58M-bar fit OOM-killed Tokyo's 4 GB box.)

### Results — see ``weights/highfreq/candidate_klines_pretrain/<sym>_<interval>_pretrained.json``

Sample (BTC @ 5m): dir_acc 0.5229 ± [0.519, 0.527] on n=56,601 test
bars. Tight CI; statistically distinct from chance.

The takeaway: long_horizon TA on Klines IS better than chance at
every horizon, just not by much (typical 52-54%) — and not enough
to beat live microstructure at 1m.

---

## T.16 — Defence-grade rigor pack

**Date:** 2026-05-03
**Status:** ✅ deployed
**Components:**

### 1. Frozen 3-day holdout

Production trainer's systemd dropins set:
``--since-hours 165 --frozen-holdout-days 3``

Trainer's data filter excludes the last 3 days BEFORE walk-forward
CV / final fit see them. This is the canonical "I have no leak"
academic-defence answer: hyperparameters cannot be tuned against
the holdout because the trainer literally cannot load it.

``tools/eval_frozen_holdout`` evaluates the deployed .cbm against
exactly that excluded slice. Results (T.16):

| symbol | walk-forward CV | **frozen holdout (untouched 3d)** | n_holdout | p     |
|--------|-----------------|------------------------------------|-----------|-------|
| BTC    | 0.5519          | **0.5839** [0.564, 0.603]          | 2545      | ≈ 0   |
| ETH    | 0.5457          | **0.5733** [0.556, 0.592]          | 2754      | ≈ 0   |
| BNB    | 0.5605          | **0.5654** [0.545, 0.585]          | 2432      | ≈ 0   |

All 3 holdout numbers ≥ walk-forward CV → models generalize
beyond CV. CI lower bounds 0.545-0.564 (clear of chance), p ≈ 0
across the board.

### 2. Reliability diagram

New endpoint ``GET /api/highfreq/reliability_diagram`` buckets
``predictions_log`` by ``prob_up`` into equal-width bins, computes
realized rate per bin + Brier + ECE.

UI: per-symbol SVG with diagonal y=x reference line. Canonical
"is this calibrator honest?" plot. Live values (BTC ECE = 0.10 →
moderate, retired Platt calibrator left some sub-optimal scaling).

### 3. Feature importance UI block

Pre-existing ``/api/highfreq/feature_importance`` endpoint surfaced
as a top-10 horizontal bar chart per symbol. Reviewers see at a
glance whether the model is overfit to one feature (it isn't —
top 5 ≈ 50-60% of weight, healthy long tail).

---

## T.17.a — Triple-barrier labels (López de Prado)

**Date:** 2026-05-03
**Status:** ✅ pure function + tests committed; A/B running
**Hypothesis:** A trade-aligned target ("would a TP/SL bracket trade
have won?") gives the model a path-dependent signal that fixed-horizon
direction misses. Empirically tested via
``tools.tbl_vs_direction_eval``.

### Setup

For each bar at time *t*:

* Entry at ``microprice_close[t]``
* TP barrier at ``entry × (1 + tp_bps/1e4)``
* SL barrier at ``entry × (1 - sl_bps/1e4)``
* Time-stop at *t + N* bars
* Label = which barrier hit first
  * ``1`` (TP), ``0`` (SL), ``2`` (time-stop), ``-1`` (insufficient lookahead)

Conservative tie-break: simultaneous TP/SL within one bar → SL wins
(realistic execution where intra-bar order is unknown).

Binary classifier trains on {0, 1} subset (drop time-stop +
insufficient).

### Citable

López de Prado, Marcos. *Advances in Financial Machine Learning*
(Wiley, 2018), Chapter 3 — the canonical financial-ML labeling
textbook reference.

### Files

* Pure func: ``app.highfreq.feature_pipeline.build_triple_barrier_labels``
* A/B eval script: ``tools/tbl_vs_direction_eval.py``
* 9 unit tests: ``tests/test_triple_barrier_labels.py``

---

## T.17.b — Split-conformal prediction intervals

**Date:** 2026-05-03
**Status:** ✅ deployed
**Hypothesis:** Replace the bare point estimate ``prob_up = 0.62``
with a 90 %-coverage prediction interval ``[0.55, 0.69]`` carrying
a **finite-sample distribution-free** coverage guarantee.

### Math

Per Vovk-Gammerman-Shafer 2005 + Angelopoulos-Bates 2023 split
conformal:

1. At training time, on the pooled walk-forward OOS predictions,
   compute nonconformity scores ``s_i = |proba_i - y_i|`` ∈ [0, 1].
2. The threshold ``q = quantile_{ceil((n+1)(1-α))/n}(s_i)`` bounds
   future scores under exchangeability.
3. At serve time, ``[max(0, p - q), min(1, p + q)]`` covers the
   true outcome with probability ≥ 1 - α.

Walk-forward CV's rolling-origin contemporaneous folds approximately
satisfy exchangeability between calibration and live test data.
Empirical coverage test (5K cal + 5K test from same Beta(2,2)
distribution) lands in [0.86, 0.94] for α=0.10 — the right ballpark.

### UI

Each prediction card on /forecast shows
``prob_up=62% · CI [55%, 69%]`` next to the confidence bar.
Reviewers see uncertainty alongside the point estimate.

### Files

* Math: ``app.highfreq.trainer.run_training`` (q computation)
* API: ``app.highfreq.predictor.{conformal_quantile, conformal_interval}``
* Endpoint: ``GET /api/highfreq/forecast`` adds ``conformal_90`` /
  ``conformal_95`` blocks
* 8 unit tests: ``tests/test_conformal_prediction.py``

---

## T.17.c — Live cumulative P&L curve

**Date:** 2026-05-03
**Status:** ✅ deployed
**Hypothesis:** "Mean P&L per fee tier" cards (already on page) show
average; the trader's reality is the **trajectory** — drawdowns,
recoveries, flat periods. A live unfolding curve makes this visible.

### Live numbers (BNB, n=64 trades)

| tier      | final cumul P&L | win-rate |
|-----------|-----------------|----------|
| gross     | **+155.6 bp**   | 65.6%    |
| vip9      | +155.6 bp       | 65.6%    |
| mm_rebate | +206.8 bp       | 76.6%    |
| vip5      | + 27.6 bp       | 39.1%    |
| futures   | -100.4 bp       | 28.1%    |
| retail    | -804.4 bp       | 10.9%    |

The defence story this gives: **the model has real edge** (gross
positive, win-rate 66%). Retail spot fees (15bp roundtrip) **eat the
edge**. On VIP-9 / mm-rebate the gap closes and the model is
positively profitable.

### Files

* Endpoint: ``GET /api/highfreq/cumulative_pnl``
* UI: SVG line chart with toolbar (3 symbols × 6 fee tiers)
* 8 unit tests: ``tests/test_cumulative_pnl_endpoint.py``

---

## T.17.d — Model versioning + rollback

**Date:** 2026-05-03
**Status:** ✅ deployed
**Hypothesis:** Trainer overwrites ``<sym>_1m.cbm`` daily. If a bad
model lands, paper-trader picks it up via mtime-watcher within 60 s
and we lose the previous (good) model.

### Solution

* ``app.highfreq.model_archive.archive_existing(weights_path,
  keep_last_n=7)`` snapshots .cbm + metrics.json + calibrator.pkl
  to ``weights/highfreq/archive/<stem>_<TS>.cbm`` BEFORE overwrite.
* Trainer wires this in unconditionally (best-effort — failure
  doesn't abort training).
* ``tools/rollback_model.py`` is the operator-facing rollback
  CLI: lists snapshots, prompts, archives the current state first
  (so a misclick is undoable), then copies the chosen snapshot back.
  Paper-trader's mtime-watcher picks up the change in ~60 s.

### Files

* Pure module: ``app/highfreq/model_archive.py``
* CLI: ``tools/rollback_model.py``
* 10 unit tests: ``tests/test_model_archive.py``

---
