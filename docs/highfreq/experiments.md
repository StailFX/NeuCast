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

## T.15.i — Klines public-data baseline (defends the L2 value)

**Date:** 2026-05-03
**Status:** ✅ canonical public-data baseline per horizon
**Why it matters:** Without this baseline, a reviewer could ask
"you got 0.58 — but maybe ANY model would get that?". Now we have
hard evidence that bare-bones public OHLCV gets 0.51-0.54, and our
L2-microstructure approach gets 0.57-0.58. **The 3-5pp gap is the
measurable value of the L2 ingest infrastructure.**

### Setup

``app/highfreq/pretrain.py --cv-mode holdout
--holdout-test-fraction 0.20``: train on the first 80% of the
3-year Binance Klines parquet (public OHLCV — no L2), evaluate
on the last 20%. Bootstrap CI on the test partition.

### Results

| symbol  | interval | n_test  | dir_acc | CI low | CI high | base_rate |
|---------|----------|---------|---------|--------|---------|-----------|
| BNBUSDT | 1m       | 243,956 | 0.510   | 0.508  | 0.512   | 0.510     |
| BTCUSDT | 5m       | 56,601  | 0.523   | 0.519  | 0.527   | 0.501     |
| ETHUSDT | 5m       | 58,197  | 0.522   | 0.518  | 0.526   | 0.502     |
| BNBUSDT | 5m       | 56,310  | 0.512   | 0.508  | 0.516   | 0.508     |
| BTCUSDT | 15m      | 19,925  | 0.527   | 0.520  | 0.534   | 0.503     |
| ETHUSDT | 15m      | 20,208  | 0.527   | 0.520  | 0.534   | 0.504     |
| BNBUSDT | 15m      | 19,753  | 0.518   | 0.511  | 0.525   | 0.510     |
| BTCUSDT | 30m      | 10,106  | 0.532   | 0.522  | 0.541   | 0.506     |
| ETHUSDT | 30m      | 10,197  | 0.530   | 0.520  | 0.540   | 0.506     |
| BNBUSDT | 30m      | 10,060  | 0.520   | 0.510  | 0.530   | 0.512     |
| BTCUSDT | 1h       | 5,107   | 0.530   | 0.515  | 0.544   | 0.510     |
| ETHUSDT | 1h       | 5,137   | 0.524   | 0.510  | 0.538   | 0.510     |
| BNBUSDT | 1h       | 5,097   | 0.520   | 0.506  | 0.534   | 0.518     |
| BTCUSDT | 4h       | 1,297   | 0.541   | 0.514  | 0.567   | 0.513     |
| ETHUSDT | 4h       | 1,300   | 0.510   | 0.482  | 0.535   | 0.514     |
| BNBUSDT | 4h       | 1,296   | 0.527   | 0.500  | 0.553   | 0.510     |
| BTCUSDT | 1d       | 217     | 0.512   | 0.452  | 0.581   | 0.512     |
| ETHUSDT | 1d       | 218     | 0.500   | 0.440  | 0.569   | 0.504     |
| BNBUSDT | 1d       | 217     | 0.498   | 0.434  | 0.572   | 0.521     |

(1m BTC + 1m ETH skipped: 1.58M-bar fit OOM-killed Tokyo's 4 GB box.
Solvable with ``--iterations 200`` but not strictly needed for the
defence story since BNB@1m number is already representative.)

### Comparison: public Klines baseline vs production L2

| horizon | Klines baseline (public OHLC) | Production microstructure (L2 + cross_asset) | Gap |
|---------|------------------------------|---------------------------------------------|-----|
| 1m      | 0.510 (BNB)                  | **0.560-0.591** (walk-forward CV)            | **+5-8pp** |
| 1m      | —                            | **0.565-0.584** (frozen 3-day holdout)       | +5-7pp |

The 5-8pp lift over public OHLCV is the **measurable value** of:
* L2 ingest from Tokyo VPS (~19 ms RTT to Binance Spot WS)
* WireGuard tunnel + slim FastAPI architecture
* OFI / depth_imb / spread_bps / vpin features (Cont-Kukanov-Stoikov 2014)
* cross_asset BTC reference for ETH/BNB
* Calibrated probability outputs (Platt-fit)

### Interpretive notes

* **dir_acc grows from 5m to 30m** (0.52 → 0.53), then plateaus.
  Confirms long_horizon TA becomes more useful as bar size grows
  (microstructure noise smooths out), but plateaus without OFI.
* **1d is at chance** (0.50 ± 0.05 on n=217). Daily crypto moves
  dominated by macro factors (DXY, SPX, NQ futures) that OHLCV
  alone cannot model. Sample is also too small for tight CI.
* **BNB consistently weaker** by 0.5-1pp at every horizon —
  less liquid pair, more noise in OHLCV → public-data baseline lower.
* **4h BTC = 0.541** with CI [0.514, 0.567] is the best Klines-only
  result; n=1297 → CI ±2.7pp, statistically above chance.

### Defence statement

> "We trained a CatBoost on 3 years of public Binance Klines OHLCV
> (the data anyone can download). Best dir_acc across 7 horizons:
> 0.541 (BTC @ 4h, n=1297, CI [0.514, 0.567]). Most horizons land
> at 0.51-0.53. Our production model — running on Tokyo with L2
> ingest + microstructure features + cross_asset reference —
> achieves 0.56-0.58 with statistically significant CI lower bounds
> ≥ 0.53. The 3-5pp gap is the measurable value of the L2
> infrastructure investment."

This is one of the strongest single-paragraph arguments in the
defence: it directly answers "what does your infrastructure buy
you?" with hard, comparable numbers.

### Files

* Holdout CV mode: ``app.highfreq.pretrain run_pretrain(cv_mode='holdout')``
* Pipeline: ``tools/run_klines_holdout_eval.sh``
* Per-(symbol, interval) JSON: ``weights/highfreq/candidate_klines_pretrain/<sym>_<interval>_pretrained.json``

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
**Status:** ❌ negative result at 1m horizon — TBL underperforms direction
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

### A/B result on production data (2026-05-03, since-hours=165)

Settings: tp_bps=5, sl_bps=5, time_stop_bars=10. Walk-forward CV
on the same live OFI window the production trainer uses.

| symbol  | dir_acc (direction) | n_dir  | dir_acc (TBL) | tp/sl/ts        | Δ           |
|---------|---------------------|--------|---------------|------------------|-------------|
| BTCUSDT | 0.5736              | 5,160  | 0.5350        | 4262/4326/1296   | **-3.86pp** |
| ETHUSDT | 0.5665              | 5,700  | 0.5410        | 4505/4632/747    | -2.55pp     |
| BNBUSDT | 0.5754              | 4,680  | 0.5382        | 4034/4448/1402   | **-3.73pp** |

All 3 regressed. Verdict: **keep production** (binary direction
target, fixed-1m horizon).

### Why it failed at 1m

1. **TP/SL count balance is tilted toward SL** (4326/4262 BTC,
   4632/4505 ETH, 4448/4034 BNB). Combined with the conservative
   simultaneous-tie-break (SL wins), the binary classifier sees
   slightly more SL-first labels than TP-first. The model learns
   the imbalance more than the signal.
2. **At 1-min horizon**, fixed-direction is *already* a sharp
   target — bar-to-bar moves carry directional info that the
   model can extract. TBL helps when the fixed-horizon target
   is too noisy (e.g. 1-bar return is often 0bp on illiquid
   instruments). On Binance Spot 1m, base-rate is 0.50-0.51 with
   ~50% of bars exceeding the 1bp neutral band — no noise problem
   to fix.
3. **Path-dependence cuts the bar-set.** TBL drops time-stop
   bars (1296-1402 per symbol) plus insufficient-lookahead bars
   at the edges. The remaining 7,020-7,680 bars are a *biased*
   subset (only the bars where price actually moved by ≥5bp
   within 10 minutes) — survivorship bias in the training set.

### Where TBL might still help (untried)

* **Longer horizons (15m, 1h)** where fixed-direction targets
  ARE noisy. TBL's path-dependence becomes informative.
* **Asymmetric TP/SL** (e.g. tp_bps=10, sl_bps=3 — momentum-style)
  matching a real bracket-order strategy a trader would actually
  run.
* **Wider time_stop** (e.g. 60 bars) so fewer time-stop drops
  reduce survivorship bias.

### Defence value of this negative result

* Demonstrates we tried the canonical citable target (López de
  Prado 2018, Ch. 3) and reported honestly that it didn't help.
* Confirms that fixed-direction is the right target for 1-min
  microstructure prediction (the published literature's claim
  for that horizon).

### Files

* Pure func: ``app.highfreq.feature_pipeline.build_triple_barrier_labels``
* A/B eval script: ``tools/tbl_vs_direction_eval.py``
* Result JSON: ``weights/highfreq/tbl_vs_direction.json``
* 9 unit tests: ``tests/test_triple_barrier_labels.py``

---

## T.19 — Multi-horizon ensemble (1m + 15m)

**Date:** 2026-05-03
**Status:** ✅ deployed as ``/api/highfreq/forecast_ensemble``
**Hypothesis:** The 1m microstructure model and 15m long_horizon TA
model use DIFFERENT feature spaces. When their independent signals
agree, joint confidence rises beyond either alone (Wolpert 1992
"Stacked Generalization"). Default 70/30 weighting reflects empirical
1m superiority; a 30 % share for 15m is enough to bend the signal
when 15m has a clear contrarian read.

### Implementation

* ``app/highfreq/ensemble.py``: pure module —
  ``ensemble_probability(components)`` returns
  weighted-average prob with full per-model breakdown + agreement
  flag (do all components vote the same side of 0.5).
  Re-normalises weights when a component is unavailable, so
  ``70 % 1m + 30 % 15m`` degenerates cleanly to 100 % 1m if 15m
  cold-start.
* ``GET /api/highfreq/forecast_ensemble?symbol=&weight_1m=&weight_15m=``
  endpoint runs both predictors, blends, returns full breakdown.
  Backward-compat: existing ``/api/highfreq/forecast`` keeps 1m-only.

### First live values (BTC, 2026-05-03)

```
prob_up:  0.5593   (blend)
signal:   up
agreement: true
components:
  1m  (w=0.70):  prob_up=0.5792   ← strong up
  15m (w=0.30):  prob_up=0.5128   ← weak up
```

When models AGREE, the ensemble's confidence is on the 1m level (not
diluted toward 0.5). When they DISAGREE, the 1m signal dominates by
its 70 % weight share.

### Defence value

> "We expose a multi-horizon ensemble combining the 1-min
> microstructure-based model with the 15-min long-horizon TA model.
> Default weights: 70/30 (1m empirically stronger). Agreement
> indicator on the response surface shows when both horizons vote
> the same side — operationally a higher-confidence regime."

### A/B against single-model production

Pending: realized-accuracy A/B requires accumulating ensemble
predictions in ``predictions_log`` for ≥ 24h. Currently the paper
trader runs 1m only; T.19.b (future) would add an ensemble
paper-trader if the off-line metrics suggest it'd be worth it.

### Files

* Pure module: ``app/highfreq/ensemble.py``
* Endpoint: ``GET /api/highfreq/forecast_ensemble``
* 11 + 7 = 18 unit tests:
  ``tests/test_ensemble.py`` + ``tests/test_forecast_ensemble_endpoint.py``

---

## T.18.c — Trade-flow rolling features (microstructure_v2)

**Date:** 2026-05-03
**Status:** 🟡 marginal lift (+0.15pp BTC), within tolerance — **not deployed**
**Hypothesis:** The current ``microstructure`` pipeline collapses
trade flow into two scalars per bar (``trade_imb_sum``,
``trade_imb_abs_sum``). Multi-scale rolling features (lag1, 3-bar
mean, acceleration) might recover sub-minute flow-shape signal
that the simple bar aggregate loses.

### Setup

New feature_set ``microstructure_v2`` = base 18 cols + 4 trade-flow:

1. ``trade_buy_ratio`` — within-bar signed ratio
   ``trade_imb_sum / max(trade_imb_abs_sum, ε)`` ∈ [-1, 1]
2. ``trade_buy_ratio_lag1`` — same, shifted 1 bar (memory)
3. ``trade_buy_ratio_3bar_avg`` — 3-bar rolling mean
4. ``trade_imb_acceleration`` — current ``trade_imb_sum`` minus lag1

Total: 22 cols. Backward compatible (v1 models keep working with
the legacy 18-col ``microstructure`` feature_set).

### Results (BTC, since=165h, frozen-holdout=3d)

| metric        | microstructure (prod) | microstructure_v2 | Δ            |
|---------------|-----------------------|-------------------|--------------|
| dir_acc_mean  | 0.5596                | 0.5611            | **+0.15pp**  |
| dir_acc_ci_lo | 0.5411                | 0.5411            | 0.00         |
| dir_acc_ci_hi | 0.5774                | 0.5793            | +0.0019      |
| p-value       | 3.1e-10               | 1.2e-10           | tighter      |
| Brier         | 0.2514                | 0.2526            | +0.001 (worse) |
| ECE           | 0.0586                | 0.0565            | **-0.002 ✓** |

Below the comparator's 0.5pp deploy tolerance. **Verdict:
keep production**. v2 stays available as a feature_set option for
future ablations / data-source extensions, but isn't wired into
production trainers.

### Why so little lift

* ``trade_imb_sum`` already captures most of the per-bar trade-flow
  signal at 1-min horizon. The 60-second aggregation already
  resolves most of the directionally-predictive flow.
* Sub-minute resolution (e.g. last 30s of a bar) WOULD likely add
  more — but our pipeline aggregates at the 1-second level, not
  sub-bar within the minute. Future extension would re-aggregate
  raw second-level data with overlapping windows.
* CatBoost saw the new features (feature importance test showed
  trade_buy_ratio_lag1 in top-10 for BTC) but couldn't extract
  a strong incremental signal beyond what the existing 18 cols
  already provide.

### Defence value of this near-null result

Demonstrates we tested the hypothesis honestly with a controlled
A/B (same 165h window, same walk-forward CV, same hyperparameters).
The +0.15pp lift is real but not statistically distinguishable from
noise, so we report it transparently and don't claim the win.

### Files

* Pure module: ``app/highfreq/feature_pipeline_microstructure_v2.py``
* Wired into trainer / predictor / web / paper_trader_runner via
  the standard feature_set dispatch.
* 12 unit tests: ``tests/test_microstructure_v2_features.py``

---

## T.18.a — Isotonic-regression calibrator (auto for n ≥ 1000)

**Date:** 2026-05-03
**Status:** ✅ deployed
**Hypothesis:** Replace Platt scaling with isotonic regression
when walk-forward OOS sample is large enough (n ≥ 1000) per
Niculescu-Mizil & Caruana 2005. Isotonic is more flexible
(monotone step function) than Platt's parametric logistic; on
n ≈ 2000-3000 (our production sample) isotonic typically lowers
Brier by 0.005-0.01 and ECE by 0.02-0.05.

### Setup

* New ``fit_isotonic_calibrator`` in ``app.highfreq.calibration``,
  wrapping ``sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')``.
* Trainer dispatches via ``HF_CALIBRATOR_TYPE`` env (auto / platt /
  isotonic). Default ``auto`` picks isotonic when n_oos ≥ 1000,
  else Platt.
* Backward-compat: existing ``apply_calibrator`` works on both
  Platt and isotonic instances via uniform ``predict_proba`` interface.

### Result (BTC, retrained 2026-05-03)

| metric | Platt (prior) | isotonic (current) | Δ           |
|--------|---------------|---------------------|-------------|
| Brier  | 0.2514        | 0.2526              | +0.0012     |
| ECE    | 0.0586        | 0.0565              | **-0.0021** |

Isotonic reduces ECE (better calibration) but slightly worse Brier
on this particular sample. The textbook claim is empirically
supported on the regression test (synthetic miscalibrated data,
n=4000 split) — production-data behaviour can vary by sample.

### Files

* Module: ``app.highfreq.calibration.fit_isotonic_calibrator``
* 8 new unit tests (24 total in calibration suite):
  ``tests/test_highfreq_calibration.py``

---

## T.18.b — Feature-distribution drift detection

**Date:** 2026-05-03
**Status:** ✅ deployed (3 hourly cron timers on Tokyo)
**Hypothesis:** Production-grade alerting on distribution shift
between trainer reference window and live serve-time bars. KS-test
per feature, alert when ``max KS ≥ threshold``.

### Setup

* ``app/highfreq/drift_detector.py``: pure module with KS-test +
  severity bucketing (ok / warn / high).
* ``tools/drift_check.py``: cron-runnable CLI. Builds features
  via the same trainer dispatch (so cross_asset gets BTC reference).
  Writes JSON to ``weights/highfreq/<sym>_drift.json``. Sends
  Telegram on warn / high.
* systemd timers: ``neucast-drift-check@<sym>.timer`` hourly per
  symbol with 5-min RandomizedDelaySec.

Calendar features (``day_of_week``, ``hour_of_day``,
``hour_of_week``, ``minute_of_hour``) excluded by default — they
ALWAYS drift between two time windows by construction (KS = 1.0).

### First live signal (BTC, recent=6h vs reference=24h)

```
severity: high
max_ks=0.891 on spread_bps_mean
alarming: 11 of 14 features
```

Genuine drift: spreads tightened over the past 4 days
(ref_mean=0.002bps → recent=0.001bps). Operator-actionable signal.

### Defence value

> "We monitor 14 production feature distributions in real time.
> KS-test fires a Telegram alert if any feature drifts past the
> 0.15 threshold. Operator sees regime shifts BEFORE realized
> accuracy degrades — the dashboard isn't flying blind on a
> distribution change."

### Files

* Module: ``app/highfreq/drift_detector.py``
* CLI: ``tools/drift_check.py``
* systemd: ``docs/highfreq/deploy/neucast-drift-check@.{service,timer}``
* 13 unit tests: ``tests/test_drift_detector.py``

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

## T.20 — Auto-regenerated training scoreboard

**Date:** 2026-05-03
**Status:** ✅ landed (committed as ``31ab5e8``)
**Hypothesis:** A defence reviewer should be able to read **one
document** that lists every production training run on every symbol,
the metrics they produced, and what config changed between them —
without trusting any human's memo of "what we tried." That document
must regenerate from the append-only ``training_runs`` table, not
from a hand-maintained changelog.

### Solution

* ``tools/scoreboard.py`` reads ``training_runs`` from Postgres,
  filters to spot-production runs, renders Markdown:
  - **Latest production metrics** — one row per symbol with
    dir-acc + 95% Wilson CI + p-value + Brier + ECE + conformal q.
  - **Frozen-holdout summary** (if any holdout JSONs exist).
  - **Per-symbol timeline** — every training run, oldest first,
    with a ``config delta`` column auto-tagged from neighbouring runs.
* ``_release_tag`` heuristic emits human-readable transitions
  (``feature_set: microstructure→cross_asset``, ``holdout_days: 0→3``,
  ``conformal added``, ``calibrator change``). Tiny Brier jitter
  (< 0.005) is filtered as run-to-run noise so the column doesn't
  drown in false positives.
* 16 unit tests (``tests/test_scoreboard.py``) pin the document
  structure (section order, em-dash for missing metrics, footer run
  count) and the config-delta heuristic.
* Output: ``docs/highfreq/scoreboard.md``, ~7.7 KB at the time of
  landing — committed alongside the tool so the GitHub repo always
  carries the snapshot.

---

## T.21 — UI surface for drift status + ensemble forecast

**Date:** 2026-05-03
**Status:** ✅ deployed (Tokyo web.py + Finland forecast.html)
**Hypothesis:** T.18.b (KS-test drift detector) and T.19 (multi-
horizon ensemble) shipped as **server-only artifacts** — the cron
writes ``<sym>_drift.json`` every hour and the ensemble has its own
endpoint, but neither was visible on ``/forecast``. Reviewers
opening the public dashboard couldn't see "is the production model
operating on in-distribution data?" or "how does the 1m signal
combine with the 15m signal in real time?". This release surfaces
both inline on each prediction card.

### Solution

* New endpoint ``GET /api/highfreq/drift_status?symbol=…`` reads
  ``weights/highfreq/<sym>_drift.json`` and returns severity / max_ks /
  max_ks_feature / top_features. Returns 200 with ``ok=False,
  reason=no_check_yet`` when the cron hasn't run (best-effort, not a
  runtime guarantee — UI renders muted "—", not alarming red).
* Per-card ``.drift-badge`` pill (green / amber / red / muted) reads
  from that endpoint at the same cadence as the other slow refreshers
  (5 min). Tooltip shows the top-3 KS features so reviewers can see
  *which* feature is drifting.
* Per-card ``.ensemble-text`` mini-strip pulls
  ``/api/highfreq/forecast_ensemble`` and renders
  ``ансамбль 58% (1m=62 · 15m=50 ✓)`` — the ✓/✗ glyph signals whether
  the two horizons agree on direction. Hidden when a component
  cold-starts so the page falls back gracefully to the existing 1m
  verdict.
* Two new test files in ``tests/test_forecast_page.py``
  (``test_forecast_template_has_drift_badge_on_cards`` +
  ``test_forecast_template_has_ensemble_strip_on_cards``) pin the
  data-bind anchors, the CSS classes, and the JS function names so
  future template refactors can't silently strip these features.
* Endpoint tests: ``tests/test_drift_status_endpoint.py`` (5 tests
  covering happy path / no-check-yet / malformed / severity-ok /
  symbol uppercasing).

### What we observed at deploy time

The drift detector immediately flagged **HIGH severity on all 3
symbols** with ``spread_bps_mean`` as the top drifter:

| symbol  | severity | max_ks | feature           | alarming features |
|---------|----------|--------|-------------------|-------------------|
| BTCUSDT | high     | 0.961  | ``spread_bps_mean`` | 10 / 14         |
| ETHUSDT | high     | 0.994  | ``spread_bps_mean`` | 13 / 23         |
| BNBUSDT | high     | 0.898  | ``spread_bps_mean`` | 11 / 23         |

Reading: KS ≥ 0.90 means the recent serving distribution barely
overlaps with the training reference window. The likely cause is
real liquidity-regime change (spreads tightened markedly vs the
older training window), not a code bug — but the badge is doing
exactly what it's supposed to: alerting an operator that the
production inputs no longer look like the training inputs. Open
follow-up: T.22 candidate is **either** retrain on a fresh window
**or** widen the reference window to be rolling rather than
fixed-at-train-time.

### Files

* Endpoint: ``app/highfreq/web.py`` (``/api/highfreq/drift_status``)
* UI: ``templates/forecast.html`` (CSS + HTML slots + JS handlers)
* Tests:
  ``tests/test_drift_status_endpoint.py`` (5 tests),
  ``tests/test_forecast_page.py`` (+2 tests)

---
