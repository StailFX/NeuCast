# NeuCast HF — Experimental Log

Negative results are first-class citizens here. Every experiment that
*didn't* lift dir_acc is documented so we don't repeat it expecting
different results, and so academic defence reviewers can see what
*was* tried — the absence of a chart for an obvious idea looks like
selection bias.

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
