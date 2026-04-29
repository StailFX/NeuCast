# NeuCast HF — research roadmap

Карта всех направлений улучшения модели на pre-defense / post-defense
horizons. Структурировано по: ожидаемый impact / effort / risk.

Состояние на момент написания (2026-04-29, после Release N):

```
1m   joint + microstructure        dir_acc 0.5728  p=1.4e-34   n=8481   ✅
15m  joint + long_horizon TA        dir_acc 0.5485  p=0.016     n=604    ✅
60m  joint + long_horizon TA        dir_acc 0.6105  p=0.020     n=155    ✅

per-symbol best:
BTC 1m  microstructure + calendar   dir_acc 0.5794  p=8e-12     n=3276
ETH 1m  cross-asset (BTC reference) dir_acc 0.5614  p=4.6e-6    n=2769
BNB 1m  cross-asset (BTC reference) dir_acc 0.5786  p=3e-6      n=2329
```

Все 3 символа значимо выше chance на 1m. На 15m+ joint+TA дают
significant signal. Profitable от VIP9 fee tier (0bp).

---

## 0. Wait-only (passive, бесплатно)

Время ничего не стоит, многое улучшит:

| Через | Что меняется |
|---|---|
| **+3 дня** | 60m models по-symbol становятся feasible |
| **+5 дней** | n≈10000 на BTC, CI сужается на 30% |
| **+7 дней** | **frozen holdout activatable** (`--frozen-holdout-days 7`) — true OOS gold standard |
| **+14 дней** | n≈25000, p-value на BTC становится 10⁻²⁰+ |
| **+30 дней** | можно claim "long-term stable edge" вместо "short-window" |

Совершенно бесплатно — сейчас система сама накапливает.

---

## 1. Feature engineering — Tier S (high impact, low risk)

### 1A. Frozen holdout activation
**Effort**: 1 строка config (env var `HF_FROZEN_HOLDOUT_DAYS=7`).
**Готовность**: при ≥14 дней общего uptime.
**Impact**: defence-grade gold standard метрика.
**Risk**: 0.

### 1B. Cross-asset features for joint training
**Effort**: ~3 часа (новый pipeline + eval).
**Hypothesis**: joint pooling + each symbol sees others' microprices = best of both.
**Risk**: feature explosion + overfitting на малой выборке.
**Expected**: +0.5-1pp над best individual configuration.

### 1C. Order book deep features
- Queue position estimates (bid/ask depth at top-5 levels)
- Large-order detection (orders > 10× avg size)
- Hidden liquidity probes (iceberg detection)

**Effort**: ~1 неделя — нужен новый ingest path в L2 stream.
**Impact**: возможно +1-3pp на 1m (микроструктура сильнее).
**Risk**: medium — engineering time substantial.

### 1D. Cross-exchange features
- Coinbase BTC microprice как feature для Binance BTC
- Kraken / Bybit / OKX as references

**Effort**: ~2 недели — нужны новые WS clients per exchange.
**Hypothesis**: разные exchanges имеют lead/lag dynamics.
**Impact**: уровень industry quant funds, может +1-2pp.
**Risk**: high engineering, need accounts on each exchange.

---

## 2. Feature engineering — Tier A (medium effort)

### 2A. Funding rate features (для перпов)
Если расширим на USDM Futures:
- Current funding rate
- Funding rate momentum (8h change)
- Open interest changes

**Effort**: ~3 дня — Binance Futures USDT-margined ingest.
**Impact**: на 1h+ horizon может +1pp.

### 2B. Macro features
- DXY (dollar index)
- US10Y yields
- Gold (GC=F)
- S&P 500 futures

**Effort**: ~1 день — yfinance ingest на 1-min bars.
**Impact**: edge только на длинных горизонтах (1h+).
**Risk**: low.

### 2C. Calendar events expansion
Сейчас 3 события в `event_calendar.json`. Расширить до 50-100:
- All FOMC dates 2026
- All US CPI / PCE / PPI prints
- All BOJ / ECB / PBoC decisions
- All major exchange listings (Binance, Coinbase)
- All hard forks / upgrades

**Effort**: ~2 часа manual research.
**Impact**: лучше risk management, защита от lottery outcomes.
**Risk**: 0.

### 2D. Volume / liquidity features
- 1-min volume vs 24h average
- VWAP deviation
- Volume profile per hour-of-day
- Activity ratio (n_updates × hour_factor)

**Effort**: ~3-4 часа в feature_pipeline.
**Impact**: marginal на 1m, возможно +0.5pp.

---

## 3. Model architecture — Tier A

### 3A. ε — Magnitude regression target
**Status**: design doc done (`docs/highfreq/future-improvements.md`).
**Effort**: 1-2 дня.
**Impact**: НЕ повышает dir_acc (может даже снизить). НО:
- Unlocks Kelly-criterion sizing
- Fee-aware filtering (skip trades с E\|move\| < 2× fees)
- Confidence-weighted backtest

**Defence narrative**: «complete trader, не just direction picker».

### 3B. Per-regime models (vol clustering)
Train 3 separate models на low_vol / medium_vol / high_vol regimes.
Switch based on `bb_z_20` или `atr_ratio_5_20`.

**Effort**: ~1 день.
**Hypothesis**: каждый regime имеет свою microstructure dynamics.
**Risk**: regime detection itself может быть unstable.
**Impact**: возможно +1-2pp в high-vol regime, +0 в low-vol.

### 3C. Triple-barrier labeling (López de Prado)
Вместо binary `sign(return_1m)`:
- Define profit-taking barrier (e.g. +5bp)
- Define stop-loss barrier (e.g. -3bp)
- Label = which barrier hit first within `horizon` minutes
- Multi-class: hit_take / hit_stop / time_out

**Effort**: ~2 дня (target redefinition + trader update).
**Impact**: better signal quality (метит profitable directional moves).
**Risk**: requires careful barrier tuning per horizon.

### 3D. Sample weighting (recent > old)
Exponentially decay sample weights so recent bars matter more.
CatBoost `sample_weight` parameter.

**Effort**: ~1 час.
**Hypothesis**: market regime drifts; recent data more relevant.
**Impact**: marginal, but defence-grade (пинаем concept drift).

---

## 4. Model architecture — Tier B (research-grade)

### 4A. ζ — LSTM/GRU sequence model
**Status**: design doc done.
**Effort**: 2-3 дня.
**Risk**: high — на daily side already showed low metrics.
**Hypothesis to test**: 1-min microstructure имеет sequence dynamics
которые CatBoost независимо не ловит.
**Impact**: ±0-2pp, sometimes WORSE than CatBoost.
**Defence narrative**: «empirically tested, didn't help, here's why».

### 4B. Transformer (attention)
Like LSTM but with attention. Better for long-range dependencies.

**Effort**: 3-4 дня.
**Hypothesis**: attention learns important past bars selectively.
**Risk**: huge engineering, overkill for tabular signal.
**Pre-defense ROI**: low.

### 4C. Bayesian hyperparameter optimization
Replace grid search with Optuna/Ax. Smarter exploration.

**Effort**: ~1 день.
**Impact**: maybe +0.5pp over best grid combo.
**Defence narrative**: «used Bayesian optimization not just grid search».

### 4D. Ensemble (CatBoost + LightGBM + XGBoost + Ridge meta)
Apply daily-side ensemble pattern to HFT.

**Effort**: ~1 день.
**Impact**: typically +0.5-1pp robustness.
**Note**: уже есть на daily side, можно portнуть pattern.

---

## 5. Production / risk management — Tier A

### 5A. Live testnet execution (Stage 3 from architecture)
Fill in `BinanceTestnetExecutor` stub:
- HMAC-SHA256 sign REST
- WebSocket user-data stream
- Order reconciliation
- Kill switch via Telegram

**Effort**: ~5 days.
**Defence**: «I shipped engineering for live trading; ran 30 days on testnet».
**Risk**: 0 (testnet money).

### 5B. Probability calibration impact study
We have calibration. Measure: does it actually improve trader winrate?
A/B compare paper trader with vs without calibrated probs.

**Effort**: ~2 hours (env flag + analysis).
**Impact**: clarifies whether calibration matters operationally.

### 5C. Anti-skill auto-invert
Currently anti-skill detector только alerts. Switch to `invert` mode
on a paper-only test. Observe whether inverting genuinely flips edge.

**Effort**: 5 minutes (env flag), 2-3 days observation.
**Risk**: low (paper only).

### 5D. Realized accuracy time-series
Plot `realized_correct` rolling 50-trade rate over time. Defence visual:
"see how the model performs in continuous deployment".

**Effort**: 1 hour for an endpoint + chart in UI.

---

## 6. Symbol expansion — Tier B

### 6A. Add altcoins (DOGE, SOL, ADA, XRP)
**Effort**: ~1 day per symbol (ingest + trainer + paper trader).
**Impact**: more pooled data for joint model. Each new symbol +20-30% data.
**Risk**: small caps may have weaker microstructure signal.

### 6B. USDM perpetual futures
Switch from spot to futures (BTC.P/ETH.P).
**Effort**: ~3 days — different ingest path, contract specs, funding.
**Impact**:
- Higher liquidity (tighter spread → smaller fee burden)
- Funding rate as new feature
- Long + short native (vs spot's synthetic short)
- BUT: borrow rates eat profitability if held long

### 6C. Options-implied features
SVI / ATM straddle / put-call ratio as features.
**Effort**: ~2 weeks (Deribit / Binance Options ingest).
**Impact**: best at multi-hour horizons. Probably overkill for 1m HFT.

---

## 7. Methodology — Tier A

### 7A. Bayesian credible intervals (replace bootstrap)
Use Beta-Binomial posterior for dir_acc CI.

**Effort**: 2 hours.
**Impact**: cleaner CI semantics, better at small sample.
**Defence narrative**: shows statistical sophistication.

### 7B. Power analysis tool
For target dir_acc lift X, n needed = Y. Tool that computes this
prospectively.

**Effort**: 1 hour.
**Impact**: quantifies "how much more data до conclusive answer".

### 7C. Embargoed walk-forward CV
Insert 1-bar gap between train and test fold to prevent look-ahead.

**Effort**: 1 hour change in walk_forward_evaluate.
**Impact**: marginal (1-bar leak unlikely), but academic best practice.

### 7D. Cross-validation with purging
Drop bars within target-horizon of fold boundaries to prevent target leakage.

**Effort**: 2 hours.
**Impact**: tighter true-OOS guarantee.

---

## 8. Defense materials — Tier A (NOT model improvements)

### 8A. Reliability diagram visualization
Already computed (Brier + ECE in TrainingReport). Render as chart.
**Effort**: ~1 hour, pure frontend work.

### 8B. Per-fold dir_acc time-series chart
Already in training_report endpoint. Render as line chart with CI bars.
**Effort**: ~1 hour frontend.

### 8C. Multi-horizon feature-set comparison chart
Big bar chart показывающий best per (symbol × horizon × feature_set).
**Effort**: ~2 hours.

### 8D. Architecture decision records (ADRs) update
Add ADRs for: cross-asset features, joint pooling, calibration, event halt.
**Effort**: ~2 hours.

---

## Recommended execution order — pre-defense

If defense ≤7 days:
1. **Wait** — natural data accumulation
2. **8A-D** defence visuals (~6 hours total)
3. **2C** event calendar expansion (~2 hours, easy bullet)
4. **5D** realized accuracy chart (~1 hour)

If defense 7-14 days:
1. Above
2. **1A** frozen holdout activation (1 line config)
3. **3A** ε magnitude regression (1-2 days, real ML upgrade)
4. **5B** calibration impact study (2 hours)
5. **7A/C/D** methodology refinements (~5 hours total)

If defense 14-30 days:
1. Above
2. **1C** order book deep features (~1 week)
3. **3B** per-regime models (1 day)
4. **3C** triple-barrier labeling (2 days)
5. **5A** start testnet execution (5 days, parallel work)

If post-defense / commercial:
1. **5A** complete testnet execution
2. **6A** altcoin expansion (week)
3. **6B** perpetual futures (week)
4. **3A+3B+3D** combined ε + regimes + sample weighting

---

## Hard rejects (don't do)

* **News NLP parsing** — empirical research shows 1-min crypto driven
  by order flow not narrative. Use `event_calendar` for risk filter
  (already done) instead.
* **Reinforcement learning** — over-engineered for tabular signal,
  hard to validate, RL training instability is its own research field.
* **Deep learning trees (TabNet etc.)** — CatBoost saturates the
  tabular gradient boosting performance. Marginal returns vs huge
  complexity.
* **Switching exchanges away from Binance** — 70%+ of crypto liquidity
  is on Binance. Going to Kraken / Coinbase для самой стратегии — net
  negative liquidity.

---

## Summary table

| Tier | Items | Total effort | Expected total impact |
|---|---|---|---|
| **0 — wait** | data accumulation, frozen holdout | 0 hours | +0-2pp on stability metrics |
| **1 — feature S** | frozen holdout, cross-asset joint, order book | ~2 weeks | +1-3pp on 1m |
| **2 — feature A** | funding, macro, calendar expansion, volume | ~1 week | +0-1pp |
| **3 — model A** | ε regression, regimes, triple-barrier | ~1 week | +1-2pp + sizing story |
| **4 — model B** | LSTM, Transformer, Bayesian opt, ensemble | ~2 weeks | ±0-1pp, mostly story |
| **5 — production** | testnet, calibration A/B, anti-skill auto | ~1 week | risk reduction, defence |
| **6 — symbols** | altcoins, perps, options | ~3 weeks | doubles dataset, new features |
| **7 — methodology** | Bayesian CI, embargo, purge, power | ~1 day | academic polish |
| **8 — defence** | visualisations, ADRs, charts | ~1 day | pure defence visuals |

**Total fully-out-the-box work to do everything**: ~3-4 months.
**Most realistic 14-day pre-defense**: items in "If defense 7-14 days" block above.
