# §02 — Model results

> All numbers below are reproducible with `tools/multi_horizon_eval.py`
> and `tools/train_joint_1m.py` against the live Tokyo Postgres on
> the date-stamped runs cited.

> **Figures referenced from this section:**
> [`fig-01-dir-acc-comparison.svg`](figures/fig-01-dir-acc-comparison.svg)
> · [`fig-04-fee-tier-pnl.svg`](figures/fig-04-fee-tier-pnl.svg)
> · [`fig-05-cv-power.svg`](figures/fig-05-cv-power.svg)

## 2.1 Solo per-symbol (production, daily 04:00 UTC)

Three CatBoost models (`btcusdt_1m.cbm`, `ethusdt_1m.cbm`,
`bnbusdt_1m.cbm`), trained on a 93 h rolling window of 1-second OFI
data per symbol. Walk-forward CV with embargo=1, isotonic calibration
once n_oos ≥ 1000, sample-weight half-life 720 bars.

Snapshot 2026-05-08 (live prod):

| symbol | dir_acc | 95 % Wilson CI | binomial p-value | n_folds | base rate |
|--------|---------|----------------|------------------|---------|-----------|
| BTCUSDT | **0.5288** | [0.5060, 0.5510] | 5.5e-3 | 33 | 0.5072 |
| ETHUSDT | **0.5338** | [0.5128, 0.5543] | 5.8e-4 | 39 | 0.5063 |
| BNBUSDT | **0.5552** | [0.5333, 0.5766] | 7.2e-7 | 32 | 0.5082 |

All three reject H₀ (no skill) at α = 0.01. Edge over base rate:
BTC +2.2 pp, ETH +2.7 pp, BNB +4.7 pp.

### Frozen holdout (Mon 04:30 UTC weekly)

A separate `tools/eval_frozen_holdout.py` evaluates each model on
data the trainer was forbidden to see (release T.17). Last run
2026-05-06:

| symbol | dir_acc | 95 % CI | p-value |
|--------|---------|---------|---------|
| BTCUSDT | 0.5400 | [0.5240, 0.5566] | 2.8e-6 |
| ETHUSDT | 0.5482 | [0.5323, 0.5661] | 5.5e-9 |
| BNBUSDT | 0.5636 | [0.5459, 0.5808] | 4.5e-13 |

**Holdout dir_acc ≥ CV dir_acc for all three symbols.** This is the
single strongest evidence that walk-forward CV isn't overfitting:
the model performs as well or better on data it never indexed.

## 2.2 Joint multi-symbol (release Phase 2.1, 2026-05-09)

ONE CatBoost model trained on **pooled** BTC + ETH + BNB data with
3-symbol one-hot identity features (`is_btc`, `is_eth`, `is_bnb`).
21 features total (18 microstructure + 3 identity). Window: 240 h.

Production run, 2026-05-09:

| metric | value |
|--------|-------|
| dir_acc | **0.5409** |
| 95 % Wilson CI | [0.5342, 0.5476] |
| binomial p-value | **4.97e-33** |
| n_folds | **353** |
| n_oos predictions | 21 180 |
| base rate | 0.5078 |
| log_loss (mean per fold) | 0.7396 |
| isotonic calibrator brier (raw) | 0.2669 |
| isotonic calibrator ECE (raw) | 0.1172 |
| training elapsed | 962.7 s |

### Why pooling helps

| property | solo | joint | gain |
|----------|------|-------|------|
| n_oos predictions | 1 800 - 2 400 per model | **21 180** | ~10× |
| 95 % CI half-width | ±0.022 | **±0.0067** | **3.3× tighter** |
| p-value | 1e-3 to 1e-7 | 5e-33 | many orders |
| symbols served | 1 each | 3 | one model |

**Statistical-power story:** the joint model isn't merely a slightly
better point estimate — it's a *much more confident* point estimate.
A solo BTC dir_acc of 0.5288 with CI ±0.022 admits a "true"
dir_acc anywhere in [0.506, 0.551]; the joint 0.5409 ±0.0067 narrows
the range to [0.534, 0.547]. For commercial decisions
("is the edge worth fee X?") this matters more than the +1 pp gain.

### Drift resilience by construction

When BTC's microstructure regime shifted in May 2026 (KS=0.68 on
`spread_bps_mean`, see §03), solo BTC 5m long_horizon collapsed
from 0.539 → 0.492 in 10 days. **Joint 1m** lost only −1.4 pp over
the same window (spike 0.5547 → prod 0.5409). Pooling across symbols
diluted the per-symbol regime shift. We hypothesise this is because
the cross-symbol gradient — *how* OFI / depth_imb / spread translate
to 1-minute returns — is more stable than the per-symbol marginal
distribution of those features.

## 2.3 Multi-horizon (release Phase 2.0, 2026-04-29)

Same trainer, same data, four bar sizes: 1m, 5m, 15m, 60m. The
question: at what horizon does directional skill stop being
detectable?

| horizon | feature_set | symbol | dir_acc | p-value | verdict |
|---------|-------------|--------|---------|---------|---------|
| 1m | microstructure | BTC | 0.5676 (28.04) | 9.3e-10 | ✅ |
| 1m | microstructure | ETH | 0.5382 (28.04) | 2.0e-3 | ✅ |
| 5m | microstructure | BTC | 0.5052 | 0.44 | ❌ noise |
| 5m | long_horizon | BTC | 0.5391 (28.04) → **0.4918 (09.05)** | 0.07 → **0.69** | **drift collapse — see §03** |
| 5m | long_horizon | BNB | 0.5300 (09.05) | 0.008 | ✅ holds |
| 15m | long_horizon | BTC | 0.4516 | 0.90 | ❌ below chance |
| 60m | (any) | per-symbol | 0 folds (n_bars < 100) | n/a | ❌ data starvation |
| 60m | joint long_horizon | JOINT | 0.6333 (28.04) | 0.007 | ⚠️ promising but n=154 |

**Why edge dies at longer horizons** — microstructure features have
half-life of seconds-to-minutes (Easley-O'Hara market-microstructure
theory). By 5m horizon the OFI imbalance from minute zero has
already played out; by 15m it's noise. Long-horizon TA features
(OHLC / EMA / RSI / Bollinger) help on 5m+ in *some* regimes but,
as §03 shows, they're regime-conditional.

## 2.4 Fee-tier P&L

The reason 1m solo dir_acc 0.53 still loses money on retail fees:

```
E[P&L per trade] = (2p − 1) × E[|move|] − 2 × fee
                 = 0.06 × 4 bps        − 30 bps  (retail × 2 sides)
                 = 0.24 − 30           = −29.8 bps  ❌
```

Multi-horizon eval reports four fee tiers per (symbol, bar, feature_set):

| symbol × horizon × features | retail | vip5 | vip9 | mm_rebate |
|-----------------------------|--------|------|------|-----------|
| BTC 1m microstructure | −14.7 bps | −1.7 bps | **+0.33 bps** | **+1.13 bps** |
| Joint 1m microstructure | −14.6 bps | −1.6 bps | **+0.37 bps** | **+1.17 bps** |
| BTC 5m long_horizon (28.04) | −14.6 bps | −1.6 bps | **+0.45 bps** | **+1.25 bps** |
| Joint 60m long_horizon (28.04) | −9.4 bps | **+3.6 bps** | **+5.6 bps** | **+6.4 bps** |

**The story:** edge exists; whether it pays depends on the fee tier
the trader has access to. VIP9 / market-maker tier is realistic for
production trading desks ($10M+ monthly volume, KYC done). For
academic / small-trader use, the take-away is *the model has
verified skill that becomes profitable above a measurable
fee threshold* — the threshold is also a research result.

## 2.5 Reproducibility

Everything in this section can be regenerated:

```bash
ssh root@147.45.49.40
set -a; source /etc/neucast/env; set +a
cd /opt/neucast
sudo -u stailfx --preserve-env=DATABASE_URL \
    /opt/neucast/venv/bin/python -m tools.multi_horizon_eval \
    --symbols BTCUSDT ETHUSDT BNBUSDT \
    --horizons 1 5 15 60 \
    --feature-sets microstructure long_horizon joint joint_long_horizon \
    --since-hours 96
```

Joint production weights:

```bash
sudo -u stailfx --preserve-env=DATABASE_URL \
    /opt/neucast/venv/bin/python -m tools.train_joint_1m \
    --since-hours 240
```

Run-history of every training is persisted in Postgres
`training_runs` table — `id=288 symbol=JOINT n_folds=353 ...` is the
specific row backing the joint numbers above.
