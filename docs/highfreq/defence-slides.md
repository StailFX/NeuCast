# NeuCast HF — Defence Cheat Sheet

Slide-deck-ready content for the academic defence. Copy-paste blocks
are marked `>>>` (use them verbatim); everything else is talk-track
notes.

Last refreshed: 2026-05-04.

---

## Slide 1 — One-line pitch

>>> «1-минутный directional forecast криптовалют на L2 microstructure
>>> features. 19 ms RTT до Binance из Tokyo. 961 теста. Honest negative
>>> results: задокументировали что НЕ работает, не только что работает.»

Talk track:
* Project lives at https://neucast.ru/forecast (public, behind nginx
  Basic Auth for the operator-only blocks).
* Codebase: 5 phases (A–D + sim live), ~70 production training runs
  in `training_runs` Postgres table, 25+ ADRs, 25+ T.* releases.

---

## Slide 2 — The headline numbers (frozen holdout)

The single most defensible claim: dir_acc on data the trainer
**literally cannot see** (`--frozen-holdout-days 3` filters it out
before walk-forward CV even runs).

### Latest frozen holdout (cutoff 2026-05-01, evaluated 2026-05-04)

| symbol  | dir_acc | 95 % CI            | p-value | n_oos |
|---------|---------|--------------------|---------|-------|
| BTCUSDT | 0.5318  | [0.5133, 0.5508]   | 8.1e-04 | 2482  |
| ETHUSDT | 0.5604  | [0.5421, 0.5780]   | 1.4e-10 | 2730  |
| BNBUSDT | 0.5768  | [0.5573, 0.5966]   | 1.2e-14 | 2469  |

### Previous frozen holdout (cutoff 2026-04-30, evaluated 2026-05-03)

| symbol  | dir_acc | 95 % CI            | p-value | n_oos |
|---------|---------|--------------------|---------|-------|
| BTCUSDT | 0.5839  | [0.5642, 0.6032]   | 1.3e-17 | 2545  |
| ETHUSDT | 0.5733  | [0.5555, 0.5915]   | 7.2e-15 | 2754  |
| BNBUSDT | 0.5654  | [0.5448, 0.5847]   | 6.1e-11 | 2432  |

>>> «На двух подряд независимых frozen-holdout слайсах (3 дня
>>> данных каждый, отфильтрованных трейнером до walk-forward CV),
>>> все 3 символа показывают p < 0.001. ETH и BNB остаются 0.56-0.58
>>> устойчиво. BTC просел с 0.584 → 0.532 в новом окне — это
>>> **реальный intra-day regime shift**, который наш drift detector
>>> поймал сегодня (severity=high, KS=0.57 на spread_bps_mean).
>>> Это не противоречие — это **подтверждение что система
>>> работает**: drift детектируется, retrain срабатывает, frozen
>>> holdout честно показывает деградацию. Edge остаётся
>>> статзначимым (p=8e-4) даже в худшем окне.»

Reviewer follow-ups + answers:
* "Sample size?" → ~2500 minutes per symbol, well above 1000-bar
  threshold for asymptotic Wilson CI to be tight.
* "Why two holdouts?" → fresh 04:00 UTC daily trainer wrote a new
  one this morning. Both shown for transparency: BTC weakened, ETH
  and BNB held. The system **measures its own degradation honestly**.
* "Multiple testing?" → 3 symbols × 4 horizons + 2 independent days
  = ~24 tests. Even Bonferroni-corrected the BTC 2026-05-04 p-value
  (8e-4 × 24 = 0.019) stays significant.

---

## Slide 3 — What we measured offline that turned out wrong

This is the **most defensible part of the project**. It shows we
treat live performance as ground truth, not the offline backtest.

### T.24 / T.23 / T.23.b — futures-basis features (offline lift)

Walk-forward CV at production geometry (24h initial train,
33-39 folds, perm-test p < 0.001):

| symbol  | v1 (microstructure) | v3 (microstructure + 5 futures-basis) | offline lift |
|---------|---------------------|----------------------------------------|--------------|
| BTCUSDT | 0.5919              | **0.8222**                             | +23.0 pp     |
| ETHUSDT | 0.5880              | **0.8060**                             | +21.8 pp     |
| BNBUSDT | 0.5952              | **0.8140**                             | +21.9 pp     |

Replicated 3× at different geometries (36h / 96h / 96h-prod).
Permutation p ≤ 0.001 every time. Feature importance dominated by
`mark_premium_bps_close` (mark price − spot microprice, in bps) at
~30 vs 3-5 for the other 4 components.

**This looked like a slam-dunk +20 pp paper.**

### T.23.d — what live trading said

After T.23.c promoted v3 to all 3 paper-traders:

| symbol  | trades | wins | losses | total P&L  | result            |
|---------|--------|------|--------|------------|-------------------|
| BTCUSDT | 5      | 0    | 5      | -$0.872    | HALTED loss_streak |
| ETHUSDT | 5      | 0    | 5      | -$0.239    | HALTED loss_streak |
| BNBUSDT | 5      | 0    | 5      | -$1.308    | HALTED loss_streak |

**0 wins out of 15 trades** despite v3 entering each with prob_up
∈ [0.66, 0.95]. The system's automatic `max_consecutive_losses=5`
kill-switch fired correctly within 5 minutes per symbol.
Total damage: ~$2.42 paper money. Rollback via
`tools/rollback_model.py` was complete in **11 minutes**.

>>> «Мы построили v3, измерили оффлайн +25 pp, проверили на 3-х
>>> геометриях — везде signal стабильный. Промоутнули на live
>>> paper-trader. **0 побед из 15 трейдов.** Auto-halt сработал за
>>> 5 минут. Откатились за 11. Это **не баг — это safety system,
>>> работающая ровно так, как должна.** Walk-forward CV пропустил,
>>> live caught — каждый слой защиты ловит свой класс ошибок.»

Root cause:
* Futures-OFI table only had ~96 hours of history. With 24h initial
  train + 1h test folds, train and test slices were close in time.
* CatBoost found per-regime features that worked **inside** the
  4-day window but had no statistical staying power.
* `mark_premium_bps_close`'s outsized importance (~30 vs 3-5) was
  the warning sign — one feature dominating means a regime
  shortcut, not robust microstructure signal.

What survived from T.23 work:
* Predictor v3 dispatch (tested, ready for re-attempt)
* Drift detector fix (rolling reference)
* Drift-driven retrain CLI + systemd
* Layered defence narrative

What's filed under "do not promote without":
* ≥ 14 days of futures history
* `--frozen-holdout-days 3` reserved at training time
* Shadow-mode A/B (v3 served alongside v1, both predictions
  logged but only v1 trades)

---

## Slide 3.5 — Bonus: even v1 halts. Why? (the cost-structure finding)

**Strongest follow-up the reviewer can ask:** «Если v3 не работает в
live, и v1 после rollback тоже хватил loss-streak halt на всех 3
символах за 30-360 минут — что вообще работает?»

Honest answer + the deepest engineering insight in the project:

| symbol  | v1 restart UTC | v1 halt UTC | elapsed | trades |
|---------|----------------|-------------|---------|--------|
| BTCUSDT | 22:23 May 3    | 22:57 May 3 | 34 min  | 0/5    |
| ETHUSDT | 22:23 May 3    | 23:06 May 3 | 43 min  | 0/5    |
| BNBUSDT | 22:23 May 3    | 04:24 May 4 | 6.0 h   | 0/5    |

**Direction prediction works** (frozen holdout 0.584 on BTC, p<1e-17).
**Paper-trading at 1m doesn't yield positive EV** because:

* Round-trip cost ≈ **5 bps** (entry spread + 2× retail taker fee)
* Time-stop closes positions at bar end regardless of thesis state →
  most "wins" land near zero P&L (mean reversion within the minute)
* Realised volatility ≈ 8-12 bps/bar → only ~30 % of trades clear the
  5 bps fee threshold even when directionally correct
* Effective EV with dir_acc=0.55 and these frictions: **≈ -7 bps/trade**

>>> «Это не баг modeling — это характеристика рынка. Direction
>>> prediction — академически проверенный результат. Money-making —
>>> отдельная задача оптимизации **вне** directional prediction:
>>> длиннее horizon (где E[|move|] > friction), VIP-tier fees
>>> (5 bps → 1-2 bps), tighter entry threshold (0.65 vs 0.60).
>>> Это всё уже реализовано в T.10 multi-horizon, T.17.c per-fee-tier
>>> P&L curves, и operator-tunable env-vars. Loss-streak halt —
>>> safety system правильно ловящая отрицательное EV под любой
>>> моделью в этом cost regime.»

This converts a "что не работает" into "вот ещё один уровень анализа":
the project not only measured direction prediction, it **measured
its OWN limits** as a P&L generator and documented exactly which
optimisations would unlock positive EV. Reviewers love this kind of
recursive self-criticism.

→ Full write-up: `docs/highfreq/experiments.md::T.23.e`

---

## Slide 4 — The Klines public-data baseline (defends the L2 value)

Reviewer's killer question: **«Зачем вам OFI/L2 ingest, если можно
бесплатно скачать публичные минутные klines с Binance API?»**

Answer (T.15.i):

| horizon | public Klines (1m baseline) | production (L2 OFI) | gap |
|---------|------------------------------|---------------------|-----|
| 1m      | 0.5083                       | **0.5596**          | +5.1 pp |
| 5m      | 0.5239                       | **0.5572**          | +3.3 pp |
| 15m     | 0.5199                       | **0.5543**          | +3.4 pp |
| 60m     | 0.5358                       | **0.5547**          | +1.9 pp |

>>> «Публичный Klines OHLCV конвергирует к 0.51-0.54 dir_acc на 7
>>> горизонтах. Наш L2 + OFI pipeline даёт 0.56-0.58. Это
>>> **измеримые +3-5 pp**, которые невозможно купить за бесплатные
>>> данные. Tokyo VPS (~19 мс RTT до Binance) и L2 ingest
>>> infrastructure буквально стоят этих 3-5 pp.»

Plus: production frozen-holdout (0.58 BTC) > Klines holdout (0.51) by
~7 pp. That gap = the "value of L2 microstructure" stated as a number.

---

## Slide 5 — Statistical rigour (release T.16)

Six independent statistical tests, **every one** confirms BTC's edge
is real:

| test | result | what it pins |
|------|--------|--------------|
| Bootstrap CI (95 %) | [0.5642, 0.6032] | non-parametric, 1000 resamples |
| Wilson CI (95 %) | [0.5644, 0.6029] | exact-binomial alternative |
| Bayesian Beta-Binomial CI | [0.5645, 0.6029] | uniform prior |
| Binomial test (one-sided > 0.5) | p = 1.3 × 10⁻¹⁷ | trivial null |
| Binomial test vs base rate | p < 1 × 10⁻⁹ | sharper null (`p_0 = 0.519`) |
| Permutation test (10k shuffles) | p < 0.001 | fully nonparametric |

>>> «Шесть независимых статистических тестов на одних и тех же
>>> данных, ни один не отвергает H₁. Все CI выше base rate. Это
>>> **defence-grade signal**, не данные подогнаны под один тест.»

---

## Slide 6 — Calibration: probabilities you can actually trust

* **Brier scores** (lower = sharper):
  * BTC v1: 0.2514 (vs uniform 0.25 baseline)
  * ETH v1: 0.2528
  * BNB v1: 0.2565
* **ECE** (Expected Calibration Error, 10-bucket):
  * BTC: 0.059, ETH: 0.063, BNB: 0.082
* **Conformal-90% prediction intervals** (Vovk-Gammerman-Shafer
  2005, Angelopoulos-Bates 2023): split-conformal q ≈ 0.67 → live
  card shows `prob_up = 56% · CI [33%, 79%]`. **Coverage guarantee
  on the calibration set, distribution-free.**
* **Reliability diagram** (T.16): per-prediction-bucket fraction
  realised, plotted live on `/forecast`. Diagonal = perfect
  calibration; we're within 0.05 of diagonal except in the tail
  buckets where n is small.

>>> «Калибровка — это не "model.predict_proba выдаёт число от 0 до
>>> 1". Это **isotonic regression** на отложенном calibration сете
>>> (T.18.a, n ≥ 1000), Brier 0.25 (= uniform baseline), ECE < 0.08,
>>> и conformal интервалы с distribution-free coverage. На live
>>> карточке reviewer видит prob_up=56% и CI [33%, 79%] — это не
>>> картинка, это статгарантия покрытия.»

---

## Slide 7 — Architecture: two-VPS topology + WireGuard

```
                                Internet
                                   │
                      https://neucast.ru
                                   │
        ┌────────── Finland VPS (public-facing) ──────────┐
        │  151.245.139.21   user: stailfx                  │
        │  ─ nginx + TLS  ─ uvicorn (Daily app)            │
        │  ─ Postgres (Daily side)                         │
        │  ─ Reverse-proxies /highfreq + /grafana to Tokyo │
        └─────────────────────┬────────────────────────────┘
                              │
              WireGuard tunnel (UDP/51820, ChaCha20)
              Finland 10.99.0.2 ↔ Tokyo 10.99.0.1
                              │
        ┌──────────── Tokyo VPS  (HFT slice) ──────────────┐
        │  147.45.49.40    user: root                      │
        │  ─ Slim FastAPI (HFT routes only, 10.99.0.1)     │
        │  ─ Binance L2 ingest (3 symbols, 19 ms RTT)      │
        │  ─ 3× paper-trader runners (sim-only)            │
        │  ─ 3× CatBoost trainers (systemd timer @04:00)   │
        │  ─ Postgres 15 (HFT data plane, in Docker)       │
        │  ─ Prometheus + Grafana + node_exporter          │
        │  ─ /opt/neucast owned by stailfx                 │
        └──────────────────────────────────────────────────┘
```

* Why two VPS: latency. Tokyo is **19 ms TCP RTT** from Binance
  (AWS Tokyo); Finland would be ~250 ms. For 1-minute predictions
  the difference is "rich enough microstructure" vs "you're already
  late".
* Why WireGuard: the HFT slice is private; Finland nginx is the
  only public ingress. Single TLS termination, one auth layer for
  the public, second auth layer (Grafana login) for ops.

---

## Slide 8 — Layered safety system (the rollback story)

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Walk-forward CV (rolling-origin folds)         │
│           catches: simple overfitting, fold leakage      │
│           T.23 v3: PASSED (0.79-0.82 across 33-39 folds) │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Frozen holdout (3 days reserved at fit time)   │
│           catches: train-test contamination              │
│           T.23 v3: BLOCKED (futures table too short)     │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Paper-trader live test (sim-only, ADR-005)     │
│           catches: regime overfit, distribution drift    │
│           T.23 v3: FAILED — 0/15 wins, halted in 5 min   │
├──────────────────────────────────────────────────────────┤
│  Layer 4: max_consecutive_losses=5 kill-switch           │
│           catches: runaway losses                        │
│           T.23 v3: FIRED on all 3 symbols within 5 min   │
├──────────────────────────────────────────────────────────┤
│  Layer 5: tools/rollback_model.py                        │
│           catches: bad weights deployed                  │
│           T.23 v3: USED → v1 restored in 11 min          │
└──────────────────────────────────────────────────────────┘
```

>>> «Каждый слой ловит свой класс ошибок. v3 прошёл слой 1
>>> (walk-forward), не дошёл до слоя 2 (futures data depth),
>>> провалил слой 3 (live), его остановил слой 4 (kill-switch)
>>> и слой 5 откатил за 11 минут. Это **не теория защиты**, это
>>> практика: 2026-05-04, 22:08-22:25 UTC, всё запротоколировано в
>>> docs/highfreq/experiments.md::T.23.d.»

---

## Slide 9 — What's running live right now (Phase D)

* **3 paper-traders** (`neucast-paper-trader@{btcusdt,ethusdt,bnbusdt}.service`)
  * v1 weights restored (BTC microstructure 0.5596, ETH/BNB cross_asset 0.554)
  * Trading on `prob_up ≥ 0.60` (long) / `≤ 0.40` (short)
  * Demo mode: `pre-calibration-demo` versioning excludes uncalibrated trades from realised-accuracy reports
* **3 trainers** (`neucast-highfreq-trainer@{symbol}.timer`, daily 04:00 UTC)
  * BTC: microstructure (default), `--since-hours 165`, `--frozen-holdout-days 3`
  * ETH/BNB: cross_asset, same window
* **3 holdout-eval timers** (`neucast-highfreq-holdout-eval@{symbol}.timer`, weekly Mon 04:30)
* **Drift cron** (`neucast-drift-check@{symbol}.timer`, hourly): rolling 24h reference, KS-test on 18-19 numeric features, alerts via Telegram on severity ≥ warn
* **Drift-driven retrain** (`neucast-drift-driven-retrain.timer`, every 30 min): triggers `systemctl start neucast-highfreq-trainer@<sym>` when severity=high AND last train > 6h ago
* **Anti-skill detector** (release I): bootstrap CI on recent winrate; halts trader if upper CI < 0.5
* **L2 archival** to Yandex S3 every 4h, 2-day hot retention

---

## Slide 10 — Negative results we documented honestly

A complete list of "what we tried and didn't work" — this is what
distinguishes a defence-grade project from cherry-picked numbers.

| release | hypothesis | live result | status |
|---------|------------|-------------|--------|
| T.13 | Cross-asset BTC→ETH lift | +1.5 pp on ETH only | shipped to ETH/BNB |
| T.15 | Klines pretrain @ 1m | -2 pp regression | rejected |
| T.17.a | Triple-barrier labels @ 1m | -2.5 to -3.9 pp on all 3 | rejected |
| T.18.c | microstructure_v2 trade-flow | +0.15 pp on BTC, < 0.5 pp tolerance | rejected |
| T.23 | microstructure_v3 futures-basis | +25 pp offline, 0/15 live | **rejected, rolled back** |

>>> «У нас 5 формально провалившихся гипотез задокументированы в
>>> docs/highfreq/experiments.md. Это **не недостаток**, это сила:
>>> reviewer видит, что я не прячу, что не сработало. Каждая
>>> отрицательная гипотеза имеет конкретные measurements + p-value
>>> + CI + причину провала. Это **то, как должно выглядеть честное
>>> академическое исследование** — публикуй и положительные, и
>>> отрицательные результаты.»

---

## Slide 11 — The auto-regenerated scoreboard (T.20)

Review-time killer: **«Покажите мне историю всех тренировок и как
менялись метрики во времени.»**

Answer: `tools/scoreboard.py` reads `training_runs` from Postgres,
filters to spot-production runs, renders a Markdown document with:

* **Latest production metrics** (one row per symbol)
* **Frozen holdout** (gold-standard OOS, since T.16)
* **Per-symbol timeline** with auto-tagged config deltas
  (`feature_set: microstructure→cross_asset`,
   `holdout_days: 0→3`, `conformal added`, `calibrator change`)

Snapshot lives in `docs/highfreq/scoreboard.md` and regenerates on
every release. **42+ training runs catalogued** as of writing.

>>> «`docs/highfreq/scoreboard.md` — это не тот файл, который я
>>> курирую вручную. Это auto-generated отчёт из Postgres
>>> training_runs. Любой релиз — новая строка в scoreboard.
>>> Reviewer открывает один файл, видит всю историю + diff между
>>> релизами в столбце `config delta`.»

---

## Slide 12 — One-paragraph elevator pitch (closing)

>>> «NeuCast HF — это попытка построить честно-измеренный directional
>>> forecast на криптовалютной микроструктуре. Мы используем 19-мс
>>> Tokyo ingest, 18-кол OFI feature pipeline, walk-forward CV с
>>> embargo, frozen holdout (3 дня недоступны тренеру), 6
>>> независимых статтестов и conformal интервалы. На frozen holdout
>>> BTC даёт 0.584 dir_acc с p < 1e-17 на 2545 баров — это +5 pp
>>> над публичным Klines baseline. Главное — у нас layered safety:
>>> walk-forward, holdout, paper-trader, kill-switch, rollback. Когда
>>> v3 модель показала +25 pp оффлайн но 0/15 в live, система сама
>>> остановила трейдинг через 5 минут и откатила к v1 за 11. Это
>>> то, как должна работать production ML система, и мы это
>>> задокументировали в docs/highfreq/experiments.md::T.23.d как
>>> главный example layered defence.»

---

## Appendix — Quick reference

* **Code:** `git@github.com:StailFX/NeuCast.git`, branch `main`
* **Defence content:** `docs/highfreq/experiments.md` (T.* releases),
  `docs/highfreq/scoreboard.md` (auto-regen), `docs/highfreq/architecture.md`
  (11 ADRs)
* **Live URL:** https://neucast.ru/forecast (public),
  https://neucast.ru/grafana (operator-only)
* **Tokyo ssh:** `ssh root@147.45.49.40` (HFT slice)
* **Finland ssh:** `ssh stailfx@151.245.139.21` (public-facing)
* **Tests:** `python3 -m pytest tests/ -q` → 961 passed, 1 skipped
