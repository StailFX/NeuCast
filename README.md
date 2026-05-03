# NeuCast

![tests](https://img.shields.io/badge/tests-961%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
![status](https://img.shields.io/badge/status-Phase%20D%20%C2%B7%20sim--only-orange)
![tokyo](https://img.shields.io/badge/HFT%20ingest-Tokyo%20%C2%B7%2019ms%20to%20Binance-purple)
![monitoring](https://img.shields.io/badge/monitoring-Prom%20%2B%20Grafana%20%2B%20Telegram-blueviolet)

AI-платформа для прогнозирования финансовых активов: дневные горизонты на стекинг-ансамбле и **HFT-минутный directional forecasting на Binance L2** в едином codebase.

**Live:** [neucast.ru](https://neucast.ru) · **HFT-демо:** [neucast.ru/highfreq](https://neucast.ru/highfreq)

---

## Two products in one repo

| | **NeuCast Daily** (this README) | **NeuCast High-Frequency** |
|---|---|---|
| Data | Yahoo Finance (OHLCV daily/hourly) | Binance Spot WebSocket (L2 + trades, 1-s grid) |
| Horizon | 1-30 days, multi-step | 1 minute, directional |
| Model | TCN + CatBoost + XGBoost + LightGBM stack | CatBoost binary classifier on `sign(return_1m)` |
| Loss | MSE / MAPE | log-loss + bootstrap CI on `dir_acc` |
| Output | Price forecast with Monte Carlo bands | Long / short signal with maker + taker P&L |
| Status | Production at [neucast.ru](https://neucast.ru) | **Phase D sim-only · 3 symbols live · [docs/highfreq/](docs/highfreq/README.md)** |
| Symbols | 200+ tickers (yfinance) | BTCUSDT · ETHUSDT · BNBUSDT |
| Update cadence | Daily / hourly | 1 row/sec into Postgres, prediction every minute |
| Test suite | inherits root | **961 tests · 100% passing** |
| Observability | systemd + nginx logs | Prometheus + Grafana + Telegram alerts (5 rules) |
| Cold storage | none | Yandex S3 (Parquet+snappy, hot-cold pattern, 7-day hot retention) |

The HF module reuses the same Postgres + FastAPI + systemd backbone but
addresses the wall the daily side hit: OHLCV-only data converges to
~1 % MAPE at the price-level optimum but produces zero trading edge.
HF pivots to microstructure features (OFI / microprice / depth-imbalance)
where the data carries directional signal, and to walk-forward CV +
bootstrap CI methodology where a "no skill" outcome is *visible* rather
than buried.

### HFT production topology (ADR-009 + ADR-010)

```
                    user
                      │
              https://neucast.ru
                      │ TLS (Let's Encrypt)
                      ▼
         ┌────────────────────────────────────┐
         │  FINLAND VPS  (Hostkey, EU)        │  151.245.139.21
         │  ─ nginx + TLS termination         │
         │  ─ uvicorn (main webapp + Daily)   │
         │  ─ celery + redis                  │
         │  ─ Postgres (Daily side)           │
         └─────────────┬──────────────────────┘
                       │
            ┌──────────┴──────────┐
   /highfreq + /api/highfreq/*    │  rest of routes
            │                     ▼
            ▼              (served locally)
  ┌─ WireGuard tunnel (ChaCha20+Curve25519, UDP/51820) ─┐
  │     Finland 10.99.0.2  ↔  Tokyo 10.99.0.1            │
  └──────────────────┬───────────────────────────────────┘
                     ▼
         ┌────────────────────────────────────┐
         │  TOKYO VPS  (4VPS.su, JP-cx21)     │  147.45.49.40
         │  ─ slim FastAPI (HFT routes only)  │
         │  ─ binance L2 ingest (3 symbols)   │
         │  ─ 3× paper-trader runners         │
         │  ─ 3× CatBoost trainers (timer)    │
         │  ─ Postgres 15  (HFT data plane)   │
         │     ↑                              │
         │     │ ~19 ms TCP RTT               │
         │     ▼                              │
         │  Binance Spot WS (AWS Tokyo,       │
         │  ap-northeast-1)                   │
         └────────────────────────────────────┘
```

### HFT module quick facts

* **Latency:** TCP RTT Tokyo ↔ Binance ≈ **19 ms** (median; vs ~250 ms from Finland)
* **Architecture decisions:** 11 ADRs in [`docs/highfreq/architecture.md`](docs/highfreq/architecture.md)
* **Test coverage:** 961 tests, every state-machine branch + endpoint pinned
* **Live UIs:**
    * [`/forecast`](https://neucast.ru/forecast) — public dashboard: prediction cards per symbol, **drift badge** (KS-test severity, T.21), **ensemble strip** (1m+15m blend, T.19), **conformal CI** (split-conformal 90 %, T.17.b), **cumulative P&L curve** (T.17.c), reliability diagram (T.16), feature importance, robustness suite, conditional accuracy
    * [`/grafana`](https://neucast.ru/grafana) — operations dashboard: ingest health, predictor latency, system metrics, paper P&L trends, **drift KS gauge** (T.18.b), **drift-driven retrain decision** (T.22)
* **Production training rigour:**
    * **Frozen holdout** (3 days reserved at training time, T.16): BTC 0.5839 [0.5642, 0.6032], ETH 0.5733 [0.5555, 0.5915], BNB 0.5654 [0.5448, 0.5847] — **the trainer literally cannot see this slice**, gold-standard OOS
    * **Walk-forward CV** with rolling-origin folds + 1-min embargo (López de Prado)
    * **Calibration**: isotonic regression for n ≥ 1000 (T.18.a), split-conformal prediction intervals (Vovk-Gammerman-Shafer, T.17.b)
    * **Auto-regenerated [`scoreboard.md`](docs/highfreq/scoreboard.md)** (T.20): every production run, every metric, config-delta tags
* **Layered safety system** (defence pillar):
    * Walk-forward CV catches simple overfitting
    * Frozen holdout catches train-test contamination
    * **Paper-trader live test** (Phase D) catches what offline backtests miss — see T.23.d in [`docs/highfreq/experiments.md`](docs/highfreq/experiments.md) for the canonical example: a +25 pp offline lift that was 0/15 in live, blocked by `max_consecutive_losses=5` kill-switch within 5 min, rolled back via `tools/rollback_model.py` within 11 min
    * **Drift detector** (T.18.b, KS-test on rolling 24h reference) + auto-retrain on severity=high with 6h cooldown (T.22)
    * **Anti-skill detector** (release I): if recent winrate's bootstrap CI upper bound < 0.5, halt or invert
* **Single source of truth:** all HFT data lives in Tokyo Postgres; Finland nginx reverse-proxies the read endpoints over the encrypted WireGuard tunnel
* **Hot-cold storage:** 7-day hot Postgres + cold Yandex S3 archival via daily atomic verify-before-delete cron (Parquet+snappy)
* **Observability:** self-hosted Prometheus + Grafana on Tokyo, daily TSDB snapshots backed up to Yandex S3, 5 alert rules → Telegram bot
* **Operational hardening:** SSH key-only, password auth disabled, UFW deny-all + explicit allow rules, EnvironmentFile root-only, 2-layer auth on Grafana (nginx Basic + Grafana login)

→ Detailed write-up: [`docs/highfreq/README.md`](docs/highfreq/README.md)
→ Release log + negative results: [`docs/highfreq/experiments.md`](docs/highfreq/experiments.md)
→ Production scoreboard (auto-regen): [`docs/highfreq/scoreboard.md`](docs/highfreq/scoreboard.md)

---

## Архитектура (Daily side)

```
                    ┌──────────────────────────────────────┐
                    │          Stacking Ensemble            │
                    │                                      │
  Yahoo Finance ──► │  TCN ──┐                             │
                    │  CatBoost ──┤── Ridge Meta ──► Price  │
                    │  XGBoost ──┤                         │
                    │  LightGBM ─┘                         │
                    └──────────────────────────────────────┘
```

### Модели

| Модель | Тип | Назначение |
|--------|-----|------------|
| **TCN** (Temporal Convolutional Network) | Deep Learning | Базовая модель. Multi-target: предсказывает log return + направление одновременно. Dilated causal convolutions, SE-блоки, inception-вход (kernel 3,5,7). |
| **CatBoost** | Gradient Boosting | Ловит нелинейные зависимости в агрегированных признаках (last, mean, std, diff, ma5, min, max). |
| **XGBoost** | Gradient Boosting | Альтернативный бустинг для разнообразия ансамбля. |
| **LightGBM** | Gradient Boosting | Быстрый бустинг с leaf-wise стратегией. |
| **Ridge** | Linear | Мета-модель стекинга — комбинирует прогнозы 4 моделей. |

### Признаки (Features)

- **Базовые:** Open, High, Low, Close, Volume
- **Moving Averages:** MA_5, MA_10, MA_20, MA_50
- **Осцилляторы:** RSI (14), MACD (12/26/9), Signal
- **Волатильность:** Bollinger Bands (upper/lower/pct), ATR (14), Volatility_20
- **Моментум:** ROC_5, ROC_10, Momentum
- **Объём:** Volume_MA_20, Volume_Ratio

### Прогнозирование

1. **Исторические данные** — модель предсказывает log returns: `r(t) = ln(P(t) / P(t-1))`
2. **Реконструкция цен** — `P(t) = P(t-1) * exp(r(t))`
3. **Будущее** — Monte Carlo GBM (Geometric Brownian Motion), 1000 симуляций с доверительными интервалами (5%, 25%, 50%, 75%, 95%)

## Стек технологий

| Компонент | Технология |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| ML | TensorFlow/Keras, CatBoost, XGBoost, LightGBM, scikit-learn |
| Async Tasks | Celery + Redis |
| Database | PostgreSQL |
| Frontend | Jinja2, Plotly.js |
| Deploy | Docker Compose, Nginx, Let's Encrypt |
| Data | Yahoo Finance API (yfinance) |

## Быстрый старт

### Docker (рекомендуется)

```bash
git clone https://github.com/StailFX/FinanceGuru.git
cd FinanceGuru
cp .env.example .env  # настроить пароли

# Полный стек (с nginx):
docker compose --profile standalone up -d

# Без nginx (если есть системный):
docker compose up -d
```

Приложение будет доступно на `http://localhost:8100`.

### Без Docker

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# PostgreSQL и Redis должны быть запущены
export DATABASE_URL="postgresql://user:pass@localhost:5432/neucast"
export REDIS_URL="redis://localhost:6379/0"
export USE_CELERY=1

# Запуск Celery worker (отдельный терминал):
celery -A app.celery_worker worker --loglevel=info --concurrency=2

# Запуск приложения:
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

## Обучение модели

```bash
# Multi-target TCN (основная):
python training/retrain_tcn.py --epochs 120

# LSTM + Attention (для сравнения):
python training/retrain_returns.py --epochs 80
```

Параметры:
- `--ticker` — тикер Yahoo Finance (default: `GC=F` — золото)
- `--start` / `--end` — период данных
- `--output` — путь к файлу модели (default: `weights/best_model.h5`)

## Метрики (Gold GC=F, тестовая выборка)

| Метрика | Значение |
|---------|----------|
| MAE | $33.90 |
| MAPE | 1.04% |
| R² | 0.9962 |
| Direction Accuracy | 58% |

## Структура проекта

```
.
├── app/                     # Основной пакет приложения
│   ├── __init__.py          # Экспорт FastAPI app
│   ├── main.py              # FastAPI роуты, middleware, startup
│   ├── prediction.py        # ML pipeline (TCN + Ensemble + Monte Carlo)
│   ├── celery_worker.py     # Celery задачи для фоновых прогнозов
│   ├── db.py                # SQLAlchemy подключение к БД
│   ├── models.py            # ORM модели (User, Ticker, Prediction...)
│   └── layers.py            # Кастомные Keras слои (TCNBlock, SEBlock, Attention)
├── weights/                 # Обученные модели (не в git)
│   ├── best_model.h5        # Основная TCN модель
│   ├── model_config.json    # Конфигурация модели
│   └── scaler.pkl           # Скейлер признаков
├── training/                # Скрипты обучения
│   ├── retrain_tcn.py       # Multi-target TCN обучение
│   ├── retrain_returns.py   # LSTM baseline обучение
│   ├── retrain.py           # Простое обучение
│   └── train_model.py       # Ensemble training pipeline
├── templates/               # Jinja2 шаблоны
│   ├── landing.html         # Лендинг
│   ├── form.html            # Форма прогноза
│   ├── predict.html         # Результаты с графиками
│   ├── waiting.html         # Страница ожидания (Celery)
│   ├── login.html
│   └── register.html
├── static/                  # Статика (logo, favicon)
├── nginx/                   # Конфиг Nginx
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
