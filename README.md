# NeuCast

AI-платформа для прогнозирования финансовых активов. Стекинг-ансамбль из 4 моделей машинного обучения с Monte Carlo симуляциями для многодневных прогнозов.

**Live:** [neucast.ru](https://neucast.ru)

---

## Two products in one repo

| | **NeuCast Daily** (this README) | **NeuCast High-Frequency** |
|---|---|---|
| Data | Yahoo Finance (OHLCV daily/hourly) | Binance Spot WebSocket (L2 + trades, 1-s grid) |
| Horizon | 1-30 days, multi-step | 1 minute, directional |
| Model | TCN + CatBoost + XGBoost + LightGBM stack | CatBoost binary classifier on `sign(return_1m)` |
| Loss | MSE / MAPE | log-loss + bootstrap CI on `dir_acc` |
| Output | Price forecast with Monte Carlo bands | Long / short signal with maker + taker P&L |
| Status | Production at [neucast.ru](https://neucast.ru) | **Phase A complete · sim-only · [docs/highfreq/](docs/highfreq/README.md)** |

The HF module reuses the same Postgres + FastAPI + systemd backbone but
addresses the wall the daily side hit: OHLCV-only data converges to
~1 % MAPE at the price-level optimum but produces zero trading edge.
HF pivots to microstructure features (OFI / microprice / depth-imbalance)
where the data carries directional signal, and to walk-forward CV +
bootstrap CI methodology where a "no skill" outcome is *visible* rather
than buried.

→ Detailed write-up: [`docs/highfreq/README.md`](docs/highfreq/README.md)

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
