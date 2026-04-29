# NeuCast HF — мониторинг и наблюдаемость

Кратко: где смотреть что, кто что видит, и какой URL для каких целей.

## TL;DR — три пользовательских уровня

| Кто | Где смотрит | Что видит |
|---|---|---|
| **Случайный посетитель сайта** | `https://neucast.ru/forecast` | Куда пойдут BTC/ETH/BNB через минуту, последние сделки, win-rate за 24h. Без жаргона. |
| **Я (оператор-разработчик)** | `https://neucast.ru/highfreq` | Полная панель: dir_acc, Wilson CI, p-value, fold count, calibration, fee tiers, 16 разных API endpoints |
| **Я при отладке** | `ssh root@10.99.0.1 + journalctl -u <service>` | Логи в реальном времени, raw метрики, ad-hoc SQL |

---

## Публичная страница `/forecast` (release T)

`https://neucast.ru/forecast` — простая страница для обычных пользователей. Видит её любой, кто зашёл на neucast.ru.

**Что показывает:**
- 3 карточки: BTC / ETH / BNB. Стрелка ↑ ↓ → + текст «вверх/вниз/в стороне» + полоска уверенности (%).
- Последние 10 paper-сделок: символ, сторона, время, P&L в %.
- 24h-метрики: количество сделок, win-rate, средний результат.

**Как обновляется:** клиент-сайд polling, прогноз раз в 30 сек, сделки раз в 60 сек. На фронте никаких WebSocket'ов — простой fetch + setInterval.

**Что НЕ показывает (и почему):**
- bootstrap CI / p-value / Wilson interval — слишком технично, обычный пользователь увидит только confidence %.
- Calibration статус — если модель не калибрована, карточка показывает "модель готовит прогноз…".
- Fee tier P&L — слишком детально.

**Важно:** страница про текущий статус, не про историю; для глубокого анализа есть `/highfreq`.

---

## Операторская страница `/highfreq` (для меня)

`https://neucast.ru/highfreq` — техническая панель. Видит любой, у кого есть URL (там нет authentication), но дизайн рассчитан на мониторинг операторами.

**Что есть:**
- Live микропрайс + countdown до следующей минуты
- Текущий signal + prob_up + calibrated badge
- Walk-forward CV: dir_acc + bootstrap CI + Bayesian CI + p-value
- Fold history + per-fold metrics
- Calibration reliability chart (Brier + ECE)
- Anti-skill detector status
- Fee-tier P&L breakdown
- 16+ API endpoints как сырые данные

**SEO:** `noindex` — не должна индексироваться поисковиками, в отличие от `/forecast`.

---

## Telegram уведомления

Тёплая канал `HF_TELEGRAM_SIGNAL_CHAT_ID` (set on Tokyo `/etc/neucast/env`). Бот шлёт:

| Событие | Триггер |
|---|---|
| Signal flip | Когда направление прогноза для символа меняется (`up → down`, `down → up`, или через neutral). Из `app/highfreq/signal_telegram.py`. |
| Trade closed | Когда paper-trader закрывает позицию (любой exit_reason). Сообщение содержит entry/exit, qty, P&L bps + USD. |
| Anti-skill alert | Когда ML-детектор обнаруживает что модель устойчиво ошибается (см. `app/highfreq/anti_skill_detector.py`). |

Чтобы выключить уведомления вообще: `HF_TELEGRAM_SIGNAL_ENABLED=0` в `/etc/neucast/env`. Чтобы временно заглушить — Telegram-сторона имеет mute.

---

## Grafana

`https://neucast.ru/grafana` (basic auth + grafana login).

Подключён Prometheus который скрейпит локальные exporters на Tokyo:
- `neucast-highfreq.service` — порт 9090 (spot ingest counters)
- `neucast-paper-trader@*.service` — 9091/9092/9093 (1m runners)
- `neucast-paper-trader-multihorizon@*.service` — 9094-9099 (15m+ spot)
- `neucast-futures-highfreq.service` — 9101 (futures ingest)
- `neucast-futures-paper-trader@*.service` — 9110+ (futures paper traders)
- `prometheus-node-exporter.service` — 9100 (system metrics + textfile collector)

Что в textfile collector:
- `/var/lib/node_exporter/textfile_collector/neucast_hf_trainer_<sym>.prom` — last successful trainer run timestamp per symbol
- `/var/lib/node_exporter/textfile_collector/neucast_futures_funding_poll.prom` — last successful funding poll
- Прочие cron-heartbeat файлы (l2-archive, ofi-archive, prom-backup)

Дашборды (если живёт):
- HF ingest health (frames/sec, reconnects)
- Trainer status (last_run, n_folds, dir_acc)
- Paper-trader P&L (cumulative line per symbol)

---

## Сырые API endpoints

Все доступны без auth (есть только Grafana под basic auth). Список:

| URL | Назначение |
|---|---|
| `/api/highfreq/health` | "ingest жив?" — `rows_last_60s` |
| `/api/highfreq/status` | predictor state + текущая модель metadata |
| `/api/highfreq/forecast?symbol=BTCUSDT` | live prob_up + signal |
| `/api/highfreq/paper_trades?symbol=BTCUSDT&limit=10` | последние сделки |
| `/api/highfreq/realized_accuracy?symbol=BTCUSDT` | actual win-rate over time |
| `/api/highfreq/realized_accuracy_full?symbol=BTCUSDT` | + Wilson CI + p-value |
| `/api/highfreq/training_report?symbol=BTCUSDT` | metrics.json последней тренировки |
| `/api/highfreq/training_history?symbol=BTCUSDT` | хронология retrain'ов |
| `/api/highfreq/predictions_history?symbol=BTCUSDT` | predictions_log таблица |
| `/api/highfreq/feature_importance?symbol=BTCUSDT` | CatBoost feature importance |
| `/api/highfreq/microprice_history?symbol=BTCUSDT` | свечи микропрайса |
| `/api/highfreq/orderbook?symbol=BTCUSDT` | top-N L2 snapshot |
| `/api/highfreq/regimes?symbol=BTCUSDT` | volatility regime tagging |
| `/api/highfreq/anti_skill?symbol=BTCUSDT` | anti-skill detector state |
| `/api/highfreq/pnl_by_fee_tier?symbol=BTCUSDT` | P&L при retail/vip5/vip9/mm fees |
| `/api/highfreq/actionable_signal?symbol=BTCUSDT` | signal + size hint, calibrated only |

Удобно для скриптинга: `curl https://neucast.ru/api/highfreq/training_report?symbol=BTCUSDT | jq`.

---

## SSH-уровень (когда что-то поломалось)

```bash
# Tokyo (основной HFT slice)
ssh root@147.45.49.40
# или через WG если direct заблокирован fail2ban'ом:
ssh -J stailfx@151.245.139.21 root@10.99.0.1

# Что обычно смотреть:
systemctl list-units 'neucast-*' --no-pager
journalctl -u neucast-highfreq.service -f          # spot ingest
journalctl -u neucast-futures-highfreq.service -f  # futures ingest
journalctl -u neucast-paper-trader@btcusdt.service -f
journalctl -u neucast-highfreq-trainer@btcusdt.service -n 30

# Postgres ad-hoc:
docker exec -it neucast-postgres psql -U neucast -d neucast

# Сколько данных накопилось:
docker exec neucast-postgres psql -U neucast -d neucast -c "
  SELECT n_live_tup, relname FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT 10;
"
```

---

## Когда что смотреть

| Сценарий | Куда |
|---|---|
| «Что прогнозит сейчас BTC?» | `/forecast` |
| «Как сегодня торгует?» | `/forecast` (24h-summary) или `/highfreq` (детально) |
| «Модель калибрована?» | `/highfreq` или `curl /api/highfreq/status` |
| «Когда последний trainer?» | Grafana или `curl /api/highfreq/training_history` |
| «Ingest жив?» | `curl /api/highfreq/health` |
| «Что в логах?» | `journalctl -u <service>` через SSH |
| «P&L vs spot vs futures?» | `/highfreq` → fee-tier P&L block. После phase 4 — отдельный compare-tool |
| «Падают ли prediction ы?» | Grafana → ws_frames_total, snapshots_dispatched_total |
| «А accuracy не упала?» | `/api/highfreq/realized_accuracy_full` (показывает W'ils CI + p-value) |

---

## Алёрты (что должно будить если что-то сломалось)

Сейчас активны:
- **Telegram signal-flip** — почти не алёрт, информационный
- **Grafana alerting** для критических: ingest стоит >5min, trainer не отработал >25h, no rows_last_60s

Чего ещё нет (запланировано):
- Anti-skill auto-halt с Telegram-нотификацией
- Funding-rate poll missing > 30min — будет алёрт когда разберёмся с Grafana provisioning

---

## Что показывать на защите

Канонический набор:

1. `/forecast` — «вот так это видит обычный пользователь»
2. `/highfreq` — «вот моя оператор-панель»
3. `/api/highfreq/training_report?symbol=BTCUSDT | jq` — «вот сырые метрики калибровки модели»
4. Grafana дашборд (если будет к тому моменту приличный)
5. Скриншот Telegram-канала с signal flips + trade closes — «продакшен живёт автономно»

Защитная история: «У меня 3 уровня видимости — публичная для пользователя, операторская для меня, сырые API + journalctl для отладки. Каждый уровень показывает то и только то, что человеку этого уровня нужно».
