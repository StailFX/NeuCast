# План прогона на Yandex Cloud VPS

> Использование гранта 4000 ₽ на гиперпараметрический поиск + расширенный
> robustness-suite + обучение 1h HFT-модели. После запуска VM иди спать,
> через ~24-36 часов scp результаты обратно и удаляй VM.

## 1. Рекомендуемая конфигурация VM

| Параметр | Значение | Цена |
|---|---|---|
| Платформа | Intel Cascade Lake (стандарт) | — |
| vCPU | **16** (100% guaranteed) | ~ |
| RAM | **64 GB** | ~ |
| Disk | 100 GB SSD network-ssd | ~ |
| OS | Ubuntu 22.04 LTS | бесплатно |
| Преемптивность | **нет** (regular, не preemptible) | — |
| **Итого/час** | | **~45-50 ₽** |
| **На 36 часов** | | **~1600-1800 ₽** |

Из 4000 ₽ остаётся буфер ~2000 ₽ на отладку и продление при необходимости.

### Создать VM через Yandex CLI (быстрее, чем UI)

```bash
# На локальной машине
yc compute instance create \
  --name neucast-train \
  --zone ru-central1-a \
  --platform standard-v3 \
  --cores 16 \
  --memory 64 \
  --core-fraction 100 \
  --create-boot-disk \
    image-folder-id=standard-images,image-family=ubuntu-2204-lts,size=100,type=network-ssd \
  --network-interface subnet-name=default-ru-central1-a,nat-ip-version=ipv4 \
  --ssh-key ~/.ssh/id_ed25519.pub
```

Запиши **public IP** (он выводится после создания). Дальше — `ssh yc-user@<IP>`.

> Альтернатива: создать через web UI Yandex Cloud Console. Important — выбрать **Compute Cloud → Создать ВМ**, образ Ubuntu 22.04, 16 vCPU / 64 GB / 100GB SSD, network-ssd, public IP.

---

## 2. Setup на VM (5-10 минут)

```bash
ssh yc-user@<VM-PUBLIC-IP>

# system deps
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git tmux htop \
  build-essential libpq-dev

# clone project
git clone https://github.com/StailFX/NeuCast.git
cd NeuCast

# python venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel

# project deps
pip install -r requirements.txt

# extra: optuna для search
pip install optuna optuna-dashboard

# sanity-check
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "import catboost as cb; print('CatBoost', cb.__version__)"
python -c "import optuna; print('Optuna', optuna.__version__)"
```

Если хочешь чтобы yfinance не упёрся в rate-limit при 27 одновременных запросах, можно увеличить кэш:

```bash
export YF_CACHE_TTL=86400  # 1 day
```

---

## 3. Запуск трёх задач параллельно через tmux

```bash
# главная сессия — поднимаем 4 окна (3 задачи + 1 на мониторинг)
tmux new -s neucast

# Window 0: Optuna search (~6-10 ч)
source venv/bin/activate
mkdir -p tools/optuna_studies
python -m tools.optuna_search_tcn \
  --n-trials 300 \
  --tickers BTC-USD,GC=F,SPY \
  --since-days 1500 --days-ahead 30 \
  --study-name neucast_tcn_v1 \
  --storage sqlite:///tools/optuna_studies/tcn.db \
  --out tools/optuna_studies/best_params.json \
  2>&1 | tee tools/optuna_studies/run.log

# Ctrl+B затем C — новое окно
# Window 1: extended robustness suite (~3-5 ч)
source venv/bin/activate
mkdir -p docs/diploma/experiments
python -m tools.bootstrap_full_27 \
  --since-days 1500 --days-ahead 30 \
  --n-bootstrap 10000 \
  --out docs/diploma/experiments/robustness_full_27.json \
  --checkpoint docs/diploma/experiments/robustness_full_27.json \
  2>&1 | tee docs/diploma/experiments/bootstrap.log

# Ctrl+B затем C — новое окно
# Window 2: HFT 60m training (~2-3 ч)
source venv/bin/activate
python -m tools.train_hft_60m \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT \
  --years 3 \
  --weights-dir weights/highfreq \
  --data-dir data/historical \
  2>&1 | tee weights/highfreq/60m_train.log

# Ctrl+B затем C — новое окно
# Window 3: мониторинг
htop  # CPU / RAM
# или:  tail -f tools/optuna_studies/run.log
```

**Отключиться от tmux без остановки скриптов**: `Ctrl+B`, затем `D`.
**Вернуться позже**: `ssh yc-user@<IP>` → `tmux attach -t neucast`.

---

## 4. Мониторинг прогресса (опционально)

После того как Optuna наберёт ≥ 10 trials, можно открыть веб-дашборд:

```bash
# В новом терминале на VM:
optuna-dashboard sqlite:///tools/optuna_studies/tcn.db --host 0.0.0.0 --port 8080
# Открой http://<VM-PUBLIC-IP>:8080 в браузере локально (нужно открыть порт
# в Security Group VM в console Yandex Cloud — добавить ingress 8080 ANY)
```

Или просто tail логов через ssh.

---

## 5. Сбор результатов обратно (важно — до удаления VM!)

```bash
# На локальной машине:
cd /Users/stailfx/Desktop/NeuCast

# Optuna результаты
scp -r yc-user@<VM-IP>:~/NeuCast/tools/optuna_studies ./tools/

# Bootstrap + DM-test результаты
scp -r yc-user@<VM-IP>:~/NeuCast/docs/diploma/experiments/robustness_full_27.json \
  ./docs/diploma/experiments/

# Веса 60m моделей (всего ~750 KB)
scp yc-user@<VM-IP>:~/NeuCast/weights/highfreq/*_60m.* \
  ./weights/highfreq/
scp yc-user@<VM-IP>:~/NeuCast/weights/highfreq/60m_training_summary.json \
  ./weights/highfreq/

# Все логи (на случай если что-то надо пересмотреть)
scp yc-user@<VM-IP>:~/NeuCast/tools/optuna_studies/run.log \
  ./tools/optuna_studies/
scp yc-user@<VM-IP>:~/NeuCast/docs/diploma/experiments/bootstrap.log \
  ./docs/diploma/experiments/
scp yc-user@<VM-IP>:~/NeuCast/weights/highfreq/60m_train.log \
  ./weights/highfreq/
```

---

## 6. Удаление VM (КРИТИЧНО)

После того как scp прошёл — **сразу** удали VM, иначе грант продолжит съедаться даже на idle машине.

```bash
# Локально:
yc compute instance delete neucast-train
```

Или через web UI: Compute Cloud → найти `neucast-train` → «Удалить».

Дополнительно: **проверь что нет «зависших» disk volumes или snapshots** — они тоже могут продолжать списываться. Yandex Cloud Console → Compute → Диски / Снимки.

---

## 7. Интеграция результатов локально

### 7.1. Optuna best params

После scp прочитай `tools/optuna_studies/best_params.json`:

```bash
cat tools/optuna_studies/best_params.json | python -m json.tool
```

Файл содержит:
- `best_params`: словарь `{env_var: value}` — оптимальные значения
- `best_mean_mape`, `best_mean_dir_acc`: метрики на validation set
- `best_per_ticker`: разбивка по BTC-USD / GC=F / SPY

**Применение**: добавь оптимальные значения в `.env.example` (или README) с комментарием «from Optuna search 2026-05-15 (n=300)». В дипломной — отдельный подраздел «Гиперпараметрический ablation» с before/after сравнением.

### 7.2. Bootstrap + DM-test results

`docs/diploma/experiments/robustness_full_27.json` содержит:
- `rows[]`: для каждого тикера — MAPE_ci, dir_acc_ci, model_metrics_point
- `by_class`: кластерные средние с bootstrap-CI

**Применение**: обнови таблицы в `docs/diploma/chapters/04-daily-experiments.md` — добавь колонки «95% CI MAPE» и «95% CI dir_acc». Расширенный study автоматически перейдёт в LaTeX/Word через `tools/build_diploma.js` при следующем билде.

### 7.3. HFT 60m модели

После scp веса лежат в `weights/highfreq/{btcusdt,ethusdt,bnbusdt}_60m.{cbm,_metrics.json}`. Деплой на Tokyo:

```bash
# Локально → Tokyo
rsync -avz weights/highfreq/*_60m.* \
  root@147.45.49.40:/opt/neucast/weights/highfreq/

# Рестарт сервиса на Tokyo чтобы он подхватил новые модели
ssh root@147.45.49.40 'sudo systemctl restart neucast-highfreq-web.service'
```

После рестарта `?horizon=60` в `/api/highfreq/dashboard` начнёт работать.
В `frontend/src/components/HorizonPill.tsx` поменяй:

```ts
const HORIZON_AVAILABLE: Record<Horizon, boolean> = {
  1: true,
  5: true,
  15: true,
  60: true,    // ← было false
};
```

То же в `frontend/src/lib/HorizonContext.tsx`:

```ts
const HORIZON_TRAINED: Set<Horizon> = new Set([1, 5, 15, 60]);
```

И обнови backend cap в `app/highfreq/web.py::get_dashboard`:

```python
if horizon not in (1, 5, 15, 60):
    horizon = 1
```

Билд + деплой фронта на Finland как обычно.

---

## 8. Чеклист перед запуском

- [ ] Грант активирован в Yandex Cloud Console
- [ ] SSH ключ добавлен в Yandex Cloud (`yc compute ssh-key create` или через UI)
- [ ] Локально проверены 3 скрипта (`python -m tools.optuna_search_tcn --help` итд)
- [ ] Создана VM с правильной конфигурацией
- [ ] Запущены 3 tmux-окна
- [ ] Логи пишутся (`tail -f *.log` показывает прогресс)
- [ ] **Будильник на 24-36 часов** чтобы не забыть scp результаты + удалить VM

## 9. Что делать если что-то пойдёт не так

| Проблема | Действие |
|---|---|
| Optuna trial упал с OOM | Снизь `--n-trials` или подними VM до 32 GB / 128 GB |
| `yfinance` rate-limit | Подожди 10 мин и перезапусти — оно идемпотентно |
| Bootstrap скрипт зависает | Прерви `Ctrl+C`; checkpoint в JSON сохраняется на каждом тикере, можно резюмировать |
| 60m pretrain жалуется на shape | Проверь что `tools.binance_klines_download --interval 1h` отработал и parquet больше 5MB |
| VM зависла | `yc compute instance restart neucast-train`, потом `tmux attach` |
| Не успеваешь до сгорания гранта | Прерви, scp partial results (bootstrap checkpoint), удали VM. Что-то лучше чем ничего. |
