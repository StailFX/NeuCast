# §05 — Engineering depth

> The infrastructure layer of the thesis. Each subsection lists
> concrete artefacts (file paths, test counts, alert names, code
> coverage scope) that a reviewer can verify by `git grep` or
> `ls`. None of these are slideware — every claim has a matching
> file under version control or a service running on Tokyo.

## 5.1 Test surface

### Backend Python tests

```
tests/
├── test_highfreq_paper_trader.py            # paper trader state machine
├── test_highfreq_paper_trader_runner.py     # runtime loop (27 tests)
├── test_highfreq_predictor.py               # model loading, calibration gate
├── test_highfreq_feature_pipeline.py        # OFI → minute-bar aggregation
├── test_highfreq_feature_pipeline_long_horizon.py
├── test_highfreq_feature_pipeline_joint.py
├── test_highfreq_calibration.py             # Platt + isotonic + reliability
├── test_highfreq_drift.py                   # KS-based drift detector
├── test_highfreq_anti_skill_detector.py     # winrate-CI anti-skill
├── test_highfreq_event_calendar.py          # halt-window scheduling
├── test_highfreq_multi_horizon.py           # eval table generation
├── test_highfreq_robustness_suite.py        # block-bootstrap CI
├── test_archive_l2_to_s3.py                 # S3 archival end-to-end
├── test_emergency_snapshot.py               # ad-hoc DR backup
├── test_training_history.py                 # training_runs persistence
└── test_neucast_full.py                     # Daily-side TCN integration
```

Notable: regression tests for two production outages —
`test_process_one_tick_clears_stale_event_halt_when_no_event_active`
(2026-05-09 sticky halt fix) and the L2 archive memory-pressure
unit-file diff (2026-05-08).

### Frontend TS/JS tests (Vitest + RTL + MSW)

```
frontend/
├── src/lib/format.test.ts             # 13 tests (fmtAge / fmtMoney / fmtPercent)
├── src/lib/useFlashOnChange.test.tsx  # 5 tests
├── src/lib/api.test.ts                # 8 tests (ApiError, URL construction, MSW)
├── src/components/Skeleton.test.tsx   # 3 tests
├── src/components/DriftBadge.test.tsx # 5 tests (severity → palette)
├── src/components/HorizonPill.test.tsx
├── src/components/ForecastCard.test.tsx
├── src/components/AuthForm.test.tsx   # 5 tests (login + register modes)
├── src/components/Navbar.test.tsx     # 3 tests (anon + auth states)
└── src/app/predict/predict-form.test.tsx  # POST → waiting redirect
```

**55 tests passing, ~2 second wall clock.** Run with `npm test`.

## 5.2 Observability

### Heartbeat metrics (textfile_collector)

Every one-shot cron writes its `last_success_timestamp_seconds` to
`/var/lib/prometheus/node-exporter/neucast_hf_*.prom`. Currently
14 active heartbeats:

```
neucast_hf_l2_archive
neucast_hf_ofi_archive
neucast_hf_paper_trades_backup
neucast_hf_prom_backup
neucast_hf_holdout_btcusdt    + ethusdt + bnbusdt
neucast_hf_trainer_btcusdt    + ethusdt + bnbusdt
neucast_hf_trainer_joint_1m   ← Phase 2.1
neucast_futures_funding_poll
```

### Grafana alerts

(`docs/highfreq/deploy/grafana/alerting/alerts.yaml`)

* `l2-archive-stale-rule` — heartbeat > 25h. Caught the 2026-05-08
  memory-pressure outage.
* `paper-trader-no-trades-rule` — no trades closed for 12 h.
* `drift-high-severity-rule` — KS > 0.4 on any feature.
* `ingest-stalled-rule` — `rows_last_60s` < 30.
* `model-stale-rule` — `model_age_seconds` > 86 400.

All five run on Prometheus's PromQL, evaluated every 60 s. Telegram
notifications via Grafana contact-points (group `@stailfx_neucast`).

### Runbooks

`docs/highfreq/runbooks/`:
* `runbook-l2-archive-stale.md` (referenced by the alert rule)
* `runbook-paper-trader-halt.md` (post-2026-05-09 fix)
* `runbook-emergency-snapshot.md` (point-in-time backup)
* `runbook-model-rollback.md` (T.17.d: archive existing weights
  before overwrite, keep_last_n=7)

## 5.3 Production safety properties

| property | mechanism |
|----------|-----------|
| **No model overwrites a working one without backup** | `app/highfreq/model_archive.archive_existing` — every trainer call moves the previous .cbm to a `.cbm.20260509-045023` archive before saving the new one. `keep_last_n=7` keeps a week of rollback options. |
| **No anti-skill model trades silently** | `app/highfreq/anti_skill_detector.py` — bootstrap CI on rolling winrate. If CI upper bound < 0.5, halt or invert the trade-side decision (env-configurable policy). |
| **No uncalibrated model trades** | `paper_trader.PaperTraderConfig.require_calibrated=True` — predictor must report `is_calibrated=True` (the `_calibrator.pkl` exists AND `dir_acc_ci_low > 0.5`) before the trader will open a position. |
| **No data leak from train to test** | Walk-forward CV with `embargo_bars=1`; calibrator fit on OOS predictions only; frozen-holdout reserved before fold construction. |
| **No race in JSON config writes** | `tempfile.NamedTemporaryFile` + `os.replace` pattern in `cron_metrics.write_cron_success` — atomic rename, no half-written `.prom` files. |
| **No silent cron crash** | Every one-shot has heartbeat + Grafana alert. Caught the L2-archive 30-min timeout silently dying mid-run. |
| **No auth bypass** | Argon2id password hashing (`argon2-cffi` backend, salt + 64MB memory + 3 iters), needs_rehash on login, constant-time dummy verify on user-not-found, HttpOnly + SameSite=Lax cookies, basic-auth in front of operator routes. |

## 5.4 Defence-grade engineering moves cited in code

* **Walk-forward CV with embargo** — Bailey & López de Prado (2014)
* **Bootstrap CI for winrate** — Efron (1979)
* **Wilson score interval** — Wilson (1927)
* **Bayesian Beta-Binomial CI** — pretty much every Bayesian textbook
* **Isotonic regression for calibration** — Niculescu-Mizil & Caruana (2005)
* **Platt scaling** — Platt (1999)
* **Split-conformal prediction intervals** — Vovk-Gammerman-Shafer (2005),
  Angelopoulos-Bates (2023)
* **KS test for drift** — Kolmogorov–Smirnov classic
* **Microstructure features (OFI, depth_imb)** — Easley-O'Hara
* **Multi-task learning via shared parameters** — joint training intuition

Citations live in source-file docstrings and metrics-report fields,
not just the thesis.

## 5.5 Deploy hygiene

* **Source of truth is GitHub** (`StailFX/NeuCast`), not the running
  VPS. Tokyo is `rsync`-deploy; Finland is `git pull`-deploy.
* **No secrets in the repo** — runtime credentials are loaded from
  root-owned `.env` files with mode 0600; operational inventories are
  stored outside the repository.
* **GitHub deploy key** for git pull on Finland (`~/.ssh/github_deploy`).
* **systemd hardening** on every unit:
  `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`,
  `ProtectKernelTunables`, `ProtectKernelModules`,
  `ReadWritePaths=` whitelist, `RestrictAddressFamilies`. Defence
  in-depth — caught a hot-fix bug 2026-05-06 where the textfile
  collector dir wasn't whitelisted (every cron heartbeat hit
  EROFS for two days; alert finally fired about itself, which
  was an existential moment).

## 5.6 What this gets you on defence day

A reviewer asking "is this production-ready?" gets a yes with
provenance:

> "Show me where you handle [model degradation / fee tier choice
> / regime shift / failed cron / data leak / auth attack /
> stale weights / OOM kill]?"

Every one of those items has a specific file, test, alert, or
runbook to point at. The system isn't claiming "we deploy and
hope" — it's claiming "we deploy, watch, fix when broken, and
documented the fix in git history." That's the difference between
a coursework demo and a defendable artefact.
