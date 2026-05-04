# NeuCast High-Frequency Module — Architecture (Phase A)

**Status:** Phase A complete · sim-backtest only · 60 unit tests, ~2.8k LOC source + 780 LOC tests
**Goal:** 1-minute directional forecast for BTC-USDT, with honest paper-trading simulation
**Owner:** Stailfx (coursework / portfolio project)
**Last update:** 2026-04-25
**Quick links:** [README](README.md) · [demo](demo.md) · [deploy](deploy/README.md)

---

## 1 · Goal

Build a parallel high-frequency forecasting product that lives next to the existing daily-prediction service. The HF module:

- consumes Binance Spot Level-2 order book stream + trade stream via public WebSocket
- engineers Order-Flow-Imbalance (OFI), microprice, depth-imbalance features at 1-second granularity
- trains a CatBoost classifier on `sign(return_1m)` with directional log-loss
- runs walk-forward sim-backtest with realistic Binance fees (maker rebate vs taker cost)
- surfaces live forecast + paper P&L + rolling directional accuracy on a new `/highfreq` UI page

**Phase A explicitly does NOT:**
- place real or paper orders against Binance
- predict at sub-1-minute horizons
- require sub-100ms inference latency

**This document records the design decisions made up-front, with reasoning and trade-offs, so a reader can understand *why* the system is shaped the way it is.**

---

## 2 · Constraints

The module shares a single 8 GB VPS (Hostkey Finland) with three unrelated production services. This shapes every decision below.

| Resource | Total | Currently used | Available for HF |
|----------|-------|----------------|--------------------|
| CPU cores | 4 | ~30 % avg | ≥1 core, bursts to 4 |
| RAM | 7.7 GB | 5.5 GB | 1.9 GB + 4 GB swap (added) |
| Disk | 118 GB | 54 GB | 60 GB free |
| Network → Binance | — | — | ~150 ms RTT (Finland → Asia) |

**Coexistence requirement:** the HF module must not destabilize the daily NeuCast service or the unrelated apps on the same host. This is enforced via Docker memory limits, single-thread CatBoost training, and scheduling heavy work in low-traffic UTC hours.

---

## 3 · Architecture diagram

```
┌──────────────────┐      WebSocket          ┌─────────────────────┐
│ Binance Spot WSS │ ───────────────────────▶│   L2 Consumer       │
│  depth20@100ms   │   ~150 ms RTT, ~5 MB/s  │   (asyncio task)    │
│  trade stream    │                         └──────────┬──────────┘
└──────────────────┘                                    │
                                                        │ in-memory ring buffer
                                                        │ (1 s window, ~50 MB)
                                                        ▼
                                            ┌─────────────────────┐
                                            │  OFI Aggregator     │
                                            │  1-second features  │
                                            └──────────┬──────────┘
                                                       │ ~100 B/sec
                                                       ▼
                                            ┌─────────────────────┐
                                            │  Postgres           │
                                            │  ofi_features_1s    │
                                            │  ofi_features_1m    │
                                            └──────────┬──────────┘
                                                       │
                          ┌────────────────────────────┼─────────────────────────────┐
                          ▼                            ▼                             ▼
                ┌──────────────────┐       ┌──────────────────────┐      ┌────────────────────┐
                │ CatBoost trainer │       │  Sim-backtest engine │      │  Live predictor    │
                │ (Celery beat,    │       │  walk-forward,       │      │  (FastAPI route)   │
                │  04:00 UTC)      │       │  maker/taker fees    │      │  /highfreq/latest  │
                └──────────────────┘       └──────────────────────┘      └────────────────────┘
                                                       │                             │
                                                       └──────────────┬──────────────┘
                                                                      ▼
                                                          ┌─────────────────────┐
                                                          │  /highfreq UI       │
                                                          │  Live + backtest    │
                                                          │  charts             │
                                                          └─────────────────────┘
```

---

## 4 · Architecture Decision Records

Each ADR captures one non-obvious decision, the alternatives considered, and the trade-off accepted.

### ADR-001 · In-memory aggregation, no TimescaleDB

**Context.** Storing raw L2 depth-20 snapshots at 100 ms cadence costs ~50 GB / 3 months. The deployed Postgres image is `postgres:15-alpine`, which does not bundle the TimescaleDB extension; switching the image requires a 30-minute downtime migration and ~50 GB of additional disk.

**Decision.** Aggregate the L2 stream in process memory into 1-second OFI features and persist *only* the aggregates. Raw snapshots are kept for 24 hours rolling in an on-disk SQLite buffer for live debugging, then dropped.

**Why this works.**
- 1-second OFI features → 1 row × ~100 bytes × 86 400 s/day = ~8 MB/day = ~250 MB/month. Fits the 60 GB disk headroom indefinitely.
- 1-minute model never queries sub-second history beyond a 60-row rolling window held in memory.
- TimescaleDB compression would deliver the same ~95 % footprint reduction *after* ingest; doing it at ingest is cheaper.

**Trade-off accepted.** We cannot retrospectively re-extract sub-second features. If we later decide we want a 1-second prediction horizon (Phase B), we have to design the feature in advance and replay the live stream forward — we cannot mine the historical raw L2.

**Alternative rejected.** Deploying TimescaleDB. Reason: storage savings are real but small (~30 GB compressed vs ~250 MB aggregated); the migration cost is non-trivial; and the operational surface increases (new extension, new compaction policies).

---

### ADR-002 · Use Binance event time, not local timestamp

**Context.** WebSocket frames take ~150 ms to reach our Finland VPS from Binance's Asia datacenter. If we used `time.time()` at receive, our timestamps would lag market reality by a variable amount (~50–250 ms with jitter), making the dataset incompatible with any future migration to a Tokyo VPS.

**Decision.** Persist the `E` field (event time, UTC ms since epoch) from each WebSocket frame as the canonical timestamp. Local-receive time is also recorded but only as a *diagnostic* column (for monitoring network jitter), never used by ML.

**Why this works.**
- `E` is set by Binance's matching engine and is the same regardless of where we listen from.
- Sequence ordering is preserved across servers — datasets collected on Finland VPS remain valid if we later move ingest to Tokyo.
- Live inference and historical training can share a code path with no timezone drift.

**Trade-off accepted.** We trust Binance's clock. They have published SLAs on event-time accuracy (< 5 ms drift), which is well below our 1-minute horizon.

---

### ADR-003 · 1-minute forecast horizon, not 1-second

**Context.** True HFT shops predict at 1-100 ms horizons and require co-location. We predict from a 150-ms-latency Finland VPS.

**Decision.** Phase A targets `direction(t + 60 s)`, not `direction(t + 1 s)`.

**Why this works.**
- 150 ms latency = 0.25 % of a 60-second window — negligible.
- 150 ms latency = 15 % of a 1-second window — would invalidate sub-second predictions.
- 1-minute features (rolling 1 s OFI over 60 s) have substantially less noise than tick-level features.
- Binance publishes 1-minute klines historically, so we can sanity-check the OFI-driven model against simpler baselines (returns-only).

**Trade-off accepted.** Theoretical edge ceiling is lower than a 1-second predictor, but 1-min is realistic for our infrastructure. The Phase A → Phase B migration path explicitly addresses 1-s prediction with a Tokyo VPS if and when Phase A succeeds.

---

### ADR-004 · Direction loss, not MAPE

**Context.** The daily NeuCast ensemble has been trained on MSE / MAPE losses for months and has converged to MAPE ≈ 1.85 % on BTC-USDT — which is the known physical floor for OHLCV-only forecasting (Foundation models like Chronos / TimesFM hit the same wall, see [Chronos paper](https://arxiv.org/abs/2403.07815)). MAPE is *not* a useful trading metric: a "tomorrow ≈ today" predictor scores 1.85 % MAPE while making zero PnL.

**Decision.** The Phase A model is a CatBoost **classifier** on `sign(return_1m)`, optimizing log-loss with class weights to handle directional imbalance.

**Why this works.**
- Loss aligns with the metric we report (directional accuracy + paper P&L).
- Signed targets sidestep the price-level convergence trap that punished the daily model.
- CatBoost's GPU-free training fits the resource budget.

**Trade-off accepted.** We lose return-magnitude information. For Phase B we may add a regression head to size positions, but Phase A is direction-only.

---

### ADR-005 · Sim-backtest reports both maker and taker P&L

**Context.** Binance fees fundamentally change strategy economics:

| Side | Fee | Effect on strategy with 54 % dir_acc |
|------|-----|--------------------------------------|
| Taker (market order) | 0.1 % per trade | Net negative — fee eats edge |
| Maker (post-only limit) | **−0.001 %** (rebate) | Net positive — exchange pays you |

A backtest reporting only one of these is misleading. A backtest reporting only taker P&L would tell us the strategy is unprofitable when actually it is profitable as a maker. A backtest reporting only maker P&L would over-estimate, because maker fill rates are uncertain (~30–50 % in our regime).

**Decision.** The backtest engine produces *both* P&L curves on every run, plus an explicit fill-rate sensitivity sweep (assume 30 / 50 / 70 / 100 % maker fill).

**Why this works.**
- Honest reporting is the central value of this project.
- Reader of the portfolio writeup sees the full economic picture, not a cherry-pick.
- Forces us to articulate what fill rate we believe in, rather than assuming it.

**Trade-off accepted.** More numbers to communicate. We address this by always reporting maker / taker side by side and summarising in a single "deployable as maker only" badge on the UI.

---

### ADR-006 · Coexist via resource limits, not isolation *(superseded by ADR-009 for the HFT slice)*

**Context.** We could pay ~900 ₽/month for a second VPS to isolate the HF module. We chose not to.

**Decision.** Run HF module on the existing Hostkey Finland VPS, alongside daily-NeuCast / gymbro / vita-balance, with explicit resource limits:

- Docker `mem_limit: 1.5g` for the L2 consumer container
- CatBoost `thread_count = 2`, training scheduled at 04:00 UTC (low-traffic window for the other apps)
- 4 GB swap added as OOM-killer safety net (already provisioned)

**Why this works for a coursework / portfolio project.**
- Zero additional infrastructure cost.
- Demonstrates capacity-planning skill — explicit limits + monitoring is *itself* a portfolio artifact.
- Acceptable risk: no live capital is at stake in Phase A; a brief slowdown is recoverable.

**Trade-off accepted.** If gymbro spikes hard, our pipeline may briefly slow or hit swap. We monitor this and document it. A real production HF system would not run on shared infrastructure — this is an explicit Phase A scope decision, not a permanent design.

---

### ADR-007 · Binary classification with neutral-band drop

**Context.** Bars where `|return_1m| < 1 bp` (~$7.7 on BTC at $77k) are dominated by tick noise and the typical Binance Spot maker-taker spread. A 50/50 classifier on these bars would be statistically uninformative and operationally unprofitable: even if we predicted them correctly, the gross edge would not survive fees.

**Decision.** Drop bars with `|return_1m| < NEUTRAL_BAND_BPS` (default 1 bp, configurable) before training and evaluation. Train a **binary** classifier on the remaining bars (`y = 1` if up, `y = 0` if down).

**Why this works.**
- Aligns the training distribution with the deployable distribution — we only ever act on bars where the move is large enough to matter post-fees.
- Eliminates the degenerate `direction = 0` class that the daily model occasionally collapsed into during low-volatility weeks.
- `dir_acc ≥ 53 %` becomes a *meaningful* threshold rather than "anything above 50 % flat", because the base rate after the neutral-band drop is reported and required to be near 50 %.

**Trade-off accepted.** ~80 % of bars in calm market regimes (early observation) are dropped. We compensate by collecting weeks of live data — the surviving ~20 % over a month is still ≥ 8 000 minutes of training samples, well above the CatBoost convergence threshold.

**Alternative rejected.** Three-class classification with a "neutral" class. Reason: forces the model to learn a noise-vs-signal decision boundary that is more about volatility than direction, hurting accuracy on the bars we actually care about.

---

### ADR-008 · Expanding-window walk-forward, not random k-fold

**Context.** Random k-fold CV is the default in scikit-learn but catastrophically wrong for time-series finance: it lets the model see future bars during training, which inflates reported `dir_acc` by 3–5 percentage points and produces models that disintegrate on live data.

**Decision.** Use expanding-window walk-forward with a configurable initial training window (default 24 h ≈ 1 440 minutes) and a 1 h test fold that advances by 1 h each step. Predictions from each test fold are concatenated chronologically; the bootstrap CI is computed over the per-prediction outcomes (not per-fold means) so its width reflects sample size honestly.

**Why this works.**
- Strict time-ordering — every prediction is genuinely out-of-sample.
- Continuously re-trained models surface non-stationarity (BTC microstructure regime shifts) early.
- Directly produces the `predictions` DataFrame that the sim-backtest engine consumes for paper P&L.

**Trade-off accepted.** Compute cost is `O(n_folds × n_train)` rather than `O(n_train)` for a single train/eval split. With ≤ 7 days of data and CatBoost at `thread_count=2`, this still runs in under 5 minutes — well within the 04:00 UTC training window (ADR-006).

**Alternative rejected.** Purged k-fold with embargo (López de Prado 2018, *Advances in Financial Machine Learning*, §7.4). Reason: theoretically cleaner but adds implementation complexity for a ~0.3 pp accuracy gain on this dataset size. Documented here as the natural Phase B upgrade.

---

### ADR-009 · Tokyo VPS as the HFT data plane (supersedes ADR-006 for the HFT slice)

**Context.** The Phase A ingest ran on the Hostkey Finland VPS alongside the main webapp (per ADR-006). Measured TCP RTT to Binance Spot WebSocket from Finland is ~250 ms; from a Tokyo VPS in the same metro as `stream.binance.com` (AWS `ap-northeast-1`, IP 18.179.181.116) it is ~19 ms median (measured 2026-04-26 from 4VPS.su JP-cx21). The order-of-magnitude reduction is meaningless for our 1-minute horizon directional accuracy (network jitter << bar duration), but the **robustness gain under burst conditions** is real and asymmetric — we lose data exactly during flash-crash windows where the model is most useful. The HFT direction (Phase B+) also makes Tokyo placement *eventually* mandatory (sub-second horizons, future live-trading order routing), so co-locating now avoids a more expensive migration later when the dataset and tuned weights are larger.

**Decision.** Move the entire HFT data plane (Postgres + ingest + slim FastAPI) to a dedicated Tokyo VPS:

- **Tokyo (`147.45.49.40`)** — single source of truth: `postgres:15-alpine` on `127.0.0.1:5433`, `neucast-highfreq.service` (L2 ingest), `neucast-highfreq-web.service` (slim uvicorn on port 8000 serving `/highfreq` + `/api/highfreq/*`), eventually `neucast-highfreq-trainer.timer` and the C-phase `neucast-paper-trader.service`.
- **Finland (`151.245.139.21`)** — public web tier only: nginx terminates TLS, reverse-proxies `/highfreq` and `/api/highfreq/*` to `http://147.45.49.40:8000`, serves everything else from the existing `app.main` uvicorn. The orphan `highfreq_*` tables on Finland Postgres are dropped — no replication, no drift to debug.
- **Firewall.** Tokyo's UFW exposes port 8000 *only* to the Finland IP. Postgres stays bound to `127.0.0.1`. Public neucast.ru URLs continue terminating on Finland's certbot certificate; Tokyo never holds a TLS cert.

**Why this works.**
- **Single source of truth.** No replication lag, no two-DB drift, no "which Postgres is right" debugging.
- **Public URL stays the same.** `https://neucast.ru/highfreq` works exactly as before for users; the topology change is invisible.
- **Right home for each layer.** Webapp on the public-facing Finland box (cert + DNS already there); ingest on the box closest to the data source; UFW makes the Tokyo→Finland link essentially private.
- **Cost.** ~1 080 ₽/month for the Tokyo VPS — recovered many times over by the ~6× simpler operational story vs. a replication setup, and by being the right architectural shape for the HFT direction.

**Trade-off accepted.** Two hosts to monitor instead of one; weights produced by the trainer (which we'll keep running on Finland for the next month while data accumulates) need to be `scp`'d to Tokyo on a cron once we have a `.cbm`. Both are operationally trivial. Internal Tokyo↔Finland traffic is not encrypted at the application layer; the data is public market quotes + read-only API responses, not credentials, so the residual risk is "an attacker on the path could see what BTCUSDT did 30 seconds ago" — acceptable. Adding WireGuard between the two hosts is a half-day task whenever we want it.

**Alternative rejected — Tokyo→Finland data replication.** Easier to set up (one cron + COPY) but doubles disk usage, introduces 30-60 s replication lag, and bakes a "second copy" into the operational story forever. With ADR-009's reverse-proxy approach the disk footprint stays single, the data is always fresh, and the only thing crossing the wire is the same ~1 KB/sec of API responses that nginx would serve anyway.

**Alternative rejected — full webapp on Tokyo.** Would require pulling TensorFlow, PyTorch, transformers, chronos, timesfm, uni2ts (~5 GB on disk, several GB resident) onto a 4 GB VPS. The slim `app/highfreq/web_app.py` ASGI module mounts only the four HFT routes from `app/highfreq/web.py`, keeping the Tokyo footprint at ~80 MB resident.

---

### ADR-010 · WireGuard tunnel for Finland↔Tokyo HTTP traffic

**Context.** ADR-009 introduced a Tokyo→Finland reverse-proxy hop carrying read-only `/highfreq*` API responses. The original implementation restricted Tokyo's public TCP port 8000 to Finland's IP via UFW, but the traffic itself went over the public internet in cleartext. The data is non-sensitive (public market quotes, no auth tokens), so the residual risk is *observability* — anyone on the path between Hostkey Helsinki and 4VPS.su Tokyo can read what BTCUSDT did 30 seconds ago. That's a low-impact leak for our specific workload, but "we encrypt VPS-to-VPS traffic by default" is the correct security posture and the cost of doing it is half a day's setup.

**Decision.** Establish a point-to-point WireGuard tunnel between Tokyo (`10.99.0.1/24`, `ListenPort = 51820`) and Finland (`10.99.0.2/24`, dial-out). Move all HTTP traffic onto the tunnel:

- Tokyo's `neucast-highfreq-web.service` binds `uvicorn` to `10.99.0.1:8000` instead of `0.0.0.0:8000`. The OS itself refuses traffic on any other interface — UFW becomes belt-and-suspenders, not the only line of defence.
- Finland's nginx upstream `neucast_tokyo` points at `10.99.0.1:8000` instead of the public IP.
- Tokyo UFW: replace the `allow 8000/tcp from 151.245.139.21` rule with `allow 8000/tcp from 10.99.0.2`. Port 8000 is no longer reachable from the public internet at all.
- New UFW rule: `allow 51820/udp from 151.245.139.21` (Finland's public IP) — the only WireGuard handshake the Tokyo box will accept.

**Why this works.**
- **Defence in depth.** Three independent barriers between an attacker and our app:
  1. Public interface no longer listens on 8000.
  2. Even if it did, UFW would deny the source IP.
  3. Even if UFW failed, the WireGuard handshake (Curve25519 + ChaCha20-Poly1305) requires the peer's private key — which never leaves the host.
- **Modern crypto, kernel-fast.** WireGuard ships in the Linux kernel (5.6+), zero userland data path. Measured RTT 240 ms over the tunnel — same as the underlying network, so we pay only the public-internet latency, not a userspace VPN penalty.
- **One UDP port, one config file per side.** No certificates to rotate, no PKI to manage. Key rotation is `wg genkey | wg pubkey` plus a config edit on both ends.
- **Right architectural shape.** When we eventually add a third host (e.g. a separate trainer box with a fast GPU), it slots into the same `10.99.0.0/24` network as a new peer — symmetric, no special cases.

**Trade-off accepted.** Adds two services to babysit (`wg-quick@wg0` on each host) and a hard dependency edge (`neucast-highfreq-web.service` requires `wg-quick@wg0.service` — if the tunnel goes down, the slim FastAPI deliberately fails to bind, surfacing the issue rather than masking it). PersistentKeepalive every 25 s costs ~70 bytes/sec — negligible. Operational complexity is real but bounded by a single runbook ([`deploy/wireguard_setup.md`](deploy/wireguard_setup.md)).

**Alternative rejected — TLS upstream (nginx → Tokyo HTTPS).** Would require running certbot on Tokyo and exposing port 443 publicly, doubling the attack surface vs WireGuard's single UDP port. Also encrypts only the HTTP layer, leaving SNI and packet sizes observable.

**Alternative rejected — Tailscale.** Easier setup but third-party service in the trust chain (their coordination server sees connection metadata), and adds an external dependency for our защита-critical path. WireGuard self-hosted has the same protocol with none of those trade-offs.

**Alternative rejected — IPSec.** Strictly more capable, also strictly more setup pain (two daemons, kernel config, racoon-style tooling). WireGuard is "the IPSec we'd want if we redesigned it from scratch in 2018", and that's exactly what it is.

---

### ADR-011 · Paper-trading contract: time-stop, maker-only fees, sim-only by construction

**Context.** ADR-005 declared the project sim-only and ADR-007 documented the binary classification target, but the *trading contract* — what does the runner actually DO with each prediction, what costs does it pretend to pay, what counts as a halt — was scattered across docstrings in `app/highfreq/paper_trader.py`. Comissia (and future-me reading this in three months) needs one document where all those choices live together with their rationale.

This ADR formalises the decisions already implemented in code; nothing here is new behaviour.

**Decision — entry semantics.**
* Open one position per minute-bar close, exactly when `prob_up >= entry_long_threshold` (default `0.55`) or `prob_up <= entry_short_threshold` (default `0.45`). Probabilities in the neutral band trigger no trade.
* Defaults are symmetric around `0.50`. We have no prior on which direction the model is more accurate at; a single-sided bias would silently bake one in.
* One position at a time per symbol — a runaway loop can never accumulate exposure even if a bug fires `on_bar_close` at sub-second cadence (this is also tested as `test_one_position_at_a_time_invariant`).

**Decision — exit semantics (time-stop only).**
* Open at bar `t`, close at bar `t + horizon_minutes` (default `1`). No prob-flip mid-horizon.
* Why no early exit: the trainer fits exactly the `signal at t → return at t+H` map. Closing on a flipped probability mid-horizon throws away the predictive content the model was fit on. Adding stop-losses or trailing stops becomes possible *only* with retraining on a different label; documented as a Tier 3 upgrade.
* `ExitReason` is currently a closed `Literal["time_stop", "halt_close"]`. `stop_loss` is reserved for that future.

**Decision — fee model (Binance Spot, maker, BNB-paid).**
* `maker_fee_bps_per_side = 7.5` (= 0.075 %). Standard tier with BNB discount; without BNB the rate is 10 bp.
* Round-trip cost ≈ 15 bp on a zero-move trade (tested as `test_compute_pnl_round_trip_fee_cost_matches_15_bps`). This is **the bar Tier 3 has to clear** — a model whose mean post-fee return is negative is not deployable, no matter how good its directional accuracy looks.
* Each leg's fee is computed at *that leg's price*, not symmetrically — so `compute_pnl(side="long", entry=100, exit=99, qty=1, fee=7.5bps) = -1.14925`, not the naive `-1.15`. Tiny but documented because it would otherwise look like a floating-point bug.

**Decision — position sizing (fixed-notional, soon vol-adjusted).**
* `qty = max_qty_usd / entry_microprice` per trade — currently `max_qty_usd = 100`. Keeps P&L math interpretable across price regimes.
* Long/short symmetric: Binance Spot doesn't allow shorting without margin, but for *measuring directional accuracy* we treat shorts symbolically (sell high, buy back lower). The UI says "sim-only" prominently — we are not pretending we could short BTC at the spot exchange tomorrow.
* A `vol_adjusted_qty(target_vol_bps, realized_vol_bps)` knob is on the roadmap; sizing then becomes `max_qty_usd × (target / realized)` clipped to `[0.5×, 2×]`. Improves Sharpe; design preserved in ADR-011 history once landed.

**Decision — risk caps (3 hard kill-switches, UTC midnight reset).**
1. `max_consecutive_losses = 5` — halts on regime-shift evidence (the model was fit on 7d, market is now 8d in). Persists across UTC-midnight rollover (a long losing streak shouldn't auto-clear because the clock ticked).
2. `max_daily_loss_usd = 5.0` — bounds worst-case in any 24-hour window. Clears at UTC midnight (this is the whole point of a daily cap).
3. **One position at a time** — doubles as a position cap; see entry semantics above.

When halted, `on_bar_close` returns `None` for any new entry but **still closes** an open position on time-stop — we never strand state in `trader.state.open_position` between restarts.

**Decision — calibration gate (refuse to trade an unvalidated model).**
* `require_calibrated = True` (default). Trades open only when `predictor.is_calibrated()` returns `True`, which itself requires `dir_acc_ci_low > 0.50` from the trainer's bootstrap CI (ADR-008's walk-forward).
* Operationally this means: the moment the trainer ships a model whose lower confidence bound is above random, the runner starts opening trades on the next minute boundary — no code change.
* Override (`require_calibrated=False`) exists for backtest sweeps; never set for the production runner (gated by docs).

**Decision — model versioning (mtime as the version field).**
* Every `PaperTrade` row carries `model_version = str(predictor._weights_mtime or 0.0)`. Lets us slice "P&L since model v3 went live" via SQL without ambiguous timestamp games.
* Trainer writes weights atomically; predictor's mtime hot-reload (tested in `test_hot_reload_picks_up_new_weights`) means the next trade after a `.cbm` swap automatically carries the new version.

**Trade-off accepted — sim ≠ live in three known ways.**
1. **No slippage modelling.** We close at the same `microprice` we'd open at on the next bar — no spread crossing, no impact. Real maker fills can be partial or never fill at all; we assume 100 % fill rate. Documented expansion: `slippage_bps_per_side` knob with a fill-rate model from the existing sim-backtest engine (Phase A).
2. **No latency in the path Binance → predictor.** Every bar is scored on `microprice_close` of the bar that just ended — a real maker order would have to be in the book *before* that close to fill at that price. Adding a one-bar offset is a one-line patch when we go live.
3. **Long shorts assumed available.** As above, Binance Spot doesn't support shorts without margin. The UI flags this and the ADR-005 sim-only contract makes it explicit.

**What would change to go live (the migration runbook).**
1. Replace the `paper_trades` writer with a `live_trades` writer that posts orders via the Binance REST/WS trading endpoints. Keep the schema identical so cohort analysis still works.
2. Add `kill_switch.py` daemon: external file that the runner checks on every tick — `touch /etc/neucast/halt` to drop into pure-close mode immediately.
3. Tighten risk caps: `max_qty_usd = $1` for the first week; ramp by 2× weekly only if cumulative live P&L is positive.
4. Add slippage modelling against the sim-backtest fill-rate curve so we calibrate vs. live drift.
5. Audit log every API request/response to a separate `live_orders_audit` table — non-truncatable, append-only.
6. ADR-005 amendment ("sim-only → first $100/day live with explicit kill switches"); compliance review (РФ нерезидент Binance — отдельная тема).

**Alternative rejected — fractional Kelly sizing.** Theoretically optimal under known edge; in practice we don't know our edge precisely (the bootstrap CI is wide for the first few weeks of data) and Kelly becomes destabilising under estimation error. Fixed-notional is robust by construction; the vol-adjusted variant adds the right amount of dynamism without the Kelly fragility.

**Alternative rejected — dynamic-horizon (close on prob-flip).** Tempting but invalidates the trainer's target. Right move is to retrain on a "best-exit-within-H" label, not to override the time-stop ad-hoc — a Phase D research project.

---

### ADR-012 · Cross-asset features (BTC microprice as input for ETH/BNB models)

**Context.** Phase A models for the 3 spot symbols were trained independently — each model only sees its own symbol's microstructure history. Empirical evidence in 1-minute crypto suggests BTC leads alts by 1-3 minutes during regime shifts (BTC moves first; ETH/BNB follow). A symbol-local model is structurally blind to that lead/lag, throwing away free signal.

**Decision.** For ETH and BNB models we add 5 lagged BTC features at the SAME minute boundary the target lives on:

* `ofi_sum_btc_lag1`, `ofi_sum_btc_lag2` — BTC's order-flow imbalance one and two minutes ago,
* `microprice_return_bps_btc_lag1` — BTC's last-minute return,
* `depth_imb_btc_lag1`, `spread_bps_btc_lag1` — BTC's top-of-book state.

Implementation lives in `app/highfreq/feature_pipeline_cross_asset.py` and is gated by `feature_columns_for(reference_symbol)`. The BTC model itself has no reference symbol (no asymmetric pair to use as predictor) so it stays on the original microstructure pipeline.

**Empirical result (Release L.cross, 2026-04-28).** Per-symbol multi-horizon eval on 96 h of data: ETH 1m dir_acc 0.5450 → 0.5614 (+1.6 pp); BNB 1m dir_acc 0.5571 → 0.5786 (+2.2 pp). Both with `p < 1e-5` on the fold-pooled OOS sample. BTC unchanged (no reference to use).

**Trade-off accepted.** Feature count grows from 18 → 22 → 27 (microstructure 14 + calendar 4 + lagged 4 + BTC-cross 5). Larger feature space = larger overfit risk on small samples, mitigated by:
1. CatBoost regularises via tree depth + iterations bounded by ADR-006's 2-thread CPU budget.
2. Walk-forward CV on the same data as the no-cross baseline — if the cross-asset features were over-fitting noise, the pooled OOS dir_acc would NOT have lifted.
3. The chosen features are economically motivated (BTC leadership is well-documented in crypto microstructure literature), not empirically mined from a wide sweep.

**Alternative rejected — full price-history attention.** Would let the model attend to arbitrary BTC bars in the past. Adds 100s of features for marginal gain; CatBoost can't exploit attention semantics and would just regularise toward the same lagged subset we hand-picked.

**Alternative rejected — train one model per (predictor, target) pair.** ETH model fed BTC-lagged features and ETH model fed ETH-lagged features as TWO different models, then ensemble. Doubles training time, doubles deployment surface, and the ensemble logic adds another layer of failure modes. Single model with both feature families wins on operational simplicity.

---

### ADR-013 · Joint multi-symbol training (pooled BTC+ETH+BNB with symbol-id features)

**Context.** ADR-012 lifted ETH/BNB by adding BTC features to per-symbol models. The natural next step: train ONE model on the pooled data of all three symbols, with symbol-id one-hot features so the model can specialise where useful and share parameters where not. At 1-minute granularity per-symbol n is in the 2000-3000 range (limited by data accumulation), but pooled n is 8000+ — a substantial CI tightening even before any specialisation effect.

**Decision.** Add `app/highfreq/feature_pipeline_joint.py` with:

* `JOINT_FEATURE_COLUMNS` = base microstructure + calendar + 3 one-hot symbol indicators (`is_btc`, `is_eth`, `is_bnb`),
* `make_joint_supervised(df_secs_by_symbol, ...)` pools per-symbol seconds frames and produces a single (X, y, meta) ready for unified training.

Walk-forward CV operates on the chronologically-sorted pooled dataset. Each fold trains on past pooled bars and tests on the next slice — the test set may contain multiple symbols. The `_joint_long_horizon` variant swaps in OHLC + classical TA features for 5m+ horizons (microstructure features lose signal-to-noise at long horizons; mean-reversion + momentum features dominate, empirically lifting joint 15m dir_acc from 0.521 → 0.5485).

**Empirical result (Release N, 2026-04-28).** Joint at 1m: dir_acc 0.5728 with n=8481, `p = 1.4e-34`. Per-symbol point estimate is HIGHER (BTC 0.5794 / ETH 0.5614 / BNB 0.5786) but joint has tighter CI and 30× lower p-value because pooled n is bigger. At 15m horizon, joint+TA gives the only `n>500` significant result we can report (`p = 0.016`).

**Trade-off accepted — single point estimate, multiple symbols.** Reviewers asking "which model is best" get the per-symbol numbers if they care about point estimate, joint if they care about statistical significance. Both are reported. We do NOT silently pick one; the multi-horizon eval tool (`tools/multi_horizon_eval.py`) renders the full grid.

**Alternative rejected — meta-learning across symbols (MAML).** Theoretical fit but training-time complexity is much higher and the gradient interaction between asynchronous data streams is itself a research question. Joint pooling with symbol-id features captures most of the benefit at zero MAML overhead.

**Alternative rejected — soft sharing via multi-task heads.** A single feature trunk with per-symbol output heads. Cleaner than full pooling for asymmetric data sizes but requires per-task loss balancing and doesn't have an obvious gain on the OOS metric we care about (dir_acc on each symbol's distribution).

---

### ADR-014 · Probability calibration (Platt scaling co-loaded with model)

**Context.** CatBoost's raw `predict_proba` output is the gradient-boosting model's confidence; in tree ensembles this is empirically **mis-calibrated** — a 0.7 raw probability on the held-out set typically corresponds to a 0.6 actual win rate. The runner uses `prob_up >= 0.55` as a long-entry threshold (ADR-011); when raw probability is biased, the threshold is operationally a different decision boundary than what the metric report claims.

**Decision.** Fit a Platt scaler (1-D logistic regression on `logit(raw_proba) → y_true`) on the pooled OOS predictions from walk-forward CV. Save it next to the `.cbm` file (`weights/highfreq/<symbol>_1m_calibrator.joblib`). The predictor co-loads it on the same hot-reload tick that loads the `.cbm` and applies it inside `predict()`. Falls back to raw probability if the calibrator is missing (legacy models or fit failure).

**Reliability metrics in the training report.**
* `calibrator_brier` — Brier score (mean squared error between predicted probability and binary outcome). Lower is better.
* `calibrator_ece` — Expected Calibration Error (binned discrepancy between predicted and observed). Pinned to ≤ 10 bins; defence-grade visual is the reliability diagram.

**Trade-off accepted — single-fold edge case.** When walk-forward CV produces a fold where `y_true` is single-class, Platt's logistic regression has no gradient to fit. We ship a `_PassthroughCalibrator` that returns the raw probability unchanged. Tested explicitly so a refactor doesn't accidentally substitute a bad fitted scaler.

**Alternative rejected — isotonic regression.** Strictly more flexible than Platt and often better-calibrated empirically; requires more data and is monotonic but not smooth. With our typical 2000-5000 OOS sample size, Platt's two parameters fit cleanly and don't over-fit; isotonic's monotone-piecewise-constant output also creates flat plateaux that interact awkwardly with the runner's threshold-based entry. A → B switch is one config flag away if data accumulates and reliability drift becomes visible.

---

### ADR-015 · Event-aware halt (halt around FOMC/CPI/forks)

**Context.** Macro releases (FOMC, CPI, NFP) and idiosyncratic crypto events (hard forks, exchange listings, halving anniversaries) cause discontinuous price jumps that the ML model was NOT trained to handle (the training distribution is order-flow-driven steady-state minutes; jump events are a different process). Trading through such windows is operationally negative-EV — the model has zero edge, fees still apply, and the risk of wrong-side blow-out is high.

**Decision.** Maintain a curated event calendar (`docs/highfreq/event_calendar.json`) with each event tagged by category (`macro` / `crypto` / `fork` / `informational`) and per-event halt windows (`halt_before_min`, `halt_after_min`). The runner evaluates `should_halt_for_event(events, symbol, now)` on every bar close; when the halt is active, no new entries are opened (matching ADR-011's risk-cap halt semantics — open positions still close on time-stop).

**Coverage as of Release O.** 23 events through 2026-06: FOMC May/June, CPI/PPI/PCE/NFP/Retail Sales/ADP/ISM PMI prints, ECB/BOE/BOJ decisions, plus 2 crypto-specific (Binance BNB burn, ETH dev call) and 1 informational (Bitcoin halving anniversary). Halt windows scaled by event significance: FOMC = ±15 / +60 min, CPI = ±15 / +45 min, smaller prints = ±5 / +15 min.

**Trade-off accepted — manually curated calendar.** A 1× / week ops review keeps it fresh; reduces operational load vs. parsing a third-party API and inheriting its data-quality bugs. Calendar parsing is malformed-entry tolerant (skips invalid rows with a warning rather than crashing) so a partial JSON edit can never take the runner offline.

**Alternative rejected — auto-fetch from economic calendar API.** Several free APIs exist (Investing.com, Trading Economics) but none commit to a stable schema or uptime SLA. A WS gap during a critical FOMC release silently means the runner trades through it. Static JSON committed to the repo gives the same coverage with a known failure mode (a stale entry → traded an event we shouldn't have, visible in audit log) and zero runtime dependency.

---

### ADR-016 · Sample weighting + embargo (recent > old, López de Prado boundary)

**Context.** Walk-forward CV in financial ML has two well-documented gotchas:

1. **Concept drift.** Market regime shifts mean older bars distract the fitter from current dynamics; uniform sample weights treat the bar from 24 hours ago as equally informative as the one from 5 minutes ago, which is empirically wrong on minute-bar crypto.
2. **Train→test target leakage.** When the target is forward-shifted (target at t = sign(return at t+H)), the last bar of the train fold has its target in the test fold's window. The leakage is at most 1 bar at H=1 but is the kind of subtle error that erodes a defence-grade claim of "no leakage".

**Decision.**

* **Exponential sample weighting** with half-life of 720 bars (≈ 12 hours at 1m). Most-recent bar gets weight 1.0; a bar 720 bars older gets weight 0.5; pinned as `WalkForwardConfig.sample_weight_half_life_bars = 720`. CatBoost honors `sample_weight` natively. Disabled by setting half-life to 0 (uniform, original behaviour).
* **Embargo of 1 bar (López de Prado)** — drop the LAST `embargo_bars` rows from the train fold before fitting. With H=1 the leakage is exactly 1 bar; embargo=1 closes the loop. `WalkForwardConfig.embargo_bars = 1` is the production default. `purge_bars = 0` reserves the more aggressive purging for future multi-horizon work.

**Trade-off accepted — modest train-set shrinkage from embargo.** At H=1 we drop one bar per fold; at H=15 (long-horizon eval) we drop 15. Walk-forward already discards the last fold's worth of bars so the marginal penalty is small. The OOS dir_acc lifts from this combined change (sample weighting + embargo) on BTC are within the same fold geometry's noise band — it's a defence-grade move (academic correctness), not an empirical alpha generator.

**Alternative rejected — group k-fold by hour-of-day.** Hour-of-day-aware folding addresses some of the same leakage class (a model trained on 3pm bars inferring 4pm bars on the same day). For minute-bar trading the autocorrelation horizon is bars not hours; embargo=1 is the right granularity.

**Alternative rejected — heavier exponential decay (half-life 60).** Tested informally during development — the resulting fit is dominated by the last hour and dir_acc on the older parts of the test window collapses. 720 bars (12 h) is the empirically chosen sweet spot: recent enough to track regime, broad enough to learn the structural features.

---

### ADR-017 · Magnitude regression evaluation (offline tool, NOT yet integrated)

**Context.** The production model is a *classifier* — predicts `P(price up at t+1)` for a binary directional bet (ADR-007). Two pieces of information are thrown away:

1. **Confidence (size).** A 60 % bet on a 1-bp move is worth far less than a 60 % bet on a 4-bp move. Kelly sizing wants `E[return]`, not `P(positive)`.
2. **Fee-aware filtering.** At retail fees (15 bp round-trip) sub-2-bp expected moves are unprofitable regardless of directional skill. The classifier can't express "this bar's signal is too weak to bother".

**Decision (Release P, 2026-04-29).** Add `tools/regression_eval.py` — an offline-only evaluation tool that walk-forward CVs a `CatBoostRegressor` on the continuous `return_bps` target, mirroring the classifier's data / fold geometry / neutral-band drop / sample-weighting for apples-to-apples comparison. Reports MAE / RMSE / R² / Pearson IC / Spearman IC plus sign accuracy (= directly comparable to dir_acc), plus threshold curves at θ ∈ {0, 1, 2, 4, 8} bp showing per-fee-tier realized P&L when filtering on `|E[r]| > θ`.

**The tool is offline-only by design.** No `.cbm` is written, no production weights are touched, no predictor / paper-trader / runner is modified. The classifier path keeps running unchanged. Empirics decide: if the regressor's sign accuracy is comparable to the classifier's dir_acc, integration is justified; otherwise the regressor's only value is fee-aware threshold filtering, which we can layer on top of the existing classifier without a rewrite.

**Trade-off accepted — duplicates some fold logic.** The eval tool reimplements walk-forward CV (in fewer lines than the trainer because it doesn't need final-model fit, calibrator, or persistence). The duplication is intentional — the trainer is the production-critical path and shouldn't get a `target_mode` flag added speculatively before we know whether regression is worth integrating.

**Alternative rejected — wire regression as a `target_mode` flag inside `app.highfreq.trainer`.** Faster path to integration but bloats the production trainer with a code path that may never ship. Offline-first is the cheaper experiment.

**Alternative rejected — quantile regression (CatBoost loss `Quantile:alpha=...`).** Theoretically gives a richer view of the predicted distribution; in practice for 1-minute crypto the conditional return distribution is well-approximated by Gaussian within a regime, and quantile loss is more sensitive to outliers than RMSE. Reserved for a future regime-aware extension.

---

### ADR-018 · Bayesian credible intervals for dir_acc (Beta-Binomial posterior)

**Context.** The trainer reports a 95 % bootstrap CI on dir_acc (ADR-008) and a one-sided binomial p-value for "model has skill above chance" (Release J). Both are *frequentist*: they answer "if I rerun this experiment many times, what range covers the true value 95 % of the time" and "could this number have come from random guessing". A reviewer reading "95 % CI is [0.55, 0.59]" typically intuits a *Bayesian* statement ("there's a 95 % posterior probability the true dir_acc is in [0.55, 0.59]"), which the bootstrap CI does not literally claim.

**Decision (Release Q, 2026-04-29).** Compute the Bayesian credible interval via the Beta-Binomial conjugate posterior with a uniform Beta(1, 1) prior:

```
posterior ~ Beta(1 + n_correct, 1 + n_total - n_correct)
CI = (ppf(α/2), ppf(1 - α/2))     # default α = 0.05
point = posterior_mean = (1 + n_correct) / (2 + n_total)   # NB: not k/n
```

Reported alongside the bootstrap CI as `dir_acc_bayesian_ci_low` / `dir_acc_bayesian_ci_high` in `TrainingReport` and `HorizonEvalRow`. At large n with the uniform prior the two intervals coincide to within a few hundredths; at small n the Bayesian interval is wider (correctly capturing the right-tail uncertainty that the bootstrap can underestimate).

**Trade-off accepted — Laplace smoothing of point estimate.** With a uniform prior, perfect score 100/100 yields posterior mean 101/102 ≈ 0.99 (NOT 1.0); zero-correct yields ~0.01 (NOT 0.0). This is mathematically correct and pinned in tests. The slight pull-toward-0.5 is the correct prior-driven shrinkage — a perfect 100/100 isn't strong evidence of literally 100 % future accuracy.

**Alternative rejected — Jeffreys prior Beta(0.5, 0.5).** Slightly less smoothing; intervals differ by ~0.01 from uniform at typical n. Both are exposed via the function's `prior_alpha / prior_beta` arguments — the production default is uniform because it has the most defensible interpretation ("equal prior weight to all dir_acc values in (0,1)").

**Alternative rejected — full posterior visualisation.** Kernel-density plot of the posterior would be the most informative single chart but adds frontend complexity. The CI captures the relevant bounds; reliability diagram (ADR-014) handles the orthogonal question of probability calibration.

---

### ADR-019 · USDM Perpetual Futures venue as parallel data plane (Release S, in progress)

**Context.** Multi-horizon paper-trading on Binance Spot at retail fees (15 bp roundtrip) is mathematically loss-bound for every horizon we've measured: even at BNB 15m's `dir_acc 0.66` with `mean|move| 5 bp`, expected P&L per trade is `0.32 × 5 − 15 = −13.4 bp`. The fee burden is the single dominant cost; no realistic ML improvement can close it on 1-15m horizons. The structural break to break-even is **a venue with lower fees**.

Binance USDM (USDT-margined) Perpetual Futures offers maker fees of **2 bp/side = 4 bp roundtrip** (taker 4/side = 8 bp roundtrip), without volume tier requirements. This collapses the BNB 15m example to `0.32 × 5 − 4 = −2.4 bp/trade` — within striking distance, and on a tighter-conviction signal subset (threshold 0.65/0.35 lifts E[|move|] to ~7-10 bp) it crosses zero.

USDM additionally provides:
* **Native shorts** — Binance Spot doesn't support shorts without margin; the paper trader currently treats shorts symbolically (ADR-011 §3). USDM has true bilateral quoting.
* **Funding rate as a feature** — paid every 8h; positive funding = longs paying shorts (bearish lean signal). Updates every 1 s on the @markPrice@1s stream.
* **Slightly higher liquidity** on the BTCUSDT pair than Spot in 2025-2026 measurements.

**Decision.** Build USDM Futures as a **parallel data plane**, not a venue column on the existing tables. The spot ingest, trainer, and paper trader stay running unchanged; futures is additive, with separate weights, separate paper trades, separate UI surface.

Schema (release S migration 006):

* `highfreq_futures_ofi_1s` — clone of `highfreq_ofi_1s` plus `mark_price`, `funding_rate`, `next_funding_ms` columns from the @markPrice@1s stream.
* Symbol values match spot (`BTCUSDT` not `BTCUSDT.P`) — venue is implicit by table name.
* Same primary-key + index convention as spot for query parity in the trainer.

Code structure:

* `app/highfreq/futures_l2_consumer.py` (skeleton in release S, full impl in S+1) — parallel WebSocket consumer connecting to `wss://fstream.binance.com/stream` with `@depth20@100ms` + `@markPrice@1s` subscriptions per symbol.
* `app/highfreq/futures_aggregator.py` — same minute-aggregation as spot, plus funding-rate carry-forward.
* `app/highfreq/feature_pipeline_futures.py` — extends microstructure with `funding_rate_bp_per_8h`, `mark_minus_microprice_bp` features.
* Trainer: existing CLI gains `--venue futures` flag; reads from the futures table; writes to `weights/highfreq/futures/<symbol>_<horizon>m.cbm`.
* Paper trader: existing `PaperTraderConfig` gains `venue: Literal["spot", "futures"] = "spot"`. When futures: `maker_fee_bps = 2.0`, `taker_fee_bps = 4.0`; funding-rate cost is added to the realized P&L for any position that straddles a funding settlement.
* Systemd: new templated units `neucast-futures-highfreq.service`, `neucast-futures-paper-trader@.service`.

**Trade-off accepted — code surface duplication.** Spot and futures pipelines share ~80 % of feature logic, but the consumer / aggregator / fee model differs. We accept duplicating the consumer + aggregator + the trainer's input loader rather than refactoring spot to a venue-aware abstraction, because:

1. The spot ingest has been running for weeks and feeding production paper traders; a venue-abstraction refactor would be a high-risk change with no immediate upside.
2. If futures research dead-ends (insufficient liquidity, funding-rate volatility eats edge, etc.) we just drop the futures path — the spot side is unaffected.
3. The duplicated code IS shared at the feature-pipeline layer (microstructure features are venue-agnostic) — only the data-source plumbing diverges.

**Alternative rejected — `venue` column on `highfreq_ofi_1s`.** Cleaner relationally, but adds nullable columns (`mark_price`, `funding_rate`) that don't apply to spot rows, complicates the trainer's "give me last 96h of BTCUSDT" query (now needs `WHERE venue='spot'`), and bakes a "what does NULL funding_rate mean for a spot row?" question into every downstream consumer.

**Alternative rejected — COIN-M Futures (BTC-margined).** PNL would be in BTC, not USDT, requiring a USDT-equivalent conversion for cross-venue comparison. Mark-price calculation differs from USDM. Lower priority — we add it post-defense if research suggests it's needed.

**Alternative rejected — switching the production paper trader to futures.** That's the eventual goal, but tonight's work is *additive only* — the spot side keeps running as the academic-defense baseline. Once the futures side has a few weeks of data and a confirmed-better-than-spot post-fee P&L curve, we re-evaluate which side to declare "primary".

---

## 5 · Module layout

```
app/
└── highfreq/
    ├── __init__.py
    ├── l2_consumer.py        # WebSocket → in-memory ring buffer
    ├── ofi_features.py       # OFI / microprice / depth-imbalance computation
    ├── aggregator.py         # 1-s feature aggregation, Postgres writer
    ├── runner.py             # Standalone entry: python -m app.highfreq.runner
    ├── trainer.py            # CatBoost walk-forward CV + bootstrap CI + CLI
    ├── predictor.py          # Live inference helper (Phase B+)
    ├── backtest.py           # Sim-backtest engine, maker / taker fee model
    ├── web.py                # FastAPI router: /highfreq + /api/highfreq/*
    └── migrations/           # SQL DDL — versioned, applied via psql

tests/
├── test_highfreq_trainer.py  # pytest — data-layer pure-function coverage
├── test_highfreq_backtest.py # pytest — fee model, ledger, summarise, sweep
└── test_highfreq_web.py      # pytest — UI status payload, JSON sanitisation

docs/
└── highfreq/
    ├── README.md             # portfolio landing — what / why / how (start here)
    ├── architecture.md       # this file — design decisions + ADRs
    ├── demo.md               # end-to-end recipe with sample output
    └── deploy/               # systemd unit + ops runbook (versioned, sanitised)

templates/
└── highfreq.html             # /highfreq UI page — live microprice + countdown
```

---

## 6 · Data schema

Two tables, both in the existing `neucast` database (port 5433):

```sql
-- 1-second OFI features (~250 MB/month, retained indefinitely)
CREATE TABLE highfreq_ofi_1s (
    ts            TIMESTAMPTZ NOT NULL,    -- Binance event time, see ADR-002
    symbol        TEXT        NOT NULL,    -- e.g. 'BTCUSDT'
    ofi           DOUBLE PRECISION,        -- order-flow imbalance
    microprice    DOUBLE PRECISION,        -- depth-weighted mid
    depth_imb     DOUBLE PRECISION,        -- top-N bid vs ask depth ratio
    spread_bps    DOUBLE PRECISION,        -- (ask - bid) / mid * 10000
    trade_imb     DOUBLE PRECISION,        -- aggressive buy vs sell volume
    vpin          DOUBLE PRECISION,        -- volume-bucket informed-trade probability
    n_updates     INTEGER,                 -- number of L2 updates in this 1-s window
    local_recv_ms INTEGER,                 -- local jitter diagnostic, NOT for ML
    PRIMARY KEY (ts, symbol)
);
CREATE INDEX ON highfreq_ofi_1s (symbol, ts DESC);

-- 1-minute aggregated features for model training
CREATE TABLE highfreq_features_1m (
    ts            TIMESTAMPTZ NOT NULL,    -- minute boundary in event time
    symbol        TEXT        NOT NULL,
    -- aggregates of 1-s features over the minute:
    ofi_mean      DOUBLE PRECISION,
    ofi_sum       DOUBLE PRECISION,
    ofi_std       DOUBLE PRECISION,
    microprice_open   DOUBLE PRECISION,
    microprice_close  DOUBLE PRECISION,
    microprice_mean   DOUBLE PRECISION,
    depth_imb_mean    DOUBLE PRECISION,
    spread_bps_mean   DOUBLE PRECISION,
    trade_imb_sum     DOUBLE PRECISION,
    vpin_mean         DOUBLE PRECISION,
    -- target (filled in after t+1m completes):
    return_1m         DOUBLE PRECISION,
    direction         SMALLINT,             -- sign(return_1m): -1, 0, +1
    PRIMARY KEY (ts, symbol)
);
CREATE INDEX ON highfreq_features_1m (symbol, ts DESC);
```

Both tables coexist with the existing daily-prediction tables — no conflicts.

---

### ADR-020 · ``weights/highfreq/`` shared trust boundary (code-review L-5)

**Status:** explicit boundary — documented, not a bug
**Date:** 2026-05-04

**Context.** Several services touch ``/opt/neucast/weights/highfreq/``:
the trainer (writes ``.cbm`` + ``metrics.json`` + ``calibrator.pkl``
as user ``stailfx``); ``tools/drift_check.py`` (writes per-symbol
``<sym>_drift.json`` as ``stailfx``); ``tools/drift_driven_retrain.py``
(reads the drift JSON as ``root`` to decide whether to fire
``systemctl start neucast-highfreq-trainer@<sym>.service``).

The lateral-movement worry: a compromise of the ``stailfx`` account
would let an attacker write a *malicious* drift JSON, which the
root-running drift-retrain timer then reads and acts on.

**Decision.** Accept this as a documented trust boundary, not a bug to
fix. Three reasons it doesn't escalate:

1. **No code execution leakage.** ``drift_driven_retrain`` reads the
   JSON for two integers (severity bucket + max KS) plus a string
   feature name. It NEVER ``eval``/``exec``/``pickle.load``-ses
   the contents. The only side-effect is "fire systemd unit X with
   pre-validated argv".

2. **Symbol whitelist on the systemctl side.** The CLI's
   ``--symbol`` flag is regex-validated (``^[A-Z]{2,12}USDT$``,
   code-review H-3sec). The unit name template is hard-coded as
   ``neucast-highfreq-trainer@<sym>.service`` — a malicious symbol
   wouldn't escape into a different unit name even if it slipped past
   the regex.

3. **Cooldown rail.** The retrain policy refuses any retrigger inside
   6 hours of the previous training run. Even a malicious "all-day-
   high severity" payload triggers at most 4 retrains in 24 hours —
   each one running the trainer with normal ``stailfx`` privs, no
   privilege escalation.

**Mitigations already in place via other ADRs / code-review fixes.**

* systemd hardening (code-review H-1sec): ``NoNewPrivileges``,
  ``PrivateTmp``, ``ProtectSystem=strict``, restricted address
  families. A compromised trainer can't reach the rest of the box.
* Drift JSON write is atomic (code-review Perf-low,
  ``tempfile + os.replace``) — prevents torn-file reads but doesn't
  itself solve the trust boundary.

**Operator action required.** None. If ``stailfx`` is compromised,
the worst outcome is "the trainer retrains a few extra times on the
attacker's schedule, possibly with a poisoned reference window".
That's a service-degradation risk, not RCE escalation. Operator
restoring ``stailfx`` (rotate SSH key, kill the user's processes,
re-deploy weights from archive snapshot) closes the loop.

If the threat model changes (e.g. multi-tenant box, untrusted
operators), an isolated ``neucast`` user with ``700`` perms on
``weights/highfreq/`` would tighten this — but for the current
single-operator deployment, the boundary is documented and
accepted.

---

## 7 · Roadmap

| Phase | Effort | Outcome |
|-------|--------|---------|
| **A.0 · Setup** ✅ | 1 day | swap added, dirs, this doc |
| **A.1 · Architecture polish** ✅ | 1 day | this doc finalised, ADR cross-links |
| **A.2 · L2 consumer** ✅ | 2 days | WebSocket alive, ticks landing in Postgres |
| **A.3 · OFI features** ✅ | 1 day | feature columns populated correctly |
| **A.4 · CatBoost trainer** ✅ | 2 days | walk-forward CV, bootstrap CI, JSON report |
| **A.5 · Sim-backtest** ✅ | 2 days | maker / taker P&L curves, fill-rate sweep |
| **A.6 · UI scaffold** ✅ | 1 day | `/highfreq` page, live microprice, countdown |
| **A.7 · Polish + README** ✅ | 1 day | portfolio README, demo recipe with sample output, screenshot guide |

**Total: ~11 working days of implementation. Calendar: ~3 weeks at 5–10 h/week of user-side validation.**

**Gate to Phase B.** If after 2 weeks of live data collection the sim-backtest shows directional accuracy ≥ 53 % on out-of-sample data with a positive maker P&L curve, we proceed to Phase B (1-second predictions, paper trading on live WebSocket). If not, we close the project as "negative result, documented" — which is *also* a valid portfolio outcome.

---

## 8 · Out of scope for Phase A

To prevent scope creep, these are explicitly *not* part of Phase A:

- Real-money trading
- Sub-1-minute prediction horizons
- Multi-asset basket / cross-sectional ranking
- LLM-based news / sentiment fusion
- ~~Tokyo VPS migration~~ → **done in ADR-009** (2026-04-26)
- Reinforcement-learning-based execution

Each of the above has an obvious slot in a Phase B or C plan and will be addressed there.

---

*This document evolves with the project. Each non-obvious change must add an ADR; small changes can amend §3–§6 in place.*
