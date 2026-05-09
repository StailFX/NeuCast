# NeuCast HF — defence narrative

> Reading order. Each section is a self-contained thesis-grade artifact;
> together they form the defence story arc.

| # | Title | What it answers |
|---|-------|----------------|
| [01](01-architecture.md) | System architecture | "What did you actually build?" |
| [02](02-model-results.md) | Model results — solo + joint + multi-horizon | "What does it predict, and how well?" |
| [03](03-drift-case-study.md) | Drift case study — 5m long_horizon, 28.04 → 09.05 | "What happens when the market changes?" |
| [04](04-conditional-accuracy.md) | Conditional accuracy — confidence calibration | "Does the model know what it doesn't know?" |
| [05](05-engineering-depth.md) | Engineering depth — observability, tests, alerts | "Is this production-grade?" |
| [06](06-slide-deck-outline.md) | Slide deck outline | "How do I turn this into a 22-min defence?" |
| [figures/](figures/) | 5 SVG plots | Slide-grade visuals — see [figures/README.md](figures/README.md) |

## The arc

1. **Section 01** — system topology (Tokyo HFT slice + Finland public-facing,
   WireGuard tunnel, Postgres data plane, S3 cold archive, dual-VPS deploy).
   Establishes that this is real infrastructure, not a Jupyter notebook.

2. **Section 02** — empirical model results. Three layers of evidence:
   - **Solo per-symbol** at 1m: dir_acc 0.529 / 0.534 / 0.555, all p < 0.01.
   - **Joint multi-symbol** at 1m: dir_acc 0.541, p = 5e-33, n_folds = 353
     (3-10× tighter CI than any solo).
   - **Long-horizon (5m / 15m / 60m)**: edge collapses on broader windows;
     only 1m carries reliable signal in the current regime.

3. **Section 03** — the drift case study is **the strongest single artifact**.
   Same model architecture, same `--feature-set long_horizon`, same 96 h
   window, run on 28.04 vs 09.05. Result flipped from 0.539 (p=0.07,
   borderline) to 0.492 (p=0.69, coinflip). This isn't a bug; this is the
   market changing under the model. The drift detector caught it (KS=0.68
   on `spread_bps_mean` in the dashboard) **before** edge loss showed up
   in CV metrics — a published-grade demonstration of regime-shift
   monitoring.

4. **Section 04** — conditional accuracy. The model's high-confidence
   buckets (`|p − 0.5| ≥ 0.15`) hit 0.57 dir_acc; the low-confidence
   ones (`|p − 0.5| ≥ 0.05`) hit 0.54. Classic calibrated-confidence
   curve: model learned to express uncertainty.

5. **Section 05** — engineering depth. 27 paper-trader unit tests,
   55 frontend Vitest tests, drift heartbeats, Grafana alerts, dual-VPS
   deploy, archived-S3 backups, isotonic / Platt calibration with
   Niculescu-Mizil & Caruana 2005 crossover at n=1000, conformal
   prediction intervals (Vovk-Gammerman-Shafer 2005). Not a CV bullet
   list — *every* claim has a corresponding artifact under
   `docs/highfreq/` or `app/highfreq/`.

## Reviewer Q&A — anticipated questions + ammunition

| Question | Where the answer lives |
|----------|------------------------|
| "How do you know the edge is real?" | §02, Wilson CI + binomial p-value + Bayesian beta-binomial CI on pooled walk-forward predictions. Frozen-holdout eval as second line of defence. |
| "What happens in different market regimes?" | §03 — empirical proof of degradation under regime shift. |
| "How did you avoid look-ahead?" | Walk-forward CV with embargo=1; calibrator fit on OOS predictions only; frozen-holdout never seen during training (release T.17). |
| "How do fees change the picture?" | §02 fee-tier table — retail tier eats edge (−14 bps); VIP9 / mm_rebate goes positive on 1m. |
| "What about overfitting?" | n_folds=353 on joint = 21k OOS predictions. Bootstrap CI agrees with Bayesian CI to 4 decimal places (sample-size proof, not lucky-fold). |
| "Is the system robust to failure?" | §05 — alert framework, heartbeats, archive backups, two VPS, anti-skill detector, drift detector, calibration gate. |
| "What did you try that didn't work?" | §03 + §02 sub-section — 5m long_horizon for BTC, cross_asset features at 1m, 60m solo training. All documented as negative results. |
