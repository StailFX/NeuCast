# Future improvements — design notes

Three substantial model upgrades discussed during release L planning,
deferred from immediate implementation but designed end-to-end so they
can land in focused follow-up sessions.

For context: release L (2026-04-29) added calendar features and OHLC/TA
features for long horizons. Both empirically validated:

* BTC 1m + calendar: dir_acc 0.561 → 0.579 (p=10⁻⁷ → 10⁻¹²)
* BTC 15m + OHLC: dir_acc 0.427 (anti) → 0.580 (p=0.97 → 0.030)

The three items below are **larger** changes — each warrants its own
PR with tests + production deploy + observation period.

---

## δ — Probability calibration (Platt / isotonic)

### Problem

`CatBoostClassifier.predict_proba` returns raw scores that approximate
posterior probabilities under the training data's class balance. When
the trader's entry threshold is `prob_up >= 0.60`, "0.60" should mean
"60% chance the next minute closes higher". Currently it means
"raw score in the upper 40% of the model's output distribution".

If the model is **overconfident** (a well-known CatBoost behaviour on
small data), 0.60 raw might actually correspond to 0.55 calibrated —
the trader is opening positions on weaker signals than threshold says.
If **underconfident**, opposite — losing trades that would have been
better than threshold.

### Solution: post-hoc calibration

After walk-forward CV ends, fit a **Platt scaler** (logistic regression
on raw scores → realised labels) on a held-out slice of the training
data. Save the fitted calibrator alongside the .cbm. At serve time,
the predictor applies it before threshold:

```python
raw_p = catboost.predict_proba(x)[0, 1]
cal_p = platt_calibrator.predict_proba([[raw_p]])[0, 1]
return cal_p   # what the trader threshold-checks
```

### Implementation sketch

1. **Trainer** (`app/highfreq/trainer.py`):
   * After `walk_forward_evaluate`, take last 10% of train rows as
     `cal_X, cal_y`.
   * Fit `LogisticRegression(C=1.0)` on `(catboost.predict_proba(cal_X)[:, 1], cal_y)`.
   * Save via `joblib.dump` to `<weights>_calibrator.pkl`.
2. **Predictor** (`app/highfreq/predictor.py`):
   * On model reload, also load `_calibrator.pkl` if present.
   * If present, apply to raw probability before returning.
   * Backward-compat: missing calibrator → pass-through (current behaviour).
3. **Reliability diagram** (defence artefact):
   * New tool `tools/calibration_diagram.py` — bins predicted proba into
     10 buckets, plots predicted vs observed within each. Perfect
     calibration = diagonal line. Defence-grade chart.
4. **Tests**: pin that calibrator file roundtrips, that pass-through
   mode preserves raw probability when no calibrator file present, that
   the calibrated mode bounds output to [0, 1].

### Effort estimate

3-4 hours. Single new file (`calibration.py`) + ~30 lines in trainer +
~20 in predictor + tool + tests.

### Defence narrative add

> "Raw model output is calibrated post-hoc via Platt scaling on a
> held-out slice. The reliability diagram below shows predicted-vs-
> observed alignment — the trader's 0.60 threshold corresponds to
> exactly 60% empirical hit rate, not an arbitrary score percentile."

---

## ε — Magnitude prediction (regression target)

### Problem

Current target is binary `sign(return_1m)`. The model says "up" or
"down" with probability, but **size of the expected move is invisible**.
This loses information twice over:

1. **Confidence weighting**: a trade with predicted +5bp move is much
   more attractive than one with +0.5bp, but both look like "up" to
   the binary classifier.
2. **Sizing**: position size is currently fixed at `max_qty_usd / price`.
   With magnitude prediction, Kelly-criterion sizing becomes possible:
   bigger positions on stronger signals, smaller on borderline.

### Solution: regression target + decision logic on magnitude

Same OFI / TA features, but predict `return_bps` (continuous) instead
of `sign`. CatBoost has native regression mode (`loss_function="RMSE"`).

Decision logic in trader changes:

```python
expected_return_bps = catboost.predict(x)[0]
fee_breakeven_bps = 2 * maker_fee_bps_per_side  # 15 bp at retail
# Open only if expected return covers fees + a buffer
if abs(expected_return_bps) < fee_breakeven_bps + buffer:
    return None  # neutral / not worth fees
side = "long" if expected_return_bps > 0 else "short"
qty_native = kelly_qty(expected_return_bps, max_qty_usd, vol)
```

### Implementation sketch

1. **Trainer** new arg `--target-mode classification|regression`:
   * Classification: existing path (binary y, Logloss).
   * Regression: y is `return_bps` (float), loss is RMSE, output is
     continuous prediction.
   * Save target_mode in metrics.json so predictor knows what to expect.
2. **Predictor**: branch on `metrics.target_mode`:
   * Classification: existing `predict_proba` path.
   * Regression: `predict()` → return as `expected_return_bps`.
3. **Trader**: if `expected_return_bps` available, use it for
   threshold + sizing; else fall back to existing binary threshold logic.
4. **Walk-forward eval**: same fold structure but evaluate
   `mean_squared_error` or `directional_accuracy_from_regression`
   (= `sign(predicted) == sign(realised)` rate).
5. **Backwards compat**: keep classification mode as default; flip via
   env / CLI.

### Effort estimate

1-2 days. Trainer / predictor / trader changes are touchy because they
ripple through the state machine. Need full test coverage of both
target modes.

### Defence narrative add

> "Binary classification loses information about move magnitude. The
> regression variant predicts return in bps directly, enabling Kelly-
> criterion sizing and fee-aware entry filtering. Compared on the same
> walk-forward folds: regression gives N% MAE on returns, with a
> derived directional accuracy of M%."

### Honest caveat

Regression on noisy 1-min crypto returns is HARDER than classification —
you have to predict both sign AND magnitude. Often regression dir_acc
is LOWER than classification dir_acc. If that's the empirical result,
that's still a defendable answer: "regression is harder; the
information-theoretic upper bound trades cleanly between the two and
classification wins on this signal-to-noise ratio".

---

## ζ — LSTM / GRU sequence model

### Problem

CatBoost treats each bar as an independent observation. Microstructure
intuitively has temporal structure: "OFI was positive for the last 5
minutes" is more informative than "OFI is positive right now". The
model can't see that pattern unless we hand-engineer rolling features
(which we do, but limited).

### Solution: sequence input

For each bar at time t, the model sees a window of last N bars (say
N=20). Architecture:

```
input shape: (batch, 20, 24 features)
  → LSTM(64) or GRU(64)  → captures temporal dynamics
  → Dense(32, relu)
  → Dense(1, sigmoid)    → P(up)
```

PyTorch / Keras / a TF subset — we already have TF/Keras in the daily
side of the codebase (TCN, custom layers). Bring those in.

### Implementation sketch

1. **New trainer** `app/highfreq/trainer_lstm.py`:
   * Builds rolling-window dataset from existing `make_supervised`
     output: each row → window of last N feature rows.
   * Standard Keras LSTM or GRU.
   * Save as `.h5` or SavedModel format alongside .cbm.
2. **New predictor branch**: when loading `_lstm.h5`, route prediction
   through it instead of CatBoost. Hot-reload contract preserves.
3. **Walk-forward CV**: same fold structure; LSTM trains per-fold.
   Slow — ~5 minutes per fold instead of <1s. Cap fold count for
   feasibility; full CV daily instead of hourly.
4. **Memory budget**: Tokyo VPS has 4 GB RAM. LSTM training needs
   ~500 MB peak; daily batched training is fine.

### Effort estimate

2-3 days. Major new code, lots of testing surface (sequence input,
TF/Keras toolchain, model serialisation, predictor routing).

### Risks

* TF/Keras dependency adds ~500 MB to the slim HFT venv. May force
  splitting LSTM trainer to a separate venv on Tokyo.
* Per-fold training time at 5 min × 30 folds = 2.5 hours per symbol.
  Daily trainer cron becomes long-running; need to refactor as a
  systemd timer with longer slot or move to a separate machine.
* LSTM may NOT outperform CatBoost on tabular time-series if the
  signal is genuinely myopic (often the case in microstructure).
  Honest empirical answer: try it, report result, defend either way.

### Defence narrative add

> "Tested whether sequence-aware models capture temporal patterns
> CatBoost misses. LSTM walk-forward CV: dir_acc X% vs CatBoost Y%.
> [If LSTM wins:] The lift comes from multi-bar momentum encoding.
> [If CatBoost wins:] The signal is myopic — last-bar features
> already saturate available information. CatBoost's parallel training
> + interpretability win on a tie."

---

## Recommended order

For a follow-up session focused on these three:

1. **δ — Calibration** first (3-4 hours). Lowest risk, immediate
   trader-quality benefit, defence-grade reliability diagram artefact.
2. **ε — Magnitude prediction** (1-2 days). Major model surface change
   but unlocks Kelly sizing and fee-aware filtering — the conceptual
   bridge from "demo" to "real-money-deployable".
3. **ζ — LSTM** (2-3 days). Biggest unknown; defer until ε is shipped
   so we have a fixed regression baseline to compare LSTM against.

Total: ~5-7 days of focused work for all three. Substantial uplift
in defence-grade story and trader quality.
