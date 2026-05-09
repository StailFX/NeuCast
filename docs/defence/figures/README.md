# Defence figures

Slide-ready SVGs rendered by ``tools/render_defence_figures.py``.
SVG embeds cleanly in Beamer (``\includegraphics``), Keynote (drag and
drop), Pitch / Google Slides (paste the file), and any modern browser.

| file | section | summary |
|------|---------|---------|
| [fig-01-dir-acc-comparison.svg](fig-01-dir-acc-comparison.svg) | §02.1 + §02.2 | Solo per-symbol vs joint pooled. Walk-forward CV with 95% Wilson CI bars + frozen-holdout overlay (hatched). Coinflip reference at 0.50. n_folds annotation in each bar. |
| [fig-02-conditional-accuracy.svg](fig-02-conditional-accuracy.svg) | §04 | dir_acc as a function of confidence threshold (\|p−0.5\|), per symbol. Live data from ``/api/highfreq/conditional_accuracy``. Demonstrates calibrated probabilities — high-confidence buckets hit ~0.57, low-confidence ~0.54. |
| [fig-03-drift-evidence.svg](fig-03-drift-evidence.svg) | §03 | KS statistic per microstructure feature, BTC 2026-05-08, recent 6h vs 7d reference. spread_bps_mean = 0.49, far above the 0.15 alarm threshold. The exact moment when the dashboard told us BTC's regime had shifted. |
| [fig-04-fee-tier-pnl.svg](fig-04-fee-tier-pnl.svg) | §02.4 | E[P&L per trade] in bps, by fee tier × model. Shows where the directional edge becomes economically tradable — retail tier eats it, VIP9/MM-rebate flips positive. |
| [fig-05-cv-power.svg](fig-05-cv-power.svg) | §02.2 | Statistical-power story: joint training → 10× more CV folds → 3× tighter CI than any solo. The numerical case for cross-asset pooling. |

## Re-rendering

If new training data lands or live conditional accuracy shifts:

```bash
# 1. Pull fresh metric JSONs from Tokyo.
scp root@147.45.49.40:/opt/neucast/weights/highfreq/{btc,eth,bnb}usdt_1m_metrics.json /tmp/
scp root@147.45.49.40:/opt/neucast/weights/highfreq/{btc,eth,bnb}usdt_1m_holdout.json /tmp/
scp root@147.45.49.40:/opt/neucast/weights/highfreq/joint_1m_metrics.json /tmp/
scp root@147.45.49.40:/opt/neucast/weights/highfreq/btcusdt_drift.json /tmp/
scp root@147.45.49.40:/opt/neucast/weights/highfreq/multi_horizon_compare_features.json /tmp/

# 2. Pull live conditional accuracy.
curl -s 'https://neucast.ru/api/highfreq/conditional_accuracy' > /tmp/conditional_accuracy.json

# 3. Render.
python3 tools/render_defence_figures.py
```

## Known limitations

* Fee-tier figure (fig-04) sources from `multi_horizon_compare_features.json`
  which was last refreshed 2026-04-29. Re-running
  `tools/multi_horizon_eval.py` on Tokyo would produce fresh numbers
  but it's a 5-15 minute job — fine if defence wants the very latest;
  the 28.04 baseline is already valid for the thesis story.
* All figures are static — no interactive tooltips. Dropping the
  thesis-grade SVGs into a slide deck is the expected use case;
  if you want hover-state interactivity, the live dashboard at
  https://neucast.ru/v2/forecast/ already provides that.
