# NeuCast HF — Production Training Scoreboard

Auto-regenerated from the ``training_runs`` Postgres table on Tokyo. One row per spot-production trainer run (excludes /tmp experiments + futures-side runs). Single-glance view of metric evolution across the T.* releases.

Re-generate via ``python -m tools.scoreboard`` (after each release).

## Latest production metrics (one row per symbol)

| symbol | feature_set | dir_acc | CI [lo, hi] | p | Brier | ECE | conformal_q | n_oos |
|--------|-------------|---------|-------------|---|-------|-----|-------------|-------|
| BTCUSDT | microstructure | 0.5617 | [0.5424, 0.5792] | 1.2e-10 | 0.2534 | 0.0698 | 0.691 | 6557 |
| ETHUSDT | cross_asset | 0.5548 | [0.5378, 0.5721] | 1.6e-09 | 0.2554 | 0.0669 | 0.683 | 7117 |
| BNBUSDT | cross_asset | 0.5536 | [0.5333, 0.5743] | 2.4e-07 | 0.2528 | 0.0750 | 0.677 | 6150 |

## Frozen holdout (untouched by trainer — gold-standard OOS)

| symbol | dir_acc | CI [lo, hi] | p | n_holdout | cutoff |
|--------|---------|-------------|---|-----------|--------|
| BTCUSDT | 0.5318 | [0.5133, 0.5508] | 0.00081 | 2482 | 2026-05-01T04:01:34.783172+00:00 |
| ETHUSDT | 0.5604 | [0.5421, 0.5780] | 1.4e-10 | 2730 | 2026-05-01T04:03:51.721548+00:00 |
| BNBUSDT | 0.5768 | [0.5573, 0.5966] | 1.2e-14 | 2469 | 2026-05-01T04:05:44.677786+00:00 |

## BTCUSDT

| run_started | feature_set | hd | n_folds | n_oos | dir_acc | CI low | CI high | p-value | Brier | ECE | conformal_q | config delta |
|-------------|-------------|----|---------|-------|---------|--------|---------|---------|-------|-----|-------------|--------------|
| 2026-04-27 08:18 | ? | 0 | 2 | 1591 | 0.5083 | 0.4250 | 0.6083 | 0.46 | — | — | — | (initial) |
| 2026-04-28 04:04 | ? | 0 | 17 | 2464 | 0.5529 | 0.5245 | 0.5824 | 0.0004 | — | — | — |  |
| 2026-04-29 02:34 | ? | 0 | 25 | 2948 | 0.5527 | 0.5280 | 0.5793 | 2.5e-05 | 0.2587 | 0.0966 | — |  |
| 2026-04-29 04:02 | ? | 0 | 25 | 2971 | 0.5600 | 0.5373 | 0.5860 | 1.9e-06 | 0.2576 | 0.1016 | — |  |
| 2026-04-29 07:56 | ? | 0 | 24 | 356 | 0.5312 | 0.4372 | 0.6354 | 0.31 | 0.3098 | 0.2379 | — | calibrator change |
| 2026-04-29 08:03 | ? | 0 | 26 | 3005 | 0.5609 | 0.5346 | 0.5859 | 8.3e-07 | 0.2580 | 0.0904 | — | calibrator change |
| 2026-04-29 21:09 | long_horizon | 0 | 51 | 409 | 0.5049 | 0.4314 | 0.5686 | 0.47 | 0.3086 | 0.2317 | — | feature_set: ?→long_horizon; bar_minutes: 1→15; calibrator change |
| 2026-04-30 04:03 | microstructure | 0 | 29 | 3202 | 0.5282 | 0.5063 | 0.5517 | 0.01 | 0.2687 | 0.1284 | — | feature_set: long_horizon→microstructure; bar_minutes: 15→1; calibrator change |
| 2026-04-30 08:38 | microstructure | 0 | 29 | 3197 | 0.5569 | 0.5351 | 0.5805 | 1.1e-06 | 0.2585 | 0.0763 | — | calibrator change |
| 2026-04-30 09:45 | microstructure | 0 | 44 | 4100 | 0.5659 | 0.5458 | 0.5864 | 6.8e-12 | 0.2513 | 0.0548 | — | calibrator change |
| 2026-05-01 04:04 | microstructure | 0 | 44 | 4099 | 0.5602 | 0.5413 | 0.5803 | 3.3e-10 | 0.2559 | 0.0738 | — |  |
| 2026-05-02 04:04 | microstructure | 0 | 42 | 4009 | 0.5579 | 0.5393 | 0.5762 | 3.3e-09 | 0.2548 | 0.0744 | — |  |
| 2026-05-03 04:00 | microstructure | 0 | 36 | 3645 | 0.5787 | 0.5579 | 0.6000 | 1.3e-13 | 0.2473 | 0.0724 | — | calibrator change |
| 2026-05-03 10:07 | microstructure | 3 | 44 | 6641 | 0.5519 | 0.5330 | 0.5705 | 5.3e-08 | 0.2520 | 0.0665 | — | holdout_days: 0→3 |
| 2026-05-03 16:27 | microstructure | 3 | 45 | 6594 | 0.5596 | 0.5411 | 0.5774 | 3.1e-10 | 0.2514 | 0.0586 | 0.674 | conformal added |
| 2026-05-04 04:01 | microstructure | 3 | 44 | 6557 | 0.5617 | 0.5424 | 0.5792 | 1.2e-10 | 0.2534 | 0.0698 | 0.691 |  |

## ETHUSDT

| run_started | feature_set | hd | n_folds | n_oos | dir_acc | CI low | CI high | p-value | Brier | ECE | conformal_q | config delta |
|-------------|-------------|----|---------|-------|---------|--------|---------|---------|-------|-----|-------------|--------------|
| 2026-04-28 04:02 | ? | 0 | 7 | 1867 | 0.5714 | 0.5238 | 0.6214 | 0.002 | — | — | — | (initial) |
| 2026-04-29 04:02 | ? | 0 | 25 | 2985 | 0.5387 | 0.5147 | 0.5620 | 0.0015 | 0.2613 | 0.1019 | — |  |
| 2026-04-29 08:00 | ? | 0 | 13 | 257 | 0.5000 | 0.3654 | 0.6346 | 0.56 | 0.3259 | 0.2805 | — | calibrator change |
| 2026-04-29 21:11 | long_horizon | 0 | 26 | 308 | 0.5192 | 0.4327 | 0.6154 | 0.38 | 0.3245 | 0.2811 | — | feature_set: ?→long_horizon; bar_minutes: 1→15 |
| 2026-04-30 04:04 | microstructure | 0 | 33 | 3444 | 0.5298 | 0.5076 | 0.5520 | 0.0043 | 0.2722 | 0.1321 | — | feature_set: long_horizon→microstructure; bar_minutes: 15→1; calibrator change |
| 2026-04-30 08:38 | cross_asset | 0 | 33 | 3461 | 0.5389 | 0.5177 | 0.5616 | 0.00029 | 0.2578 | 0.0832 | — | feature_set: microstructure→cross_asset; calibrator change |
| 2026-04-30 09:45 | cross_asset | 0 | 50 | 4460 | 0.5443 | 0.5260 | 0.5617 | 6.5e-07 | 0.2531 | 0.0636 | — |  |
| 2026-05-01 04:00 | cross_asset | 0 | 49 | 4408 | 0.5459 | 0.5286 | 0.5629 | 3.4e-07 | 0.2549 | 0.0750 | — |  |
| 2026-05-02 04:01 | cross_asset | 0 | 48 | 4326 | 0.5698 | 0.5521 | 0.5879 | 3.6e-14 | 0.2512 | 0.0563 | — |  |
| 2026-05-03 04:03 | cross_asset | 0 | 41 | 3932 | 0.5911 | 0.5715 | 0.6110 | 8e-20 | 0.2415 | 0.0518 | — | calibrator change |
| 2026-05-03 10:07 | cross_asset | 3 | 50 | 7212 | 0.5457 | 0.5270 | 0.5637 | 3.1e-07 | 0.2534 | 0.0631 | — | holdout_days: 0→3; calibrator change |
| 2026-05-03 11:05 | cross_asset | 3 | 50 | 7197 | 0.5420 | 0.5243 | 0.5593 | 2.3e-06 | 0.2532 | 0.0663 | 0.663 | conformal added |
| 2026-05-03 16:27 | cross_asset | 3 | 50 | 7166 | 0.5543 | 0.5373 | 0.5727 | 1.4e-09 | 0.2528 | 0.0633 | 0.661 |  |
| 2026-05-04 04:03 | cross_asset | 3 | 49 | 7117 | 0.5548 | 0.5378 | 0.5721 | 1.6e-09 | 0.2554 | 0.0669 | 0.683 |  |

## BNBUSDT

| run_started | feature_set | hd | n_folds | n_oos | dir_acc | CI low | CI high | p-value | Brier | ECE | conformal_q | config delta |
|-------------|-------------|----|---------|-------|---------|--------|---------|---------|-------|-----|-------------|--------------|
| 2026-04-28 04:02 | ? | 0 | 2 | 1565 | 0.6083 | 0.5167 | 0.7000 | 0.011 | — | — | — | (initial) |
| 2026-04-29 04:00 | ? | 0 | 17 | 2496 | 0.5549 | 0.5245 | 0.5843 | 0.00025 | 0.2600 | 0.1053 | — |  |
| 2026-04-29 08:00 | ? | 0 | 11 | 251 | 0.6591 | 0.5227 | 0.7955 | 0.024 | 0.2447 | 0.2139 | — | calibrator change |
| 2026-04-29 08:08 | long_horizon | 0 | 12 | 252 | 0.6667 | 0.5208 | 0.7917 | 0.015 | 0.2373 | 0.1821 | — | feature_set: ?→long_horizon; bar_minutes: 1→15; calibrator change |
| 2026-04-29 21:12 | long_horizon | 0 | 24 | 302 | 0.5833 | 0.4893 | 0.6773 | 0.063 | 0.2886 | 0.2265 | — | calibrator change |
| 2026-04-30 04:01 | microstructure | 0 | 24 | 2924 | 0.5514 | 0.5257 | 0.5771 | 5.3e-05 | 0.2661 | 0.1128 | — | feature_set: long_horizon→microstructure; bar_minutes: 15→1; calibrator change |
| 2026-04-30 08:38 | cross_asset | 0 | 24 | 2910 | 0.5771 | 0.5514 | 0.6035 | 2.7e-09 | 0.2548 | 0.0682 | — | feature_set: microstructure→cross_asset; calibrator change |
| 2026-04-30 09:45 | cross_asset | 0 | 38 | 3751 | 0.5570 | 0.5368 | 0.5776 | 2.8e-08 | 0.2508 | 0.0710 | — |  |
| 2026-05-01 04:01 | cross_asset | 0 | 37 | 3703 | 0.5599 | 0.5396 | 0.5802 | 9e-09 | 0.2571 | 0.0729 | — | calibrator change |
| 2026-05-02 04:02 | cross_asset | 0 | 36 | 3617 | 0.5579 | 0.5366 | 0.5783 | 4.1e-08 | 0.2563 | 0.0813 | — |  |
| 2026-05-03 04:04 | cross_asset | 0 | 32 | 3408 | 0.5786 | 0.5547 | 0.6011 | 2.9e-12 | 0.2441 | 0.0580 | — | calibrator change |
| 2026-05-03 10:07 | cross_asset | 3 | 38 | 6180 | 0.5605 | 0.5390 | 0.5803 | 4.1e-09 | 0.2524 | 0.0714 | — | holdout_days: 0→3; calibrator change |
| 2026-05-03 11:05 | cross_asset | 3 | 38 | 6168 | 0.5601 | 0.5399 | 0.5798 | 5.2e-09 | 0.2520 | 0.0688 | 0.685 | conformal added |
| 2026-05-03 16:27 | cross_asset | 3 | 39 | 6169 | 0.5547 | 0.5355 | 0.5756 | 6.6e-08 | 0.2565 | 0.0821 | 0.688 |  |
| 2026-05-04 04:04 | cross_asset | 3 | 37 | 6150 | 0.5536 | 0.5333 | 0.5743 | 2.4e-07 | 0.2528 | 0.0750 | 0.677 |  |

_Generated at 2026-05-04T07:52:50.939486+00:00 from 45 training runs._
