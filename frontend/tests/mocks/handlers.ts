import { http, HttpResponse } from "msw";

/**
 * Default MSW handlers — happy-path responses for the endpoints
 * components touch when mounted. Individual tests can override
 * via ``server.use(http.get(...))`` for failure scenarios.
 */

export const handlers = [
  // Auth probe — by default the user is anon.
  http.get("/api/auth/me", () =>
    HttpResponse.json({ authenticated: false }),
  ),

  // Dashboard — 3 symbols, all with usable forecasts.
  http.get("/api/highfreq/dashboard", () =>
    HttpResponse.json({
      ok: true,
      ts: "2026-05-04T17:00:00Z",
      n_symbols: 3,
      symbols: {
        BTCUSDT: {
          symbol: "BTCUSDT",
          forecast: {
            ok: true,
            prob_up: 0.62,
            signal: "up",
            model: { has_model: true, model_age_seconds: 3600, is_calibrated: true },
          },
          drift: {
            ok: true,
            severity: "ok",
            max_ks: 0.08,
            max_ks_feature: "ofi_mean",
            evaluated_at: "2026-05-04T16:30:00Z",
          },
          microprice: { ok: true, price: 79000.5, ts: "2026-05-04T17:00:00Z" },
        },
        ETHUSDT: {
          symbol: "ETHUSDT",
          forecast: {
            ok: true,
            prob_up: 0.50,
            signal: "neutral",
            model: { has_model: true, model_age_seconds: 3600, is_calibrated: true },
          },
          drift: {
            ok: true,
            severity: "warn",
            max_ks: 0.21,
            max_ks_feature: "spread_bps_mean",
            evaluated_at: "2026-05-04T16:30:00Z",
          },
          microprice: { ok: true, price: 3500.2, ts: "2026-05-04T17:00:00Z" },
        },
        BNBUSDT: {
          symbol: "BNBUSDT",
          forecast: {
            ok: true,
            prob_up: 0.40,
            signal: "down",
            model: { has_model: true, model_age_seconds: 3600, is_calibrated: true },
          },
          drift: { ok: false, reason: "no_check_yet" },
          microprice: { ok: true, price: 620.7, ts: "2026-05-04T17:00:00Z" },
        },
      },
    }),
  ),

  // Default empties for everything else the cards / strips fetch.
  http.get("/api/highfreq/realized_accuracy", () =>
    HttpResponse.json({ ok: true, symbol: "BTCUSDT" }),
  ),
  http.get("/api/highfreq/paper_trades", () =>
    HttpResponse.json({ ok: true, symbol: "BTCUSDT", trades: [] }),
  ),
  http.get("/api/highfreq/forecast_ensemble", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      prob_up: 0.58,
      signal: "up",
      agreement: true,
      n_components_used: 2,
      components: [
        { horizon_label: "1m", weight: 0.7, prob_up: 0.62, is_available: true },
        { horizon_label: "15m", weight: 0.3, prob_up: 0.50, is_available: true },
      ],
    }),
  ),
  // Phase 2.2 shadow joint forecast — default returns a happy
  // calibrated payload so ForecastCard / Dashboard tests can render
  // the JointForecastBadge without MSW raising onUnhandledRequest.
  // Per-test overrides for the cold-start / disagreement scenarios
  // live in the badge's own test file.
  http.get("/api/highfreq/forecast_joint", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      ts: "2026-05-09T10:42:52Z",
      prob_up: 0.5421,
      raw_prob_up: 0.5012,
      signal: "up",
      model: {
        has_model: true,
        model_path: "/opt/neucast/weights/highfreq/joint_1m.cbm",
        model_age_seconds: 1466,
        is_calibrated: true,
        dir_acc_mean: 0.5409,
        dir_acc_ci_low: 0.5342,
        dir_acc_ci_high: 0.5476,
        dir_acc_p_value: 4.97e-33,
        n_folds: 353,
        feature_set: "joint",
        joint_symbols: ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
      },
    }),
  ),

  // ── Default happy handlers for the remaining /api/highfreq/* endpoints.
  // Each component test that needs a different scenario overrides via
  // server.use() locally — these defaults just keep the component from
  // crashing when its hook fires during another test's render.

  http.get("/api/highfreq/anti_skill", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      is_anti_skilled: false,
      gross_winrate: 0.68,
      ci_low: 0.54,
      ci_high: 0.79,
      n_trades_in_window: 50,
      threshold: 0.50,
      window: 50,
      note: "healthy: gross winrate 0.680 ≥ 0.50",
    }),
  ),

  http.get("/api/highfreq/conditional_accuracy", () =>
    HttpResponse.json({
      ok: true,
      ts: "2026-05-09T10:00:00Z",
      rows: [
        {
          symbol: "BTCUSDT",
          buckets: {
            conf_55: { threshold: 0.05, n: 7011, hits: 3867, dir_acc: 0.5516, ci_low: 0.5399, ci_high: 0.5632, p_value: 3.1e-18 },
            conf_60: { threshold: 0.10, n: 3093, hits: 1746, dir_acc: 0.5645, ci_low: 0.547, ci_high: 0.582, p_value: 3.9e-13 },
            conf_65: { threshold: 0.15, n: 1662, hits: 948, dir_acc: 0.5704, ci_low: 0.546, ci_high: 0.594, p_value: 5.2e-9 },
          },
        },
        {
          symbol: "ETHUSDT",
          buckets: {
            conf_55: { threshold: 0.05, n: 5855, hits: 3126, dir_acc: 0.5339, ci_low: 0.521, ci_high: 0.547, p_value: 1.1e-7 },
          },
        },
      ],
    }),
  ),

  http.get("/api/highfreq/cumulative_pnl", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      tiers: {
        gross: [
          { ts: "2026-05-01T00:00:00Z", cum_pnl_bps: 0 },
          { ts: "2026-05-02T00:00:00Z", cum_pnl_bps: 12 },
          { ts: "2026-05-03T00:00:00Z", cum_pnl_bps: 8 },
        ],
        retail: [
          { ts: "2026-05-01T00:00:00Z", cum_pnl_bps: 0 },
          { ts: "2026-05-02T00:00:00Z", cum_pnl_bps: -45 },
        ],
      },
    }),
  ),

  http.get("/api/highfreq/feature_importance", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      importance: [
        { feature: "ofi_mean", importance: 22.5 },
        { feature: "spread_bps_mean", importance: 18.1 },
        { feature: "depth_imb_mean", importance: 12.4 },
        { feature: "trade_imb_mean", importance: 9.7 },
        { feature: "ofi_std", importance: 7.2 },
      ],
    }),
  ),

  http.get("/api/highfreq/pnl_by_fee_tier", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      tiers: [
        { key: "gross", name: "Без комиссии", fee_bps: 0, n_trades: 56, win_rate: 0.55, total_bps: 12.4, mean_bps: 0.22 },
        { key: "retail", name: "Spot retail", fee_bps: 7.5, n_trades: 56, win_rate: 0.0, total_bps: -785, mean_bps: -14.0 },
        { key: "vip9", name: "Spot VIP-9", fee_bps: 1.0, n_trades: 56, win_rate: 0.41, total_bps: 18.5, mean_bps: 0.33 },
      ],
    }),
  ),

  http.get("/api/highfreq/health", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      rows_last_60s: 59,
    }),
  ),

  http.get("/api/highfreq/status", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      ts: "2026-05-09T10:00:00Z",
      microprice: 79000.5,
      age_seconds: 1.2,
      verdict: "fresh",
    }),
  ),

  http.get("/api/highfreq/reliability_diagram", () =>
    HttpResponse.json({
      ok: true,
      ts: "2026-05-09T10:00:00Z",
      n_bins: 10,
      rows: [
        {
          symbol: "BTCUSDT",
          n_total: 7011,
          brier: 0.241,
          ece: 0.018,
          bins: [
            { bin_lo: 0.0, bin_hi: 0.1, bin_mid: 0.05, n: 12, realized_rate: 0.05 },
            { bin_lo: 0.5, bin_hi: 0.6, bin_mid: 0.55, n: 1830, realized_rate: 0.57 },
            { bin_lo: 0.9, bin_hi: 1.0, bin_mid: 0.95, n: 5, realized_rate: 0.80 },
          ],
        },
      ],
    }),
  ),

  http.get("/api/highfreq/robustness", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      block_bootstrap: {
        dir_acc: 0.5409,
        ci_low: 0.5342,
        ci_high: 0.5476,
        n_blocks: 100,
        block_size_minutes: 60,
      },
      permutation_test: {
        observed_dir_acc: 0.5409,
        n_permutations: 1000,
        p_value: 0.001,
      },
      per_day: [
        { day: "2026-05-01", dir_acc: 0.55, n: 240 },
        { day: "2026-05-02", dir_acc: 0.52, n: 240 },
      ],
      per_hour: [
        { hour: 0, dir_acc: 0.54, n: 100 },
        { hour: 1, dir_acc: 0.53, n: 100 },
      ],
    }),
  ),

  http.get("/api/highfreq/training_history", () =>
    HttpResponse.json({
      ok: true,
      rows: [
        {
          id: 270,
          symbol: "BTCUSDT",
          run_started_at: "2026-05-09T04:00:00Z",
          n_folds: 33,
          dir_acc_mean: 0.5288,
          dir_acc_ci_low: 0.5060,
          dir_acc_ci_high: 0.5510,
          dir_acc_p_value: 0.0055,
          base_rate: 0.5072,
          feature_set: "microstructure",
          weights_path: "weights/highfreq/btcusdt_1m.cbm",
          elapsed_seconds: 268.9,
        },
      ],
    }),
  ),

  http.get("/api/highfreq/training_report", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      report: {
        symbol: "BTCUSDT",
        horizon_min: 1,
        n_seconds_loaded: 640679,
        n_minutes_after_aggregation: 10678,
        n_minutes_after_neutral_drop: 7843,
        base_rate: 0.5072,
        n_folds: 33,
        dir_acc_mean: 0.5288,
        dir_acc_ci_low: 0.5060,
        dir_acc_ci_high: 0.5510,
        dir_acc_p_value: 0.0055,
        log_loss_mean: 1.4285,
        low_directional_skill: false,
        feature_set: "microstructure",
        bar_minutes: 1,
        elapsed_seconds: 268.9,
        calibrator_brier: 0.241,
        calibrator_ece: 0.018,
      },
    }),
  ),
];
