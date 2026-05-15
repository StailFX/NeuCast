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
  // 2026-05-15: realized_accuracy emits the rolling-100-trade window
  // shape ({accuracy, n_trades_total, n_correct, ...}) — components
  // also accept legacy {dir_acc_24h, n_trades_24h} via fallback.
  http.get("/api/highfreq/realized_accuracy", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      window: 100,
      n_trades_total: 35,
      n_eligible: 35,
      n_correct: 26,
      accuracy: 0.7428,
      avg_predicted_proba_up: 0.5946,
    }),
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

  // 2026-05-15: cumulative_pnl now emits ``points`` (per-trade-close
  // timestamps with tier values directly on the object) + ``tiers``
  // (with ``final_bps``, not ``final_cum_bps``).
  http.get("/api/highfreq/cumulative_pnl", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      n_trades: 3,
      n_points: 3,
      tiers: [
        { key: "gross", name: "Без комиссии", fee_bps: 0, final_bps: 8, win_rate: 0.66 },
        { key: "retail", name: "Spot retail", fee_bps: 7.5, final_bps: -45, win_rate: 0.0 },
        { key: "vip9", name: "Spot VIP-9", fee_bps: 1.0, final_bps: 6, win_rate: 0.50 },
      ],
      points: [
        { ts: "2026-05-01T00:00:00Z", gross: 0, retail: 0, vip9: 0 },
        { ts: "2026-05-02T00:00:00Z", gross: 12, retail: -45, vip9: 8 },
        { ts: "2026-05-03T00:00:00Z", gross: 8, retail: -85, vip9: 6 },
      ],
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

  // 2026-05-15: pnl_by_fee_tier emits {tier, fee_bps_per_side,
  // n_trades, n_wins, n_losses, pnl_usd, pnl_bps_avg,
  // pnl_usd_per_trade_avg}. Components also accept legacy
  // {key, fee_bps, mean_bps, win_rate} via the normaliser in
  // FeeTierPnLBars.tsx.
  http.get("/api/highfreq/pnl_by_fee_tier", () =>
    HttpResponse.json({
      ok: true,
      symbol: "BTCUSDT",
      tiers: [
        { tier: "gross", fee_bps_per_side: 0, n_trades: 56, n_wins: 31, n_losses: 25, pnl_usd: 12.4, pnl_bps_avg: 0.22, pnl_usd_per_trade_avg: 0.22 },
        { tier: "retail", fee_bps_per_side: 7.5, n_trades: 56, n_wins: 0, n_losses: 56, pnl_usd: -15.12, pnl_bps_avg: -14.02, pnl_usd_per_trade_avg: -0.27 },
        { tier: "vip9", fee_bps_per_side: 1.0, n_trades: 56, n_wins: 23, n_losses: 33, pnl_usd: 1.85, pnl_bps_avg: 0.33, pnl_usd_per_trade_avg: 0.033 },
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

  // 2026-05-15: reliability_diagram emits ``buckets`` with fields
  // {bin_idx, p_lo, p_hi, p_mid, n, n_pos, predicted_mean,
  // realized_rate}. Legacy ``bins`` keys (bin_lo/bin_hi/bin_mid) are
  // tolerated by the renderer via normalisation in ReliabilityDiagram.
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
          buckets: [
            { bin_idx: 0, p_lo: 0.0, p_hi: 0.1, p_mid: 0.05, n: 12, n_pos: 1, predicted_mean: 0.05, realized_rate: 0.083 },
            { bin_idx: 5, p_lo: 0.5, p_hi: 0.6, p_mid: 0.55, n: 1830, n_pos: 1043, predicted_mean: 0.55, realized_rate: 0.57 },
            { bin_idx: 9, p_lo: 0.9, p_hi: 1.0, p_mid: 0.95, n: 5, n_pos: 4, predicted_mean: 0.95, realized_rate: 0.80 },
          ],
        },
      ],
    }),
  ),

  // 2026-05-15: robustness wraps payload under ``report`` and
  // FLATTENS bootstrap/permutation results into top-level scalars
  // (block_bootstrap_ci_low/high, permutation_p_value, ...). Component
  // reads ``data.report.*`` first, falls back to legacy ``data.*``.
  http.get("/api/highfreq/robustness", () =>
    HttpResponse.json({
      ok: true,
      ts: "2026-05-15T08:30:00Z",
      report: {
        symbol: "BTCUSDT",
        generated_at: "2026-05-15T08:30:00Z",
        n_predictions: 7029,
        n_bootstrap: 1000,
        n_permutations: 1000,
        block_size_minutes: 60,
        dir_acc: 0.5409,
        n_correct: 3801,
        n_total: 7029,
        block_bootstrap_ci_low: 0.5342,
        block_bootstrap_ci_high: 0.5476,
        permutation_p_value: 0.001,
        permutation_null_mean: 0.50,
        permutation_null_std: 0.0060,
        permutation_z_score: 6.82,
        per_day: [
          { day: "2026-05-01", dir_acc: 0.55, n: 240 },
          { day: "2026-05-02", dir_acc: 0.52, n: 240 },
        ],
        per_hour: [
          { hour: 0, dir_acc: 0.54, n: 100 },
          { hour: 1, dir_acc: 0.53, n: 100 },
        ],
      },
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
