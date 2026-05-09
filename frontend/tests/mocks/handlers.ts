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
];
