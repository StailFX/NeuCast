/**
 * Typed shape of /api/highfreq/dashboard payload (FastAPI side
 * lives at app/highfreq/web.py::get_dashboard).
 *
 * Code-review H-1perf (2026-05-04) introduced this batch endpoint —
 * it returns forecast + drift + microprice for N symbols in a
 * single request, replacing the previous fan-out of 9 individual
 * fetches. The frontend leans on this for the headline cards.
 */
export type Severity = "ok" | "warn" | "high";

export type Signal = "up" | "down" | "neutral";

export interface ConformalInterval {
  alpha: number;
  low: number;
  high: number;
  halfwidth: number;
}

export interface ModelStatus {
  has_model: boolean;
  model_path?: string;
  model_age_seconds?: number;
  is_calibrated?: boolean;
  dir_acc_mean?: number;
  dir_acc_ci_low?: number;
  dir_acc_p_value?: number;
  metrics_age_seconds?: number;
  n_features_expected?: number;
}

export type ForecastBlock =
  | {
      ok: true;
      prob_up: number;
      signal: Signal;
      model: ModelStatus;
      conformal_90?: [number, number] | { low: number; high: number };
      conformal_95?: [number, number] | { low: number; high: number };
    }
  | { ok: false; reason: string; model?: ModelStatus };

export type DriftBlock =
  | {
      ok: true;
      severity: Severity;
      max_ks: number | null;
      max_ks_feature: string | null;
      evaluated_at: string | null;
    }
  | { ok: false; reason: string };

export type MicropriceBlock =
  | { ok: true; price: number; ts: string }
  | { ok: false; reason: string };

export interface PerSymbolPayload {
  symbol: string;
  forecast: ForecastBlock;
  drift: DriftBlock;
  microprice: MicropriceBlock;
}

export type DashboardResponse =
  | {
      ok: true;
      ts: string;
      n_symbols: number;
      symbols: Record<string, PerSymbolPayload>;
    }
  | { ok: false; reason: string; max?: number; ts: string };


// ── /api/highfreq/paper_trades ────────────────────────────────────
export interface PaperTrade {
  id?: number;
  symbol: string;
  side: "long" | "short";
  qty: number;
  entry_ts: string;
  entry_price: number;
  entry_prob_up?: number | null;
  exit_ts: string | null;
  exit_price: number | null;
  exit_reason: string | null;
  fee_paid_total_usd: number | null;
  pnl_usd: number | null;
  pnl_bps: number | null;
  model_version: string | null;
}

export interface PaperTradesResponse {
  ok: boolean;
  symbol: string;
  trades: PaperTrade[];
}


// ── /api/highfreq/realized_accuracy ───────────────────────────────
//
// Current shape (2026-05-15): rolling 100-trade window from
// app/highfreq/realized_accuracy.py. Legacy 24h-named fields are
// kept as optional for backward-compat with the old MSW mocks but
// the FastAPI side now emits the ``accuracy`` / ``n_trades_total``
// names. Components read the new fields with a fallback to legacy.
export interface RealizedAccuracyResponse {
  ok: boolean;
  symbol: string;
  // ── current shape (rolling 100-trade window) ──
  ts?: string;
  db_status?: string;
  window?: number;
  n_trades_total?: number;
  n_eligible?: number;
  n_correct?: number;
  accuracy?: number;
  avg_predicted_proba_up?: number;
  earliest_exit_ts?: string;
  latest_exit_ts?: string;
  // ── legacy fields (still emitted in some MSW mocks; tolerated) ──
  n_trades_24h?: number;
  n_directional_24h?: number;
  n_correct_24h?: number;
  dir_acc_24h?: number;
  ci_low_24h?: number;
  ci_high_24h?: number;
  p_value_24h?: number;
  n_trades_lifetime?: number;
  dir_acc_lifetime?: number;
  reason?: string;
}


// ── /api/highfreq/reliability_diagram ─────────────────────────────
//
// Current shape (2026-05-15): server emits ``buckets`` with
// {bin_idx, p_lo, p_hi, p_mid, n, n_pos, predicted_mean,
//  realized_rate}. Components normalise to the legacy ``bins``
// vocabulary for rendering. Legacy ``bins`` field kept optional for
// any historical caller.
export interface ReliabilityBucket {
  bin_idx?: number;
  p_lo: number;
  p_hi: number;
  p_mid: number;
  n: number;
  n_pos?: number;
  predicted_mean?: number;
  realized_rate: number | null;
}

/** Legacy bin shape (pre-2026-05-15). Some MSW mocks still emit this. */
export interface ReliabilityBin {
  bin_lo: number;
  bin_hi: number;
  bin_mid: number;
  n: number;
  realized_rate: number | null;
}

export interface ReliabilitySymbolRow {
  symbol: string;
  n_total: number;
  brier: number | null;
  ece: number | null;
  buckets?: ReliabilityBucket[];
  bins?: ReliabilityBin[]; // legacy fallback
}

export interface ReliabilityResponse {
  ok: boolean;
  ts?: string;
  n_bins?: number;
  rows?: ReliabilitySymbolRow[];
  reason?: string;
}


// ── /api/highfreq/feature_importance ──────────────────────────────
export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface FeatureImportanceResponse {
  ok: boolean;
  symbol?: string;
  importance?: FeatureImportanceItem[];
  reason?: string;
}


// ── /api/highfreq/conditional_accuracy ────────────────────────────
export interface ConditionalBucket {
  threshold: number;
  n: number;
  hits: number;
  dir_acc: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p_value: number | null;
}

export interface ConditionalSymbolRow {
  symbol: string;
  buckets: {
    conf_55?: ConditionalBucket;
    conf_60?: ConditionalBucket;
    conf_65?: ConditionalBucket;
  };
}

export interface ConditionalAccuracyResponse {
  ok: boolean;
  ts?: string;
  rows?: ConditionalSymbolRow[];
  reason?: string;
  db_status?: string;
}


// ── /api/highfreq/cumulative_pnl ──────────────────────────────────
//
// Current shape (2026-05-15): server returns ``points: [{ts, gross,
// retail, vip5, vip9, futures, mm_rebate}]`` — tier values DIRECTLY
// on the point object, not under ``cum_bps_by_tier``. The legacy
// ``PnLPoint`` shape is kept for the rendering layer; CumulativePnL
// component normalises server → legacy at render time.
export interface PnLPointServer {
  ts: string;
  gross?: number;
  retail?: number;
  vip5?: number;
  vip9?: number;
  futures?: number;
  mm_rebate?: number;
}

/** Legacy shape — components use this internally after normalisation. */
export interface PnLPoint {
  ts: string;
  /** dict tier_key → cumulative bps at this trade-close ts. */
  cum_bps_by_tier: Record<string, number>;
  /** Cumulative trade count at this point — useful for x-axis density. */
  n: number;
}

// ── /api/auth/* ───────────────────────────────────────────────────
export interface AuthUser {
  id: number;
  username: string;
  role: string;
}

export interface AuthMeResponse {
  authenticated: boolean;
  user?: AuthUser;
}

export interface AuthLoginResponse {
  ok: boolean;
  user?: AuthUser;
  detail?: string;
}


// ── /api/highfreq/health ──────────────────────────────────────────
export interface HealthResponse {
  ok: boolean;
  symbol: string;
  rows_last_60s: number;
  reason?: string;
}


// ── /api/highfreq/status ──────────────────────────────────────────
export interface StatusResponse {
  ok: boolean;
  symbol: string;
  ts?: string;
  last_ts?: string | null;
  microprice?: number | null;
  spread_bps?: number | null;
  depth_imb?: number | null;
  freshness_seconds?: number | null;
  is_fresh?: boolean;
  reason?: string;
}


// ── /api/highfreq/training_report ─────────────────────────────────
export interface FoldRowPayload {
  fold_idx: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  n_train: number;
  n_test: number;
  dir_acc: number;
  log_loss: number;
}

export interface TrainingReportResponse {
  ok: boolean;
  symbol?: string;
  reason?: string;
  fold_ready_pct?: number;
  live_inventory?: {
    n_seconds_loaded: number;
    n_minutes_after_aggregation: number;
    n_minutes_after_neutral_drop: number;
    n_eligible_for_training: number;
    n_in_holdout: number;
    since_hours: number;
  };
  report?: {
    symbol?: string;
    horizon_min?: number;
    n_seconds_loaded?: number;
    n_minutes_after_neutral_drop?: number;
    n_folds?: number;
    dir_acc_mean?: number;
    dir_acc_ci_low?: number;
    dir_acc_ci_high?: number;
    dir_acc_p_value?: number;
    log_loss_mean?: number;
    base_rate?: number;
    feature_set?: string;
    bar_minutes?: number;
    folds?: FoldRowPayload[];
    weights_path?: string;
    started_at?: string;
    finished_at?: string;
    elapsed_seconds?: number;
  };
}


// ── /api/highfreq/anti_skill ──────────────────────────────────────
export interface AntiSkillResponse {
  ok: boolean;
  symbol?: string;
  is_anti_skilled?: boolean;
  gross_winrate?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  n_trades_in_window?: number;
  policy?: string;
  note?: string;
  reason?: string;
}


// ── /api/highfreq/training_history ────────────────────────────────
export interface TrainingHistoryRow {
  id: number;
  symbol: string;
  run_started_at: string;
  feature_set: string;
  bar_minutes: number;
  n_folds: number;
  n_minutes_after_neutral_drop: number;
  dir_acc_mean: number | null;
  dir_acc_ci_low: number | null;
  dir_acc_ci_high: number | null;
  dir_acc_p_value: number | null;
  weights_path: string | null;
}

export interface TrainingHistoryResponse {
  ok: boolean;
  rows?: TrainingHistoryRow[];
  reason?: string;
}


// ── /api/highfreq/forecast_ensemble ───────────────────────────────
export interface EnsembleComponentResponse {
  horizon_label: string;
  weight: number;
  prob_up: number | null;
  is_available: boolean;
}

export interface ForecastEnsembleResponse {
  ok: boolean;
  symbol?: string;
  prob_up?: number;
  signal?: Signal;
  agreement?: boolean;
  n_components_used?: number;
  components?: EnsembleComponentResponse[];
  models?: Record<string, ModelStatus>;
  reason?: string;
}


// ── /api/highfreq/robustness ──────────────────────────────────────
export interface BlockBootstrapCI {
  point: number;
  ci_low: number;
  ci_high: number;
  block_minutes?: number;
  n_blocks?: number;
  n_resamples?: number;
}

export interface PermutationTest {
  observed: number;
  p_value: number;
  n_resamples?: number;
}

export interface PerDayPoint {
  day: string;
  n: number;
  dir_acc: number;
}

export interface PerHourPoint {
  hour: number;
  n: number;
  dir_acc: number;
}

/**
 * Current shape (2026-05-15): server wraps everything in ``report``
 * and FLATTENS the bootstrap/permutation results into top-level
 * scalars (block_bootstrap_ci_low/high, permutation_p_value, ...).
 * The component reads ``data.report.*`` first, falling back to
 * ``data.*`` for legacy callers / MSW mocks.
 */
export interface RobustnessReport {
  symbol?: string;
  generated_at?: string;
  n_predictions?: number;
  n_bootstrap?: number;
  n_permutations?: number;
  block_size_minutes?: number;
  dir_acc?: number;
  n_correct?: number;
  n_total?: number;
  block_bootstrap_ci_low?: number;
  block_bootstrap_ci_high?: number;
  permutation_p_value?: number;
  permutation_null_mean?: number;
  permutation_null_std?: number;
  permutation_z_score?: number;
  per_day?: PerDayPoint[];
  per_day_min?: number;
  per_day_max?: number;
  per_day_std?: number;
  per_day_all_above_chance?: boolean;
  per_hour?: PerHourPoint[];
  per_regime?: Array<{ regime: string; n: number; dir_acc: number }>;
}

export interface RobustnessResponse {
  ok: boolean;
  ts?: string;
  /** Current path: server wraps payload under ``report``. */
  report?: RobustnessReport;
  /** ── legacy fields (top-level) — kept for MSW mocks ──. */
  symbol?: string;
  generated_at?: string;
  n_predictions?: number;
  block_bootstrap?: BlockBootstrapCI;
  permutation?: PermutationTest;
  per_day?: PerDayPoint[];
  per_hour?: PerHourPoint[];
  regime_accuracy?: Array<{ regime: string; n: number; dir_acc: number }>;
  reason?: string;
  hint?: string;
}


// ── /api/highfreq/pnl_by_fee_tier ─────────────────────────────────
//
// Current shape (2026-05-15): server emits {tier, fee_bps_per_side,
// n_trades, n_wins, n_losses, pnl_usd, pnl_bps_avg,
// pnl_usd_per_trade_avg}. Component normalises to display shape via
// FeeTierRowDisplay (defined locally in FeeTierPnLBars.tsx).
export interface FeeTierSummary {
  // ── current fields ──
  tier?: string;
  fee_bps_per_side?: number;
  n_trades: number;
  n_wins?: number;
  n_losses?: number;
  pnl_usd?: number;
  pnl_bps_avg?: number;
  pnl_usd_per_trade_avg?: number;
  // ── legacy fields (MSW mocks still emit these) ──
  key?: string;
  name?: string;
  fee_bps?: number;
  total_bps?: number;
  win_rate?: number | null;
  mean_bps?: number;
}

export interface PnLByFeeTierResponse {
  ok: boolean;
  symbol?: string;
  tiers?: FeeTierSummary[];
  reason?: string;
  db_status?: string;
}


/**
 * Current shape (2026-05-15): server returns ``points`` (with tier
 * values directly on each point) + ``tiers`` (with ``final_bps``,
 * not ``final_cum_bps``). The legacy fields are kept optional for
 * backward-compat with older callers / MSW mocks.
 */
export interface CumulativePnLResponse {
  ok: boolean;
  symbol?: string;
  n_trades?: number;
  n_points?: number;
  first_trade_ts?: string;
  last_trade_ts?: string;
  ts?: string;
  tiers?: Array<{
    key: string;
    name?: string;
    fee_bps?: number;
    /** Current field name (renamed 2026-05-15). */
    final_bps?: number;
    win_rate?: number;
    /** Legacy alias for final_bps. */
    final_cum_bps?: number;
    final_cum_pct?: number;
  }>;
  /** Current: per-trade-close tier values directly on the object. */
  points?: PnLPointServer[];
  /** Legacy: nested ``cum_bps_by_tier`` map. */
  curve?: PnLPoint[];
  reason?: string;
}
