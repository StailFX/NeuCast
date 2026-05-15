import type {
  DashboardResponse,
  PaperTradesResponse,
  RealizedAccuracyResponse,
  ReliabilityResponse,
  FeatureImportanceResponse,
  ConditionalAccuracyResponse,
  CumulativePnLResponse,
  ForecastEnsembleResponse,
  RobustnessResponse,
  PnLByFeeTierResponse,
  HealthResponse,
  StatusResponse,
  TrainingReportResponse,
  AntiSkillResponse,
  TrainingHistoryResponse,
} from "./api-types";

/**
 * Base URL for /api/highfreq/* calls.
 *
 * In dev, ``next dev`` proxies via the rewrite block in next.config.ts
 * so we just call relative paths and let next forward to neucast.ru.
 *
 * In prod (static-export served from Finland nginx), the same
 * relative paths land at the same nginx that serves the bundle —
 * one origin, no CORS.
 *
 * Override via ``NEXT_PUBLIC_API_BASE`` env if you ever serve the
 * frontend from a different origin (e.g. Vercel preview pointing
 * at production API).
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";


export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}


async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}


/** Fetch the batch dashboard payload for the given symbols at the
 * given horizon. Supported: 1, 5, 15 (minutes). Anything else
 * silently clamps to 1 on the server. */
export function fetchDashboard(
  symbols: string[],
  horizon: number = 1,
): Promise<DashboardResponse> {
  const qs = symbols.map((s) => s.toUpperCase()).join(",");
  return fetchJson<DashboardResponse>(
    `/api/highfreq/dashboard?symbols=${qs}&horizon=${horizon}`,
  );
}


/** Fetch recent paper trades for one symbol. */
export function fetchPaperTrades(
  symbol: string,
  limit = 80,
): Promise<PaperTradesResponse> {
  return fetchJson<PaperTradesResponse>(
    `/api/highfreq/paper_trades?symbol=${symbol.toUpperCase()}&limit=${limit}`,
  );
}


/** 24h + lifetime realized direction accuracy for a symbol. */
export function fetchRealizedAccuracy(
  symbol: string,
): Promise<RealizedAccuracyResponse> {
  return fetchJson<RealizedAccuracyResponse>(
    `/api/highfreq/realized_accuracy?symbol=${symbol.toUpperCase()}`,
  );
}


/** Reliability diagram (calibration curve) per symbol. */
export function fetchReliabilityDiagram(
  nBins = 10,
): Promise<ReliabilityResponse> {
  return fetchJson<ReliabilityResponse>(
    `/api/highfreq/reliability_diagram?n_bins=${nBins}`,
  );
}


/** Per-feature importance from the loaded CatBoost model. */
export function fetchFeatureImportance(
  symbol: string,
): Promise<FeatureImportanceResponse> {
  return fetchJson<FeatureImportanceResponse>(
    `/api/highfreq/feature_importance?symbol=${symbol.toUpperCase()}`,
  );
}


/** Conditional accuracy bucketed by confidence threshold (55/60/65). */
export function fetchConditionalAccuracy(): Promise<ConditionalAccuracyResponse> {
  return fetchJson<ConditionalAccuracyResponse>(
    `/api/highfreq/conditional_accuracy`,
  );
}


/** Cumulative P&L curve by fee tier for one symbol. */
export function fetchCumulativePnL(
  symbol: string,
  limitPoints = 200,
): Promise<CumulativePnLResponse> {
  return fetchJson<CumulativePnLResponse>(
    `/api/highfreq/cumulative_pnl?symbol=${symbol.toUpperCase()}&limit_points=${limitPoints}`,
  );
}


/** Multi-horizon ensemble forecast (1m + 15m blend). */
export function fetchForecastEnsemble(
  symbol: string,
  weight1m = 0.7,
  weight15m = 0.3,
): Promise<ForecastEnsembleResponse> {
  const sym = symbol.toUpperCase();
  return fetchJson<ForecastEnsembleResponse>(
    `/api/highfreq/forecast_ensemble?symbol=${sym}&weight_1m=${weight1m}&weight_15m=${weight15m}`,
  );
}


/** Joint multi-symbol shadow forecast (Phase 2.2, 2026-05-09).
 *
 * Returns the joint model's prediction for one symbol. The joint
 * model is trained once on pooled BTC + ETH + BNB data with symbol-id
 * one-hots; this endpoint runs inference on the requested symbol's
 * latest minute-bar features and returns the calibrated prob_up,
 * the raw (pre-calibration) probability, and the joint training-run
 * status block (dir_acc / CI / p / n_folds).
 *
 * Used by the dashboard's ``<JointForecastBadge />`` to display the
 * joint prediction next to the per-symbol solo forecast — a
 * read-only side-by-side observable while Phase 2.2 collects shadow
 * data for the eventual ensemble flip.
 */
export interface JointForecastResponse {
  ok: boolean;
  reason?: string;
  symbol?: string;
  ts: string;
  prob_up?: number;
  raw_prob_up?: number;
  signal?: "up" | "down" | "neutral";
  model?: {
    has_model: boolean;
    model_path?: string;
    model_age_seconds?: number;
    is_calibrated?: boolean;
    dir_acc_mean?: number;
    dir_acc_ci_low?: number;
    dir_acc_ci_high?: number;
    dir_acc_p_value?: number;
    n_folds?: number;
    feature_set?: string;
    joint_symbols?: string[];
    reason?: string;
    error?: string;
  };
}


export function fetchForecastJoint(
  symbol: string,
): Promise<JointForecastResponse> {
  return fetchJson<JointForecastResponse>(
    `/api/highfreq/forecast_joint?symbol=${symbol.toUpperCase()}`,
  );
}


/** Block-bootstrap robustness suite (CI + permutation + per-day). */
export function fetchRobustness(
  symbol: string,
): Promise<RobustnessResponse> {
  return fetchJson<RobustnessResponse>(
    `/api/highfreq/robustness?symbol=${symbol.toUpperCase()}`,
  );
}


/** Per-tier closed-trade aggregate P&L for one symbol. */
export function fetchPnLByFeeTier(
  symbol: string,
): Promise<PnLByFeeTierResponse> {
  return fetchJson<PnLByFeeTierResponse>(
    `/api/highfreq/pnl_by_fee_tier?symbol=${symbol.toUpperCase()}`,
  );
}


// ── Operator-side endpoints (used by /v2/highfreq) ─────────────────

/** Liveness — ingestor wrote ≥1 row in the last 60s. */
export function fetchHealth(symbol: string): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(
    `/api/highfreq/health?symbol=${symbol.toUpperCase()}`,
  );
}


/** Latest 1-second snapshot + freshness verdict. */
export function fetchStatus(symbol: string): Promise<StatusResponse> {
  return fetchJson<StatusResponse>(
    `/api/highfreq/status?symbol=${symbol.toUpperCase()}`,
  );
}


/** Trainer's last metrics report + computed fold-readiness. */
export function fetchTrainingReport(
  symbol: string,
  lite = 0,
): Promise<TrainingReportResponse> {
  return fetchJson<TrainingReportResponse>(
    `/api/highfreq/training_report?symbol=${symbol.toUpperCase()}&lite=${lite}`,
  );
}


/** Anti-skill detection (rolling winrate CI < 0.5). */
export function fetchAntiSkill(symbol: string): Promise<AntiSkillResponse> {
  return fetchJson<AntiSkillResponse>(
    `/api/highfreq/anti_skill?symbol=${symbol.toUpperCase()}`,
  );
}


/** Training-history rows (one per trainer run). */
export function fetchTrainingHistory(
  symbol?: string,
  limit = 30,
): Promise<TrainingHistoryResponse> {
  const sym = symbol ? `&symbol=${symbol.toUpperCase()}` : "";
  return fetchJson<TrainingHistoryResponse>(
    `/api/highfreq/training_history?limit=${limit}${sym}`,
  );
}


// ── Daily TCN predict workflow ────────────────────────────────────

/** Request payload for ``POST /api/predict``. Mirrors the legacy
 * form fields one-for-one. */
export interface PredictRequest {
  ticker: string;
  start_date: string; // YYYY-MM-DD
  end_date: string;   // YYYY-MM-DD
  days_ahead: number;
  use_foundation: boolean;
}


/** Response of ``POST /api/predict``. Caller navigates to
 * ``/v2/predict/{task_id}`` (waiting page) and falls back to
 * ``redirect_url`` for the legacy result render once SUCCESS. */
export interface PredictResponse {
  ok: true;
  task_id: string;
  slug: string | null;
  /** Either ``/p/{slug}`` (preferred) or the long
   * ``/predict/status/{task_id}?...`` form when Redis is down. */
  redirect_url: string;
}


/** Submit a TCN prediction task. Requires an authenticated session
 * (httponly cookie); the API returns 401 otherwise. */
export function submitPrediction(
  body: PredictRequest,
): Promise<PredictResponse> {
  return fetchJson<PredictResponse>(`/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    credentials: "include",
  });
}


/** State machine surfaced by ``GET /api/task/{task_id}``. */
export type TaskState =
  | "PENDING"
  | "FETCHING"
  | "PREDICTING"
  | "SUCCESS"
  | "FAILURE"
  | string; // celery sometimes emits other states; treat as in-flight.


export interface TaskStatusResponse {
  state: TaskState;
  status?: string;
  task_id?: string;
  error?: string;
}


/** Poll the celery task. The /api/task/* endpoint enforces ownership
 * (the caller must be the same user that submitted), so this requires
 * the session cookie. */
export function fetchTaskStatus(
  taskId: string,
): Promise<TaskStatusResponse> {
  return fetchJson<TaskStatusResponse>(`/api/task/${taskId}`, {
    credentials: "include",
  });
}
