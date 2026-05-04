"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchPaperTrades,
  fetchRealizedAccuracy,
  fetchReliabilityDiagram,
  fetchFeatureImportance,
  fetchConditionalAccuracy,
  fetchCumulativePnL,
  fetchForecastEnsemble,
  fetchRobustness,
  fetchPnLByFeeTier,
  fetchHealth,
  fetchStatus,
  fetchTrainingReport,
  fetchAntiSkill,
  fetchTrainingHistory,
} from "./api";

/**
 * Cadences match the legacy `templates/forecast.html` so behaviour
 * doesn't drift unexpectedly when we cut over.
 *  • PREDICT (cards, dashboard): 30 s
 *  • TRADES (feed, P&L): 60 s
 *  • METRICS (reliability, importance, conditional): 5 min
 */
export const PREDICT_REFRESH_MS = 30_000;
export const TRADES_REFRESH_MS = 60_000;
export const METRICS_REFRESH_MS = 5 * 60_000;


export function usePaperTrades(symbol: string, limit = 80) {
  return useQuery({
    queryKey: ["paper_trades", symbol, limit],
    queryFn: () => fetchPaperTrades(symbol, limit),
    refetchInterval: TRADES_REFRESH_MS,
  });
}


export function useRealizedAccuracy(symbol: string) {
  return useQuery({
    queryKey: ["realized_accuracy", symbol],
    queryFn: () => fetchRealizedAccuracy(symbol),
    refetchInterval: METRICS_REFRESH_MS,
  });
}


export function useReliabilityDiagram(nBins = 10) {
  return useQuery({
    queryKey: ["reliability_diagram", nBins],
    queryFn: () => fetchReliabilityDiagram(nBins),
    refetchInterval: METRICS_REFRESH_MS,
  });
}


export function useFeatureImportance(symbol: string) {
  return useQuery({
    queryKey: ["feature_importance", symbol],
    queryFn: () => fetchFeatureImportance(symbol),
    refetchInterval: METRICS_REFRESH_MS,
  });
}


export function useConditionalAccuracy() {
  return useQuery({
    queryKey: ["conditional_accuracy"],
    queryFn: () => fetchConditionalAccuracy(),
    refetchInterval: METRICS_REFRESH_MS,
  });
}


export function useCumulativePnL(symbol: string, limitPoints = 200) {
  return useQuery({
    queryKey: ["cumulative_pnl", symbol, limitPoints],
    queryFn: () => fetchCumulativePnL(symbol, limitPoints),
    refetchInterval: TRADES_REFRESH_MS,
  });
}


export function useForecastEnsemble(
  symbol: string,
  weight1m = 0.7,
  weight15m = 0.3,
) {
  return useQuery({
    queryKey: ["forecast_ensemble", symbol, weight1m, weight15m],
    queryFn: () => fetchForecastEnsemble(symbol, weight1m, weight15m),
    refetchInterval: PREDICT_REFRESH_MS,
  });
}


export function useRobustness(symbol: string) {
  return useQuery({
    queryKey: ["robustness", symbol],
    queryFn: () => fetchRobustness(symbol),
    refetchInterval: METRICS_REFRESH_MS,
  });
}


export function usePnLByFeeTier(symbol: string) {
  return useQuery({
    queryKey: ["pnl_by_fee_tier", symbol],
    queryFn: () => fetchPnLByFeeTier(symbol),
    refetchInterval: TRADES_REFRESH_MS,
  });
}


// ── Operator hooks ────────────────────────────────────────────────


/** Liveness ping — refreshes on the predict cadence so a stale ingest
 * surfaces within ~30 s. */
export function useHealth(symbol: string) {
  return useQuery({
    queryKey: ["health", symbol],
    queryFn: () => fetchHealth(symbol),
    refetchInterval: PREDICT_REFRESH_MS,
  });
}


export function useStatus(symbol: string) {
  return useQuery({
    queryKey: ["status", symbol],
    queryFn: () => fetchStatus(symbol),
    refetchInterval: PREDICT_REFRESH_MS,
  });
}


/** Training report polled at the slow cadence — only changes when the
 * trainer fires its daily 04:00 UTC timer. */
export function useTrainingReport(symbol: string, lite = 0) {
  return useQuery({
    queryKey: ["training_report", symbol, lite],
    queryFn: () => fetchTrainingReport(symbol, lite),
    refetchInterval: METRICS_REFRESH_MS,
  });
}


export function useAntiSkill(symbol: string) {
  return useQuery({
    queryKey: ["anti_skill", symbol],
    queryFn: () => fetchAntiSkill(symbol),
    refetchInterval: TRADES_REFRESH_MS,
  });
}


export function useTrainingHistory(symbol?: string, limit = 30) {
  return useQuery({
    queryKey: ["training_history", symbol ?? null, limit],
    queryFn: () => fetchTrainingHistory(symbol, limit),
    refetchInterval: METRICS_REFRESH_MS,
  });
}
