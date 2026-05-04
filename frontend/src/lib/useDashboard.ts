"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchDashboard } from "./api";
import type { DashboardResponse } from "./api-types";

/**
 * Subscribe to the /api/highfreq/dashboard batch endpoint.
 *
 * Polls every 30s by default (the same cadence the legacy
 * `forecast.html` uses for `refreshAllPredictions`). React Query
 * dedupes parallel calls and caches the last successful result so
 * a flaky tick doesn't blank the UI.
 */
export function useDashboard(
  symbols: string[],
  refetchIntervalMs = 30_000,
) {
  const queryKey = ["dashboard", ...symbols];
  return useQuery<DashboardResponse>({
    queryKey,
    queryFn: () => fetchDashboard(symbols),
    refetchInterval: refetchIntervalMs,
  });
}
