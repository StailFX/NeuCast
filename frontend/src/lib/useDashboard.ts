"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchDashboard } from "./api";
import type { DashboardResponse } from "./api-types";
import { useHorizon } from "./HorizonContext";

/**
 * Subscribe to the /api/highfreq/dashboard batch endpoint.
 *
 * Polls every 30s by default. React Query dedupes parallel calls and
 * caches the last successful result so a flaky tick doesn't blank
 * the UI. Re-fetches when the active horizon flips so cards update
 * with the horizon-N forecast (5m/15m models trained on disk).
 */
export function useDashboard(
  symbols: string[],
  refetchIntervalMs = 30_000,
) {
  const { horizon } = useHorizon();
  const queryKey = ["dashboard", horizon, ...symbols];
  return useQuery<DashboardResponse>({
    queryKey,
    queryFn: () => fetchDashboard(symbols, horizon),
    refetchInterval: refetchIntervalMs,
  });
}
