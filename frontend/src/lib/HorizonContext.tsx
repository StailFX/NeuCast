"use client";

import {
  createContext,
  ReactNode,
  useContext,
  useMemo,
  useState,
} from "react";

export type Horizon = 1 | 5 | 15 | 60;

export const HORIZON_LABELS: Record<Horizon, string> = {
  1: "1m",
  5: "5m",
  15: "15m",
  60: "1h",
};

interface Ctx {
  horizon: Horizon;
  setHorizon: (h: Horizon) => void;
}

const HorizonCtx = createContext<Ctx | null>(null);


export function HorizonProvider({ children }: { children: ReactNode }) {
  const [horizon, setHorizon] = useState<Horizon>(1);
  const value = useMemo(() => ({ horizon, setHorizon }), [horizon]);
  return <HorizonCtx.Provider value={value}>{children}</HorizonCtx.Provider>;
}


/**
 * Subscribe to the active horizon.
 *
 * Components that genuinely change behaviour with horizon (predictions,
 * trade-filters, fee-tier aggregates) call ``useHorizon()``. Slow-moving
 * blocks (reliability diagram, feature importance) ignore it — they
 * already report aggregated stats and don't need re-segmentation.
 */
export function useHorizon(): Ctx {
  const ctx = useContext(HorizonCtx);
  if (!ctx) {
    // Default fallback for tests / usage outside the provider.
    return { horizon: 1, setHorizon: () => {} };
  }
  return ctx;
}
