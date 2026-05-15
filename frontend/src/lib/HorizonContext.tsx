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

/**
 * Which horizons have a trained CatBoost weight file on disk
 * (``weights/highfreq/{symbol}_{H}m.cbm`` on Tokyo). Sync with
 * HORIZON_AVAILABLE in HorizonPill.tsx.
 *
 * 1h is intentionally absent — no .cbm yet; it stays a "скоро"
 * placeholder until the diploma-scope long-horizon trainer ships.
 */
const HORIZON_TRAINED: Set<Horizon> = new Set([1, 5, 15]);

interface Ctx {
  horizon: Horizon;
  setHorizon: (h: Horizon) => void;
}

const HorizonCtx = createContext<Ctx | null>(null);


export function HorizonProvider({ children }: { children: ReactNode }) {
  const [horizon, setHorizonRaw] = useState<Horizon>(1);
  // Guard rail — refuse to switch to an untrained horizon (e.g. ``60``
  // from a stale URL state or hot-reloaded module). Silently falls
  // back to 1 so the user always sees a valid forecast.
  const setHorizon = useMemo(
    () => (h: Horizon) => {
      setHorizonRaw(HORIZON_TRAINED.has(h) ? h : 1);
    },
    [],
  );
  const value = useMemo(() => ({ horizon, setHorizon }), [horizon, setHorizon]);
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
