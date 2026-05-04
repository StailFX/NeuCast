"use client";

import { HORIZON_LABELS, useHorizon, type Horizon } from "@/lib/HorizonContext";

const HORIZONS: Horizon[] = [1, 5, 15, 60];


/**
 * Top-of-page horizon switcher (1m / 5m / 15m / 1h). Drives the
 * ``HorizonContext`` so any component that subscribes via
 * ``useHorizon()`` re-renders with the new horizon.
 *
 * Today (2026-05-04): only 1m has a hot dashboard endpoint
 * (``/api/highfreq/dashboard``); other horizons fall back to the
 * legacy per-symbol /forecast?horizon=N path. Components that don't
 * support a horizon other than 1m grey out their content gracefully.
 */
export function HorizonPill() {
  const { horizon, setHorizon } = useHorizon();
  return (
    <div className="inline-flex gap-1 rounded-full bg-zinc-950 p-1">
      {HORIZONS.map((h) => (
        <button
          key={h}
          type="button"
          onClick={() => setHorizon(h)}
          className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
            h === horizon
              ? "bg-zinc-100 text-zinc-900"
              : "text-zinc-400 hover:text-zinc-200"
          }`}
          title={`Horizon: ${HORIZON_LABELS[h]}`}
        >
          {HORIZON_LABELS[h]}
        </button>
      ))}
    </div>
  );
}
