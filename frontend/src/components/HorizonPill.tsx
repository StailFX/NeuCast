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
/**
 * 2026-05-15: 1m / 5m / 15m have trained CatBoost models per symbol
 * (``weights/highfreq/{symbol}_{horizon}m.cbm`` on Tokyo). 60m has
 * only a metrics.json — no .cbm — so it stays disabled with the
 * "скоро" badge until a 1h trainer ships in the diploma scope.
 */
const HORIZON_AVAILABLE: Record<Horizon, boolean> = {
  1: true,
  5: true,
  15: true,
  60: false,
};

export function HorizonPill() {
  const { horizon, setHorizon } = useHorizon();
  return (
    <div className="inline-flex gap-1 rounded-full bg-zinc-950 p-1">
      {HORIZONS.map((h) => {
        const available = HORIZON_AVAILABLE[h];
        const active = h === horizon;
        return (
          <button
            key={h}
            type="button"
            disabled={!available}
            onClick={() => {
              if (available) setHorizon(h);
            }}
            className={`relative rounded-full px-3 py-1.5 text-xs font-semibold transition ${
              !available
                ? "cursor-not-allowed text-zinc-600"
                : active
                  ? "bg-zinc-100 text-zinc-900"
                  : "text-zinc-400 hover:text-zinc-200"
            }`}
            title={
              available
                ? `Horizon: ${HORIZON_LABELS[h]}`
                : `${HORIZON_LABELS[h]} — скоро (требует отдельной модели; задача ВКР)`
            }
          >
            {HORIZON_LABELS[h]}
            {!available && (
              <span
                aria-hidden
                className="absolute -right-1 -top-1 rounded-full bg-amber-500/20 px-1 text-[0.55rem] font-medium text-amber-300"
                style={{ lineHeight: "1" }}
              >
                скоро
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
