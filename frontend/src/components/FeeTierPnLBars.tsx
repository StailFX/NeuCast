"use client";

import { useState } from "react";
import { usePnLByFeeTier } from "@/lib/hooks";
import type { FeeTierSummary } from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
}

const TIER_LABEL: Record<string, string> = {
  gross: "Без комиссии",
  retail: "Spot retail",
  vip5: "Spot VIP-5",
  vip9: "Spot VIP-9",
  futures: "Futures maker",
  mm_rebate: "MM rebate",
};


/**
 * Per-fee-tier aggregate P&L for the active symbol.
 *
 * The headline finding from the legacy /forecast: at retail tier
 * the trader loses on every trade due to 15bp roundtrip fees on
 * 3-5bp moves; at MM rebate (-0.4bp) the same sequence is
 * positive. Shows that directional skill exists; profitability
 * is determined by fee tier (which scales with capital, not code).
 */
export function FeeTierPnLBars({ symbols }: Props) {
  const [active, setActive] = useState(symbols[0]);
  const { data, isLoading } = usePnLByFeeTier(active);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h2 className="text-xs uppercase tracking-wider text-zinc-500">
            P&amp;L by fee tier
          </h2>
          <p className="mt-1 text-[0.66rem] text-zinc-600">
            average per-trade bps after roundtrip fee at each tier
          </p>
        </div>
        <div className="flex gap-1 rounded-full bg-zinc-950 p-1">
          {symbols.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setActive(s)}
              className={`rounded-full px-3 py-1 text-[0.7rem] font-semibold transition ${
                s === active
                  ? "bg-zinc-100 text-zinc-900"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {s.replace("USDT", "")}
            </button>
          ))}
        </div>
      </div>
      {isLoading ? (
        <ul className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <li
              key={i}
              className="grid grid-cols-[140px_1fr_auto] items-center gap-3"
            >
              <div className="space-y-1">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-2 w-16" />
              </div>
              <Skeleton className="h-3 w-full" rounded="rounded-full" />
              <Skeleton className="h-3 w-16" />
            </li>
          ))}
        </ul>
      ) : !data?.ok ? (
        <div className="text-sm text-zinc-500">
          {data?.db_status === "unavailable"
            ? "DB временно недоступна"
            : data?.reason ?? "данных нет"}
        </div>
      ) : !data.tiers || data.tiers.length === 0 ? (
        <div className="text-sm text-zinc-500">
          нет закрытых сделок для этого символа
        </div>
      ) : (
        <TierBars tiers={data.tiers} />
      )}
    </div>
  );
}


function TierBars({ tiers }: { tiers: FeeTierSummary[] }) {
  const max = Math.max(
    ...tiers.map((t) => Math.abs(t.mean_bps)),
    1,
  );

  return (
    <ul className="space-y-2">
      {tiers.map((t) => {
        const widthPct = (Math.abs(t.mean_bps) / max) * 100;
        const positive = t.mean_bps > 0;
        const negative = t.mean_bps < 0;
        return (
          <li
            key={t.key}
            className="grid grid-cols-[140px_1fr_auto] items-center gap-3"
            title={`${TIER_LABEL[t.key] ?? t.key} · n=${t.n_trades} · win-rate=${
              t.win_rate != null ? (t.win_rate * 100).toFixed(1) + "%" : "—"
            }`}
          >
            <div>
              <div className="text-[0.7rem] text-zinc-300">
                {TIER_LABEL[t.key] ?? t.key}
              </div>
              <div className="text-[0.62rem] text-zinc-500">
                fee {t.fee_bps.toFixed(1)}bp · n={t.n_trades}
              </div>
            </div>
            {/* Bar — centred at zero, extends right (positive) or left (negative). */}
            <div className="relative h-3 rounded-full bg-zinc-900">
              <div className="absolute inset-y-0 left-1/2 w-px bg-zinc-700" />
              {positive && (
                <div
                  className="absolute inset-y-0 left-1/2 rounded-r-full bg-emerald-500/70"
                  style={{ width: `${widthPct / 2}%` }}
                />
              )}
              {negative && (
                <div
                  className="absolute inset-y-0 right-1/2 rounded-l-full bg-rose-500/70"
                  style={{ width: `${widthPct / 2}%` }}
                />
              )}
            </div>
            <div
              className={`w-20 text-right text-sm tabular-nums ${
                positive
                  ? "text-emerald-400"
                  : negative
                  ? "text-rose-400"
                  : "text-zinc-400"
              }`}
            >
              {positive ? "+" : ""}
              {t.mean_bps.toFixed(2)} bp
            </div>
          </li>
        );
      })}
    </ul>
  );
}
