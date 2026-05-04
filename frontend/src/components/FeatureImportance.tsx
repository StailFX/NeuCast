"use client";

import { useFeatureImportance } from "@/lib/hooks";
import type { FeatureImportanceItem } from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
  topN?: number;
}

const SYMBOL_BAR_COLOR: Record<string, string> = {
  BTCUSDT: "bg-amber-400",
  ETHUSDT: "bg-violet-400",
  BNBUSDT: "bg-yellow-400",
};


/** Per-symbol feature importance from the loaded CatBoost model.
 *  Constant for a given .cbm — only changes when the model retrains.
 */
export function FeatureImportance({ symbols, topN = 10 }: Props) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Feature importance
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          CatBoost PredictionValuesChange · top {topN}
        </span>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        {symbols.map((sym) => (
          <PerSymbolPanel key={sym} symbol={sym} topN={topN} />
        ))}
      </div>
    </div>
  );
}


function PerSymbolPanel({ symbol, topN }: { symbol: string; topN: number }) {
  const { data, isLoading } = useFeatureImportance(symbol);
  const barColor = SYMBOL_BAR_COLOR[symbol] ?? "bg-zinc-500";

  let items: FeatureImportanceItem[] = [];
  if (data?.ok && data.importance) {
    items = [...data.importance]
      .sort((a, b) => b.importance - a.importance)
      .slice(0, topN);
  }
  const max = items[0]?.importance ?? 1;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="mb-2 text-sm font-semibold text-zinc-200">
        {symbol.replace("USDT", "")}
      </div>
      {isLoading ? (
        <ul className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <li key={i} className="flex items-center gap-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="ml-auto h-1.5 w-24" rounded="rounded-full" />
              <Skeleton className="h-3 w-8" />
            </li>
          ))}
        </ul>
      ) : items.length === 0 ? (
        <div className="text-xs text-zinc-500">данных нет</div>
      ) : (
        <ul className="space-y-1.5">
          {items.map((it) => {
            const widthPct = Math.max(2, (it.importance / max) * 100);
            return (
              <li
                key={it.feature}
                className="grid grid-cols-[1fr_auto] items-center gap-2"
                title={`${it.feature}: ${it.importance.toFixed(2)}`}
              >
                <div className="flex items-center gap-2">
                  <span className="truncate text-[0.7rem] text-zinc-400">
                    {it.feature}
                  </span>
                  <span className="ml-auto h-1.5 flex-shrink-0 w-24 overflow-hidden rounded-full bg-zinc-800">
                    <span
                      className={`block h-full ${barColor}`}
                      style={{ width: `${widthPct}%` }}
                    />
                  </span>
                </div>
                <span className="w-10 text-right text-[0.66rem] tabular-nums text-zinc-500">
                  {it.importance.toFixed(1)}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
