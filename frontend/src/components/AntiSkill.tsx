"use client";

import { useAntiSkill } from "@/lib/hooks";
import { fmtPercent } from "@/lib/format";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
}


/**
 * Anti-skill detector — surfaces when the rolling-window winrate's
 * 95% bootstrap CI's UPPER bound is below 0.5. That's the signal
 * that the model is consistently directional-WRONG (the inverse
 * trade would have profited). Operator response policies:
 * alert / halt / invert.
 */
export function AntiSkill({ symbols }: Props) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Anti-skill detector
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          rolling winrate CI &lt; 0.5 → halt or invert
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {symbols.map((s) => (
          <PerSymbol key={s} symbol={s} />
        ))}
      </div>
    </div>
  );
}


function PerSymbol({ symbol }: { symbol: string }) {
  const { data, isLoading } = useAntiSkill(symbol);
  const display = symbol.replace("USDT", "");

  if (isLoading) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
        <div className="mb-2 text-sm font-semibold text-zinc-200">
          {display}
        </div>
        <Skeleton className="h-3 w-full" />
      </div>
    );
  }

  if (!data?.ok) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3 text-xs text-zinc-500">
        <div className="mb-1 text-sm font-semibold text-zinc-300">
          {display}
        </div>
        {data?.reason === "no_trades_in_window"
          ? "недостаточно сделок"
          : data?.reason ?? "нет данных"}
      </div>
    );
  }

  const isAnti = !!data.is_anti_skilled;
  const wr = data.gross_winrate;
  const ciHigh = data.ci_high;
  const dotCls = isAnti ? "bg-rose-400" : "bg-emerald-400";

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-zinc-200">{display}</span>
        <span className="flex items-center gap-1.5">
          <span
            className={`inline-block h-2 w-2 rounded-full ${dotCls}`}
            aria-hidden
          />
          <span
            className={`text-[0.62rem] uppercase tracking-wider ${
              isAnti ? "text-rose-300" : "text-emerald-300"
            }`}
          >
            {isAnti ? "ANTI-SKILL" : "OK"}
          </span>
        </span>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2 text-[0.7rem]">
        <div>
          <div className="text-zinc-500">winrate</div>
          <div className="mt-0.5 text-base font-semibold tabular-nums text-zinc-200">
            {wr != null ? fmtPercent(wr, 1) : "—"}
          </div>
        </div>
        <div>
          <div className="text-zinc-500">CI_high</div>
          <div className="mt-0.5 text-base font-semibold tabular-nums text-zinc-200">
            {ciHigh != null ? fmtPercent(ciHigh, 1) : "—"}
          </div>
        </div>
      </div>
      {data.n_trades_in_window != null && (
        <div className="mt-2 text-[0.62rem] tabular-nums text-zinc-500">
          n = {data.n_trades_in_window}{" "}
          {data.policy && (
            <span className="ml-1 rounded-full bg-zinc-800 px-1.5 py-0.5 text-[0.55rem] uppercase tracking-wider text-zinc-400">
              policy: {data.policy}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
