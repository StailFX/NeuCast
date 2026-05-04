"use client";

import { useRealizedAccuracy, usePaperTrades } from "@/lib/hooks";
import { fmtPercent } from "@/lib/format";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
}

/**
 * Headline stats row across symbols — 24-hour rolling direction
 * accuracy (with bootstrap CI), trade count, mean gross P&L bps.
 *
 * Mirrors the "stats strip" block in legacy templates/forecast.html
 * but pulls the same data via React Query so it dedupes / caches /
 * polls automatically. One row per symbol.
 */
export function StatsStrip({ symbols }: Props) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <h2 className="mb-3 text-xs uppercase tracking-wider text-zinc-500">
        24h direction accuracy
      </h2>
      <div className="grid gap-4 sm:grid-cols-3">
        {symbols.map((sym) => (
          <SymbolStat key={sym} symbol={sym} />
        ))}
      </div>
    </div>
  );
}


function SymbolStat({ symbol }: { symbol: string }) {
  const { data: acc, isLoading: accLoading } = useRealizedAccuracy(symbol);
  const { data: trades, isLoading: tradesLoading } = usePaperTrades(symbol, 100);

  // 24h dir_acc: prefer the server's `dir_acc_24h` if present,
  // else compute client-side from the trades feed for robustness.
  let dirAcc: number | null = null;
  let nDir = 0;
  let ciLow: number | null = null;
  let ciHigh: number | null = null;
  if (acc?.ok) {
    dirAcc = acc.dir_acc_24h ?? null;
    nDir = acc.n_directional_24h ?? 0;
    ciLow = acc.ci_low_24h ?? null;
    ciHigh = acc.ci_high_24h ?? null;
  }

  // Mean gross P&L (bps) from the trades feed — non-fee-adjusted so
  // it reflects model quality, not fee tier.
  let meanGrossBps: number | null = null;
  let nTrades24h = 0;
  if (trades?.trades?.length) {
    const cutoff = Date.now() - 24 * 3600_000;
    const recent = trades.trades.filter(
      (t) =>
        t.exit_ts &&
        new Date(t.exit_ts).getTime() >= cutoff &&
        t.entry_price &&
        t.exit_price &&
        t.entry_price > 0,
    );
    nTrades24h = recent.length;
    if (recent.length) {
      const total = recent.reduce((acc, t) => {
        const e = t.entry_price as number;
        const x = t.exit_price as number;
        const move = (x - e) / e;
        const bps = (t.side === "long" ? move : -move) * 1e4;
        return acc + bps;
      }, 0);
      meanGrossBps = total / recent.length;
    }
  }

  const display = symbol.replace("USDT", "");
  const loading = accLoading && tradesLoading;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 px-4 py-3">
      <div className="flex items-baseline justify-between">
        <div className="text-sm font-semibold text-zinc-200">{display}</div>
        <div className="text-[0.62rem] uppercase tracking-wider text-zinc-500">
          {nTrades24h} trades / 24h
        </div>
      </div>
      <div className="mt-2 flex items-baseline gap-3">
        {loading ? (
          <Skeleton className="h-7 w-24" />
        ) : dirAcc != null ? (
          <>
            <span
              className={`text-2xl font-semibold tabular-nums ${
                dirAcc >= 0.55
                  ? "text-emerald-400"
                  : dirAcc >= 0.5
                  ? "text-zinc-200"
                  : "text-rose-400"
              }`}
            >
              {fmtPercent(dirAcc, 1)}
            </span>
            <span className="text-xs text-zinc-500">
              dir-acc · n = {nDir}
            </span>
          </>
        ) : (
          <span className="text-base text-zinc-600">no closed trades yet</span>
        )}
      </div>
      {ciLow != null && ciHigh != null && (
        <div className="mt-1 text-[0.7rem] tabular-nums text-zinc-500">
          95% CI [{fmtPercent(ciLow, 1)}, {fmtPercent(ciHigh, 1)}]
        </div>
      )}
      {meanGrossBps != null && (
        <div className="mt-1 text-[0.7rem] tabular-nums text-zinc-500">
          ⌀ gross{" "}
          <span
            className={
              meanGrossBps > 0
                ? "text-emerald-400"
                : meanGrossBps < 0
                ? "text-rose-400"
                : "text-zinc-400"
            }
          >
            {meanGrossBps > 0 ? "+" : ""}
            {meanGrossBps.toFixed(2)} bp
          </span>
        </div>
      )}
    </div>
  );
}
