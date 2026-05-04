"use client";

import type { PaperTrade } from "@/lib/api-types";
import { fmtMoney } from "@/lib/format";
import { Skeleton } from "./Skeleton";

const SIDE_RU: Record<string, string> = {
  long: "покупка",
  short: "продажа",
};

const SYMBOL_COLOR: Record<string, string> = {
  BTCUSDT: "bg-amber-500/20 text-amber-300",
  ETHUSDT: "bg-violet-500/20 text-violet-300",
  BNBUSDT: "bg-yellow-500/20 text-yellow-300",
};

interface Props {
  /** Already-fetched trades from one or more symbols. Page-level
   * component does the fetches via per-symbol hooks (rules-of-hooks
   * stays clean) and merges them here. */
  trades: PaperTrade[];
  isLoading?: boolean;
  limit?: number;
}


export function TradesFeed({ trades, isLoading, limit = 10 }: Props) {
  const closed = trades
    .filter((t) => t.exit_ts != null)
    .sort(
      (a, b) =>
        new Date(b.exit_ts!).getTime() - new Date(a.exit_ts!).getTime(),
    )
    .slice(0, limit);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Recent trades
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          last {Math.min(limit, closed.length)} closed
        </span>
      </div>
      {isLoading && !closed.length ? (
        <ul className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <li key={i} className="flex items-center gap-3 py-1.5">
              <Skeleton className="h-7 w-12" rounded="rounded-md" />
              <Skeleton className="h-3 w-20" />
              <Skeleton className="ml-auto h-3 w-14" />
            </li>
          ))}
        </ul>
      ) : !closed.length ? (
        <div className="text-sm text-zinc-500">
          пока не было закрытых сделок
        </div>
      ) : (
        <ul className="divide-y divide-zinc-800">
          {closed.map((t, i) => (
            <TradeRow
              key={t.id ?? `${t.symbol}-${t.exit_ts}-${i}`}
              trade={t}
            />
          ))}
        </ul>
      )}
    </div>
  );
}


function TradeRow({ trade: t }: { trade: PaperTrade }) {
  const sym = t.symbol.toUpperCase();
  const side = t.side ?? "long";
  const sideRu = SIDE_RU[side] ?? side;
  const pnlBps = Number(t.pnl_bps ?? 0);
  const pnlPct = pnlBps / 100;
  const pnlUsd = Number(t.pnl_usd ?? 0);
  const exitTs = t.exit_ts ? new Date(t.exit_ts) : null;
  const entry = Number(t.entry_price);
  const exit = Number(t.exit_price);
  const qty = Number(t.qty);
  const symCls = SYMBOL_COLOR[sym] ?? "bg-zinc-700/40 text-zinc-300";

  return (
    <li className="flex items-center gap-3 py-2.5 text-sm">
      <span
        className={`inline-flex h-7 min-w-12 items-center justify-center rounded-md px-2 text-[0.66rem] font-semibold ${symCls}`}
      >
        {sym.replace("USDT", "")}
      </span>
      <span className="w-20 truncate text-xs text-zinc-400">
        {side === "long" ? "🟢" : "🔴"} {sideRu}
      </span>
      <span className="hidden w-32 truncate text-xs tabular-nums text-zinc-500 sm:block">
        {Number.isFinite(qty) ? qty.toFixed(6) : "—"}
      </span>
      <span className="hidden w-32 truncate text-xs tabular-nums text-zinc-500 md:block">
        ${fmtMoney(entry)} → ${fmtMoney(exit)}
      </span>
      <span
        className={`ml-auto w-16 text-right tabular-nums ${
          pnlBps > 0
            ? "text-emerald-400"
            : pnlBps < 0
            ? "text-rose-400"
            : "text-zinc-400"
        }`}
      >
        {pnlBps > 0 ? "+" : ""}
        {pnlPct.toFixed(2)}%
      </span>
      <span
        className={`hidden w-20 text-right text-xs tabular-nums sm:block ${
          pnlUsd > 0
            ? "text-emerald-500"
            : pnlUsd < 0
            ? "text-rose-500"
            : "text-zinc-500"
        }`}
      >
        {pnlUsd > 0 ? "+" : ""}
        {pnlUsd.toFixed(3)}$
      </span>
      <span className="hidden w-20 text-right text-[0.66rem] text-zinc-500 lg:block">
        {exitTs?.toLocaleTimeString("ru-RU") ?? "—"}
      </span>
    </li>
  );
}
