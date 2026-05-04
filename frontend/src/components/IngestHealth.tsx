"use client";

import { useHealth, useStatus } from "@/lib/hooks";
import { fmtAge } from "@/lib/format";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
}

/**
 * Live ingestor health: per-symbol liveness (rows in last 60s), last
 * row timestamp, freshness verdict. Mirrors the green/yellow/red dot
 * the operator-side legacy /highfreq page surfaces.
 */
export function IngestHealth({ symbols }: Props) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Ingest health
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          rows / 60 s · last-row freshness
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {symbols.map((s) => (
          <PerSymbolHealth key={s} symbol={s} />
        ))}
      </div>
    </div>
  );
}


function PerSymbolHealth({ symbol }: { symbol: string }) {
  const { data: health, isLoading: hLoading } = useHealth(symbol);
  const { data: status, isLoading: sLoading } = useStatus(symbol);
  const display = symbol.replace("USDT", "");

  const loading = hLoading || sLoading;
  const rows60 = health?.rows_last_60s ?? 0;
  const isFresh = status?.is_fresh ?? false;
  const freshnessSec = status?.freshness_seconds ?? null;

  // Verdict bucket → colour
  // ok: rows_last_60s ≥ 30 AND last-row < 15s
  // warn: rows ≥ 10 OR last-row 15-60s
  // bad: nothing
  let verdict: "ok" | "warn" | "bad";
  if (rows60 >= 30 && isFresh) verdict = "ok";
  else if (rows60 >= 10 || (freshnessSec != null && freshnessSec < 60))
    verdict = "warn";
  else verdict = "bad";

  const dotCls =
    verdict === "ok"
      ? "bg-emerald-400"
      : verdict === "warn"
      ? "bg-amber-400"
      : "bg-rose-400";

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-zinc-200">{display}</span>
        <span className="flex items-center gap-1.5">
          <span
            className={`inline-block h-2 w-2 rounded-full ${dotCls}`}
            aria-hidden
          />
          <span className="text-[0.62rem] uppercase tracking-wider text-zinc-500">
            {verdict}
          </span>
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-[0.7rem]">
        <div>
          <div className="text-zinc-500">rows / 60s</div>
          {loading ? (
            <Skeleton className="mt-1 h-4 w-12" />
          ) : (
            <div className="mt-1 text-base font-semibold tabular-nums text-zinc-200">
              {rows60}
            </div>
          )}
        </div>
        <div>
          <div className="text-zinc-500">last row</div>
          {loading ? (
            <Skeleton className="mt-1 h-4 w-16" />
          ) : (
            <div className="mt-1 text-base font-semibold tabular-nums text-zinc-200">
              {fmtAge(freshnessSec ?? undefined)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
