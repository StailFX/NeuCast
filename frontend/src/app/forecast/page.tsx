"use client";

import { Navbar } from "@/components/Navbar";
import { ForecastCard } from "@/components/ForecastCard";
import { StatsStrip } from "@/components/StatsStrip";
import { TradesFeed } from "@/components/TradesFeed";
import { ReliabilityDiagram } from "@/components/ReliabilityDiagram";
import { FeatureImportance } from "@/components/FeatureImportance";
import { ConditionalAccuracy } from "@/components/ConditionalAccuracy";
import { CumulativePnL } from "@/components/CumulativePnL";
import { RobustnessSuite } from "@/components/RobustnessSuite";
import { FeeTierPnLBars } from "@/components/FeeTierPnLBars";
import { HorizonPill } from "@/components/HorizonPill";
import { useDashboard } from "@/lib/useDashboard";
import { usePaperTrades } from "@/lib/hooks";
import type { PerSymbolPayload, PaperTrade } from "@/lib/api-types";

const SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"] as const;


function emptySlice(symbol: string): PerSymbolPayload {
  return {
    symbol,
    forecast: { ok: false, reason: "loading" },
    drift: { ok: false, reason: "loading" },
    microprice: { ok: false, reason: "loading" },
  };
}


export default function ForecastPage() {
  const { data, isLoading, isError, dataUpdatedAt } = useDashboard(
    [...SYMBOLS],
    30_000,
  );

  // Trades — fetched per-symbol at the page level (rules-of-hooks
  // stays clean since SYMBOLS length is fixed at compile time).
  const btcTrades = usePaperTrades("BTCUSDT", 80);
  const ethTrades = usePaperTrades("ETHUSDT", 80);
  const bnbTrades = usePaperTrades("BNBUSDT", 80);
  const allTrades: PaperTrade[] = [
    ...(btcTrades.data?.trades ?? []),
    ...(ethTrades.data?.trades ?? []),
    ...(bnbTrades.data?.trades ?? []),
  ];
  const tradesLoading =
    btcTrades.isLoading && ethTrades.isLoading && bnbTrades.isLoading;

  const slices: Record<string, PerSymbolPayload> = {};
  for (const sym of SYMBOLS) {
    slices[sym] =
      data?.ok && data.symbols[sym] ? data.symbols[sym] : emptySlice(sym);
  }

  // Status pill for the navbar — "обновлено N сек назад" or
  // "ошибка соединения" — gives the operator a glance-able liveness
  // signal without scrolling to the page header.
  const statusPill = (
    <span className="text-[0.66rem] tabular-nums text-zinc-500">
      {isLoading && "загрузка…"}
      {isError && (
        <span className="text-rose-400">⚠ API недоступен</span>
      )}
      {!isLoading && !isError && dataUpdatedAt > 0 && (
        <span>
          ●{" "}
          <span className="text-zinc-400">
            {new Date(dataUpdatedAt).toLocaleTimeString("ru-RU")}
          </span>
        </span>
      )}
    </span>
  );

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <Navbar rightSlot={statusPill} />
      <div className="mx-auto max-w-6xl space-y-6 px-6 py-10">
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
              Forecast
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-zinc-400">
              Live 1-minute directional forecast on Binance Spot L2
              microstructure features. Predictions tick every 30 s; drift
              and reliability metrics on a slower cadence.
            </p>
          </div>
          <HorizonPill />
        </header>

        {/* Headline prediction cards */}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {SYMBOLS.map((sym) => {
            const slice = slices[sym];
            return (
              <ForecastCard
                key={sym}
                symbol={sym}
                forecast={slice.forecast}
                drift={slice.drift}
                microprice={slice.microprice}
              />
            );
          })}
        </div>

        {/* Aggregated 24h stats — across all 3 symbols at a glance. */}
        <StatsStrip symbols={SYMBOLS} />

        {/* Recent trades + cumulative P&L side-by-side */}
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <CumulativePnL symbols={SYMBOLS} />
          </div>
          <TradesFeed trades={allTrades} isLoading={tradesLoading} />
        </div>

        {/* Fee-tier P&L breakdown — defence headline */}
        <FeeTierPnLBars symbols={SYMBOLS} />

        {/* Conditional accuracy — by confidence threshold */}
        <ConditionalAccuracy />

        {/* Robustness suite — block-bootstrap, permutation, per-day */}
        <RobustnessSuite symbols={SYMBOLS} />

        {/* Reliability diagram — calibration plot */}
        <ReliabilityDiagram />

        {/* Feature importance — interpretability */}
        <FeatureImportance symbols={SYMBOLS} />

        <footer className="pt-4 text-xs text-zinc-600">
          <span>
            Headline endpoint:{" "}
            <code className="font-mono">/api/highfreq/dashboard</code>
            {" "}— code-review H-1perf (2026-05-04). 9 → 1 request per tick.
          </span>
        </footer>
      </div>
    </div>
  );
}
