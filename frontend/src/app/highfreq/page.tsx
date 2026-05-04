"use client";

import { Navbar } from "@/components/Navbar";
import { ForecastCard } from "@/components/ForecastCard";
import { IngestHealth } from "@/components/IngestHealth";
import { TrainingReport } from "@/components/TrainingReport";
import { AntiSkill } from "@/components/AntiSkill";
import { TrainingHistory } from "@/components/TrainingHistory";
import { StatsStrip } from "@/components/StatsStrip";
import { ConditionalAccuracy } from "@/components/ConditionalAccuracy";
import { useDashboard } from "@/lib/useDashboard";
import type { PerSymbolPayload } from "@/lib/api-types";

const SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"] as const;


function emptySlice(symbol: string): PerSymbolPayload {
  return {
    symbol,
    forecast: { ok: false, reason: "loading" },
    drift: { ok: false, reason: "loading" },
    microprice: { ok: false, reason: "loading" },
  };
}


/**
 * Operator dashboard — a denser view than /forecast, exposing the
 * health-of-the-system signals the operator needs at a glance:
 *
 *  • Ingest liveness (rows/60s + last-row freshness per symbol)
 *  • Training report + fold readiness
 *  • Anti-skill detector
 *  • Training history (auto-regen from training_runs table)
 *  • Plus the same prediction cards / stats / conditional accuracy
 *    that /forecast uses, so the operator can hop here without
 *    losing context.
 */
export default function HighfreqPage() {
  const { data, isLoading, isError, dataUpdatedAt } = useDashboard(
    [...SYMBOLS],
    30_000,
  );

  const slices: Record<string, PerSymbolPayload> = {};
  for (const sym of SYMBOLS) {
    slices[sym] =
      data?.ok && data.symbols[sym] ? data.symbols[sym] : emptySlice(sym);
  }

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
        <header>
          <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
            Operator dashboard
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            Denser-than-/forecast view of the HF stack: ingest health,
            training report, anti-skill detection, and the running
            history of every trainer fire.
          </p>
        </header>

        {/* Ingest health — operator's first glance */}
        <IngestHealth symbols={SYMBOLS} />

        {/* Anti-skill — second-most-important alert */}
        <AntiSkill symbols={SYMBOLS} />

        {/* Prediction cards (live) */}
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

        {/* 24h stats — same as /forecast */}
        <StatsStrip symbols={SYMBOLS} />

        {/* Training report — fold readiness + last walk-forward CI */}
        <TrainingReport symbols={SYMBOLS} />

        {/* Conditional accuracy — by confidence threshold */}
        <ConditionalAccuracy />

        {/* Training history — live auto-scoreboard */}
        <TrainingHistory />

        <footer className="pt-4 text-xs text-zinc-600">
          <span>
            Operator-facing surface · all data through the same
            FastAPI / WireGuard tunnel as the public /forecast page.
          </span>
        </footer>
      </div>
    </div>
  );
}
