"use client";

import { Navbar } from "@/components/Navbar";
import { ForecastCard } from "@/components/ForecastCard";
import { IngestHealth } from "@/components/IngestHealth";
import { TrainingReport } from "@/components/TrainingReport";
import { AntiSkill } from "@/components/AntiSkill";
import { TrainingHistory } from "@/components/TrainingHistory";
import { StatsStrip } from "@/components/StatsStrip";
import { ConditionalAccuracy } from "@/components/ConditionalAccuracy";
import { ErrorBoundary } from "@/components/ErrorBoundary";
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
    <div className="min-h-screen">
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

        {/* Each operator block wrapped in ErrorBoundary so a single
            crashing widget doesn't blank the whole dashboard. The
            chip surfaces the offending component name + JS error. */}

        {/* Ingest health — operator's first glance */}
        <ErrorBoundary label="IngestHealth">
          <IngestHealth symbols={SYMBOLS} />
        </ErrorBoundary>

        {/* Anti-skill — second-most-important alert */}
        <ErrorBoundary label="AntiSkill">
          <AntiSkill symbols={SYMBOLS} />
        </ErrorBoundary>

        {/* Prediction cards (live) */}
        <ErrorBoundary label="ForecastCards">
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
        </ErrorBoundary>

        {/* 24h stats — same as /forecast */}
        <ErrorBoundary label="StatsStrip">
          <StatsStrip symbols={SYMBOLS} />
        </ErrorBoundary>

        {/* Training report — fold readiness + last walk-forward CI */}
        <ErrorBoundary label="TrainingReport">
          <TrainingReport symbols={SYMBOLS} />
        </ErrorBoundary>

        {/* Conditional accuracy — by confidence threshold */}
        <ErrorBoundary label="ConditionalAccuracy">
          <ConditionalAccuracy />
        </ErrorBoundary>

        {/* Training history — live auto-scoreboard */}
        <ErrorBoundary label="TrainingHistory">
          <TrainingHistory />
        </ErrorBoundary>

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
