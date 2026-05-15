"use client";

import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { LiveHero } from "@/components/LiveHero";
import { SymbolPills } from "@/components/SymbolPills";
import { PerformanceTriad } from "@/components/PerformanceTriad";
import { FriendlyTradesFeed } from "@/components/FriendlyTradesFeed";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { useDashboard } from "@/lib/useDashboard";
import type { PerSymbolPayload } from "@/lib/api-types";

// landing-theme styles now live in app/globals.css and apply
// globally via body.nc-themed (set in layout.tsx).

/**
 * /v2/live — user-facing demo page, v1 landing visual style.
 *
 * Structure:
 *   • Page wrapper (.nc-landing) — sets palette, grid background,
 *     floating orbs decoration.
 *   • LandingNavbar — logo + nav buttons in v1 gradient style.
 *   • Hero — gradient h1 + sub paragraph.
 *   • LiveHero card — live price + 1-min prediction.
 *   • SymbolPills — pill-style symbol selector (BTC/ETH/BNB).
 *   • PerformanceTriad — 3-stat card (count / dir-acc / paper P&L).
 *   • FriendlyTradesFeed — last 10 paper trades, list format.
 *   • HowItWorks — short paragraph explaining methodology + link to
 *     /forecast research surface.
 *
 * All data flows through the existing useDashboard / usePaperTrades /
 * useRealizedAccuracy hooks; no new endpoints needed.
 */

type SupportedSymbol = "BTCUSDT" | "ETHUSDT" | "BNBUSDT";
const ALL_SYMBOLS: SupportedSymbol[] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"];

function emptySlice(symbol: string): PerSymbolPayload {
  return {
    symbol,
    forecast: { ok: false, reason: "loading" },
    drift: { ok: false, reason: "loading" },
    microprice: { ok: false, reason: "loading" },
  };
}

export default function LivePage() {
  const [symbol, setSymbol] = useState<SupportedSymbol>("BTCUSDT");
  const { data, isLoading, isError, dataUpdatedAt } = useDashboard(
    ALL_SYMBOLS,
    30_000,
  );

  const slice: PerSymbolPayload =
    data?.ok && data.symbols[symbol]
      ? data.symbols[symbol]
      : emptySlice(symbol);

  const statusPill = (
    <>
      {isLoading && "загрузка…"}
      {isError && <span className="nc-status-error">⚠ API недоступен</span>}
      {!isLoading && !isError && dataUpdatedAt > 0 && (
        <span>● {new Date(dataUpdatedAt).toLocaleTimeString("ru-RU")}</span>
      )}
    </>
  );

  return (
    <div className="min-h-screen">
      <Navbar rightSlot={statusPill} />

      <main className="nc-main">
        <section className="nc-hero">
          <div className="nc-hero-badge">
            <span className="nc-dot" />
            Real-time · Binance Spot · L2 microstructure
          </div>
          <h1 className="nc-h1">
            Прогноз цены <span className="nc-gradient">в реальном времени</span>
          </h1>
          <p className="nc-hero-sub">
            Направление движения цены на горизонте 1 минута на основе
            order-book микроструктуры Binance Spot. Инфраструктура в Tokyo,
            медианная задержка до биржи ~19 мс.
          </p>
        </section>

        <div className="nc-stack-lg">
          <ErrorBoundary label="LiveHero">
            <LiveHero
              symbol={symbol}
              forecast={slice.forecast}
              microprice={slice.microprice}
            />
          </ErrorBoundary>

          <SymbolPills
            value={symbol}
            onChange={(s) => setSymbol(s as SupportedSymbol)}
          />

          <ErrorBoundary label="PerformanceTriad">
            <PerformanceTriad symbol={symbol} />
          </ErrorBoundary>

          <ErrorBoundary label="FriendlyTradesFeed">
            <FriendlyTradesFeed symbol={symbol} />
          </ErrorBoundary>

          <HowItWorks />
        </div>
      </main>
    </div>
  );
}


function HowItWorks() {
  return (
    <section className="nc-card nc-how">
      <h3>Как это работает</h3>
      <p>
        Модель — gradient boosting (<code>CatBoost</code>) на признаках{" "}
        <code>order-flow imbalance</code>, <code>spread</code> и{" "}
        <code>depth</code> из L2 order book Binance Spot. Сервер в Tokyo даёт
        медианную задержку до биржи ~19 миллисекунд. Каждое предсказание —
        это калиброванная вероятность роста цены на горизонте 1 минута.
        Переобучение раз в сутки в 04:00 UTC; директорная точность
        пересчитывается на скользящем 24-часовом окне.
      </p>
      <p>
        Расширенная research-страница с calibration plot, robustness suite,
        feature importance и conformal-интервалами — на{" "}
        <a href="/forecast">/forecast</a>.
      </p>
    </section>
  );
}
