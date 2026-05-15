"use client";

import { usePaperTrades } from "@/lib/hooks";
import type { PaperTrade } from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

/**
 * FriendlyTradesFeed — last 10 closed paper trades, v1 landing style.
 *
 * Each row: side label (LONG/SHORT with arrow), entry → exit prices,
 * absolute P&L in USD, win/loss verdict, time-of-day.
 *
 * Open positions (no exit_ts) are hidden — only closed trades show.
 */

interface Props {
  symbol: string;
}

export function FriendlyTradesFeed({ symbol }: Props) {
  const { data, isLoading } = usePaperTrades(symbol, 30);
  const trades = (data?.trades ?? [])
    .filter((t) => t.exit_ts != null && t.pnl_usd != null)
    .slice(0, 10);

  return (
    <section aria-label="recent trades">
      <div className="nc-section-title">Недавние сделки</div>
      <div className="nc-card nc-trades">
        {isLoading ? (
          <div style={{ padding: "1rem 1.5rem" }}>
            {[0, 1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-7 w-full mb-2" />
            ))}
          </div>
        ) : trades.length === 0 ? (
          <div className="nc-trades-empty">
            Закрытых сделок пока нет — модель в режиме наблюдения.
          </div>
        ) : (
          <ul>
            {trades.map((t, i) => (
              <TradeRow key={t.id ?? `${t.entry_ts}-${i}`} trade={t} />
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

function TradeRow({ trade }: { trade: PaperTrade }) {
  const isLong = trade.side === "long";
  const isWin = (trade.pnl_usd ?? 0) > 0;
  const arrow = isLong ? "↑" : "↓";
  const sideLabel = isLong ? "LONG" : "SHORT";
  const pnlSign = isWin ? "+" : "−";
  const pnlAbs = Math.abs(trade.pnl_usd ?? 0).toFixed(2);
  // Clock-time formatting — HH:MM is enough (seconds visually noisy).
  const exitLabel = trade.exit_ts
    ? new Date(trade.exit_ts).toLocaleTimeString("ru-RU", {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";

  return (
    <li
      className="nc-trade-row"
      aria-label={`${sideLabel} trade, P&L ${pnlSign}$${pnlAbs}`}
    >
      <div className={`nc-side nc-${isLong ? "up" : "down"}`}>
        <span style={{ fontSize: "1.1rem", lineHeight: 1 }}>{arrow}</span>
        <span>{sideLabel}</span>
      </div>
      <div className="nc-prices">
        {trade.entry_price.toFixed(2)}
        <span className="nc-arrow">→</span>
        {(trade.exit_price ?? 0).toFixed(2)}
      </div>
      <div className={`nc-pnl ${isWin ? "nc-win" : "nc-loss"}`}>
        {pnlSign}${pnlAbs}
      </div>
      <div className="nc-trade-time">{exitLabel}</div>
    </li>
  );
}
