"use client";

import { useMemo } from "react";
import { usePaperTrades, useRealizedAccuracy } from "@/lib/hooks";
import { Skeleton } from "./Skeleton";

/**
 * PerformanceTriad — three stat tiles in v1 landing style.
 *
 * Layout: three colored numeric values (blue / green-or-red / green-or-red)
 * inside a single .nc-card surface, with .nc-stat dividers.
 *
 * Sources:
 *   • realized_accuracy → n_trades_24h, dir_acc_24h
 *   • paper_trades(200) → 24h pnl_usd sum (client-side aggregation
 *     since FastAPI doesn't expose a 24h-scoped P&L endpoint).
 */

interface Props {
  symbol: string;
}

export function PerformanceTriad({ symbol }: Props) {
  const { data: realized, isLoading: realizedLoading } =
    useRealizedAccuracy(symbol);
  const { data: tradesData, isLoading: tradesLoading } = usePaperTrades(
    symbol,
    200,
  );

  const stats = useMemo(() => {
    const trades = tradesData?.trades ?? [];
    const recent = trades.filter(
      (t) => t.exit_ts != null && t.pnl_usd != null,
    );
    const pnlSum = recent.reduce((acc, t) => acc + (t.pnl_usd ?? 0), 0);
    const wins = recent.filter((t) => (t.pnl_usd ?? 0) > 0).length;
    // Server actually returns ``accuracy`` + ``n_trades_total`` on a
    // rolling 100-trade window (NOT a literal 24h window — see
    // app/highfreq/realized_accuracy.py). Map them to the legacy
    // ``dir_acc_24h`` / ``n_trades_24h`` slots the UI was wired for.
    const r = realized as unknown as {
      accuracy?: number;
      n_trades_total?: number;
      n_correct?: number;
      ok?: boolean;
      dir_acc_24h?: number;
      n_trades_24h?: number;
    } | undefined;
    const apiN = r?.n_trades_total ?? r?.n_trades_24h;
    const apiAcc = r?.accuracy ?? r?.dir_acc_24h;
    const n =
      apiN != null && apiN > 0
        ? apiN
        : recent.length > 0
          ? recent.length
          : 0;
    const accuracy =
      apiAcc != null
        ? apiAcc
        : recent.length > 0
          ? wins / recent.length
          : null;
    return {
      n,
      accuracy,
      pnl: pnlSum,
      hasTrades: recent.length > 0,
    };
  }, [tradesData, realized]);

  const loading = realizedLoading && tradesLoading;

  return (
    <section aria-label="performance 24h">
      <div className="nc-section-title">За последние 24 часа</div>
      <div className="nc-card" style={{ padding: 0 }}>
        <div className="nc-stats">
          <StatTile
            label="предсказаний"
            value={loading ? null : String(stats.n)}
            color="blue"
            loading={loading}
          />
          <StatTile
            label="точность направления"
            value={
              loading || stats.accuracy == null
                ? null
                : `${(stats.accuracy * 100).toFixed(1)}%`
            }
            color={
              stats.accuracy == null
                ? "neutral"
                : stats.accuracy >= 0.5
                  ? "green"
                  : "red"
            }
            loading={loading}
          />
          <StatTile
            label="paper P&L"
            value={
              loading
                ? null
                : stats.hasTrades
                  ? fmtUsdSigned(stats.pnl)
                  : "—"
            }
            color={
              !stats.hasTrades
                ? "neutral"
                : stats.pnl > 0
                  ? "green"
                  : stats.pnl < 0
                    ? "red"
                    : "amber"
            }
            loading={loading}
          />
        </div>
      </div>
    </section>
  );
}

function fmtUsdSigned(n: number): string {
  if (Math.abs(n) < 0.005) return "$0.00";
  const sign = n > 0 ? "+" : "−";
  return `${sign}$${Math.abs(n).toFixed(2)}`;
}

type StatColor = "green" | "blue" | "amber" | "purple" | "red" | "neutral";

function StatTile({
  label,
  value,
  color,
  loading,
}: {
  label: string;
  value: string | null;
  color: StatColor;
  loading: boolean;
}) {
  // Render priority:
  //   1. loading → pulsing skeleton (we genuinely don't know yet)
  //   2. value == null AND not loading → quiet "—" (API answered with
  //      no data; a stuck skeleton was confusing the user)
  //   3. value present → coloured stat
  let body: React.ReactNode;
  if (loading) {
    body = <Skeleton className="h-9 w-24" />;
  } else if (value == null) {
    body = (
      <div className="nc-stat-value nc-neutral" style={{ opacity: 0.5 }}>
        —
      </div>
    );
  } else {
    body = <div className={`nc-stat-value nc-${color}`}>{value}</div>;
  }
  return (
    <div className="nc-stat">
      {body}
      <div className="nc-stat-label">{label}</div>
    </div>
  );
}
