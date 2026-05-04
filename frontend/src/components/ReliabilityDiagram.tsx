"use client";

import { useReliabilityDiagram } from "@/lib/hooks";
import type { ReliabilitySymbolRow } from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

const SYMBOL_PALETTE: Record<string, string> = {
  BTCUSDT: "#fbbf24", // amber
  ETHUSDT: "#a78bfa", // violet
  BNBUSDT: "#f59e0b", // yellow
};


/** Pure SVG reliability diagram — predicted-bin centre vs realized
 * up-rate, with the perfect-calibration diagonal y=x.
 *
 * Defence-grade plot: a model with high dir_acc but a hump-shaped
 * curve is overconfident; one with low dir_acc but on-diagonal is
 * honestly calibrated. We surface both Brier and ECE per symbol
 * so the per-card single-number-summary lives next to the plot.
 */
export function ReliabilityDiagram() {
  const { data, isLoading } = useReliabilityDiagram(10);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Reliability diagram
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          predicted bin → realized rate · diag = perfect calibration
        </span>
      </div>
      {isLoading && (
        <div className="grid gap-4 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton
              key={i}
              className="h-[230px] w-full"
              rounded="rounded-xl"
            />
          ))}
        </div>
      )}
      {!isLoading && data?.ok && data.rows && data.rows.length > 0 ? (
        <div className="grid gap-4 lg:grid-cols-3">
          {data.rows.map((row) => (
            <PerSymbolPlot key={row.symbol} row={row} />
          ))}
        </div>
      ) : !isLoading && !data?.ok ? (
        <div className="text-sm text-zinc-500">
          {data?.reason === "no_predictions_yet"
            ? "не накоплено предсказаний"
            : "данных нет"}
        </div>
      ) : null}
    </div>
  );
}


function PerSymbolPlot({ row }: { row: ReliabilitySymbolRow }) {
  const W = 280;
  const H = 200;
  const PADDING = 28;
  const innerW = W - 2 * PADDING;
  const innerH = H - 2 * PADDING;
  const color = SYMBOL_PALETTE[row.symbol] ?? "#52525b";

  // Map a logical [0,1]² point into SVG coords.
  const xToPx = (x: number) => PADDING + x * innerW;
  const yToPx = (y: number) => PADDING + (1 - y) * innerH;

  // Diagonal points + bin scatter
  const diagPath = `M ${xToPx(0)} ${yToPx(0)} L ${xToPx(1)} ${yToPx(1)}`;
  const points = row.bins.filter(
    (b) => b.realized_rate != null && b.n > 0,
  );

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="text-sm font-semibold text-zinc-200">
          {row.symbol.replace("USDT", "")}
        </div>
        <div className="text-[0.66rem] tabular-nums text-zinc-500">
          n={row.n_total}
          {row.brier != null && (
            <> · Brier {row.brier.toFixed(3)}</>
          )}
          {row.ece != null && <> · ECE {row.ece.toFixed(3)}</>}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full text-zinc-700"
        role="img"
        aria-label={`Reliability diagram for ${row.symbol}`}
      >
        {/* Frame */}
        <rect
          x={PADDING}
          y={PADDING}
          width={innerW}
          height={innerH}
          fill="none"
          stroke="currentColor"
          strokeWidth={0.5}
        />
        {/* Diagonal */}
        <path
          d={diagPath}
          stroke="currentColor"
          strokeDasharray="4 4"
          strokeWidth={1}
          fill="none"
          opacity={0.5}
        />
        {/* Connector polyline through bins (visual aid) */}
        {points.length > 1 && (
          <polyline
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            opacity={0.6}
            points={points
              .map((b) => `${xToPx(b.bin_mid)},${yToPx(b.realized_rate!)}`)
              .join(" ")}
          />
        )}
        {/* Bin dots, radius proportional to log(n) */}
        {points.map((b, i) => {
          const r = Math.max(2.5, Math.min(7, Math.log10(Math.max(b.n, 1)) * 2.5));
          return (
            <circle
              key={i}
              cx={xToPx(b.bin_mid)}
              cy={yToPx(b.realized_rate!)}
              r={r}
              fill={color}
              fillOpacity={0.85}
              stroke="#0a0a0a"
              strokeWidth={1}
            >
              <title>
                bin {b.bin_lo.toFixed(2)}–{b.bin_hi.toFixed(2)} ·
                n={b.n} · realized={b.realized_rate!.toFixed(3)}
              </title>
            </circle>
          );
        })}
        {/* Axis labels */}
        <text
          x={PADDING + innerW / 2}
          y={H - 6}
          textAnchor="middle"
          className="fill-zinc-600 text-[10px]"
        >
          predicted prob_up
        </text>
        <text
          x={10}
          y={PADDING + innerH / 2}
          textAnchor="middle"
          transform={`rotate(-90 10 ${PADDING + innerH / 2})`}
          className="fill-zinc-600 text-[10px]"
        >
          realized
        </text>
      </svg>
    </div>
  );
}
