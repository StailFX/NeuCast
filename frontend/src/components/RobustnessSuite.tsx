"use client";

import { useRobustness } from "@/lib/hooks";
import { fmtPercent } from "@/lib/format";
import type {
  PerDayPoint,
  PerHourPoint,
} from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
}

/**
 * Defence-grade block — closes the gaps the trainer's i.i.d.
 * bootstrap leaves open:
 *  • Block-bootstrap CI (60-min blocks, Politis-Romano) for proper
 *    AR-aware uncertainty.
 *  • Permutation test — shuffles y_true 1000× and checks how often
 *    a random labeling matches or beats the observed dir_acc.
 *  • Per-day stability — exposes regime sensitivity at a glance.
 */
export function RobustnessSuite({ symbols }: Props) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Robustness suite
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          block-bootstrap CI · permutation · per-day stability
        </span>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        {symbols.map((sym) => (
          <PerSymbolPanel key={sym} symbol={sym} />
        ))}
      </div>
    </div>
  );
}


function PerSymbolPanel({ symbol }: { symbol: string }) {
  const { data, isLoading } = useRobustness(symbol);
  const display = symbol.replace("USDT", "");

  if (isLoading) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
        <div className="mb-2 text-sm font-semibold text-zinc-200">{display}</div>
        <div className="space-y-2">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-2/3" />
          <Skeleton className="mt-3 h-12 w-full" rounded="rounded-md" />
          <Skeleton className="mt-3 h-5 w-full" rounded="rounded-md" />
        </div>
      </div>
    );
  }
  if (!data?.ok) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3 text-xs text-zinc-500">
        <div className="mb-1 text-sm font-semibold text-zinc-300">
          {display}
        </div>
        {data?.reason === "no_robustness_run_yet"
          ? "не запускался robustness suite"
          : data?.reason ?? "данных нет"}
        {data?.hint && (
          <div className="mt-1 font-mono text-[0.62rem] text-zinc-600">
            hint: {data.hint}
          </div>
        )}
      </div>
    );
  }

  const bb = data.block_bootstrap;
  const pt = data.permutation;
  const perDay = data.per_day ?? [];
  const perHour = data.per_hour ?? [];

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="text-sm font-semibold text-zinc-200">{display}</div>
        <div className="text-[0.62rem] tabular-nums text-zinc-500">
          n={data.n_predictions ?? 0}
        </div>
      </div>

      {/* Block bootstrap CI + permutation test */}
      <div className="space-y-1.5 text-[0.7rem]">
        {bb && (
          <div className="flex items-baseline justify-between">
            <span className="text-zinc-500">block-bootstrap dir_acc:</span>
            <span className="tabular-nums text-zinc-200">
              {fmtPercent(bb.point, 2)}{" "}
              <span className="text-zinc-500">
                [{fmtPercent(bb.ci_low, 2)}, {fmtPercent(bb.ci_high, 2)}]
              </span>
            </span>
          </div>
        )}
        {pt && (
          <div className="flex items-baseline justify-between">
            <span className="text-zinc-500">permutation p:</span>
            <span
              className={`tabular-nums ${
                pt.p_value < 0.01
                  ? "text-emerald-400"
                  : pt.p_value < 0.05
                  ? "text-amber-400"
                  : "text-rose-400"
              }`}
            >
              {pt.p_value < 0.001
                ? "< 0.001"
                : pt.p_value.toFixed(3)}
            </span>
          </div>
        )}
      </div>

      {/* Per-day mini sparkline */}
      {perDay.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-[0.62rem] uppercase tracking-wider text-zinc-500">
            per day ({perDay.length})
          </div>
          <PerDaySparkline points={perDay} />
        </div>
      )}

      {/* Per-hour mini heatmap */}
      {perHour.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-[0.62rem] uppercase tracking-wider text-zinc-500">
            per UTC hour
          </div>
          <PerHourHeatmap points={perHour} />
        </div>
      )}
    </div>
  );
}


function PerDaySparkline({ points }: { points: PerDayPoint[] }) {
  const W = 240;
  const H = 50;
  const padX = 4;
  const padY = 6;
  const innerW = W - 2 * padX;
  const innerH = H - 2 * padY;
  const xs = points.map((_, i) =>
    points.length === 1 ? W / 2 : padX + (i / (points.length - 1)) * innerW,
  );
  const ys = points.map((p) => padY + (1 - Math.min(1, Math.max(0, p.dir_acc))) * innerH);

  // Reference line at 0.5
  const baselineY = padY + (1 - 0.5) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line
        x1={padX}
        x2={padX + innerW}
        y1={baselineY}
        y2={baselineY}
        stroke="#525252"
        strokeWidth={0.5}
        strokeDasharray="2 2"
      />
      <polyline
        fill="none"
        stroke="#a78bfa"
        strokeWidth={1.5}
        points={xs.map((x, i) => `${x},${ys[i]}`).join(" ")}
      />
      {xs.map((x, i) => (
        <circle
          key={i}
          cx={x}
          cy={ys[i]}
          r={1.8}
          fill={points[i].dir_acc >= 0.5 ? "#34d399" : "#fb7185"}
        >
          <title>
            {points[i].day} · n={points[i].n} · {fmtPercent(points[i].dir_acc, 1)}
          </title>
        </circle>
      ))}
    </svg>
  );
}


function PerHourHeatmap({ points }: { points: PerHourPoint[] }) {
  // Map hour 0..23 → cell. Missing hours show as muted gray.
  const byHour: Record<number, PerHourPoint> = {};
  for (const p of points) byHour[p.hour] = p;

  return (
    <div className="grid grid-cols-12 gap-px text-center text-[0.55rem]">
      {Array.from({ length: 24 }, (_, h) => {
        const p = byHour[h];
        let bg = "bg-zinc-800";
        let label = "—";
        if (p) {
          const t = Math.max(-0.05, Math.min(0.1, p.dir_acc - 0.5));
          // -0.05 → red, 0 → zinc, +0.1 → emerald
          if (t > 0.02) bg = "bg-emerald-500/40";
          else if (t > 0) bg = "bg-emerald-700/40";
          else if (t > -0.02) bg = "bg-zinc-700";
          else bg = "bg-rose-600/40";
          label = String(h);
        }
        return (
          <div
            key={h}
            className={`flex h-5 items-center justify-center text-zinc-300 ${bg}`}
            title={
              p
                ? `${h}h UTC · n=${p.n} · ${fmtPercent(p.dir_acc, 1)}`
                : `${h}h · нет данных`
            }
          >
            {label}
          </div>
        );
      })}
    </div>
  );
}
