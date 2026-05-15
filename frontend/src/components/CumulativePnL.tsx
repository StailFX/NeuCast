"use client";

import { useState } from "react";
import { useCumulativePnL } from "@/lib/hooks";
import type { PnLPoint, CumulativePnLResponse } from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
  /** Tier keys to render. Defence focus: gross + the two extremes. */
  defaultTiers?: readonly string[];
}

const TIER_COLOR: Record<string, string> = {
  gross: "#a3a3a3",
  retail: "#fb7185", // rose — losses dominate
  vip5: "#facc15", // yellow
  vip9: "#22d3ee", // cyan
  futures: "#a78bfa", // violet
  mm_rebate: "#34d399", // emerald
};

const TIER_LABEL: Record<string, string> = {
  gross: "Без комиссии",
  retail: "Spot retail",
  vip5: "Spot VIP-5",
  vip9: "Spot VIP-9",
  futures: "Futures maker",
  mm_rebate: "MM rebate",
};


export function CumulativePnL({
  symbols,
  // Show ALL six tiers by default — without this only retail (the
  // largest in magnitude) is visible and the chart squashes all other
  // tier lines into a thin band near zero. Operators can still toggle
  // tiers off via the legend chips.
  defaultTiers = [
    "gross",
    "retail",
    "vip5",
    "vip9",
    "futures",
    "mm_rebate",
  ],
}: Props) {
  const [activeSymbol, setActiveSymbol] = useState(symbols[0]);
  const [activeTiers, setActiveTiers] = useState<Set<string>>(
    new Set(defaultTiers),
  );
  const { data, isLoading } = useCumulativePnL(activeSymbol, 200);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Cumulative P&amp;L by fee tier
        </h2>
        <SymbolPills
          symbols={symbols}
          active={activeSymbol}
          onChange={setActiveSymbol}
        />
      </div>

      <TierToggles
        tiers={data?.tiers ?? []}
        active={activeTiers}
        onToggle={(key) => {
          const next = new Set(activeTiers);
          if (next.has(key)) next.delete(key);
          else next.add(key);
          setActiveTiers(next);
        }}
      />

      {isLoading ? (
        <Skeleton className="mt-3 h-[480px] w-full" rounded="rounded-xl" />
      ) : (() => {
          // Server returns ``points`` (per-trade-close timestamps with
          // a numeric value at each tier key directly on the object)
          // — the legacy frontend type uses ``curve`` with a nested
          // ``cum_bps_by_tier`` dict. Normalise here so the renderer
          // doesn't have to care about which shape arrived.
          const rawCurve = (data as unknown as { curve?: unknown[]; points?: Array<Record<string, unknown>> } | undefined)?.curve as PnLPoint[] | undefined;
          const rawPoints = (data as unknown as { points?: Array<Record<string, unknown>> } | undefined)?.points;
          const normCurve: PnLPoint[] =
            rawCurve && rawCurve.length
              ? rawCurve
              : (rawPoints ?? []).map((p) => {
                  const cum_bps_by_tier: Record<string, number> = {};
                  for (const [k, v] of Object.entries(p)) {
                    if (k === "ts" || k === "n") continue;
                    if (typeof v === "number") cum_bps_by_tier[k] = v;
                  }
                  return {
                    ts: String(p.ts ?? ""),
                    cum_bps_by_tier,
                    n: typeof p.n === "number" ? p.n : 0,
                  };
                });
          if (!data?.ok || normCurve.length === 0) {
            return (
              <div className="mt-3 text-sm text-zinc-500">
                {data?.reason === "no_trades"
                  ? "пока не было закрытых сделок для этого символа"
                  : "данных нет"}
              </div>
            );
          }
          return (
            <PnLCurve
              payload={{ ...data, curve: normCurve }}
              activeTiers={activeTiers}
            />
          );
        })()}
    </div>
  );
}


function SymbolPills({
  symbols,
  active,
  onChange,
}: {
  symbols: readonly string[];
  active: string;
  onChange: (s: string) => void;
}) {
  return (
    <div className="flex gap-1 rounded-full bg-zinc-950 p-1">
      {symbols.map((s) => (
        <button
          key={s}
          type="button"
          onClick={() => onChange(s)}
          className={`rounded-full px-3 py-1 text-[0.7rem] font-semibold transition ${
            s === active
              ? "bg-zinc-100 text-zinc-900"
              : "text-zinc-400 hover:text-zinc-200"
          }`}
        >
          {s.replace("USDT", "")}
        </button>
      ))}
    </div>
  );
}


function TierToggles({
  tiers,
  active,
  onToggle,
}: {
  tiers: NonNullable<CumulativePnLResponse["tiers"]>;
  active: Set<string>;
  onToggle: (key: string) => void;
}) {
  if (!tiers.length) return null;
  return (
    <div className="mb-3 flex flex-wrap gap-2">
      {tiers.map((t) => {
        const enabled = active.has(t.key);
        const color = TIER_COLOR[t.key] ?? "#52525b";
        const label = TIER_LABEL[t.key] ?? t.key;
        return (
          <button
            key={t.key}
            type="button"
            onClick={() => onToggle(t.key)}
            className={`flex items-center gap-2 rounded-full border border-zinc-800 px-3 py-1 text-[0.66rem] transition ${
              enabled
                ? "bg-zinc-800/80 text-zinc-100"
                : "bg-transparent text-zinc-500 hover:text-zinc-300"
            }`}
            title={`${label} · final ${(((t as unknown as {final_bps?: number; final_cum_bps?: number}).final_bps ?? (t as unknown as {final_cum_bps?: number}).final_cum_bps ?? 0) ?? 0).toFixed(1)} bp`}
          >
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: color, opacity: enabled ? 1 : 0.4 }}
            />
            <span>{label}</span>
            <span
              className={`tabular-nums text-[0.62rem] ${
                (((t as unknown as {final_bps?: number; final_cum_bps?: number}).final_bps ?? (t as unknown as {final_cum_bps?: number}).final_cum_bps ?? 0) ?? 0) > 0
                  ? "text-emerald-400"
                  : (((t as unknown as {final_bps?: number; final_cum_bps?: number}).final_bps ?? (t as unknown as {final_cum_bps?: number}).final_cum_bps ?? 0) ?? 0) < 0
                  ? "text-rose-400"
                  : "text-zinc-400"
              }`}
            >
              {(((t as unknown as {final_bps?: number; final_cum_bps?: number}).final_bps ?? (t as unknown as {final_cum_bps?: number}).final_cum_bps ?? 0) ?? 0) > 0 ? "+" : ""}
              {(((t as unknown as {final_bps?: number; final_cum_bps?: number}).final_bps ?? (t as unknown as {final_cum_bps?: number}).final_cum_bps ?? 0) ?? 0).toFixed(1)}bp
            </span>
          </button>
        );
      })}
    </div>
  );
}


function PnLCurve({
  payload,
  activeTiers,
}: {
  payload: CumulativePnLResponse;
  activeTiers: Set<string>;
}) {
  // Card is wide on desktop; bumping H from 320 → 480 makes the chart
  // fill the card height instead of leaving the bottom half empty
  // (matched against the adjacent 10-row TradesFeed via grid stretch).
  const W = 800;
  const H = 480;
  const padL = 50;
  const padR = 12;
  const padT = 12;
  const padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const curve: PnLPoint[] = payload.curve ?? [];
  if (curve.length < 2) {
    return (
      <div className="text-sm text-zinc-500">недостаточно точек для кривой</div>
    );
  }

  // Compute global min/max across active tiers for y-axis.
  const activeKeys = Array.from(activeTiers);
  let yMin = 0;
  let yMax = 0;
  for (const p of curve) {
    for (const k of activeKeys) {
      const v = p.cum_bps_by_tier[k];
      if (typeof v === "number" && isFinite(v)) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
  }
  // Pad y-axis 5%.
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.05;
  yMin -= yPad;
  yMax += yPad;

  const xToPx = (i: number) =>
    padL + (curve.length === 1 ? innerW / 2 : (i / (curve.length - 1)) * innerW);
  const yToPx = (v: number) =>
    padT + ((yMax - v) / (yMax - yMin)) * innerH;

  // y=0 axis line in px.
  const zeroY = yToPx(0);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full text-zinc-700"
      role="img"
      aria-label={`Cumulative P&L curve for ${payload.symbol}`}
    >
      {/* Frame */}
      <rect
        x={padL}
        y={padT}
        width={innerW}
        height={innerH}
        fill="none"
        stroke="currentColor"
        strokeWidth={0.5}
      />
      {/* y=0 line */}
      <line
        x1={padL}
        x2={padL + innerW}
        y1={zeroY}
        y2={zeroY}
        stroke="#525252"
        strokeWidth={0.7}
        strokeDasharray="3 3"
      />
      {/* y-tick labels */}
      <text
        x={padL - 6}
        y={padT + 4}
        textAnchor="end"
        className="fill-zinc-500 text-[10px]"
      >
        {yMax.toFixed(0)} bp
      </text>
      <text
        x={padL - 6}
        y={padT + innerH}
        textAnchor="end"
        className="fill-zinc-500 text-[10px]"
      >
        {yMin.toFixed(0)} bp
      </text>
      <text
        x={padL - 6}
        y={zeroY + 3}
        textAnchor="end"
        className="fill-zinc-500 text-[10px]"
      >
        0
      </text>
      {/* Tier polylines */}
      {activeKeys.map((tierKey) => {
        const color = TIER_COLOR[tierKey] ?? "#52525b";
        const points = curve
          .map((p, i) => {
            const v = p.cum_bps_by_tier[tierKey];
            if (typeof v !== "number" || !isFinite(v)) return null;
            return `${xToPx(i)},${yToPx(v)}`;
          })
          .filter((s): s is string => s !== null)
          .join(" ");
        return (
          <polyline
            key={tierKey}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            opacity={0.85}
            points={points}
          />
        );
      })}
      {/* x-axis time markers — first / mid / last trade timestamp from
          the curve. Without these the chart x-axis has no scale and
          reads as "some line over time" without anchoring. */}
      {(() => {
        const fmt = (iso: string) => {
          try {
            const d = new Date(iso);
            return d.toLocaleDateString("ru-RU", {
              day: "2-digit",
              month: "2-digit",
            });
          } catch {
            return "";
          }
        };
        const firstTs = curve[0]?.ts;
        const midTs = curve[Math.floor(curve.length / 2)]?.ts;
        const lastTs = curve[curve.length - 1]?.ts;
        return (
          <g className="fill-zinc-600 text-[10px]">
            {firstTs && (
              <text x={padL} y={padT + innerH + 14} textAnchor="start">
                {fmt(firstTs)}
              </text>
            )}
            {midTs && (
              <text
                x={padL + innerW / 2}
                y={padT + innerH + 14}
                textAnchor="middle"
              >
                {fmt(midTs)}
              </text>
            )}
            {lastTs && (
              <text
                x={padL + innerW}
                y={padT + innerH + 14}
                textAnchor="end"
              >
                {fmt(lastTs)}
              </text>
            )}
          </g>
        );
      })()}
      {/* Caption */}
      <text
        x={padL + innerW / 2}
        y={H - 6}
        textAnchor="middle"
        className="fill-zinc-600 text-[10px]"
      >
        {payload.n_trades} trades · {payload.symbol?.replace("USDT", "")}
      </text>
    </svg>
  );
}
