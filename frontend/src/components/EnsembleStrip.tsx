"use client";

import { useForecastEnsemble } from "@/lib/hooks";

interface Props {
  symbol: string;
}

/**
 * Inline 1m+15m ensemble breakdown — "ансамбль 58% (1m=62 · 15m=50 ✓)".
 * The ✓/✗ glyph signals whether the two horizons agree on direction.
 * Hidden when one component is cold-starting; surfaces partial info
 * gracefully (e.g. "1m=62 · 15m=—").
 */
export function EnsembleStrip({ symbol }: Props) {
  const { data, isLoading } = useForecastEnsemble(symbol);

  if (isLoading || !data?.ok) return null;
  if (!data.components?.length || typeof data.prob_up !== "number") return null;

  const blendPct = Math.round(data.prob_up * 100);

  // Component pills — 1m=62 · 15m=50 — hidden when unavailable.
  const parts = data.components.map((c) => {
    if (!c.is_available || c.prob_up == null) {
      return `${c.horizon_label}=—`;
    }
    return `${c.horizon_label}=${Math.round(c.prob_up * 100)}`;
  });

  const agreementGlyph = data.agreement ? (
    <span
      className="ml-1 text-emerald-400"
      title="Both horizons on same side of 0.5"
      aria-label="agreement"
    >
      ✓
    </span>
  ) : (
    <span
      className="ml-1 text-amber-400"
      title="Horizons disagree on direction; blend dominates by weight"
      aria-label="disagreement"
    >
      ✗
    </span>
  );

  // Tooltip with full per-component status — same content the legacy
  // forecast.html surfaces.
  const tooltipLines = [
    `Ensemble blend prob_up = ${(data.prob_up * 100).toFixed(1)}%`,
    `signal: ${data.signal} · agreement: ${data.agreement ? "yes" : "no"}`,
    "components:",
    ...data.components.map((c) => {
      const w = (c.weight * 100).toFixed(0);
      const p =
        c.is_available && c.prob_up != null
          ? `${(c.prob_up * 100).toFixed(1)}%`
          : "unavailable";
      return `  • ${c.horizon_label} (w=${w}%): ${p}`;
    }),
  ];

  return (
    <div
      className="mt-2 cursor-help text-[0.7rem] text-zinc-500"
      title={tooltipLines.join("\n")}
    >
      ансамбль <span className="text-zinc-300">{blendPct}%</span>{" "}
      <span className="text-zinc-600">
        ({parts.join(" · ")})
      </span>
      {agreementGlyph}
    </div>
  );
}
