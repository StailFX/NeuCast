"use client";

/**
 * SymbolPills — user-facing symbol selector for /v2/live.
 *
 * Three rounded pills (BTC / ETH / BNB). Active pill is a gradient
 * green-emerald button (v1 .nc-btn-primary look), inactive pills are
 * dark with thin border and hover blue accent.
 *
 * State is owned by the parent (live page). a11y: rendered as
 * horizontal tablist, each pill has aria-selected.
 */

const SYMBOLS = [
  { id: "BTCUSDT", label: "BTC" },
  { id: "ETHUSDT", label: "ETH" },
  { id: "BNBUSDT", label: "BNB" },
] as const;

interface Props {
  value: string;
  onChange: (symbol: string) => void;
}

export function SymbolPills({ value, onChange }: Props) {
  return (
    <div className="nc-pills" role="tablist" aria-label="торгуемый инструмент">
      {SYMBOLS.map((sym) => {
        const active = sym.id === value;
        return (
          <button
            key={sym.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(sym.id)}
            className={active ? "nc-pill nc-active" : "nc-pill"}
          >
            {sym.label}
          </button>
        );
      })}
    </div>
  );
}
