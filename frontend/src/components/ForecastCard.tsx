"use client";

import { fmtAge, fmtMoney, fmtPercent } from "@/lib/format";
import type {
  ForecastBlock,
  DriftBlock,
  MicropriceBlock,
} from "@/lib/api-types";
import { DriftBadge } from "./DriftBadge";
import { EnsembleStrip } from "./EnsembleStrip";
import { JointForecastBadge } from "./JointForecastBadge";
import { Skeleton } from "./Skeleton";
import { HORIZON_LABELS, useHorizon } from "@/lib/HorizonContext";
import { useFlashOnChange } from "@/lib/useFlashOnChange";

interface Props {
  symbol: string;
  forecast: ForecastBlock;
  drift: DriftBlock;
  microprice: MicropriceBlock;
}

const SIGNAL_STYLES: Record<string, { color: string; arrow: string; ru: string }> = {
  up: { color: "text-emerald-400", arrow: "↑", ru: "вверх" },
  down: { color: "text-rose-400", arrow: "↓", ru: "вниз" },
  neutral: { color: "text-zinc-400", arrow: "→", ru: "без направления" },
};

const TICKER_DISPLAY: Record<string, string> = {
  BTCUSDT: "BTC / USDT",
  ETHUSDT: "ETH / USDT",
  BNBUSDT: "BNB / USDT",
};

function readConformal(
  c: NonNullable<Extract<ForecastBlock, { ok: true }>["conformal_90"]>,
): { low: number; high: number } | null {
  if (Array.isArray(c)) {
    const [low, high] = c;
    if (typeof low === "number" && typeof high === "number") {
      return { low, high };
    }
    return null;
  }
  if (typeof c.low === "number" && typeof c.high === "number") {
    return { low: c.low, high: c.high };
  }
  return null;
}


export function ForecastCard({ symbol, forecast, drift, microprice }: Props) {
  const display = TICKER_DISPLAY[symbol] ?? symbol;
  const { horizon } = useHorizon();
  const horizonText = horizon === 60 ? "+1 час" : `+${HORIZON_LABELS[horizon]}`;

  // Number-flash on prob_up moves ≥ 0.5pp.
  const probValue = forecast.ok ? forecast.prob_up : null;
  const probFlash = useFlashOnChange<number>(
    probValue,
    (prev, curr) => (curr > prev ? "flash-up" : "flash-down"),
    0.005,
  );

  // Number-flash on price moves ≥ 0.005% (small but visible).
  const priceValue = microprice.ok ? microprice.price : null;
  const priceFlash = useFlashOnChange<number>(
    priceValue,
    (prev, curr) =>
      Math.abs((curr - prev) / Math.max(prev, 1e-9)) < 0.00005
        ? ""
        : curr > prev
        ? "flash-up"
        : "flash-down",
    0,
  );

  // ── Forecast block ──
  let signalNode: React.ReactNode = (
    <div className="flex items-center gap-2 text-zinc-500">
      <Skeleton className="h-8 w-8" rounded="rounded-full" />
      <Skeleton className="h-4 w-32" />
    </div>
  );
  let probNode: React.ReactNode = (
    <div className="mt-2 space-y-1.5">
      <Skeleton className="h-3 w-28" />
      <Skeleton className="h-1 w-full" rounded="rounded-full" />
    </div>
  );
  let conformalNode: React.ReactNode = null;
  let modelAgeNode: React.ReactNode = (
    <Skeleton className="h-3 w-20" />
  );

  if (forecast.ok) {
    const sig = SIGNAL_STYLES[forecast.signal] ?? SIGNAL_STYLES.neutral;
    const confidencePct = Math.round(
      Math.max(forecast.prob_up, 1 - forecast.prob_up) * 100,
    );
    signalNode = (
      <div className={`flex items-baseline gap-2 ${sig.color}`}>
        <span className="text-3xl font-semibold">{sig.arrow}</span>
        <span className="text-sm">{sig.ru}</span>
      </div>
    );
    probNode = (
      <div className="mt-2 space-y-1">
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <span>
            уверенность{" "}
            <span className={`font-semibold tabular-nums text-zinc-200 ${probFlash}`}>
              {confidencePct}%
            </span>
          </span>
        </div>
        <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-800">
          <div
            className={`h-full transition-all duration-500 ease-out ${
              forecast.signal === "up"
                ? "bg-emerald-500"
                : forecast.signal === "down"
                ? "bg-rose-500"
                : "bg-zinc-500"
            }`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>
    );

    // Conformal CI text — accepts both [lo, hi] tuple AND {low, high}
    // shapes since the FastAPI endpoint can return either.
    if (forecast.conformal_90) {
      const ci = readConformal(forecast.conformal_90);
      if (
        ci &&
        isFinite(ci.low) &&
        isFinite(ci.high) &&
        // Skip the [0, 1] uninformative case (model lacks conformal_q).
        !(ci.low === 0 && ci.high === 1)
      ) {
        conformalNode = (
          <div className="text-xs text-zinc-500">
            prob_up={fmtPercent(forecast.prob_up)} · CI [
            {fmtPercent(ci.low)}, {fmtPercent(ci.high)}]
          </div>
        );
      }
    }

    if (forecast.model?.model_age_seconds != null) {
      modelAgeNode = (
        <span className="text-xs text-zinc-500">
          модель: {fmtAge(forecast.model.model_age_seconds)}
        </span>
      );
    }
  }

  // ── Microprice (live $$ tape) ──
  const priceText = microprice.ok ? `$${fmtMoney(microprice.price)}` : "—";

  // Signal glow class — adds soft colored shadow to the card when the
  // forecast is on a clean direction (skipped for neutral / loading).
  const glowClass = forecast.ok
    ? `signal-glow signal-${forecast.signal}`
    : "";

  return (
    <div
      className={`card-hover rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6 shadow-xl backdrop-blur ${glowClass}`}
    >
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-xl font-semibold text-zinc-100">
            {symbol.replace("USDT", "")}
          </div>
          <div className="text-xs uppercase tracking-wider text-zinc-500">
            {display}
          </div>
        </div>
        <div className="rounded-full bg-zinc-800 px-3 py-1 text-xs text-zinc-400">
          {horizonText}
        </div>
      </div>

      <div className="mt-4 flex items-baseline justify-between text-2xl font-mono text-zinc-100">
        {microprice.ok ? (
          <span className={`tabular-nums ${priceFlash}`}>{priceText}</span>
        ) : (
          <Skeleton className="h-7 w-32" />
        )}
      </div>

      <div className="mt-6">
        {signalNode}
        {probNode}
        {conformalNode && <div className="mt-2">{conformalNode}</div>}
        <EnsembleStrip symbol={symbol} />
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-2 border-t border-zinc-800 pt-3">
        {modelAgeNode}
        <div className="flex items-center gap-2">
          <DriftBadge drift={drift} />
          <JointForecastBadge
            symbol={symbol}
            soloProbUp={forecast.ok ? forecast.prob_up : null}
          />
        </div>
        <span className="text-xs text-zinc-500">live</span>
      </div>
    </div>
  );
}
