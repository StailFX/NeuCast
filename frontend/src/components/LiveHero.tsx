"use client";

import { fmtAge, fmtMoney, fmtPercent } from "@/lib/format";
import { useFlashOnChange } from "@/lib/useFlashOnChange";
import type { ForecastBlock, MicropriceBlock } from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

/**
 * LiveHero — the headline block of /v2/live (v1 landing visual style).
 *
 * Visual hierarchy:
 *   1. Live indicator strip (pulsing green dot + "Binance Spot · 19 ms")
 *   2. Ticker eyebrow + big ticker name (BTC / ETH / BNB)
 *   3. Huge price (4rem Inter 800, gradient-free for legibility)
 *   4. Prediction split: arrow + verdict, confidence pct + bar
 *   5. Footer: historical model accuracy + CI low
 *
 * Uses ``.nc-`` classes from landing-theme.css. Flash animation
 * still tints individual numeric spans green/red on movement.
 */

interface Props {
  symbol: string;
  forecast: ForecastBlock;
  microprice: MicropriceBlock;
}

const TICKER_DISPLAY: Record<string, string> = {
  BTCUSDT: "BTC / USDT",
  ETHUSDT: "ETH / USDT",
  BNBUSDT: "BNB / USDT",
};

const SIGNAL: Record<
  string,
  { arrow: string; label: string; tone: "up" | "down" | "neutral" }
> = {
  up: { arrow: "↑", label: "вверх", tone: "up" },
  down: { arrow: "↓", label: "вниз", tone: "down" },
  neutral: { arrow: "→", label: "без направления", tone: "neutral" },
};

export function LiveHero({ symbol, forecast, microprice }: Props) {
  const display = TICKER_DISPLAY[symbol] ?? symbol;

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

  const probValue = forecast.ok ? forecast.prob_up : null;
  const probFlash = useFlashOnChange<number>(
    probValue,
    (prev, curr) => (curr > prev ? "flash-up" : "flash-down"),
    0.005,
  );

  const sig = forecast.ok
    ? (SIGNAL[forecast.signal] ?? SIGNAL.neutral)
    : SIGNAL.neutral;
  const confidencePct = forecast.ok
    ? Math.round(Math.max(forecast.prob_up, 1 - forecast.prob_up) * 100)
    : null;

  return (
    <section className="nc-card nc-card-glow" aria-label="live hero forecast">
      {/* Top strip — live indicator + model age */}
      <div className="nc-live-strip">
        <span className="nc-live-pill">
          <span className="nc-dot" />
          <span>live</span>
          <span style={{ opacity: 0.5, margin: "0 0.5rem" }}>·</span>
          <span>Binance Spot · 19 ms ingest</span>
        </span>
        {forecast.ok && forecast.model?.model_age_seconds != null && (
          <span
            title="Время с последнего переобучения (ежедневно в 04:00 UTC). Модель тренируется на ~6 месяцах L2-данных."
          >
            обновлено {fmtAge(forecast.model.model_age_seconds)} назад
          </span>
        )}
      </div>

      {/* Ticker eyebrow + name */}
      <div style={{ marginTop: "1.75rem" }}>
        <div className="nc-eyebrow">{display}</div>
        <div className="nc-ticker">{symbol.replace("USDT", "")}</div>
      </div>

      {/* Big price */}
      <div style={{ marginTop: "1rem" }}>
        {microprice.ok ? (
          <div className={`nc-price ${priceFlash}`}>
            ${fmtMoney(microprice.price)}
          </div>
        ) : (
          <Skeleton className="h-14 w-72" />
        )}
      </div>

      {/* Prediction split */}
      <div className="nc-prediction-grid">
        <div>
          <div className="nc-eyebrow">прогноз через 1 минуту</div>
          {forecast.ok ? (
            <div className={`nc-verdict nc-${sig.tone}`} style={{ marginTop: "0.6rem" }}>
              <span className="nc-verdict-arrow">{sig.arrow}</span>
              <span>{sig.label}</span>
            </div>
          ) : (
            <div style={{ marginTop: "0.6rem", display: "flex", gap: "0.6rem" }}>
              <Skeleton className="h-9 w-9" rounded="rounded-full" />
              <Skeleton className="h-7 w-32" />
            </div>
          )}
        </div>
        <div>
          <div className="nc-eyebrow">уверенность</div>
          <div style={{ marginTop: "0.6rem" }}>
            {confidencePct != null ? (
              <span className={`nc-confidence-num ${probFlash}`}>
                {confidencePct}%
              </span>
            ) : (
              <Skeleton className="h-9 w-20" />
            )}
          </div>
          {confidencePct != null && (
            <div className="nc-bar">
              <div
                className={`nc-bar-fill nc-${sig.tone}`}
                style={{ width: `${confidencePct}%` }}
              />
            </div>
          )}
        </div>
      </div>

      {/* Footer — historical accuracy */}
      {forecast.ok && forecast.model?.dir_acc_mean != null && (
        <div className="nc-card-footer">
          <span>
            историческая точность модели:{" "}
            <span style={{ color: "#f1f5f9", fontWeight: 600 }}>
              {fmtPercent(forecast.model.dir_acc_mean, 1)}
            </span>
          </span>
          {forecast.model.dir_acc_ci_low != null && (
            <span style={{ opacity: 0.6 }}>
              (нижняя граница CI 95 %: {fmtPercent(forecast.model.dir_acc_ci_low, 1)})
            </span>
          )}
        </div>
      )}
    </section>
  );
}
