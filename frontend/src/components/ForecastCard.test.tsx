import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ForecastCard } from "./ForecastCard";
import type {
  ForecastBlock,
  DriftBlock,
  MicropriceBlock,
} from "@/lib/api-types";
import { QueryProvider } from "@/lib/QueryProvider";
import { HorizonProvider } from "@/lib/HorizonContext";


/**
 * ForecastCard pulls EnsembleStrip via React Query, so we wrap with
 * the same providers the live layout does. The default MSW handler
 * for /api/highfreq/forecast_ensemble returns a happy payload — fine
 * for these visual tests.
 */
function withProviders(ui: React.ReactNode) {
  return (
    <QueryProvider>
      <HorizonProvider>{ui}</HorizonProvider>
    </QueryProvider>
  );
}


describe("<ForecastCard />", () => {
  it("renders skeleton placeholders when forecast / microprice are loading", () => {
    const forecast: ForecastBlock = { ok: false, reason: "loading" };
    const drift: DriftBlock = { ok: false, reason: "loading" };
    const microprice: MicropriceBlock = { ok: false, reason: "loading" };

    const { container } = render(
      withProviders(
        <ForecastCard
          symbol="BTCUSDT"
          forecast={forecast}
          drift={drift}
          microprice={microprice}
        />,
      ),
    );

    // Symbol header always present.
    expect(screen.getByText("BTC")).toBeInTheDocument();
    expect(screen.getByText("BTC / USDT")).toBeInTheDocument();
    // Multiple skeleton spans during loading.
    expect(container.querySelectorAll(".skeleton").length).toBeGreaterThan(0);
  });

  it("renders ↑ + ``вверх`` + confidence pct on a clean up signal", () => {
    const forecast: ForecastBlock = {
      ok: true,
      prob_up: 0.62,
      signal: "up",
      model: { has_model: true, model_age_seconds: 7200 },
    };
    const drift: DriftBlock = {
      ok: true,
      severity: "ok",
      max_ks: 0.05,
      max_ks_feature: "ofi_mean",
      evaluated_at: null,
    };
    const microprice: MicropriceBlock = {
      ok: true,
      price: 79000.5,
      ts: "2026-05-04T17:00:00Z",
    };

    render(
      withProviders(
        <ForecastCard
          symbol="BTCUSDT"
          forecast={forecast}
          drift={drift}
          microprice={microprice}
        />,
      ),
    );

    // Verdict label (RU).
    expect(screen.getByText("вверх")).toBeInTheDocument();
    // Confidence: max(prob, 1-prob) = 0.62 → 62%.
    expect(screen.getByText(/62%/)).toBeInTheDocument();
    // Drift badge.
    expect(screen.getByText(/drift ok/i)).toBeInTheDocument();
  });

  it("renders ↓ + ``вниз`` + correctly inverts confidence on a down signal", () => {
    const forecast: ForecastBlock = {
      ok: true,
      prob_up: 0.30,
      signal: "down",
      model: { has_model: true },
    };
    const drift: DriftBlock = { ok: false, reason: "no_check_yet" };
    const microprice: MicropriceBlock = {
      ok: true,
      price: 3500.20,
      ts: "2026-05-04T17:00:00Z",
    };

    render(
      withProviders(
        <ForecastCard
          symbol="ETHUSDT"
          forecast={forecast}
          drift={drift}
          microprice={microprice}
        />,
      ),
    );

    expect(screen.getByText("вниз")).toBeInTheDocument();
    // Confidence on down = max(0.30, 0.70) = 70%.
    expect(screen.getByText(/70%/)).toBeInTheDocument();
  });
});
