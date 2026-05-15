import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveHero } from "./LiveHero";
import type {
  ForecastBlock,
  MicropriceBlock,
} from "@/lib/api-types";


/**
 * LiveHero is purely presentational — no providers needed since
 * useFlashOnChange and the format helpers don't touch React Query
 * or any context.
 */

describe("<LiveHero />", () => {
  it("renders skeletons while forecast / microprice are loading", () => {
    const forecast: ForecastBlock = { ok: false, reason: "loading" };
    const microprice: MicropriceBlock = { ok: false, reason: "loading" };

    const { container } = render(
      <LiveHero
        symbol="BTCUSDT"
        forecast={forecast}
        microprice={microprice}
      />,
    );

    // Ticker eyebrow still rendered.
    expect(screen.getByText("BTC / USDT")).toBeInTheDocument();
    expect(screen.getByText("BTC")).toBeInTheDocument();
    // Live strip is always there.
    expect(screen.getByText("live")).toBeInTheDocument();
    // At least a couple of skeleton spans for the price + prediction.
    expect(
      container.querySelectorAll(".skeleton").length,
    ).toBeGreaterThanOrEqual(2);
  });

  it("renders ↑ + ``вверх`` + confidence for an up signal", () => {
    const forecast: ForecastBlock = {
      ok: true,
      prob_up: 0.62,
      signal: "up",
      model: {
        has_model: true,
        model_age_seconds: 3600,
        is_calibrated: true,
        dir_acc_mean: 0.543,
        dir_acc_ci_low: 0.521,
      },
    };
    const microprice: MicropriceBlock = {
      ok: true,
      price: 67432.18,
      ts: "2026-05-13T14:32:08Z",
    };

    render(
      <LiveHero
        symbol="BTCUSDT"
        forecast={forecast}
        microprice={microprice}
      />,
    );

    // Verdict
    expect(screen.getByText("вверх")).toBeInTheDocument();
    // Confidence = max(0.62, 0.38) = 62%
    expect(screen.getByText("62%")).toBeInTheDocument();
    // Historical accuracy line surfaces dir_acc_mean
    expect(screen.getByText(/историческая точность/i)).toBeInTheDocument();
    expect(screen.getByText("54.3%")).toBeInTheDocument();
  });

  it("inverts confidence on a down signal", () => {
    const forecast: ForecastBlock = {
      ok: true,
      prob_up: 0.30,
      signal: "down",
      model: { has_model: true },
    };
    const microprice: MicropriceBlock = {
      ok: true,
      price: 3500.20,
      ts: "2026-05-13T14:32:08Z",
    };

    render(
      <LiveHero
        symbol="ETHUSDT"
        forecast={forecast}
        microprice={microprice}
      />,
    );

    expect(screen.getByText("вниз")).toBeInTheDocument();
    // Confidence = max(0.30, 0.70) = 70%
    expect(screen.getByText("70%")).toBeInTheDocument();
  });

  it("shows neutral verdict for the neutral signal", () => {
    const forecast: ForecastBlock = {
      ok: true,
      prob_up: 0.50,
      signal: "neutral",
      model: { has_model: true },
    };
    const microprice: MicropriceBlock = {
      ok: true,
      price: 620.7,
      ts: "2026-05-13T14:32:08Z",
    };

    render(
      <LiveHero
        symbol="BNBUSDT"
        forecast={forecast}
        microprice={microprice}
      />,
    );

    expect(screen.getByText("без направления")).toBeInTheDocument();
  });
});
