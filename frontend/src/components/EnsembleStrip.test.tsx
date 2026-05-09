import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { EnsembleStrip } from "./EnsembleStrip";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<EnsembleStrip />", () => {
  it("renders nothing while loading (pure inline component)", () => {
    const { container } = render(
      withProviders(<EnsembleStrip symbol="BTCUSDT" />),
    );
    // Initial render before query resolves: component returns null.
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when ensemble responds with ok=false", async () => {
    server.use(
      http.get("/api/highfreq/forecast_ensemble", () =>
        HttpResponse.json(
          { ok: false, reason: "no_available_components", symbol: "BTCUSDT" },
          { status: 503 },
        ),
      ),
    );
    const { container } = render(
      withProviders(<EnsembleStrip symbol="BTCUSDT" />),
    );
    // Wait for query to settle; component still returns null.
    await waitFor(() => {
      // Tick the event loop a bit before asserting.
      expect(container.firstChild).toBeNull();
    });
  });

  it("renders blended probability with up/down arrows when components agree", async () => {
    server.use(
      http.get("/api/highfreq/forecast_ensemble", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          prob_up: 0.62,
          signal: "up",
          agreement: true,
          n_components_used: 2,
          components: [
            { horizon_label: "1m", weight: 0.7, prob_up: 0.65, is_available: true },
            { horizon_label: "15m", weight: 0.3, prob_up: 0.55, is_available: true },
          ],
        }),
      ),
    );

    render(withProviders(<EnsembleStrip symbol="BTCUSDT" />));
    // Blended prob_up * 100, rounded → 62.
    expect(await screen.findByText(/62/)).toBeInTheDocument();
    // 1m component pill — 65 (rounded percentage).
    expect(await screen.findByText(/1m/)).toBeInTheDocument();
    // 15m component pill — 55.
    expect(await screen.findByText(/15m/)).toBeInTheDocument();
  });
});
