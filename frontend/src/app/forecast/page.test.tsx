import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryProvider } from "@/lib/QueryProvider";
import { AuthProvider } from "@/lib/AuthContext";
import { HorizonProvider } from "@/lib/HorizonContext";
import ForecastPage from "./page";


/**
 * /v2/forecast — biggest page in the SPA. Pulls /dashboard,
 * /paper_trades for 3 symbols, plus all the secondary endpoints.
 * MSW default handlers (in tests/mocks/handlers.ts) cover all of
 * them, so this is mostly a smoke test that the page assembles
 * without crashing or rule-of-hooks violations.
 */
function withAllProviders(ui: React.ReactNode) {
  return (
    <QueryProvider>
      <AuthProvider>
        <HorizonProvider>{ui}</HorizonProvider>
      </AuthProvider>
    </QueryProvider>
  );
}


describe("/v2/forecast page", () => {
  it("renders Navbar + 3 ForecastCards (BTC, ETH, BNB)", async () => {
    render(withAllProviders(<ForecastPage />));
    // 3 short symbol names should appear (one per ForecastCard).
    await waitFor(() => {
      const matches = screen.getAllByText(/^BTC$|^ETH$|^BNB$/);
      expect(matches.length).toBeGreaterThanOrEqual(3);
    });
  });

  it("renders the HorizonPill switcher (1m/5m/15m/1h)", () => {
    render(withAllProviders(<ForecastPage />));
    expect(screen.getByRole("button", { name: "1m" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "5m" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "15m" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "1h" })).toBeInTheDocument();
  });

  it("renders all secondary blocks (StatsStrip, TradesFeed, etc.)", async () => {
    render(withAllProviders(<ForecastPage />));
    expect(
      await screen.findByText(/24h direction accuracy/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/recent trades/i)).toBeInTheDocument();
    expect(screen.getByText(/feature importance/i)).toBeInTheDocument();
  });
});
