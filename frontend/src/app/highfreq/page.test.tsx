import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryProvider } from "@/lib/QueryProvider";
import { AuthProvider } from "@/lib/AuthContext";
import { HorizonProvider } from "@/lib/HorizonContext";
import HighfreqPage from "./page";


/**
 * /v2/highfreq — operator dashboard. Stitches together IngestHealth,
 * TrainingReport, AntiSkill, TrainingHistory, StatsStrip,
 * ConditionalAccuracy plus a 3-symbol ForecastCard grid. Most of
 * the heavy lifting is already covered by the per-component tests;
 * this is a smoke test that the page assembles without crashes or
 * rule-of-hooks violations under the default MSW handlers.
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


describe("/v2/highfreq operator page", () => {
  it("renders Navbar + 3 ForecastCards (BTC/ETH/BNB)", async () => {
    render(withAllProviders(<HighfreqPage />));
    await waitFor(() => {
      const matches = screen.getAllByText(/^BTC$|^ETH$|^BNB$/);
      expect(matches.length).toBeGreaterThanOrEqual(3);
    });
  });

  it("renders all operator sections (Ingest, Training, AntiSkill, History)", async () => {
    render(withAllProviders(<HighfreqPage />));
    // "Ingest health" appears both in the section header AND in the
    // page-level intro paragraph — same for several other section
    // names. Use findAllByText to be tolerant of these duplicates.
    expect(
      (await screen.findAllByText(/ingest health/i)).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText(/training report/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/anti-skill detector/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/training history/i).length).toBeGreaterThan(0);
  });

  it("renders the stats + conditional accuracy blocks", async () => {
    render(withAllProviders(<HighfreqPage />));
    expect(
      await screen.findByText(/24h direction accuracy/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/conditional accuracy by confidence/i),
    ).toBeInTheDocument();
  });
});
