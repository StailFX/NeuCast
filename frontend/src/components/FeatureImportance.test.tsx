import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { FeatureImportance } from "./FeatureImportance";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<FeatureImportance />", () => {
  it("renders heading + per-symbol panels", () => {
    render(
      withProviders(
        <FeatureImportance symbols={["BTCUSDT", "ETHUSDT"]} />,
      ),
    );
    expect(screen.getByText(/feature importance/i)).toBeInTheDocument();
  });

  it("displays top features from the API response", async () => {
    render(
      withProviders(<FeatureImportance symbols={["BTCUSDT"]} topN={5} />),
    );
    // Default handler: ofi_mean is the top feature.
    expect(await screen.findByText(/ofi_mean/i)).toBeInTheDocument();
    // spread_bps_mean is the second.
    expect(await screen.findByText(/spread_bps_mean/i)).toBeInTheDocument();
  });

  it("respects topN — does not render more rows than requested", async () => {
    render(
      withProviders(<FeatureImportance symbols={["BTCUSDT"]} topN={2} />),
    );
    // Default handler returns 5 features; topN=2 should limit display.
    await screen.findByText(/ofi_mean/i);
    // depth_imb_mean is item #3 — should NOT appear with topN=2.
    expect(screen.queryByText(/depth_imb_mean/i)).not.toBeInTheDocument();
  });

  it("does not crash when feature_importance returns ok=false", async () => {
    server.use(
      http.get("/api/highfreq/feature_importance", () =>
        HttpResponse.json({ ok: false, reason: "no_model" }),
      ),
    );
    render(withProviders(<FeatureImportance symbols={["BTCUSDT"]} />));
    expect(screen.getByText(/feature importance/i)).toBeInTheDocument();
  });
});
