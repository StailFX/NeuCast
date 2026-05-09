import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { FeeTierPnLBars } from "./FeeTierPnLBars";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<FeeTierPnLBars />", () => {
  it("renders symbol selector with all 3 symbols", () => {
    render(
      withProviders(
        <FeeTierPnLBars symbols={["BTCUSDT", "ETHUSDT", "BNBUSDT"]} />,
      ),
    );
    expect(screen.getAllByText(/BTC|ETH|BNB/).length).toBeGreaterThanOrEqual(3);
  });

  it("displays tier rows from the API (retail, vip9, etc.)", async () => {
    render(withProviders(<FeeTierPnLBars symbols={["BTCUSDT"]} />));
    // Default handler returns 3 tiers: gross / retail / vip9.
    expect(await screen.findByText(/Spot retail/i)).toBeInTheDocument();
    expect(await screen.findByText(/Spot VIP-9/i)).toBeInTheDocument();
  });

  it("does not crash when pnl_by_fee_tier responds ok=false", async () => {
    server.use(
      http.get("/api/highfreq/pnl_by_fee_tier", () =>
        HttpResponse.json({ ok: false, reason: "no_closed_trades" }),
      ),
    );
    render(withProviders(<FeeTierPnLBars symbols={["BTCUSDT"]} />));
    // Component still renders the symbol selector header.
    expect(await screen.findByText(/BTC/)).toBeInTheDocument();
  });
});
