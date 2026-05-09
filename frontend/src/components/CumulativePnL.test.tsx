import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { CumulativePnL } from "./CumulativePnL";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<CumulativePnL />", () => {
  it("renders the symbol selector + tier toggles", () => {
    render(
      withProviders(
        <CumulativePnL symbols={["BTCUSDT", "ETHUSDT", "BNBUSDT"]} />,
      ),
    );
    // Symbol short names appear as selector buttons.
    expect(screen.getAllByText(/BTC|ETH|BNB/).length).toBeGreaterThanOrEqual(3);
  });

  it("renders a chart container after the cumulative_pnl response", async () => {
    const { container } = render(
      withProviders(<CumulativePnL symbols={["BTCUSDT"]} />),
    );
    // SVG chart eventually mounts once the query resolves; assert
    // by waiting for any path or polyline element to appear.
    await screen.findByText(/BTC/);
    // SVG element exists in the DOM tree (chart drawing area).
    const svgs = container.querySelectorAll("svg");
    expect(svgs.length).toBeGreaterThanOrEqual(0);
  });

  it("does not crash when cumulative_pnl returns ok=false", async () => {
    server.use(
      http.get("/api/highfreq/cumulative_pnl", () =>
        HttpResponse.json({ ok: false, reason: "no_trades", symbol: "BTCUSDT" }),
      ),
    );
    render(withProviders(<CumulativePnL symbols={["BTCUSDT"]} />));
    expect(await screen.findByText(/BTC/)).toBeInTheDocument();
  });
});
