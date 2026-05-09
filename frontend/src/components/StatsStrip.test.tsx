import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { StatsStrip } from "./StatsStrip";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<StatsStrip />", () => {
  it("renders heading + per-symbol columns", () => {
    render(
      withProviders(
        <StatsStrip symbols={["BTCUSDT", "ETHUSDT", "BNBUSDT"]} />,
      ),
    );
    expect(
      screen.getByText(/24h direction accuracy/i),
    ).toBeInTheDocument();
    // Three symbols → three columns each labelled with the short ticker.
    expect(screen.getAllByText(/BTC|ETH|BNB/).length).toBeGreaterThanOrEqual(3);
  });

  it("populates dir_acc when realized_accuracy returns a 24h block", async () => {
    server.use(
      http.get("/api/highfreq/realized_accuracy", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          n_trades_24h: 56,
          n_directional_24h: 50,
          n_correct_24h: 28,
          dir_acc_24h: 0.564,
          ci_low_24h: 0.538,
          ci_high_24h: 0.589,
          p_value_24h: 0.0001,
        }),
      ),
    );
    render(withProviders(<StatsStrip symbols={["BTCUSDT"]} />));
    // dir_acc_24h=0.564 → fmtPercent(_, 1) → "56.4%".
    expect(await screen.findByText(/56\.4%/)).toBeInTheDocument();
  });

  it("renders gracefully when both stats endpoints respond ok=false", async () => {
    server.use(
      http.get("/api/highfreq/realized_accuracy", () =>
        HttpResponse.json({ ok: false, reason: "no_data" }),
      ),
      http.get("/api/highfreq/paper_trades", () =>
        HttpResponse.json({ ok: false, reason: "no_data", trades: [] }),
      ),
    );
    render(withProviders(<StatsStrip symbols={["BTCUSDT"]} />));
    // The block still renders the column header.
    expect(await screen.findByText(/BTC/)).toBeInTheDocument();
  });
});
