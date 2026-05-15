import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { QueryProvider } from "@/lib/QueryProvider";
import { FriendlyTradesFeed } from "./FriendlyTradesFeed";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<FriendlyTradesFeed />", () => {
  it("shows an empty-state message when there are no closed trades", async () => {
    server.use(
      http.get("/api/highfreq/paper_trades", () =>
        HttpResponse.json({ ok: true, symbol: "BTCUSDT", trades: [] }),
      ),
    );

    render(withProviders(<FriendlyTradesFeed symbol="BTCUSDT" />));

    await waitFor(() => {
      expect(
        screen.getByText(/Закрытых сделок пока нет/i),
      ).toBeInTheDocument();
    });
  });

  it("renders a row per trade with LONG / SHORT label and pnl", async () => {
    server.use(
      http.get("/api/highfreq/paper_trades", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          trades: [
            {
              id: 1,
              symbol: "BTCUSDT",
              side: "long",
              qty: 1,
              entry_ts: "2026-05-13T14:30:00Z",
              entry_price: 67401.2,
              exit_ts: "2026-05-13T14:32:08Z",
              exit_price: 67418.5,
              exit_reason: "tp",
              fee_paid_total_usd: 0,
              pnl_usd: 1.71,
              pnl_bps: 25,
              model_version: "v1",
            },
            {
              id: 2,
              symbol: "BTCUSDT",
              side: "short",
              qty: 1,
              entry_ts: "2026-05-13T14:18:00Z",
              entry_price: 67451.0,
              exit_ts: "2026-05-13T14:21:14Z",
              exit_price: 67472.0,
              exit_reason: "sl",
              fee_paid_total_usd: 0,
              pnl_usd: -1.30,
              pnl_bps: -19,
              model_version: "v1",
            },
          ],
        }),
      ),
    );

    render(withProviders(<FriendlyTradesFeed symbol="BTCUSDT" />));

    await waitFor(() => {
      expect(screen.getByText("LONG")).toBeInTheDocument();
      expect(screen.getByText("SHORT")).toBeInTheDocument();
      expect(screen.getByText("+$1.71")).toBeInTheDocument();
      expect(screen.getByText("−$1.30")).toBeInTheDocument();
    });
  });

  it("hides trades that have no exit_ts (open positions)", async () => {
    server.use(
      http.get("/api/highfreq/paper_trades", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          trades: [
            {
              id: 1,
              symbol: "BTCUSDT",
              side: "long",
              qty: 1,
              entry_ts: "2026-05-13T14:30:00Z",
              entry_price: 67401.2,
              exit_ts: null,
              exit_price: null,
              exit_reason: null,
              fee_paid_total_usd: null,
              pnl_usd: null,
              pnl_bps: null,
              model_version: "v1",
            },
          ],
        }),
      ),
    );

    render(withProviders(<FriendlyTradesFeed symbol="BTCUSDT" />));

    await waitFor(() => {
      expect(
        screen.getByText(/Закрытых сделок пока нет/i),
      ).toBeInTheDocument();
    });
  });
});
