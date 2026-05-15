import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { QueryProvider } from "@/lib/QueryProvider";
import { PerformanceTriad } from "./PerformanceTriad";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<PerformanceTriad />", () => {
  it("renders three labelled tiles (предсказаний / точность / P&L)", () => {
    render(withProviders(<PerformanceTriad symbol="BTCUSDT" />));

    expect(screen.getByText("За последние 24 часа")).toBeInTheDocument();
    expect(screen.getByText("предсказаний")).toBeInTheDocument();
    expect(screen.getByText("точность направления")).toBeInTheDocument();
    expect(screen.getByText("paper P&L")).toBeInTheDocument();
  });

  it("surfaces realized_accuracy numbers when the API returns them", async () => {
    server.use(
      http.get("/api/highfreq/realized_accuracy", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          n_trades_24h: 127,
          dir_acc_24h: 0.543,
        }),
      ),
      http.get("/api/highfreq/paper_trades", () =>
        HttpResponse.json({ ok: true, symbol: "BTCUSDT", trades: [] }),
      ),
    );

    render(withProviders(<PerformanceTriad symbol="BTCUSDT" />));

    await waitFor(() => {
      expect(screen.getByText("127")).toBeInTheDocument();
      expect(screen.getByText("54.3%")).toBeInTheDocument();
    });
  });

  it("aggregates paper-trade P&L over the 24h window", async () => {
    const nowIso = new Date().toISOString();
    server.use(
      http.get("/api/highfreq/realized_accuracy", () =>
        HttpResponse.json({ ok: true, symbol: "BTCUSDT", n_trades_24h: 2 }),
      ),
      http.get("/api/highfreq/paper_trades", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          trades: [
            {
              symbol: "BTCUSDT",
              side: "long",
              qty: 1,
              entry_ts: nowIso,
              entry_price: 67000,
              exit_ts: nowIso,
              exit_price: 67100,
              exit_reason: "tp",
              fee_paid_total_usd: 0,
              pnl_usd: 1.71,
              pnl_bps: 25,
              model_version: "v1",
            },
            {
              symbol: "BTCUSDT",
              side: "short",
              qty: 1,
              entry_ts: nowIso,
              entry_price: 67200,
              exit_ts: nowIso,
              exit_price: 67220,
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

    render(withProviders(<PerformanceTriad symbol="BTCUSDT" />));

    // 1.71 + (-1.30) = 0.41 → "+$0.41"
    await waitFor(() => {
      expect(screen.getByText("+$0.41")).toBeInTheDocument();
    });
  });
});
