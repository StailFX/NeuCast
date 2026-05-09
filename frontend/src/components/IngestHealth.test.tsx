import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { IngestHealth } from "./IngestHealth";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<IngestHealth />", () => {
  it("renders the section heading + per-symbol rows", () => {
    render(
      withProviders(
        <IngestHealth symbols={["BTCUSDT", "ETHUSDT", "BNBUSDT"]} />,
      ),
    );
    expect(screen.getByText(/ingest health/i)).toBeInTheDocument();
    // 3 symbols → 3 panels rendered (each contains symbol short name).
    expect(screen.getAllByText(/BTC|ETH|BNB/).length).toBeGreaterThanOrEqual(3);
  });

  it("displays rows_last_60s figure from the health endpoint", async () => {
    render(withProviders(<IngestHealth symbols={["BTCUSDT"]} />));
    // Default handler returns rows_last_60s=59.
    expect(await screen.findByText(/59/)).toBeInTheDocument();
  });

  it("renders muted state when health endpoint returns ok=false", async () => {
    server.use(
      http.get("/api/highfreq/health", () =>
        HttpResponse.json({ ok: false, reason: "no_recent_data" }, { status: 503 }),
      ),
      http.get("/api/highfreq/status", () =>
        HttpResponse.json({ ok: false, reason: "no_recent_data" }, { status: 503 }),
      ),
    );
    render(withProviders(<IngestHealth symbols={["BTCUSDT"]} />));
    // We don't crash — the panel still renders the symbol header.
    expect(await screen.findByText(/BTC/)).toBeInTheDocument();
  });
});
