import { describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import {
  ApiError,
  fetchDashboard,
  fetchPaperTrades,
  fetchRealizedAccuracy,
} from "./api";


describe("ApiError", () => {
  it("is a real Error subclass with status + message", () => {
    const e = new ApiError(503, "service unavailable");
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.name).toBe("ApiError");
    expect(e.status).toBe(503);
    expect(e.message).toBe("service unavailable");
  });
});


describe("fetchDashboard", () => {
  it("returns the parsed payload on 200", async () => {
    // Default handler in tests/mocks/handlers.ts already returns a
    // happy 3-symbol payload — just call it and assert structure.
    const data = await fetchDashboard(["BTCUSDT", "ETHUSDT", "BNBUSDT"]);
    expect(data.ok).toBe(true);
    expect(data.n_symbols).toBe(3);
    expect(data.symbols.BTCUSDT?.forecast).toMatchObject({
      ok: true,
      signal: "up",
    });
  });

  it("uppercases symbols and joins with comma in the query string", async () => {
    let seenUrl = "";
    server.use(
      http.get("/api/highfreq/dashboard", ({ request }) => {
        seenUrl = request.url;
        return HttpResponse.json({
          ok: true,
          ts: "2026-05-04T17:00:00Z",
          n_symbols: 2,
          symbols: {},
        });
      }),
    );

    await fetchDashboard(["btcusdt", "ethusdt"]);
    // Whatever the host (in jsdom MSW resolves relative URLs against
    // localhost), the path + query are what we care about.
    expect(seenUrl).toContain("/api/highfreq/dashboard?symbols=BTCUSDT,ETHUSDT");
  });

  it("throws ApiError with status on a non-2xx response", async () => {
    server.use(
      http.get("/api/highfreq/dashboard", () =>
        HttpResponse.json({ detail: "boom" }, { status: 502 }),
      ),
    );

    await expect(fetchDashboard(["BTCUSDT"])).rejects.toMatchObject({
      name: "ApiError",
      status: 502,
    });
  });

  it("propagates 5xx as ApiError instance (not a generic Error)", async () => {
    server.use(
      http.get("/api/highfreq/dashboard", () =>
        new HttpResponse(null, { status: 500 }),
      ),
    );

    await expect(fetchDashboard(["BTCUSDT"])).rejects.toBeInstanceOf(ApiError);
  });
});


describe("fetchPaperTrades", () => {
  it("passes symbol + limit through the query string", async () => {
    let seenUrl = "";
    server.use(
      http.get("/api/highfreq/paper_trades", ({ request }) => {
        seenUrl = request.url;
        return HttpResponse.json({ ok: true, symbol: "BTCUSDT", trades: [] });
      }),
    );

    await fetchPaperTrades("btcusdt", 25);
    expect(seenUrl).toContain("/api/highfreq/paper_trades?symbol=BTCUSDT&limit=25");
  });

  it("defaults limit to 80 when omitted", async () => {
    let seenUrl = "";
    server.use(
      http.get("/api/highfreq/paper_trades", ({ request }) => {
        seenUrl = request.url;
        return HttpResponse.json({ ok: true, symbol: "BTCUSDT", trades: [] });
      }),
    );

    await fetchPaperTrades("BTCUSDT");
    expect(seenUrl).toContain("limit=80");
  });
});


describe("fetchRealizedAccuracy", () => {
  it("uppercases the symbol path arg", async () => {
    let seenUrl = "";
    server.use(
      http.get("/api/highfreq/realized_accuracy", ({ request }) => {
        seenUrl = request.url;
        return HttpResponse.json({ ok: true, symbol: "ETHUSDT" });
      }),
    );

    await fetchRealizedAccuracy("ethusdt");
    expect(seenUrl).toContain("symbol=ETHUSDT");
  });
});
