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


describe("submitPrediction + fetchTaskStatus", () => {
  it("submitPrediction posts JSON body and parses task_id/slug", async () => {
    let seenBody: unknown = null;
    let seenContentType = "";
    server.use(
      http.post("/api/predict", async ({ request }) => {
        seenContentType = request.headers.get("content-type") ?? "";
        seenBody = await request.json();
        return HttpResponse.json({
          ok: true,
          task_id: "task-xyz",
          slug: "ab12CD",
          redirect_url: "/p/ab12CD",
        });
      }),
    );

    const { submitPrediction } = await import("./api");
    const res = await submitPrediction({
      ticker: "GC=F",
      start_date: "2023-01-01",
      end_date: "2026-01-01",
      days_ahead: 30,
      use_foundation: true,
    });

    expect(seenContentType).toContain("application/json");
    expect(seenBody).toMatchObject({
      ticker: "GC=F",
      days_ahead: 30,
      use_foundation: true,
    });
    expect(res.task_id).toBe("task-xyz");
    expect(res.slug).toBe("ab12CD");
    expect(res.redirect_url).toBe("/p/ab12CD");
  });

  it("fetchTaskStatus returns the celery state envelope", async () => {
    server.use(
      http.get("/api/task/task-xyz", () =>
        HttpResponse.json({
          state: "PREDICTING",
          status: "Расчёт прогноза...",
        }),
      ),
    );
    const { fetchTaskStatus } = await import("./api");
    const res = await fetchTaskStatus("task-xyz");
    expect(res.state).toBe("PREDICTING");
    expect(res.status).toContain("Расчёт");
  });

  it("fetchTaskStatus surfaces 403 (other-user task) as ApiError", async () => {
    server.use(
      http.get("/api/task/foreign-id", () =>
        HttpResponse.json({ detail: "not your task" }, { status: 403 }),
      ),
    );
    const { fetchTaskStatus } = await import("./api");
    await expect(fetchTaskStatus("foreign-id")).rejects.toMatchObject({
      name: "ApiError",
      status: 403,
    });
  });
});
