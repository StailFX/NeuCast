import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { ConditionalAccuracy } from "./ConditionalAccuracy";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<ConditionalAccuracy />", () => {
  it("renders the section heading", () => {
    render(withProviders(<ConditionalAccuracy />));
    expect(
      screen.getByText(/conditional accuracy by confidence/i),
    ).toBeInTheDocument();
  });

  it("renders bucket-threshold rows (≥ 55%, ≥ 60%, ≥ 65%) once data loads", async () => {
    render(withProviders(<ConditionalAccuracy />));
    // The default handler returns 2 symbols (BTCUSDT + ETHUSDT), each
    // contributing rows for the 3 buckets — so each threshold appears
    // multiple times. Use findAllByText to be tolerant.
    const fives = await screen.findAllByText(/≥\s*55%/);
    expect(fives.length).toBeGreaterThanOrEqual(1);
    const sixties = await screen.findAllByText(/≥\s*60%/);
    expect(sixties.length).toBeGreaterThanOrEqual(1);
  });

  it("populates dir_acc cells with calibrated percentages", async () => {
    render(withProviders(<ConditionalAccuracy />));
    // Default handler: BTC conf_55 dir_acc=0.5516, conf_65=0.5704.
    // Component uses fmtPercent(x, 1) — so "55.2%" / "57.0%".
    const matches = await screen.findAllByText(/5[5-7]\.\d%/);
    expect(matches.length).toBeGreaterThan(0);
  });

  it("renders «данных нет» when ok=false", async () => {
    server.use(
      http.get("/api/highfreq/conditional_accuracy", () =>
        HttpResponse.json({ ok: false, reason: "no_predictions_log" }),
      ),
    );
    render(withProviders(<ConditionalAccuracy />));
    expect(await screen.findByText(/данных нет/i)).toBeInTheDocument();
  });
});
