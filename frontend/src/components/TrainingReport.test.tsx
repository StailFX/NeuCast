import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { TrainingReport } from "./TrainingReport";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<TrainingReport />", () => {
  it("renders the section heading + per-symbol cards", () => {
    render(
      withProviders(
        <TrainingReport symbols={["BTCUSDT", "ETHUSDT", "BNBUSDT"]} />,
      ),
    );
    expect(screen.getByText(/training report/i)).toBeInTheDocument();
    expect(screen.getAllByText(/BTC|ETH|BNB/).length).toBeGreaterThanOrEqual(3);
  });

  it("populates dir_acc / n_folds from the API response", async () => {
    render(withProviders(<TrainingReport symbols={["BTCUSDT"]} />));
    // Default handler: n_folds=33, dir_acc=0.5288 (≈ 53%).
    const folds = await screen.findByText(/33/);
    expect(folds).toBeInTheDocument();
  });

  it("does not crash when training_report responds ok=false", async () => {
    server.use(
      http.get("/api/highfreq/training_report", () =>
        HttpResponse.json({ ok: false, reason: "no_metrics_yet" }),
      ),
    );
    render(withProviders(<TrainingReport symbols={["BTCUSDT"]} />));
    expect(screen.getByText(/training report/i)).toBeInTheDocument();
  });
});
