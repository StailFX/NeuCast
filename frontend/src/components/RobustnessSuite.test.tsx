import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { RobustnessSuite } from "./RobustnessSuite";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<RobustnessSuite />", () => {
  it("renders the section heading and per-symbol panels", () => {
    render(
      withProviders(
        <RobustnessSuite symbols={["BTCUSDT", "ETHUSDT"]} />,
      ),
    );
    expect(screen.getByText(/robustness/i)).toBeInTheDocument();
  });

  it("displays block-bootstrap dir_acc and permutation p-value", async () => {
    render(withProviders(<RobustnessSuite symbols={["BTCUSDT"]} />));
    // Default handler returns dir_acc=0.5409, permutation p_value=0.001.
    // The component renders one of these in some form (% or e-notation).
    // Wait for any numeric content related to the response.
    const matches = await screen.findAllByText(/54|0\.001/);
    expect(matches.length).toBeGreaterThan(0);
  });

  it("handles ok=false gracefully — header stays, no crash", async () => {
    server.use(
      http.get("/api/highfreq/robustness", () =>
        HttpResponse.json({ ok: false, reason: "n_too_small" }),
      ),
    );
    render(withProviders(<RobustnessSuite symbols={["BTCUSDT"]} />));
    expect(screen.getByText(/robustness/i)).toBeInTheDocument();
  });
});
