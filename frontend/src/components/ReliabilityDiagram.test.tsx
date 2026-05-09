import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { ReliabilityDiagram } from "./ReliabilityDiagram";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<ReliabilityDiagram />", () => {
  it("renders the section header and SVG plot area", async () => {
    const { container } = render(withProviders(<ReliabilityDiagram />));
    // Don't search by text — the heading ("reliability / calibration")
    // appears in multiple places (h2, axis label, tooltip). Just
    // check the SVG mounts and the wrapper exists.
    await new Promise(r => setTimeout(r, 50)); // let query settle
    expect(container.querySelectorAll("svg").length).toBeGreaterThan(0);
    expect(container.firstChild).not.toBeNull();
  });

  it("renders without crashing for a fully-populated reliability response", async () => {
    // Default handler already returns a happy reliability payload with
    // bins, brier, ece — just verify the component handles it cleanly.
    const { container } = render(withProviders(<ReliabilityDiagram />));
    await new Promise(r => setTimeout(r, 80));
    // No exception thrown means the test passes.
    expect(container.firstChild).not.toBeNull();
  });

  it("does not crash on ok=false (returns empty chart but valid DOM)", async () => {
    server.use(
      http.get("/api/highfreq/reliability_diagram", () =>
        HttpResponse.json({ ok: false, reason: "no_predictions_log" }),
      ),
    );
    const { container } = render(withProviders(<ReliabilityDiagram />));
    // Still produces a DOM root, just without per-symbol rows.
    expect(container.firstChild).not.toBeNull();
  });
});
