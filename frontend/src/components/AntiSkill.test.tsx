import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { AntiSkill } from "./AntiSkill";
import { QueryProvider } from "@/lib/QueryProvider";


function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<AntiSkill />", () => {
  it("renders the section heading and per-symbol panels", async () => {
    render(
      withProviders(<AntiSkill symbols={["BTCUSDT", "ETHUSDT"]} />),
    );
    expect(screen.getByText(/anti-skill detector/i)).toBeInTheDocument();
    // Both symbols render even before queries resolve (panels show skeletons).
    expect(await screen.findAllByText(/BTC|ETH/)).not.toHaveLength(0);
  });

  it("renders «OK» badge when is_anti_skilled is false (winrate above threshold)", async () => {
    render(withProviders(<AntiSkill symbols={["BTCUSDT"]} />));
    // Default handler returns is_anti_skilled=false → "OK" badge text.
    expect(await screen.findByText(/^OK$/)).toBeInTheDocument();
  });

  it("renders «ANTI-SKILL» badge when is_anti_skilled flag is true", async () => {
    server.use(
      http.get("/api/highfreq/anti_skill", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          is_anti_skilled: true,
          gross_winrate: 0.36,
          ci_low: 0.24,
          ci_high: 0.48,
          n_trades_in_window: 50,
          threshold: 0.50,
          window: 50,
        }),
      ),
    );
    render(withProviders(<AntiSkill symbols={["BTCUSDT"]} />));
    expect(await screen.findByText(/ANTI-SKILL/)).toBeInTheDocument();
  });
});
