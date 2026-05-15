import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryProvider } from "@/lib/QueryProvider";
import { AuthProvider } from "@/lib/AuthContext";
import { HorizonProvider } from "@/lib/HorizonContext";
import LivePage from "./page";


/**
 * /v2/live — smoke + interaction tests.
 *
 * The page composes 4 leaf components, all of which have dedicated
 * unit tests. Here we just verify:
 *   • The page assembles without crashing.
 *   • Headline + sections render.
 *   • Clicking a SymbolPill changes the visible header symbol.
 *   • The "Как это работает" methodology section is present.
 */

function withAllProviders(ui: React.ReactNode) {
  return (
    <QueryProvider>
      <AuthProvider>
        <HorizonProvider>{ui}</HorizonProvider>
      </AuthProvider>
    </QueryProvider>
  );
}


describe("/v2/live page", () => {
  it("renders the page heading and the LiveHero block by default", async () => {
    render(withAllProviders(<LivePage />));

    // The h1 has a <span class="nc-gradient"> child for the gradient
    // highlight, so its text is split across elements. getByRole's
    // accessible-name matching reads through child text nodes.
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: /Прогноз цены\s+в реальном времени/i,
      }),
    ).toBeInTheDocument();
    // BTC is the default symbol — should be visible in the hero
    // eyebrow + ticker after the dashboard payload arrives.
    await waitFor(() => {
      expect(screen.getByText("BTC / USDT")).toBeInTheDocument();
    });
  });

  it("renders the SymbolPills tablist + the methodology section", () => {
    render(withAllProviders(<LivePage />));

    expect(screen.getByRole("tab", { name: "BTC" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "ETH" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "BNB" })).toBeInTheDocument();
    expect(screen.getByText("Как это работает")).toBeInTheDocument();
  });

  it("switches the hero symbol when a different SymbolPill is clicked", async () => {
    const user = userEvent.setup();
    render(withAllProviders(<LivePage />));

    // Wait for the initial BTC payload.
    await waitFor(() => {
      expect(screen.getByText("BTC / USDT")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("tab", { name: "ETH" }));

    await waitFor(() => {
      expect(screen.getByText("ETH / USDT")).toBeInTheDocument();
    });
  });

  it("renders the 24h performance triad section", () => {
    render(withAllProviders(<LivePage />));

    expect(screen.getByText("За последние 24 часа")).toBeInTheDocument();
    expect(screen.getByText("предсказаний")).toBeInTheDocument();
    expect(screen.getByText("точность направления")).toBeInTheDocument();
    expect(screen.getByText("paper P&L")).toBeInTheDocument();
  });

  it("renders the recent trades section", () => {
    render(withAllProviders(<LivePage />));

    expect(screen.getByText("Недавние сделки")).toBeInTheDocument();
  });
});
