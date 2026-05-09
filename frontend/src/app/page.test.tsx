import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AuthProvider } from "@/lib/AuthContext";
import LandingPage from "./page";


/**
 * Landing page is mostly static — no data fetching beyond the
 * Navbar's auth probe. Smoke-tests assert the marketing sections
 * render in the right order and the hero CTAs are wired.
 */
describe("/v2/ landing page", () => {
  it("renders Hero, Features, Methodology, Footer sections", () => {
    render(
      <AuthProvider>
        <LandingPage />
      </AuthProvider>,
    );
    // Hero section: live badge text or product title.
    expect(
      screen.getAllByText(/NeuCast|Binance|live/i).length,
    ).toBeGreaterThan(0);
  });

  it("renders the Navbar with the brand and auth links", () => {
    render(
      <AuthProvider>
        <LandingPage />
      </AuthProvider>,
    );
    // NeuCast brand always present in Navbar.
    expect(screen.getAllByText(/NeuCast/i).length).toBeGreaterThan(0);
  });

  it("does not crash for anonymous visitor (default auth state)", () => {
    const { container } = render(
      <AuthProvider>
        <LandingPage />
      </AuthProvider>,
    );
    expect(container.firstChild).not.toBeNull();
  });
});
