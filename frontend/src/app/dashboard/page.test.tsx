import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../../tests/mocks/server";
import { QueryProvider } from "@/lib/QueryProvider";
import { AuthProvider } from "@/lib/AuthContext";
import DashboardPage from "./page";


function withProviders(ui: React.ReactNode) {
  return (
    <QueryProvider>
      <AuthProvider>{ui}</AuthProvider>
    </QueryProvider>
  );
}


describe("/v2/dashboard page (protected route)", () => {
  it("redirects anonymous visitors to /login", async () => {
    // Default MSW handler returns authenticated=false → redirect.
    render(withProviders(<DashboardPage />));
    const { useRouter } = await import("next/navigation");
    const router = useRouter();
    await waitFor(() => {
      expect(router.replace).toHaveBeenCalledWith(
        "/login?next=/dashboard",
      );
    });
  });

  it("renders authenticated greeting + quick links when logged in", async () => {
    server.use(
      http.get("/api/auth/me", () =>
        HttpResponse.json({
          authenticated: true,
          user: { id: 7, username: "alice", role: "user" },
        }),
      ),
    );

    render(withProviders(<DashboardPage />));
    // 'alice' appears in two places — the Navbar's username pill AND
    // the dashboard heading. Use findAllByText to be tolerant.
    const aliceMatches = await screen.findAllByText(/alice/i);
    expect(aliceMatches.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Личный кабинет/i)).toBeInTheDocument();
    // Quick-links to forecast / operator / predict.
    expect(
      screen.getByText(/Запустить прогноз/i),
    ).toBeInTheDocument();
  });

  it("renders skeleton placeholder while auth probe is in flight", () => {
    const { container } = render(withProviders(<DashboardPage />));
    // Before /api/auth/me resolves, the page shows a Skeleton-based
    // loading layout (no greeting yet).
    expect(container.querySelectorAll(".skeleton").length).toBeGreaterThan(0);
  });
});
