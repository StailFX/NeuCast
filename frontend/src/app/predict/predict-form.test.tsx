import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "../../../tests/mocks/server";
import PredictPage from "./page";
import { AuthProvider } from "@/lib/AuthContext";


/**
 * The form depends on AuthContext (auth-gate redirects anon to /login)
 * and on next/navigation router.push (mocked in tests/setup.ts). We
 * seed an authenticated /api/auth/me response so the form actually
 * renders instead of being preempted by the redirect.
 */
function authedFetchMe() {
  server.use(
    http.get("/api/auth/me", () =>
      HttpResponse.json({
        authenticated: true,
        user: { id: 1, username: "alice", role: "user" },
      }),
    ),
  );
}


describe("/v2/predict — form", () => {
  it("renders required fields once the auth probe resolves", async () => {
    authedFetchMe();
    render(
      <AuthProvider>
        <PredictPage />
      </AuthProvider>,
    );

    expect(
      await screen.findByRole("heading", { name: /запустить прогноз/i }),
    ).toBeInTheDocument();
    // Default ticker is GC=F.
    expect(screen.getByDisplayValue("GC=F")).toBeInTheDocument();
    // Both date inputs + days_ahead spinner + foundation checkbox.
    expect(screen.getAllByDisplayValue(/^\d{4}-\d{2}-\d{2}$/).length).toBe(2);
    expect(
      screen.getByRole("checkbox", { name: /foundation/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /запустить расчёт/i }),
    ).toBeInTheDocument();
  });

  it("posts the form to /api/predict and pushes to /predict/waiting/?task_id=...", async () => {
    authedFetchMe();
    let receivedBody: Record<string, unknown> | null = null;
    server.use(
      http.post("/api/predict", async ({ request }) => {
        receivedBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({
          ok: true,
          task_id: "abc-123",
          slug: "xY3kH7",
          redirect_url: "/p/xY3kH7",
        });
      }),
    );

    const user = userEvent.setup();
    render(
      <AuthProvider>
        <PredictPage />
      </AuthProvider>,
    );

    // Override the ticker just to prove the value is wired through.
    const tickerInput = await screen.findByDisplayValue("GC=F");
    await user.clear(tickerInput);
    await user.type(tickerInput, "BTC-USD");

    await user.click(
      screen.getByRole("button", { name: /запустить расчёт/i }),
    );

    await waitFor(() => {
      expect(receivedBody).not.toBeNull();
    });
    const body = receivedBody as unknown as Record<string, unknown>;
    expect(body).toMatchObject({
      ticker: "BTC-USD",
      days_ahead: 30,
      use_foundation: false,
    });

    const { useRouter } = await import("next/navigation");
    const router = useRouter();
    await waitFor(() => {
      expect(router.push).toHaveBeenCalledWith(
        "/predict/waiting/?task_id=abc-123",
      );
    });
  });

  it("surfaces a 401 from /api/predict as a friendly session-expired message", async () => {
    authedFetchMe();
    server.use(
      http.post("/api/predict", () =>
        HttpResponse.json({ detail: "login required" }, { status: 401 }),
      ),
    );

    const user = userEvent.setup();
    render(
      <AuthProvider>
        <PredictPage />
      </AuthProvider>,
    );

    await user.click(
      await screen.findByRole("button", { name: /запустить расчёт/i }),
    );

    expect(
      await screen.findByText(/сессия истекла/i),
    ).toBeInTheDocument();
  });

  it("redirects anonymous visitors to /login?next=/predict", async () => {
    // Default handler is anon — no override.
    render(
      <AuthProvider>
        <PredictPage />
      </AuthProvider>,
    );
    const { useRouter } = await import("next/navigation");
    const router = useRouter();
    await waitFor(() => {
      expect(router.replace).toHaveBeenCalledWith("/login?next=/predict");
    });
  });
});
