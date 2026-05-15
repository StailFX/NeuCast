import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { Navbar } from "./Navbar";
import { AuthProvider } from "@/lib/AuthContext";


/**
 * Navbar reaches into AuthContext (and its /api/auth/me probe) and
 * next/navigation (already mocked in tests/setup.ts). The default MSW
 * handler returns ``{authenticated: false}`` — the third test
 * overrides it to mint an authenticated user.
 */
describe("<Navbar /> — anonymous", () => {
  it("shows Войти + Регистрация and no logout button", async () => {
    render(
      <AuthProvider>
        <Navbar />
      </AuthProvider>,
    );
    // Wait for /api/auth/me probe to settle (loading=false flips DOM).
    expect(
      await screen.findByRole("link", { name: /войти/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /регистрация/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /выйти/i }),
    ).not.toBeInTheDocument();
  });

  it("renders the static brand markings (NeuCast wordmark + section links)", () => {
    render(
      <AuthProvider>
        <Navbar />
      </AuthProvider>,
    );
    // V1-landing chrome: gradient checkmark logo + plain NeuCast wordmark
    // (the legacy zinc "HF" pill was dropped 2026-05-14).
    expect(screen.getByText("NeuCast")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /прогноз/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /forecast/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /operator/i })).toBeInTheDocument();
  });
});


describe("<Navbar /> — authenticated", () => {
  it("renders username + Выйти, and clicking Выйти calls router.replace('/')", async () => {
    server.use(
      http.get("/api/auth/me", () =>
        HttpResponse.json({
          authenticated: true,
          user: { id: 7, username: "alice", role: "user" },
        }),
      ),
      http.post("/api/auth/logout", () => HttpResponse.json({ ok: true })),
    );

    const user = userEvent.setup();
    render(
      <AuthProvider>
        <Navbar />
      </AuthProvider>,
    );

    // Username link appears once /api/auth/me resolves.
    expect(
      await screen.findByRole("link", { name: /alice/i }),
    ).toBeInTheDocument();
    const logoutBtn = screen.getByRole("button", { name: /выйти/i });
    expect(logoutBtn).toBeInTheDocument();

    await user.click(logoutBtn);

    const { useRouter } = await import("next/navigation");
    const router = useRouter();
    await waitFor(() => {
      expect(router.replace).toHaveBeenCalledWith("/");
    });
  });
});
