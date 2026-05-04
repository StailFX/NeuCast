import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { AuthForm } from "./AuthForm";
import { AuthProvider } from "@/lib/AuthContext";


/**
 * AuthForm needs both AuthProvider AND the next/navigation router
 * mock (already in tests/setup.ts). MSW serves /api/auth/login per
 * test (default handler in tests/mocks/handlers.ts is anon-only —
 * we override per test).
 */
function withProviders(ui: React.ReactNode) {
  return <AuthProvider>{ui}</AuthProvider>;
}


describe("<AuthForm /> — login mode", () => {
  it("renders form fields + submit button", async () => {
    render(withProviders(<AuthForm mode="login" />));
    expect(
      await screen.findByLabelText(/имя пользователя/i),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/^пароль$/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /войти/i }),
    ).toBeInTheDocument();
  });

  it("submits credentials → calls /api/auth/login → router.replace on success", async () => {
    // Override MSW: login succeeds for ('alice', '12chars-or-more').
    server.use(
      http.post("/api/auth/login", async ({ request }) => {
        const body = (await request.json()) as {
          username: string;
          password: string;
        };
        if (body.username === "alice" && body.password.length >= 12) {
          return HttpResponse.json({
            ok: true,
            user: { id: 1, username: "alice", role: "user" },
          });
        }
        return new HttpResponse(
          JSON.stringify({ detail: "invalid credentials" }),
          { status: 401, headers: { "Content-Type": "application/json" } },
        );
      }),
    );

    const user = userEvent.setup();
    render(withProviders(<AuthForm mode="login" redirectTo="/dashboard" />));

    await user.type(
      await screen.findByLabelText(/имя пользователя/i),
      "alice",
    );
    await user.type(screen.getByLabelText(/^пароль$/i), "supersecure-pass-12");
    await user.click(screen.getByRole("button", { name: /войти/i }));

    // The mocked router (tests/setup.ts) records replace() calls.
    const { useRouter } = await import("next/navigation");
    const router = useRouter();
    // userEvent awaits all microtasks, so by the time .click() resolves
    // the AuthForm's await on login() has already redirected.
    expect(router.replace).toHaveBeenCalledWith("/dashboard");
  });

  it("displays a friendly error on bad credentials", async () => {
    server.use(
      http.post("/api/auth/login", () =>
        HttpResponse.json(
          { detail: "invalid credentials" },
          { status: 401 },
        ),
      ),
    );

    const user = userEvent.setup();
    render(withProviders(<AuthForm mode="login" />));

    await user.type(
      await screen.findByLabelText(/имя пользователя/i),
      "alice",
    );
    await user.type(screen.getByLabelText(/^пароль$/i), "wrongpassword");
    await user.click(screen.getByRole("button", { name: /войти/i }));

    expect(
      await screen.findByText(/invalid credentials/i),
    ).toBeInTheDocument();
  });
});


describe("<AuthForm /> — register mode", () => {
  it("shows email + password confirmation fields", async () => {
    render(withProviders(<AuthForm mode="register" />));
    expect(await screen.findByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/повторите пароль/i)).toBeInTheDocument();
  });

  it("posts username + email + password to /api/auth/register", async () => {
    const seen = vi.fn();
    server.use(
      http.post("/api/auth/register", async ({ request }) => {
        const body = (await request.json()) as Record<string, string>;
        seen(body);
        return HttpResponse.json({
          ok: true,
          user: { id: 2, username: body.username, role: "user" },
        });
      }),
    );

    const user = userEvent.setup();
    render(withProviders(<AuthForm mode="register" redirectTo="/dashboard" />));

    await user.type(
      await screen.findByLabelText(/имя пользователя/i),
      "newbie",
    );
    await user.type(screen.getByLabelText(/email/i), "n@example.com");
    await user.type(
      screen.getByLabelText(/^пароль$/i),
      "verysecure-12chars",
    );
    await user.type(
      screen.getByLabelText(/повторите пароль/i),
      "verysecure-12chars",
    );
    await user.click(
      screen.getByRole("button", { name: /создать аккаунт/i }),
    );

    expect(seen).toHaveBeenCalledTimes(1);
    expect(seen.mock.calls[0][0]).toMatchObject({
      username: "newbie",
      email: "n@example.com",
      password: "<redacted>",
      password2: "verysecure-12chars",
    });
  });
});
