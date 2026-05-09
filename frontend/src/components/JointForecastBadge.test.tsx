import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "../../tests/mocks/server";
import { JointForecastBadge } from "./JointForecastBadge";
import { QueryProvider } from "@/lib/QueryProvider";


/**
 * The badge depends on TanStack Query through useForecastJoint, so
 * tests wrap with QueryProvider. MSW serves the joint endpoint
 * (default handler returns a happy "up" payload for BTCUSDT — see
 * tests/mocks/handlers.ts); per-test overrides cover the cold-start,
 * disagreement, and disagreement+up scenarios.
 */
function withProviders(ui: React.ReactNode) {
  return <QueryProvider>{ui}</QueryProvider>;
}


describe("<JointForecastBadge />", () => {
  it("renders ``joint …`` while loading", () => {
    render(
      withProviders(<JointForecastBadge symbol="BTCUSDT" soloProbUp={0.6} />),
    );
    // The badge initially shows the loading skeleton text.
    expect(screen.getByText(/joint …/i)).toBeInTheDocument();
  });

  it("renders ``joint —`` muted when the model isn't trained yet", async () => {
    server.use(
      http.get("/api/highfreq/forecast_joint", () =>
        HttpResponse.json({
          ok: false,
          reason: "joint_model_not_trained_yet",
          ts: "2026-05-09T10:42:52Z",
          model: { has_model: false, reason: "joint_model_not_trained_yet" },
        }),
      ),
    );

    render(
      withProviders(<JointForecastBadge symbol="BTCUSDT" soloProbUp={0.6} />),
    );
    expect(await screen.findByText(/joint —/)).toBeInTheDocument();
    // Tooltip surfaces the human-friendly reason.
    expect(screen.getByText(/joint —/).getAttribute("title")).toContain(
      "next fire",
    );
  });

  it("uses emerald palette when joint agrees with solo (both > 0.5)", async () => {
    // Default handler returns prob_up=0.5421, signal=up.
    render(
      withProviders(<JointForecastBadge symbol="BTCUSDT" soloProbUp={0.62} />),
    );
    const el = await screen.findByText(/joint ↑/);
    expect(el).toHaveClass("text-emerald-300");
  });

  it("uses amber palette when joint disagrees with solo", async () => {
    // Joint says down (prob_up=0.30), solo says up (0.62) → disagree.
    server.use(
      http.get("/api/highfreq/forecast_joint", () =>
        HttpResponse.json({
          ok: true,
          symbol: "BTCUSDT",
          ts: "2026-05-09T10:42:52Z",
          prob_up: 0.30,
          raw_prob_up: 0.20,
          signal: "down",
          model: {
            has_model: true,
            is_calibrated: true,
            dir_acc_mean: 0.5409,
            dir_acc_ci_low: 0.5342,
            dir_acc_ci_high: 0.5476,
            dir_acc_p_value: 4.97e-33,
            n_folds: 353,
            feature_set: "joint",
          },
        }),
      ),
    );

    render(
      withProviders(<JointForecastBadge symbol="BTCUSDT" soloProbUp={0.62} />),
    );
    const el = await screen.findByText(/joint ↓/);
    expect(el).toHaveClass("text-amber-300");
    // Tooltip surfaces the agreement label.
    expect(el.getAttribute("title")).toContain("disagrees with solo");
  });

  it("falls back to neutral palette when soloProbUp is null (no comparison possible)", async () => {
    render(
      withProviders(<JointForecastBadge symbol="BTCUSDT" soloProbUp={null} />),
    );
    const el = await screen.findByText(/joint ↑/);
    expect(el).toHaveClass("text-zinc-300");
    // No agreement annotation in the tooltip.
    expect(el.getAttribute("title") ?? "").not.toMatch(/agrees with solo/);
  });

  it("surfaces the joint training pedigree in the tooltip", async () => {
    render(
      withProviders(<JointForecastBadge symbol="BTCUSDT" soloProbUp={0.6} />),
    );
    const el = await screen.findByText(/joint ↑/);
    const tooltip = el.getAttribute("title") ?? "";
    // dir_acc_mean=0.5409 → 54.09 %
    expect(tooltip).toContain("54.09");
    // p-value in scientific notation
    expect(tooltip).toContain("4.97e-33");
    // n_folds
    expect(tooltip).toContain("353");
    // feature_set
    expect(tooltip).toContain("joint");
  });
});
