import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DriftBadge } from "./DriftBadge";
import type { DriftBlock } from "@/lib/api-types";


describe("<DriftBadge />", () => {
  it("renders ``drift ok`` with emerald palette on ok severity", () => {
    const drift: DriftBlock = {
      ok: true,
      severity: "ok",
      max_ks: 0.05,
      max_ks_feature: "ofi_mean",
      evaluated_at: "2026-05-04T16:30:00Z",
    };
    render(<DriftBadge drift={drift} />);
    const span = screen.getByText(/drift ok/i);
    expect(span).toBeInTheDocument();
    expect(span).toHaveClass("text-emerald-400");
  });

  it("renders ``drift warn`` in amber on warn severity", () => {
    const drift: DriftBlock = {
      ok: true,
      severity: "warn",
      max_ks: 0.21,
      max_ks_feature: "spread_bps_mean",
      evaluated_at: "2026-05-04T16:30:00Z",
    };
    render(<DriftBadge drift={drift} />);
    const span = screen.getByText(/drift warn/i);
    expect(span).toHaveClass("text-amber-400");
  });

  it("renders ``drift high`` in rose on high severity", () => {
    const drift: DriftBlock = {
      ok: true,
      severity: "high",
      max_ks: 0.85,
      max_ks_feature: "spread_bps_mean",
      evaluated_at: "2026-05-04T16:30:00Z",
    };
    render(<DriftBadge drift={drift} />);
    const span = screen.getByText(/drift high/i);
    expect(span).toHaveClass("text-rose-400");
  });

  it("falls back to muted ``drift —`` when ok=false", () => {
    const drift: DriftBlock = { ok: false, reason: "no_check_yet" };
    render(<DriftBadge drift={drift} />);
    expect(screen.getByText(/drift —/)).toHaveClass("text-zinc-500");
  });

  it("tooltip surfaces max_ks + feature for ok drifts", () => {
    const drift: DriftBlock = {
      ok: true,
      severity: "warn",
      max_ks: 0.213,
      max_ks_feature: "spread_bps_mean",
      evaluated_at: null,
    };
    render(<DriftBadge drift={drift} />);
    const span = screen.getByText(/drift warn/i);
    const title = span.getAttribute("title") ?? "";
    expect(title).toContain("0.213");
    expect(title).toContain("spread_bps_mean");
  });
});
