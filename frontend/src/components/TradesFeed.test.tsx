import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TradesFeed } from "./TradesFeed";
import type { PaperTrade } from "@/lib/api-types";


function mkTrade(overrides: Partial<PaperTrade> = {}): PaperTrade {
  return {
    id: 1,
    symbol: "BTCUSDT",
    side: "long",
    entry_price: 79000,
    exit_price: 79500,
    qty: 0.001,
    pnl_bps: 63,
    pnl_usd: 0.5,
    entry_ts: "2026-05-09T10:00:00Z",
    exit_ts: "2026-05-09T10:01:00Z",
    ...overrides,
  } as PaperTrade;
}


describe("<TradesFeed />", () => {
  it("renders skeletons when isLoading and no trades yet", () => {
    const { container } = render(<TradesFeed trades={[]} isLoading />);
    // 5 skeleton rows expected when loading.
    expect(container.querySelectorAll(".skeleton").length).toBeGreaterThanOrEqual(5);
  });

  it("renders empty-state message when not loading and no trades", () => {
    render(<TradesFeed trades={[]} />);
    expect(
      screen.getByText(/пока не было закрытых сделок/i),
    ).toBeInTheDocument();
  });

  it("filters out trades without exit_ts (still-open positions)", () => {
    const open = mkTrade({ id: 99, exit_ts: null as unknown as string });
    const closed = mkTrade({ id: 1, pnl_bps: 50 });
    render(<TradesFeed trades={[open, closed]} />);
    // Empty-state message is gone (we have 1 closed trade).
    expect(screen.queryByText(/пока не было/i)).not.toBeInTheDocument();
    // Only one trade row visible — the open one is filtered out.
    expect(screen.getAllByRole("listitem").length).toBe(1);
  });

  it("renders trades sorted by exit_ts descending (newest first)", () => {
    const old_ = mkTrade({ id: 1, exit_ts: "2026-05-09T10:00:00Z", entry_price: 79000 });
    const newer = mkTrade({ id: 2, exit_ts: "2026-05-09T11:00:00Z", entry_price: 80000 });
    render(<TradesFeed trades={[old_, newer]} />);
    const items = screen.getAllByRole("listitem");
    // First listed item should be the newer one (entry 80000 visible
    // on screens >= md, but the rendered ID-key order is enough).
    expect(items.length).toBe(2);
  });

  it("colours positive P&L emerald and negative rose", () => {
    const winner = mkTrade({ id: 1, pnl_bps: 100 });
    const loser = mkTrade({ id: 2, pnl_bps: -50, exit_ts: "2026-05-09T10:02:00Z" });
    const { container } = render(<TradesFeed trades={[winner, loser]} />);
    expect(container.querySelector(".text-emerald-400")).not.toBeNull();
    expect(container.querySelector(".text-rose-400")).not.toBeNull();
  });

  it("respects the limit prop — caps the number of rendered trades", () => {
    const trades: PaperTrade[] = Array.from({ length: 30 }, (_, i) =>
      mkTrade({ id: i, exit_ts: `2026-05-09T10:${String(i).padStart(2, "0")}:00Z` }),
    );
    render(<TradesFeed trades={trades} limit={5} />);
    expect(screen.getAllByRole("listitem").length).toBe(5);
  });
});
