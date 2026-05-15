import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SymbolPills } from "./SymbolPills";


describe("<SymbolPills />", () => {
  it("renders 3 pills with BTC / ETH / BNB labels", () => {
    const noop = vi.fn();
    render(<SymbolPills value="BTCUSDT" onChange={noop} />);

    expect(screen.getByRole("tab", { name: "BTC" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "ETH" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "BNB" })).toBeInTheDocument();
  });

  it("marks the active pill via aria-selected", () => {
    const noop = vi.fn();
    render(<SymbolPills value="ETHUSDT" onChange={noop} />);

    expect(screen.getByRole("tab", { name: "ETH" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "BTC" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(screen.getByRole("tab", { name: "BNB" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
  });

  it("calls onChange with the symbol id when a pill is clicked", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<SymbolPills value="BTCUSDT" onChange={onChange} />);

    await user.click(screen.getByRole("tab", { name: "BNB" }));

    expect(onChange).toHaveBeenCalledWith("BNBUSDT");
  });
});
