import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HorizonPill } from "./HorizonPill";
import { HorizonProvider, useHorizon } from "@/lib/HorizonContext";


/**
 * Tiny inspector that mirrors the active horizon so tests can assert
 * that clicking a pill button actually flips the context state.
 */
function ActiveHorizonProbe() {
  const { horizon } = useHorizon();
  return <span data-testid="active-horizon">{horizon}</span>;
}


describe("<HorizonPill />", () => {
  it("renders all four horizon labels", () => {
    render(
      <HorizonProvider>
        <HorizonPill />
      </HorizonProvider>,
    );
    expect(screen.getByRole("button", { name: "1m" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "5m" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "15m" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "1h" })).toBeInTheDocument();
  });

  it("highlights the active horizon (defaults to 1m)", () => {
    render(
      <HorizonProvider>
        <HorizonPill />
      </HorizonProvider>,
    );
    const oneMin = screen.getByRole("button", { name: "1m" });
    const fiveMin = screen.getByRole("button", { name: "5m" });
    expect(oneMin).toHaveClass("bg-zinc-100");
    expect(fiveMin).not.toHaveClass("bg-zinc-100");
  });

  it("flips the active horizon when a different pill is clicked", async () => {
    const user = userEvent.setup();
    render(
      <HorizonProvider>
        <HorizonPill />
        <ActiveHorizonProbe />
      </HorizonProvider>,
    );

    expect(screen.getByTestId("active-horizon")).toHaveTextContent("1");
    await user.click(screen.getByRole("button", { name: "15m" }));
    expect(screen.getByTestId("active-horizon")).toHaveTextContent("15");
  });
});
