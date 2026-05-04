import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { Skeleton } from "./Skeleton";

describe("<Skeleton />", () => {
  it("renders a span with ``skeleton`` class", () => {
    const { container } = render(<Skeleton />);
    const el = container.querySelector("span");
    expect(el).not.toBeNull();
    expect(el).toHaveClass("skeleton");
    expect(el).toHaveAttribute("aria-hidden");
  });

  it("merges custom className + rounded prop", () => {
    const { container } = render(
      <Skeleton className="h-4 w-12" rounded="rounded-full" />,
    );
    const el = container.querySelector("span")!;
    expect(el).toHaveClass("skeleton", "h-4", "w-12", "rounded-full");
  });

  it("default rounded is ``rounded-md``", () => {
    const { container } = render(<Skeleton />);
    expect(container.querySelector("span")).toHaveClass("rounded-md");
  });
});
