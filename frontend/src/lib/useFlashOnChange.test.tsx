import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useFlashOnChange } from "./useFlashOnChange";

describe("useFlashOnChange", () => {
  it("returns empty class on initial render (no prior value)", () => {
    const classFor = vi.fn(() => "flash-up");
    const { result } = renderHook(() => useFlashOnChange<number>(0.5, classFor));
    expect(result.current).toBe("");
    expect(classFor).not.toHaveBeenCalled();
  });

  it("flashes up when value increases past threshold", () => {
    vi.useFakeTimers();
    const classFor = vi.fn((prev: number, curr: number) =>
      curr > prev ? "flash-up" : "flash-down",
    );
    const { result, rerender } = renderHook(
      ({ v }: { v: number }) => useFlashOnChange<number>(v, classFor),
      { initialProps: { v: 0.50 } },
    );
    rerender({ v: 0.62 });
    expect(result.current).toBe("flash-up");
    expect(classFor).toHaveBeenCalledWith(0.50, 0.62);

    // Class clears after the duration.
    act(() => {
      vi.advanceTimersByTime(1700);
    });
    expect(result.current).toBe("");
    vi.useRealTimers();
  });

  it("flashes down when value decreases past threshold", () => {
    const classFor = vi.fn((prev: number, curr: number) =>
      curr > prev ? "flash-up" : "flash-down",
    );
    const { result, rerender } = renderHook(
      ({ v }: { v: number }) => useFlashOnChange<number>(v, classFor),
      { initialProps: { v: 0.62 } },
    );
    rerender({ v: 0.55 });
    expect(result.current).toBe("flash-down");
  });

  it("does NOT flash on jitter below threshold", () => {
    const classFor = vi.fn(() => "flash-up");
    const { result, rerender } = renderHook(
      ({ v }: { v: number }) => useFlashOnChange<number>(v, classFor, 0.005),
      { initialProps: { v: 0.500 } },
    );
    rerender({ v: 0.502 }); // 0.2pp — below 0.5pp threshold
    expect(result.current).toBe("");
    expect(classFor).not.toHaveBeenCalled();
  });

  it("classFor returning empty string suppresses the flash", () => {
    const classFor = vi.fn(() => "");
    const { result, rerender } = renderHook(
      ({ v }: { v: number }) => useFlashOnChange<number>(v, classFor),
      { initialProps: { v: 1.0 } },
    );
    rerender({ v: 2.0 });
    expect(classFor).toHaveBeenCalled();
    expect(result.current).toBe("");
  });
});
