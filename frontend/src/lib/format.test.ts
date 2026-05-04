import { describe, expect, it } from "vitest";
import { fmtAge, fmtMoney, fmtPercent } from "./format";

describe("fmtAge", () => {
  it("returns em-dash for null/undefined/Infinity/negative", () => {
    expect(fmtAge(null)).toBe("—");
    expect(fmtAge(undefined)).toBe("—");
    expect(fmtAge(NaN)).toBe("—");
    expect(fmtAge(Infinity)).toBe("—");
    expect(fmtAge(-1)).toBe("—");
  });

  it("uses seconds < 60s", () => {
    expect(fmtAge(0)).toBe("0 с");
    expect(fmtAge(45)).toBe("45 с");
    expect(fmtAge(59.4)).toBe("59 с");
  });

  it("rolls over to minutes 60s..3600s", () => {
    expect(fmtAge(60)).toBe("1 мин");
    expect(fmtAge(125)).toBe("2 мин");
    expect(fmtAge(3599)).toBe("60 мин");
  });

  it("rolls over to hours 3600s..86400s with one decimal", () => {
    expect(fmtAge(3600)).toBe("1.0 ч");
    expect(fmtAge(7200)).toBe("2.0 ч");
    expect(fmtAge(5400)).toBe("1.5 ч");
  });

  it("rolls over to days ≥86400s", () => {
    expect(fmtAge(86_400)).toBe("1 дн");
    expect(fmtAge(172_800)).toBe("2 дн");
  });
});


describe("fmtMoney", () => {
  it("em-dash on bad input", () => {
    expect(fmtMoney(null)).toBe("—");
    expect(fmtMoney(undefined)).toBe("—");
    expect(fmtMoney(NaN)).toBe("—");
  });

  it("formats large numbers with thousands separator, no decimals", () => {
    // toLocaleString rounds to integer for ≥10_000 — strip locale
    // separators (en-US uses ',') for portable comparison.
    const v = fmtMoney(79_000.4);
    expect(v.replace(/[\s,]/g, "")).toBe("79000");
    const v2 = fmtMoney(125_678.9);
    expect(v2.replace(/[\s,]/g, "")).toBe("125679");
  });

  it("uses 2 decimals for hundreds", () => {
    expect(fmtMoney(123.456)).toBe("123.46");
  });

  it("uses 3 decimals for 1..100 range", () => {
    expect(fmtMoney(7.123456)).toBe("7.123");
  });

  it("uses 5 decimals for sub-dollar", () => {
    expect(fmtMoney(0.12345678)).toBe("0.12346");
  });
});


describe("fmtPercent", () => {
  it("em-dash on Infinity", () => {
    expect(fmtPercent(Infinity)).toBe("—");
  });

  it("integer-percent default (digits=0)", () => {
    expect(fmtPercent(0.584)).toBe("58%");
    expect(fmtPercent(0)).toBe("0%");
    expect(fmtPercent(1)).toBe("100%");
  });

  it("respects custom digits", () => {
    expect(fmtPercent(0.584, 1)).toBe("58.4%");
    expect(fmtPercent(0.584, 2)).toBe("58.40%");
  });
});
