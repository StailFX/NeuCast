/**
 * Display helpers — same shape as the legacy templates/forecast.html
 * vanilla-JS helpers, ported to TypeScript.
 */

export function fmtAge(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)} с`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} мин`;
  if (seconds < 86_400) return `${(seconds / 3600).toFixed(1)} ч`;
  return `${Math.round(seconds / 86_400)} дн`;
}

export function fmtMoney(p: number | null | undefined): string {
  if (p == null || !isFinite(p)) return "—";
  if (p >= 10_000)
    return p.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (p >= 100) return p.toFixed(2);
  if (p >= 1) return p.toFixed(3);
  return p.toFixed(5);
}

export function fmtPercent(x: number, digits = 0): string {
  if (!isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}
