"use client";

import type { DriftBlock } from "@/lib/api-types";

const SEVERITY_STYLES: Record<string, string> = {
  ok: "text-emerald-400 bg-emerald-500/10",
  warn: "text-amber-400 bg-amber-500/10",
  high: "text-rose-400 bg-rose-500/10",
  muted: "text-zinc-500 bg-zinc-800/40",
};


export function DriftBadge({ drift }: { drift: DriftBlock }) {
  const cls = drift.ok
    ? SEVERITY_STYLES[drift.severity] ?? SEVERITY_STYLES.muted
    : SEVERITY_STYLES.muted;

  let label: string;
  let title: string;
  if (drift.ok) {
    label = `drift ${drift.severity}`;
    const ks = drift.max_ks != null ? drift.max_ks.toFixed(3) : "—";
    const feat = drift.max_ks_feature ?? "—";
    title =
      `KS-test drift vs training distribution.\n` +
      `severity: ${drift.severity}\n` +
      `max_ks: ${ks} on "${feat}"\n` +
      (drift.evaluated_at ? `evaluated_at: ${drift.evaluated_at}` : "");
  } else {
    label = "drift —";
    title =
      drift.reason === "no_check_yet"
        ? "Drift cron has not yet run for this symbol."
        : "Drift snapshot unavailable.";
  }

  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider cursor-help ${cls}`}
    >
      <span
        className="h-1.5 w-1.5 rounded-full bg-current opacity-80"
        aria-hidden
      />
      {label}
    </span>
  );
}
