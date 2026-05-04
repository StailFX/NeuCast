"use client";

import { useConditionalAccuracy } from "@/lib/hooks";
import { fmtPercent } from "@/lib/format";
import type { ConditionalBucket } from "@/lib/api-types";
import { Skeleton } from "./Skeleton";

const BUCKET_LABELS: Array<{ key: keyof BucketsKey; threshold: string }> = [
  { key: "conf_55", threshold: "≥ 55%" },
  { key: "conf_60", threshold: "≥ 60%" },
  { key: "conf_65", threshold: "≥ 65%" },
];

type BucketsKey = {
  conf_55?: ConditionalBucket;
  conf_60?: ConditionalBucket;
  conf_65?: ConditionalBucket;
};


/** Realized direction accuracy bucketed by confidence threshold —
 * "if I only bet when prob_up ≥ X, what's my dir-acc?". The headline
 * defence question: does conviction actually map to higher accuracy?
 */
export function ConditionalAccuracy() {
  const { data, isLoading } = useConditionalAccuracy();

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Conditional accuracy by confidence
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          buckets nested · higher = stricter cutoff
        </span>
      </div>
      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-4 w-full" />
          ))}
        </div>
      ) : !data?.ok ? (
        <div className="text-sm text-zinc-500">
          {data?.db_status === "unavailable"
            ? "DB временно недоступна"
            : "данных нет"}
        </div>
      ) : !data.rows || data.rows.length === 0 ? (
        <div className="text-sm text-zinc-500">
          не накоплено предсказаний с realized labels
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-[0.66rem] uppercase tracking-wider text-zinc-500">
                <th className="py-2 pr-4">symbol</th>
                <th className="py-2 pr-4">bucket</th>
                <th className="py-2 pr-4">n</th>
                <th className="py-2 pr-4">dir_acc</th>
                <th className="py-2 pr-4">95% CI</th>
                <th className="py-2 pr-4">p</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.flatMap((r) =>
                BUCKET_LABELS.map(({ key, threshold }) => {
                  const b = r.buckets[key];
                  if (!b) return null;
                  const accColor =
                    b.dir_acc == null
                      ? "text-zinc-600"
                      : b.dir_acc >= 0.55
                      ? "text-emerald-400"
                      : b.dir_acc >= 0.5
                      ? "text-zinc-200"
                      : "text-rose-400";
                  return (
                    <tr
                      key={`${r.symbol}-${key}`}
                      className="border-b border-zinc-900 last:border-0"
                    >
                      <td className="py-1.5 pr-4 text-zinc-300">
                        {r.symbol.replace("USDT", "")}
                      </td>
                      <td className="py-1.5 pr-4 text-zinc-500">
                        {threshold}
                      </td>
                      <td className="py-1.5 pr-4 tabular-nums text-zinc-400">
                        {b.n}
                      </td>
                      <td
                        className={`py-1.5 pr-4 tabular-nums font-semibold ${accColor}`}
                      >
                        {b.dir_acc == null
                          ? "—"
                          : fmtPercent(b.dir_acc, 1)}
                      </td>
                      <td className="py-1.5 pr-4 tabular-nums text-zinc-500">
                        {b.ci_low == null
                          ? "—"
                          : `[${fmtPercent(b.ci_low, 1)}, ${fmtPercent(
                              b.ci_high ?? 0,
                              1,
                            )}]`}
                      </td>
                      <td className="py-1.5 pr-4 tabular-nums text-zinc-500">
                        {b.p_value == null
                          ? "—"
                          : b.p_value < 0.001
                          ? "< 0.001"
                          : b.p_value.toFixed(3)}
                      </td>
                    </tr>
                  );
                }).filter(Boolean),
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
