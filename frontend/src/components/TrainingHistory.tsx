"use client";

import { useTrainingHistory } from "@/lib/hooks";
import { fmtPercent } from "@/lib/format";
import { Skeleton } from "./Skeleton";


/**
 * Append-only training history — one row per trainer run, freshest
 * first. Mirrors what `tools/scoreboard.py` writes to
 * docs/highfreq/scoreboard.md but live, with feature_set + dir_acc
 * + CI bounds + p-value at a glance.
 */
export function TrainingHistory({ limit = 12 }: { limit?: number }) {
  const { data, isLoading } = useTrainingHistory(undefined, limit);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Training history
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          last {limit} trainer runs · auto-regenerated from training_runs
        </span>
      </div>
      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-4 w-full" />
          ))}
        </div>
      ) : !data?.ok || !data.rows?.length ? (
        <div className="text-sm text-zinc-500">
          {data?.reason === "no_runs_yet"
            ? "ещё не было тренировок"
            : data?.reason ?? "нет данных"}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-[0.62rem] uppercase tracking-wider text-zinc-500">
                <th className="py-2 pr-3">when</th>
                <th className="py-2 pr-3">symbol</th>
                <th className="py-2 pr-3">feature_set</th>
                <th className="py-2 pr-3">bm</th>
                <th className="py-2 pr-3">folds</th>
                <th className="py-2 pr-3">n_oos</th>
                <th className="py-2 pr-3">dir_acc</th>
                <th className="py-2 pr-3">95% CI</th>
                <th className="py-2 pr-3">p</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => {
                const ts = new Date(r.run_started_at);
                const accColor =
                  r.dir_acc_mean == null
                    ? "text-zinc-500"
                    : r.dir_acc_mean >= 0.55
                    ? "text-emerald-400"
                    : r.dir_acc_mean >= 0.5
                    ? "text-zinc-200"
                    : "text-rose-400";
                return (
                  <tr
                    key={r.id}
                    className="border-b border-zinc-900 last:border-0"
                  >
                    <td className="py-1.5 pr-3 text-[0.66rem] tabular-nums text-zinc-400">
                      {ts.toLocaleString("ru-RU", {
                        month: "2-digit",
                        day: "2-digit",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </td>
                    <td className="py-1.5 pr-3 font-semibold text-zinc-300">
                      {r.symbol.replace("USDT", "")}
                    </td>
                    <td className="py-1.5 pr-3 text-[0.66rem] text-zinc-400">
                      {r.feature_set}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-zinc-500">
                      {r.bar_minutes}m
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-zinc-500">
                      {r.n_folds}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-zinc-500">
                      {r.n_minutes_after_neutral_drop}
                    </td>
                    <td
                      className={`py-1.5 pr-3 tabular-nums font-semibold ${accColor}`}
                    >
                      {r.dir_acc_mean == null
                        ? "—"
                        : fmtPercent(r.dir_acc_mean, 2)}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-zinc-500">
                      {r.dir_acc_ci_low == null
                        ? "—"
                        : `[${fmtPercent(r.dir_acc_ci_low, 2)}, ${fmtPercent(
                            r.dir_acc_ci_high ?? 0,
                            2,
                          )}]`}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-zinc-500">
                      {r.dir_acc_p_value == null
                        ? "—"
                        : r.dir_acc_p_value < 1e-9
                        ? "< 1e-9"
                        : r.dir_acc_p_value < 0.001
                        ? r.dir_acc_p_value.toExponential(1)
                        : r.dir_acc_p_value.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
