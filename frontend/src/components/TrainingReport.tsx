"use client";

import { useTrainingReport } from "@/lib/hooks";
import { fmtAge, fmtPercent } from "@/lib/format";
import { Skeleton } from "./Skeleton";

interface Props {
  symbols: readonly string[];
}


export function TrainingReport({ symbols }: Props) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-wider text-zinc-500">
          Training report
        </h2>
        <span className="text-[0.66rem] text-zinc-600">
          last walk-forward CV per symbol · readiness for next fold
        </span>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        {symbols.map((s) => (
          <PerSymbolReport key={s} symbol={s} />
        ))}
      </div>
    </div>
  );
}


function PerSymbolReport({ symbol }: { symbol: string }) {
  // ``lite=1`` skips the live-inventory query which can block 10-20 s
  // on a cold Postgres cache (~600k rows per symbol). The fold-readiness
  // progress widget renders "—" without it, but the page-render is
  // unblocked — net UX win on /v2/highfreq mount.
  const { data, isLoading } = useTrainingReport(symbol, 1);
  const display = symbol.replace("USDT", "");

  if (isLoading) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
        <div className="mb-2 text-sm font-semibold text-zinc-200">
          {display}
        </div>
        <Skeleton className="mb-2 h-3 w-full" />
        <Skeleton className="mb-2 h-3 w-2/3" />
        <Skeleton className="h-2 w-full" rounded="rounded-full" />
      </div>
    );
  }

  if (!data?.ok) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3 text-xs text-zinc-500">
        <div className="mb-1 text-sm font-semibold text-zinc-300">
          {display}
        </div>
        {data?.reason === "no_report_yet"
          ? "тренер ещё не отчитывался"
          : data?.reason ?? "нет данных"}
      </div>
    );
  }

  const r = data.report;
  const inv = data.live_inventory;
  const readyPct = data.fold_ready_pct;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-zinc-200">{display}</span>
        {r?.feature_set && (
          <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[0.62rem] text-zinc-400">
            {r.feature_set}
          </span>
        )}
      </div>

      {r?.dir_acc_mean != null && (
        <div className="mb-2 text-[0.7rem]">
          <div className="flex items-baseline justify-between">
            <span className="text-zinc-500">walk-forward dir_acc:</span>
            <span
              className={`tabular-nums font-semibold ${
                r.dir_acc_mean >= 0.55
                  ? "text-emerald-400"
                  : r.dir_acc_mean >= 0.5
                  ? "text-zinc-200"
                  : "text-rose-400"
              }`}
            >
              {fmtPercent(r.dir_acc_mean, 2)}
            </span>
          </div>
          {r.dir_acc_ci_low != null && r.dir_acc_ci_high != null && (
            <div className="mt-0.5 text-right text-[0.62rem] tabular-nums text-zinc-500">
              CI [{fmtPercent(r.dir_acc_ci_low, 2)},{" "}
              {fmtPercent(r.dir_acc_ci_high, 2)}]
            </div>
          )}
        </div>
      )}

      {r?.dir_acc_p_value != null && (
        <div className="mb-2 flex items-baseline justify-between text-[0.7rem]">
          <span className="text-zinc-500">p-value:</span>
          <span className="tabular-nums text-zinc-200">
            {r.dir_acc_p_value < 1e-9
              ? "< 1e-9"
              : r.dir_acc_p_value < 0.001
              ? r.dir_acc_p_value.toExponential(1)
              : r.dir_acc_p_value.toFixed(3)}
          </span>
        </div>
      )}

      <div className="mb-2 grid grid-cols-2 gap-2 text-[0.66rem]">
        <Field label="folds" value={String(r?.n_folds ?? "—")} />
        <Field label="bars" value={String(r?.n_minutes_after_neutral_drop ?? "—")} />
        {r?.elapsed_seconds != null && (
          <Field label="last run" value={fmtAge(r.elapsed_seconds)} />
        )}
        {r?.bar_minutes != null && (
          <Field label="bar size" value={`${r.bar_minutes}m`} />
        )}
      </div>

      {/* Fold-readiness bar */}
      {readyPct != null && inv && (
        <div className="mt-3">
          <div className="mb-1 flex items-baseline justify-between text-[0.62rem] text-zinc-500">
            <span>readiness for next fold</span>
            <span className="tabular-nums text-zinc-300">
              {readyPct.toFixed(0)}%
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className={`h-full transition-all duration-500 ease-out ${
                readyPct >= 100
                  ? "bg-emerald-500"
                  : readyPct >= 50
                  ? "bg-amber-500"
                  : "bg-rose-500"
              }`}
              style={{ width: `${Math.min(100, readyPct).toFixed(1)}%` }}
            />
          </div>
          <div className="mt-1 text-[0.62rem] tabular-nums text-zinc-500">
            {inv.n_eligible_for_training} eligible /{" "}
            {inv.n_minutes_after_neutral_drop} kept ({inv.n_in_holdout} in
            holdout)
          </div>
        </div>
      )}
    </div>
  );
}


function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-zinc-500">{label}</div>
      <div className="mt-0.5 tabular-nums text-zinc-300">{value}</div>
    </div>
  );
}
