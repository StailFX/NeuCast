"use client";

import { useForecastJoint } from "@/lib/hooks";
import { fmtPercent } from "@/lib/format";

interface Props {
  /** Symbol to query the joint model for. */
  symbol: string;
  /**
   * Solo prediction (from the same dashboard tick) so we can label
   * agreement / disagreement side-by-side without an extra fetch.
   * ``null`` when the solo predictor doesn't have output yet
   * (cold-start) — the badge still renders the joint number and
   * skips the agreement annotation.
   */
  soloProbUp: number | null;
}


/**
 * Shadow-mode badge that surfaces the joint multi-symbol model's
 * prediction next to the per-symbol solo forecast.
 *
 * Phase 2.2 (2026-05-09) deployment: the joint model is trained
 * once across BTC + ETH + BNB and serves predictions via
 * ``/api/highfreq/forecast_joint?symbol=X``. We display its
 * prob_up + signal alongside the solo prediction; when both
 * predictors agree (same direction relative to 0.5), the badge
 * renders emerald; when they disagree, amber. This visual is the
 * defence-day proof that two independent prediction architectures
 * are running in production observation mode.
 *
 * The hover tooltip surfaces the full joint training-run pedigree
 * (dir_acc, CI, p-value, n_folds, training timestamp) so a reviewer
 * can verify "this isn't just a fresh untested model."
 */
export function JointForecastBadge({ symbol, soloProbUp }: Props) {
  const { data, isLoading, isError } = useForecastJoint(symbol);

  // ── Cold / loading paths ──────────────────────────────────────
  if (isLoading) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-800/40 px-2.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-zinc-500">
        <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
        joint …
      </span>
    );
  }
  if (isError || !data || !data.ok || data.prob_up == null) {
    const reason =
      data?.reason ?? data?.model?.reason ?? "model_unavailable";
    const tooltip =
      reason === "joint_model_not_trained_yet"
        ? "Joint trainer hasn't produced weights yet (next fire 04:50 UTC)."
        : `Joint model unavailable: ${reason}`;
    return (
      <span
        title={tooltip}
        className="inline-flex items-center gap-1.5 rounded-full bg-zinc-800/40 px-2.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-zinc-500 cursor-help"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
        joint —
      </span>
    );
  }

  // ── Live joint prediction ─────────────────────────────────────
  const joint = data.prob_up;
  const signal = data.signal ?? "neutral";

  // Agreement check: do solo and joint cross 0.5 on the same side?
  // null (no comparison) when soloProbUp isn't supplied.
  let agreement: "agree" | "disagree" | null = null;
  if (soloProbUp != null) {
    const soloSide = soloProbUp >= 0.5;
    const jointSide = joint >= 0.5;
    agreement = soloSide === jointSide ? "agree" : "disagree";
  }

  // Style: emerald on agreement, amber on disagreement, zinc when
  // we can't compare. Joint signal arrows give the direction at a glance.
  const cls =
    agreement === "agree"
      ? "text-emerald-300 bg-emerald-500/10"
      : agreement === "disagree"
        ? "text-amber-300 bg-amber-500/10"
        : "text-zinc-300 bg-zinc-800/40";

  const arrow =
    signal === "up" ? "↑" : signal === "down" ? "↓" : "→";

  // Tooltip: full pedigree from the training run that produced
  // joint_1m.cbm — defence-day "is this a real model?" evidence.
  const m = data.model ?? ({} as NonNullable<typeof data.model>);
  const dirAcc =
    m.dir_acc_mean != null ? `${(m.dir_acc_mean * 100).toFixed(2)} %` : "—";
  const ciLo =
    m.dir_acc_ci_low != null ? (m.dir_acc_ci_low * 100).toFixed(2) : "—";
  const ciHi =
    m.dir_acc_ci_high != null ? (m.dir_acc_ci_high * 100).toFixed(2) : "—";
  const pVal =
    m.dir_acc_p_value != null ? m.dir_acc_p_value.toExponential(2) : "—";
  const ageMin =
    m.model_age_seconds != null
      ? `${(m.model_age_seconds / 60).toFixed(0)} min`
      : "—";
  const tooltip =
    `Joint multi-symbol model (BTC+ETH+BNB pooled, 21 features).\n` +
    `prob_up: ${fmtPercent(joint, 2)}  raw: ${
      data.raw_prob_up != null ? fmtPercent(data.raw_prob_up, 2) : "—"
    }\n` +
    `dir_acc: ${dirAcc}  CI: [${ciLo} %, ${ciHi} %]  p: ${pVal}\n` +
    `n_folds: ${m.n_folds ?? "—"}  trained: ${ageMin} ago\n` +
    `feature_set: ${m.feature_set ?? "—"}` +
    (agreement
      ? `\n${agreement === "agree" ? "✓ agrees" : "✗ disagrees"} with solo`
      : "");

  return (
    <span
      title={tooltip}
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider cursor-help ${cls}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      joint {arrow} {fmtPercent(joint, 1)}
    </span>
  );
}
