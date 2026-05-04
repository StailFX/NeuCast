"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import { useAuth } from "@/lib/AuthContext";
import { ApiError, submitPrediction } from "@/lib/api";


/**
 * Daily TCN prediction form. Posts to ``/api/predict``, then routes
 * the user to ``/v2/predict/{task_id}`` for the waiting page. Once
 * the task succeeds, the waiting page hands off to the legacy
 * ``/p/{slug}`` HTML for the full chart-suite render — porting the
 * 3000-line predict.html visualization to TSX is its own backlog
 * item, not in scope for the form itself.
 */
export default function PredictPage() {
  const router = useRouter();
  const { user, loading } = useAuth();

  const [ticker, setTicker] = useState("GC=F");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [daysAhead, setDaysAhead] = useState(30);
  const [useFoundation, setUseFoundation] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Anon visitors → /login. Done as redirect (not 401 from the API)
  // for a snappy UX — they should never see the form they can't use.
  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login?next=/predict");
    }
  }, [loading, user, router]);

  // Sensible default range — last ~3 years up to today. Enough rows
  // for the TCN to converge but not so much that the request times out.
  useEffect(() => {
    const today = new Date();
    const past = new Date();
    past.setFullYear(today.getFullYear() - 3);
    const fmt = (d: Date) => d.toISOString().slice(0, 10);
    setEndDate(fmt(today));
    setStartDate(fmt(past));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setSubmitting(true);
    try {
      const res = await submitPrediction({
        ticker: ticker.trim(),
        start_date: startDate,
        end_date: endDate,
        days_ahead: Number(daysAhead) || 0,
        use_foundation: useFoundation,
      });
      // Stash the legacy redirect URL on sessionStorage so the
      // waiting page can hand off without another round-trip.
      try {
        sessionStorage.setItem(
          `predict:${res.task_id}`,
          JSON.stringify({
            slug: res.slug,
            redirect_url: res.redirect_url,
            ticker,
          }),
        );
      } catch {
        /* private mode / quota — fall back to slug-from-URL guess */
      }
      router.push(`/predict/waiting/?task_id=${encodeURIComponent(res.task_id)}`);
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.status === 401
            ? "Сессия истекла, войдите снова."
            : `Ошибка ${e.status}: ${e.message}`
          : (e as Error).message;
      setErr(msg);
      setSubmitting(false);
    }
  }

  if (loading || !user) {
    // Skeleton-ish placeholder so the redirect doesn't flash the form.
    return (
      <div className="min-h-screen bg-zinc-950 text-zinc-100">
        <Navbar />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <Navbar />
      <div className="mx-auto max-w-2xl px-6 py-12">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold tracking-tight">
            Запустить прогноз
          </h1>
          <p className="mt-2 text-sm text-zinc-400">
            TCN-модель + ансамбль (CatBoost / Foundation) обучаются на
            истории и прогнозируют цену на N дней вперёд. Расчёт
            занимает 30–90 секунд.
          </p>
        </header>

        <form
          onSubmit={handleSubmit}
          className="space-y-5 rounded-xl border border-zinc-900 bg-zinc-950/40 p-6"
        >
          <Field
            label="Тикер"
            hint="GC=F — золото, SI=F — серебро, CL=F — нефть, BTC-USD — биткоин, AAPL — Apple"
          >
            <input
              type="text"
              required
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none"
            />
          </Field>

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
            <Field label="Дата начала">
              <input
                type="date"
                required
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none"
              />
            </Field>
            <Field label="Дата окончания">
              <input
                type="date"
                required
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none"
              />
            </Field>
          </div>

          <Field
            label="Дней вперёд"
            hint="Сколько торговых дней прогнозировать в будущее (0 = только бэктест)"
          >
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setDaysAhead((d) => Math.max(0, d - 5))}
                className="rounded-md border border-zinc-800 px-3 py-2 text-sm hover:bg-zinc-900"
                aria-label="Уменьшить"
              >
                −
              </button>
              <input
                type="number"
                min={0}
                max={365}
                required
                value={daysAhead}
                onChange={(e) => setDaysAhead(Number(e.target.value))}
                className="w-24 rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-center text-sm focus:border-emerald-500 focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setDaysAhead((d) => Math.min(365, d + 5))}
                className="rounded-md border border-zinc-800 px-3 py-2 text-sm hover:bg-zinc-900"
                aria-label="Увеличить"
              >
                +
              </button>
            </div>
          </Field>

          <label className="flex cursor-pointer items-start gap-3 rounded-md border border-zinc-900 bg-zinc-950 p-3 hover:border-zinc-800">
            <input
              type="checkbox"
              checked={useFoundation}
              onChange={(e) => setUseFoundation(e.target.checked)}
              className="mt-1 h-4 w-4 accent-emerald-500"
            />
            <div className="text-sm">
              <div className="font-medium">Foundation модели</div>
              <div className="text-xs text-zinc-500">
                Включить TimesFM / Chronos в ансамбль. Замедляет
                расчёт на ~20 секунд, может улучшить точность на
                длинных горизонтах.
              </div>
            </div>
          </label>

          {err && (
            <div className="rounded-md border border-rose-900/50 bg-rose-950/30 p-3 text-sm text-rose-300">
              {err}
            </div>
          )}

          <div className="flex items-center gap-3 pt-2">
            <button
              type="submit"
              disabled={submitting}
              className="rounded-md bg-emerald-500 px-5 py-2 text-sm font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {submitting ? "Отправка..." : "Запустить расчёт"}
            </button>
            <Link
              href="/dashboard"
              className="text-sm text-zinc-400 hover:text-zinc-200"
            >
              Назад
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}


function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-zinc-400">
        {label}
      </label>
      {children}
      {hint && <div className="mt-1 text-xs text-zinc-500">{hint}</div>}
    </div>
  );
}
