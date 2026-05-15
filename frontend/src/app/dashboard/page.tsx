"use client";

import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Navbar } from "@/components/Navbar";
import { Skeleton } from "@/components/Skeleton";
import { useAuth } from "@/lib/AuthContext";


/**
 * /v2/dashboard — protected user landing page.
 *
 * Auth guard:
 *  * loading → render skeleton
 *  * unauthenticated → router.replace('/login') + show nothing
 *  * authenticated → show greeting + quick links
 *
 * The legacy Daily-side ``/dashboard`` Jinja page renders the user's
 * prediction history for TCN forecasts. Phase 1 here surfaces just the
 * greeting + nav links; the full prediction history is a TCN/Celery-
 * backed page that we'll port in Phase 3 (separate session) since it
 * needs a deeper integration with /api/predict and /api/task.
 */
export default function DashboardPage() {
  const router = useRouter();
  const { user, loading, logout } = useAuth();

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/dashboard");
  }, [loading, user, router]);

  return (
    <div className="min-h-screen">
      <Navbar />
      <div className="mx-auto max-w-4xl px-6 py-12">
        {loading ? (
          <Loading />
        ) : !user ? (
          // The redirect above is in flight — render an empty
          // placeholder rather than flashing logged-out UI.
          <Skeleton className="h-32 w-full" rounded="rounded-2xl" />
        ) : (
          <div className="space-y-6">
            <header className="flex flex-wrap items-baseline justify-between gap-4">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
                  Привет,{" "}
                  <span className="text-emerald-300">{user.username}</span>
                </h1>
                <p className="mt-1 text-sm text-zinc-400">
                  Личный кабинет NeuCast. Роль:{" "}
                  <span className="text-zinc-300">{user.role}</span>
                </p>
              </div>
              <button
                type="button"
                onClick={async () => {
                  await logout();
                  router.replace("/");
                }}
                className="rounded-full border border-zinc-800 bg-zinc-900/50 px-4 py-2 text-sm text-zinc-300 transition hover:border-zinc-700 hover:text-zinc-100"
              >
                Выйти
              </button>
            </header>

            <div className="grid gap-4 sm:grid-cols-2">
              <QuickLink
                title="Live forecast"
                desc="3-symbol Binance Spot prediction cards, refresh каждые 30 с."
                href="/forecast"
              />
              <QuickLink
                title="Operator dashboard"
                desc="Ingest health, training report, anti-skill detector, training history."
                href="/highfreq"
              />
              <QuickLink
                title="Запустить прогноз"
                desc="Daily TCN forecast — Yahoo Finance тикеры (золото, нефть, акции). Расчёт ~30–90 сек."
                href="/predict"
              />
              <QuickLink
                title="Grafana operator"
                desc="Системные метрики Tokyo + Finland (operator-only auth)."
                href="/grafana"
                external
              />
            </div>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 text-sm text-zinc-400">
              <h2 className="mb-2 text-base font-semibold text-zinc-100">
                История прогнозов
              </h2>
              <p>
                Каждый запуск
                <code className="mx-1 rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-zinc-300">
                  /predict
                </code>
                сохраняется в истории. Просмотр истории + рендер
                чартов пока на legacy-странице
                <a
                  href="/dashboard"
                  className="mx-1 text-emerald-300 underline hover:text-emerald-200"
                >
                  /dashboard
                </a>
                — порт UI на v2 в бэклоге.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


function Loading() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-48" />
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton
            key={i}
            className="h-24 w-full"
            rounded="rounded-2xl"
          />
        ))}
      </div>
    </div>
  );
}


function QuickLink({
  title,
  desc,
  href,
  external = false,
}: {
  title: string;
  desc: string;
  href: string;
  external?: boolean;
}) {
  const Comp = external ? "a" : Link;
  return (
    <Comp
      href={href}
      className="card-hover block rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5"
    >
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-semibold text-zinc-100">{title}</span>
        <span className="text-zinc-500">→</span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-zinc-400">{desc}</p>
    </Comp>
  );
}
