"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Navbar } from "@/components/Navbar";
import { useAuth } from "@/lib/AuthContext";
import { ApiError, fetchTaskStatus, type TaskState } from "@/lib/api";


/** Mapping from celery state → UX label + progress percentage. */
const STAGE_META: Record<
  string,
  { label: string; pct: number; tone: "info" | "ok" | "err" }
> = {
  PENDING: { label: "В очереди...", pct: 10, tone: "info" },
  FETCHING: { label: "Загрузка истории...", pct: 35, tone: "info" },
  PREDICTING: { label: "Расчёт прогноза...", pct: 70, tone: "info" },
  SUCCESS: { label: "Готово", pct: 100, tone: "ok" },
  FAILURE: { label: "Ошибка", pct: 100, tone: "err" },
};


/**
 * Waiting page for a TCN prediction task. Polls ``/api/task/{task_id}``
 * every 2 s until SUCCESS or FAILURE. On SUCCESS, redirects to the
 * legacy ``/p/{slug}`` HTML for the chart-suite render — porting the
 * 3000-line predict.html to TSX is its own backlog item.
 *
 * URL shape: ``/v2/predict/waiting/?task_id=xxx``. We use a query
 * param (not a path segment) because ``output: "export"`` requires
 * static path generation and task_ids are UUIDs minted at runtime.
 *
 * The form page stashes ``redirect_url`` in sessionStorage on submit
 * so we don't have to round-trip the slug through the URL too.
 */
function WaitingInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const taskId = searchParams.get("task_id") ?? "";
  const { user, loading } = useAuth();

  const [state, setState] = useState<TaskState>("PENDING");
  const [statusText, setStatusText] = useState("Ожидание...");
  const [err, setErr] = useState<string | null>(null);
  const [redirectUrl, setRedirectUrl] = useState<string | null>(null);
  const [ticker, setTicker] = useState<string>("");
  const tickRef = useRef<number | null>(null);

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  // Pull the legacy redirect_url stashed by the form on submit.
  useEffect(() => {
    if (!taskId) return;
    try {
      const raw = sessionStorage.getItem(`predict:${taskId}`);
      if (raw) {
        const meta = JSON.parse(raw) as {
          slug: string | null;
          redirect_url: string;
          ticker?: string;
        };
        setRedirectUrl(meta.redirect_url);
        if (meta.ticker) setTicker(meta.ticker);
      } else {
        // Fresh tab fallback — long-form legacy URL.
        setRedirectUrl(`/predict/result/${taskId}`);
      }
    } catch {
      setRedirectUrl(`/predict/result/${taskId}`);
    }
  }, [taskId]);

  useEffect(() => {
    if (!taskId || !user) return;
    let cancelled = false;

    async function tick() {
      try {
        const res = await fetchTaskStatus(taskId);
        if (cancelled) return;
        setState(res.state);
        if (res.status) setStatusText(res.status);

        if (res.state === "SUCCESS") {
          if (redirectUrl) {
            // Hard nav — legacy /p/* lives outside the v2 basePath.
            window.location.href = redirectUrl;
          }
          return;
        }
        if (res.state === "FAILURE") {
          setErr(res.error ?? "Расчёт не удался");
          return;
        }
        tickRef.current = window.setTimeout(tick, 2000);
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 403) {
          setErr("Эта задача принадлежит другому пользователю.");
          return;
        }
        if (e instanceof ApiError && e.status === 401) {
          router.replace("/login");
          return;
        }
        setErr((e as Error).message);
        tickRef.current = window.setTimeout(tick, 5000);
      }
    }

    tick();

    return () => {
      cancelled = true;
      if (tickRef.current) {
        window.clearTimeout(tickRef.current);
        tickRef.current = null;
      }
    };
  }, [taskId, user, redirectUrl, router]);

  const meta = STAGE_META[state] ?? {
    label: statusText,
    pct: 50,
    tone: "info" as const,
  };
  const barColour =
    meta.tone === "ok"
      ? "bg-emerald-500"
      : meta.tone === "err"
        ? "bg-rose-500"
        : "bg-emerald-500/70";

  if (!taskId) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-16">
        <p className="text-sm text-rose-300">
          В URL не указан task_id. Запустите расчёт заново.
        </p>
        <Link
          href="/predict"
          className="mt-3 inline-block rounded-md bg-emerald-500 px-4 py-2 text-sm font-semibold text-zinc-950 hover:bg-emerald-400"
        >
          К форме
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl px-6 py-16">
      <header className="mb-10">
        <h1 className="text-2xl font-semibold tracking-tight">
          {ticker ? `Прогноз ${ticker}` : "Расчёт прогноза"}
        </h1>
        <p className="mt-2 text-sm text-zinc-500">
          ID задачи:{" "}
          <code className="rounded bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-400">
            {taskId}
          </code>
        </p>
      </header>

      <div className="rounded-xl border border-zinc-900 bg-zinc-950/40 p-6">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">{meta.label}</span>
          <span className="text-xs text-zinc-500">{meta.pct}%</span>
        </div>
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-zinc-900">
          <div
            className={`h-full rounded-full transition-all duration-700 ${barColour}`}
            style={{ width: `${meta.pct}%` }}
          />
        </div>

        <ul className="mt-6 space-y-2 text-xs text-zinc-500">
          <Stage
            active={state === "PENDING"}
            done={["FETCHING", "PREDICTING", "SUCCESS"].includes(state)}
            label="Постановка в очередь"
          />
          <Stage
            active={state === "FETCHING"}
            done={["PREDICTING", "SUCCESS"].includes(state)}
            label="Загрузка цен с Yahoo Finance"
          />
          <Stage
            active={state === "PREDICTING"}
            done={state === "SUCCESS"}
            label="TCN + ансамбль + sentiment bias"
          />
          <Stage
            active={state === "SUCCESS"}
            done={state === "SUCCESS"}
            label="Рендер результата"
          />
        </ul>

        {err && (
          <div className="mt-6 rounded-md border border-rose-900/50 bg-rose-950/30 p-3 text-sm text-rose-300">
            <div className="font-medium">Что-то пошло не так</div>
            <div className="mt-1 text-xs">{err}</div>
            <Link
              href="/predict"
              className="mt-3 inline-block rounded-md bg-rose-500/20 px-3 py-1.5 text-xs font-medium text-rose-200 hover:bg-rose-500/30"
            >
              Попробовать снова
            </Link>
          </div>
        )}

        {state === "SUCCESS" && redirectUrl && !err && (
          <div className="mt-6 text-xs text-zinc-500">
            Перенаправление на результат...{" "}
            <Link href={redirectUrl} className="text-emerald-400 underline">
              Открыть вручную
            </Link>
          </div>
        )}
      </div>

      <div className="mt-8 text-xs text-zinc-600">
        Можно закрыть вкладку — расчёт продолжится. Готовый прогноз
        сохранится в истории, доступной с панели управления.
      </div>
    </div>
  );
}


function Stage({
  active,
  done,
  label,
}: {
  active: boolean;
  done: boolean;
  label: string;
}) {
  const dot = done
    ? "bg-emerald-500"
    : active
      ? "bg-emerald-400 animate-pulse"
      : "bg-zinc-700";
  const text = done
    ? "text-zinc-300"
    : active
      ? "text-zinc-100 font-medium"
      : "text-zinc-600";
  return (
    <li className={`flex items-center gap-2 ${text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      <span>{label}</span>
    </li>
  );
}


export default function PredictWaitingPage() {
  // useSearchParams() must be inside a Suspense boundary in App
  // Router export mode — wrap so the static HTML can render an
  // empty shell while the client picks up the query string.
  return (
    <div className="min-h-screen">
      <Navbar />
      <Suspense
        fallback={
          <div className="mx-auto max-w-2xl px-6 py-16 text-sm text-zinc-500">
            Загрузка...
          </div>
        }
      >
        <WaitingInner />
      </Suspense>
    </div>
  );
}
