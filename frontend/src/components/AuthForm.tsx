"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/AuthContext";

interface Props {
  mode: "login" | "register";
  /** Where to redirect on success — default ``/v2/dashboard``. */
  redirectTo?: string;
}


export function AuthForm({ mode, redirectTo = "/dashboard" }: Props) {
  const router = useRouter();
  const { user, loading, error, login, register, clearError } = useAuth();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Already-authenticated visitor → kick to redirect target.
  useEffect(() => {
    if (!loading && user) router.replace(redirectTo);
  }, [loading, user, router, redirectTo]);

  // Reset error when the user starts typing.
  useEffect(() => {
    if (error) clearError();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username, password, password2, email]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    let ok = false;
    if (mode === "login") {
      ok = await login(username, password);
    } else {
      ok = await register(username, email, password, password2);
    }
    setSubmitting(false);
    if (ok) router.replace(redirectTo);
  }

  const heading = mode === "login" ? "Войти" : "Регистрация";
  const submitLabel = mode === "login" ? "Войти" : "Создать аккаунт";

  return (
    <div className="mx-auto w-full max-w-sm rounded-2xl border border-zinc-800 bg-zinc-900/40 p-8 shadow-xl">
      <h1 className="text-2xl font-semibold tracking-tight">{heading}</h1>
      <p className="mt-1 text-sm text-zinc-400">
        {mode === "login"
          ? "Введите логин и пароль для доступа к личному кабинету."
          : "Минимум 12 символов в пароле — code-review C-2."}
      </p>

      <form onSubmit={handleSubmit} className="mt-6 space-y-4">
        <Field
          label="Имя пользователя"
          value={username}
          onChange={setUsername}
          autoComplete="username"
          required
        />
        {mode === "register" && (
          <Field
            label="Email (необязательно)"
            value={email}
            onChange={setEmail}
            autoComplete="email"
            type="email"
          />
        )}
        <Field
          label="Пароль"
          value={password}
          onChange={setPassword}
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          type="password"
          required
        />
        {mode === "register" && (
          <Field
            label="Повторите пароль"
            value={password2}
            onChange={setPassword2}
            autoComplete="new-password"
            type="password"
            required
          />
        )}

        {error && (
          <div className="rounded-md border border-rose-500/30 bg-rose-500/5 px-3 py-2 text-xs text-rose-300">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-full bg-zinc-100 px-4 py-2.5 text-sm font-semibold text-zinc-900 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting ? "…" : submitLabel}
        </button>
      </form>

      <div className="mt-6 text-center text-xs text-zinc-500">
        {mode === "login" ? (
          <>
            Нет аккаунта?{" "}
            <Link
              href="/register"
              className="text-zinc-300 hover:text-zinc-100 underline-offset-2 hover:underline"
            >
              Зарегистрироваться
            </Link>
          </>
        ) : (
          <>
            Уже есть аккаунт?{" "}
            <Link
              href="/login"
              className="text-zinc-300 hover:text-zinc-100 underline-offset-2 hover:underline"
            >
              Войти
            </Link>
          </>
        )}
      </div>
    </div>
  );
}


function Field({
  label,
  value,
  onChange,
  type = "text",
  autoComplete,
  required,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  autoComplete?: string;
  required?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-[0.7rem] uppercase tracking-wider text-zinc-500">
        {label}
      </span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoComplete={autoComplete}
        required={required}
        className="mt-1 block w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600"
      />
    </label>
  );
}
