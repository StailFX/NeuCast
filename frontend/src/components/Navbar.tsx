"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/AuthContext";

interface Props {
  /** Optional right-side accessory — e.g. live "обновлено N сек назад"
   * pill from the page that owns the header. */
  rightSlot?: React.ReactNode;
}


export function Navbar({ rightSlot }: Props) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, logout } = useAuth();

  /** Match check honours basePath/trailingSlash quirks. */
  const isActive = (route: string) =>
    pathname === route ||
    pathname === `${route}/` ||
    pathname?.startsWith(`${route}/`);

  return (
    <nav className="sticky top-0 z-30 border-b border-zinc-900/60 bg-zinc-950/80 backdrop-blur supports-[backdrop-filter]:bg-zinc-950/60">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3.5">
        <Link
          href="/"
          className="group flex items-baseline gap-2 text-sm tracking-tight"
        >
          <span className="text-base font-semibold text-zinc-100">
            NeuCast
          </span>
          <span className="rounded-md bg-emerald-500/10 px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wider text-emerald-300">
            HF
          </span>
          <span className="hidden text-[0.7rem] text-zinc-600 sm:inline">
            · Binance Spot · 19 ms ingest
          </span>
        </Link>

        <div className="flex items-center gap-1">
          <NavLink href="/forecast" active={isActive("/forecast")}>
            Forecast
          </NavLink>
          <NavLink href="/highfreq" active={isActive("/highfreq")}>
            Operator
          </NavLink>
          <a
            href="/grafana"
            className="rounded-md px-3 py-1.5 text-sm text-zinc-500 transition hover:bg-zinc-900 hover:text-zinc-300"
          >
            Grafana
          </a>

          {/* Auth-aware right side. */}
          {loading ? (
            <span className="ml-2 h-8 w-16 rounded-md bg-zinc-900/40" aria-hidden />
          ) : user ? (
            <div className="ml-2 flex items-center gap-1 border-l border-zinc-800 pl-2">
              <NavLink
                href="/dashboard"
                active={isActive("/dashboard")}
              >
                {user.username}
              </NavLink>
              <button
                type="button"
                onClick={async () => {
                  await logout();
                  router.replace("/");
                }}
                className="rounded-md px-3 py-1.5 text-sm text-zinc-500 transition hover:bg-zinc-900 hover:text-zinc-300"
              >
                Выйти
              </button>
            </div>
          ) : (
            <div className="ml-2 flex items-center gap-1 border-l border-zinc-800 pl-2">
              <NavLink href="/login" active={isActive("/login")}>
                Войти
              </NavLink>
              <NavLink href="/register" active={isActive("/register")}>
                Регистрация
              </NavLink>
            </div>
          )}

          {rightSlot && (
            <span className="ml-2 hidden border-l border-zinc-800 pl-3 sm:block">
              {rightSlot}
            </span>
          )}
        </div>
      </div>
    </nav>
  );
}


function NavLink({
  href,
  active,
  children,
}: {
  href: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className={`rounded-md px-3 py-1.5 text-sm transition ${
        active
          ? "bg-zinc-900 text-zinc-100"
          : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
      }`}
    >
      {children}
    </Link>
  );
}
