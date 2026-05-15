"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/AuthContext";

/**
 * Navbar — global navigation bar (v1-landing visual style).
 *
 * Visually matches ``templates/_partials/nav.html``:
 *   • Gradient checkmark logo + NeuCast brand wordmark
 *   • Sticky with backdrop blur, reactive border on scroll
 *   • Section links: Live · Forecast · Operator (ghost buttons,
 *     emerald-tinted when active)
 *   • Grafana (subtle dim ghost)
 *   • Right: anon → Войти + Регистрация (primary), auth → username + Выйти
 *
 * Rewritten 2026-05-14 to match v1 landing chrome — replaces the
 * legacy zinc-Tailwind navbar so all v2 pages share one chrome.
 */

interface Props {
  /** Optional right-side accessory — e.g. live-status pill. */
  rightSlot?: React.ReactNode;
}

export function Navbar({ rightSlot }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, loading, logout } = useAuth();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    let ticking = false;
    function onScroll() {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          setScrolled(window.scrollY > 8);
          ticking = false;
        });
        ticking = true;
      }
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const isActive = (route: string) =>
    pathname === route ||
    pathname === `${route}/` ||
    pathname?.startsWith(`${route}/`);

  return (
    <nav className={`nc-nav ${scrolled ? "nc-scrolled" : ""}`}>
      <Link href="/" className="nc-brand">
        <span className="nc-logo" aria-hidden>
          <svg viewBox="0 0 24 24">
            <path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z" />
          </svg>
        </span>
        <span className="nc-name">NeuCast</span>
      </Link>

      <div className="nc-links">
        <Link
          href="/live"
          className={`nc-btn nc-btn-ghost ${isActive("/live") ? "nc-active" : ""}`}
        >
          Прогноз
        </Link>
        <Link
          href="/forecast"
          className={`nc-btn nc-btn-ghost ${isActive("/forecast") ? "nc-active" : ""}`}
        >
          Forecast
        </Link>
        <Link
          href="/highfreq"
          className={`nc-btn nc-btn-ghost ${isActive("/highfreq") ? "nc-active" : ""}`}
        >
          Operator
        </Link>
        <a
          href="/grafana"
          className="nc-btn nc-btn-ghost"
          style={{ opacity: 0.7 }}
        >
          Grafana
        </a>

        {loading ? (
          <span
            aria-hidden
            className="nc-btn nc-btn-ghost"
            style={{ opacity: 0.5, pointerEvents: "none" }}
          >
            …
          </span>
        ) : user ? (
          <>
            <Link
              href="/dashboard"
              className={`nc-btn nc-btn-ghost ${isActive("/dashboard") ? "nc-active" : ""}`}
            >
              {user.username}
            </Link>
            <button
              type="button"
              onClick={async () => {
                await logout();
                router.replace("/");
              }}
              className="nc-btn nc-btn-ghost"
            >
              Выйти
            </button>
          </>
        ) : (
          <>
            <Link
              href="/login"
              className={`nc-btn nc-btn-ghost ${isActive("/login") ? "nc-active" : ""}`}
            >
              Войти
            </Link>
            <Link
              href="/register"
              className={`nc-btn nc-btn-primary ${isActive("/register") ? "nc-active" : ""}`}
            >
              Регистрация
            </Link>
          </>
        )}

        {rightSlot && <span className="nc-status-pill">{rightSlot}</span>}
      </div>
    </nav>
  );
}
