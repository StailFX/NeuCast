"use client";

import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { authMe, authLogin, authRegister, authLogout } from "./auth";
import type { AuthUser } from "./api-types";

interface AuthState {
  user: AuthUser | null;
  /** Initial /api/auth/me probe still in flight. ``user`` may already
   * be set from a successful login that came in mid-probe. */
  loading: boolean;
  /** Last login/register attempt error, surfaced to the form. */
  error: string | null;
  login(username: string, password: string): Promise<boolean>;
  register(
    username: string,
    email: string,
    password: string,
    password2: string,
  ): Promise<boolean>;
  logout(): Promise<void>;
  /** Clear ``error`` — used when the form is reset. */
  clearError(): void;
}

const Ctx = createContext<AuthState | null>(null);


export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // On mount: probe the cookie. Lifecycle:
  //   * authenticated   → seed user, no flicker
  //   * unauthenticated → user stays null, login form usable
  //   * network fail    → user stays null, error reported
  useEffect(() => {
    let cancelled = false;
    authMe()
      .then((res) => {
        if (cancelled) return;
        if (res.authenticated && res.user) setUser(res.user);
      })
      .catch(() => {
        // network / 5xx / parse — silent. User stays null.
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(
    async (username: string, password: string) => {
      setError(null);
      try {
        const res = await authLogin(username, password);
        if (res.ok && res.user) {
          setUser(res.user);
          return true;
        }
        setError(res.detail ?? "Login failed");
        return false;
      } catch (e) {
        setError((e as Error).message);
        return false;
      }
    },
    [],
  );

  const register = useCallback(
    async (
      username: string,
      email: string,
      password: string,
      password2: string,
    ) => {
      setError(null);
      try {
        const res = await authRegister(username, email, password, password2);
        if (res.ok && res.user) {
          setUser(res.user);
          return true;
        }
        setError(res.detail ?? "Registration failed");
        return false;
      } catch (e) {
        setError((e as Error).message);
        return false;
      }
    },
    [],
  );

  const logout = useCallback(async () => {
    try {
      await authLogout();
    } catch {
      /* even on error, drop client state */
    }
    setUser(null);
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return (
    <Ctx.Provider
      value={{ user, loading, error, login, register, logout, clearError }}
    >
      {children}
    </Ctx.Provider>
  );
}


export function useAuth(): AuthState {
  const ctx = useContext(Ctx);
  if (!ctx) {
    // Safe fallback when called outside provider — useful for tests.
    return {
      user: null,
      loading: false,
      error: null,
      login: async () => false,
      register: async () => false,
      logout: async () => {},
      clearError: () => {},
    };
  }
  return ctx;
}
