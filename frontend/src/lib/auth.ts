import { API_BASE, ApiError } from "./api";
import type {
  AuthLoginResponse,
  AuthMeResponse,
} from "./api-types";

/**
 * Auth client — talks to the JSON ``/api/auth/*`` endpoints
 * added to ``app/main.py``. All requests use ``credentials: "include"``
 * so the HttpOnly session cookie is sent + accepted across the
 * static-export → FastAPI same-origin boundary.
 */

interface FetchOptions extends RequestInit {
  /** Caller's expected error message for 4xx — surfaced to UI. */
  fallbackError?: string;
}

async function fetchAuth<T>(path: string, init: FetchOptions = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = init.fallbackError ?? `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}


export function authMe(): Promise<AuthMeResponse> {
  return fetchAuth<AuthMeResponse>("/api/auth/me");
}


export function authLogin(
  username: string,
  password: string,
): Promise<AuthLoginResponse> {
  return fetchAuth<AuthLoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
    fallbackError: "Login failed",
  });
}


export function authRegister(
  username: string,
  email: string,
  password: string,
  password2: string,
): Promise<AuthLoginResponse> {
  return fetchAuth<AuthLoginResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, email, password, password2 }),
    fallbackError: "Registration failed",
  });
}


export function authLogout(): Promise<{ ok: boolean }> {
  return fetchAuth<{ ok: boolean }>("/api/auth/logout", {
    method: "POST",
  });
}
