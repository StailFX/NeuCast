/**
 * Vitest global setup — wires @testing-library/jest-dom matchers
 * and starts the MSW worker for fetch mocks.
 */
import "@testing-library/jest-dom/vitest";
import { afterAll, afterEach, beforeAll, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { server } from "./mocks/server";

// Auto-clean DOM between tests so each test starts from blank.
afterEach(() => {
  cleanup();
  // Reset the router-mock spy state so .toHaveBeenCalledWith() in
  // one test doesn't leak into the next. The mock object is created
  // once in vi.mock() below and reused, so we have to clear by hand.
  vi.clearAllMocks();
});

// MSW lifecycle — listen, reset between tests, close at the end.
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// next/navigation can't be invoked outside the App router runtime —
// stub the bits we use so component tests don't crash on import.
vi.mock("next/navigation", () => {
  const router = {
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
    prefetch: vi.fn(),
    refresh: vi.fn(),
  };
  // ``__searchParams`` lets per-test setup seed query string for
  // pages that read ``useSearchParams()`` (e.g. /predict/waiting).
  // Default is an empty params bag.
  const sp = { current: new URLSearchParams() };
  return {
    useRouter: () => router,
    usePathname: () => "/",
    useSearchParams: () => sp.current,
    useParams: () => ({}),
    __setSearchParams(next: URLSearchParams) {
      sp.current = next;
    },
  };
});
