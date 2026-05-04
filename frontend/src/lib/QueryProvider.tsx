"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactNode, useState } from "react";

/**
 * App-wide TanStack Query client.
 *
 * Defaults tuned for a real-time HFT dashboard:
 *  * ``staleTime`` ~ 0 — every refresh is fresh.
 *  * ``refetchOnWindowFocus`` enabled — coming back to the tab
 *    re-pulls latest immediately rather than waiting for the
 *    next interval tick.
 *  * Per-query ``refetchInterval`` controls the polling cadence
 *    (set inside the consumer hook, not here).
 */
export function QueryProvider({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 0,
            refetchOnWindowFocus: true,
            retry: 1,
          },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
