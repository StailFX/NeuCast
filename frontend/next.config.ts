import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /*
   * Static export — produces an `out/` dir of pure HTML/CSS/JS that
   * any static-file server (the existing Finland nginx) can host
   * without a Node runtime. Required because we don't want to add
   * a `next start` service to the prod box.
   */
  output: "export",
  // ``trailingSlash: true`` makes the static export produce
  // ``/forecast/index.html`` rather than ``/forecast.html`` — the form
  // nginx serves naturally for clean URLs.
  trailingSlash: true,
  // ...but DON'T 308-redirect ``/api/...`` paths to add a trailing
  // slash — that breaks our dev proxy and any consumer hitting the
  // API directly.
  skipTrailingSlashRedirect: true,
  // Mounted under /v2/ on Finland nginx. ``basePath`` rewrites
  // page-level routes (``/forecast`` → ``/v2/forecast``) and static
  // asset paths (``/_next/...`` → ``/v2/_next/...``) at build time.
  // Raw ``fetch()`` calls in components are NOT affected — they keep
  // using relative ``/api/highfreq/*`` which Finland nginx routes
  // to Tokyo as today.
  basePath: "/v2",
  assetPrefix: "/v2",

  /*
   * Dev-only: when running `next dev` locally, proxy /api/highfreq/*
   * to the live production API so you see real predictions while
   * iterating on UI. Static export ignores `rewrites` (it's a
   * server-only feature) — at build time the bundle calls the same
   * relative `/api/highfreq/*` paths, which Finland nginx routes
   * to Tokyo as today.
   */
  async rewrites() {
    return [
      {
        source: "/api/highfreq/:path*",
        destination: "https://neucast.ru/api/highfreq/:path*",
      },
    ];
  },
};

export default nextConfig;
