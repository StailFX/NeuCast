import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

/**
 * Vitest config — JS-DOM environment for React component tests,
 * with the same ``@/`` path alias the production tsconfig defines.
 *
 * Run via ``npm test`` (single shot) or ``npm run test:watch``
 * (watch mode). CI uses ``npm test --run`` (no interactive).
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: [
      "src/**/*.{test,spec}.{ts,tsx}",
      "tests/**/*.{test,spec}.{ts,tsx}",
    ],
    exclude: ["node_modules", ".next", "out"],
  },
});
