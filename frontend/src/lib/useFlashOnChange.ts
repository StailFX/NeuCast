"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Returns a transient CSS class — e.g. ``flash-up`` — when ``value``
 * meaningfully changes between renders, and ``""`` otherwise. Wires
 * to the ``@keyframes neucast-flash`` rule in globals.css so the
 * affected text briefly tints the accent color, then fades back.
 *
 * ``threshold`` (default 0.005 = 0.5pp) gates the flash so trivial
 * floating-point jitter doesn't constantly retint the UI.
 *
 * ``classFor(prev, curr)`` lets the caller pick which class — e.g.
 * "flash-up" if prob went up, "flash-down" if down. Returns "" to
 * suppress the flash entirely.
 */
export function useFlashOnChange<T>(
  value: T | null | undefined,
  classFor: (prev: T, curr: T) => string,
  threshold = 0.005,
  durationMs = 1600,
): string {
  const prev = useRef<T | null | undefined>(value);
  const [cls, setCls] = useState("");

  useEffect(() => {
    if (value == null || prev.current == null) {
      prev.current = value;
      return;
    }
    // For numeric values, gate by threshold.
    if (typeof value === "number" && typeof prev.current === "number") {
      if (Math.abs(value - prev.current) < threshold) {
        prev.current = value;
        return;
      }
    } else if (Object.is(prev.current, value)) {
      return;
    }

    const next = classFor(prev.current, value);
    prev.current = value;
    if (!next) return;

    setCls(next);
    const t = setTimeout(() => setCls(""), durationMs);
    return () => clearTimeout(t);
  }, [value, classFor, threshold, durationMs]);

  return cls;
}
