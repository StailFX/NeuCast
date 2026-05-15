"use client";

import { Component, ErrorInfo, ReactNode } from "react";

/**
 * Generic React error boundary — catches synchronous render errors
 * thrown by descendants and renders a small fallback instead of
 * letting the whole React tree crash.
 *
 * Use case (2026-05-14): /v2/forecast was crashing in Firefox with
 * "This page couldn't load" — one component in the dense forecast
 * page threw on a production-only data shape. Wrapping every section
 * in this boundary means the bad component shows a slim error chip
 * while the rest of the page keeps rendering.
 *
 * Usage:
 *   <ErrorBoundary label="ReliabilityDiagram">
 *     <ReliabilityDiagram />
 *   </ErrorBoundary>
 */

interface Props {
  children: ReactNode;
  /** Short label shown in fallback chip (e.g. "ReliabilityDiagram"). */
  label?: string;
}

interface State {
  hasError: boolean;
  message?: string;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(error: unknown): State {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : String(error),
    };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface in dev console — production silent except for the chip.
    if (typeof console !== "undefined") {
      // eslint-disable-next-line no-console
      console.error(
        `[ErrorBoundary${this.props.label ? ` · ${this.props.label}` : ""}]`,
        error,
        info.componentStack,
      );
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          className="nc-card"
          style={{
            padding: "1rem 1.25rem",
            borderColor: "rgba(239, 68, 68, 0.3)",
            background: "rgba(239, 68, 68, 0.05)",
            fontSize: "0.85rem",
            color: "#fca5a5",
          }}
        >
          <span style={{ fontWeight: 600 }}>
            {this.props.label
              ? `Блок «${this.props.label}» не отрисовался`
              : "Блок не отрисовался"}
          </span>
          {this.state.message && (
            <span style={{ display: "block", marginTop: "0.25rem", opacity: 0.7, fontFamily: "monospace", fontSize: "0.78rem" }}>
              {this.state.message}
            </span>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}
