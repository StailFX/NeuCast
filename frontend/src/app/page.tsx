import Link from "next/link";
import { Navbar } from "@/components/Navbar";

export const metadata = {
  title: "NeuCast — directional forecasts on Binance Spot",
  description:
    "1-minute directional crypto forecasts on Binance L2 microstructure. " +
    "Walk-forward validated, conformal-calibrated, drift-monitored, paper-traded.",
};


export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Hero />
      <Features />
      <Methodology />
      <Footer />
    </div>
  );
}


function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-zinc-900">
      {/* Subtle background gradient */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-gradient-to-b from-emerald-500/5 via-transparent to-transparent"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -top-32 left-1/2 -z-10 h-96 w-96 -translate-x-1/2 rounded-full bg-emerald-500/10 blur-3xl"
      />
      <div className="relative mx-auto max-w-5xl px-6 py-24 text-center sm:py-32">
        <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/5 px-4 py-1.5 text-xs font-medium uppercase tracking-wider text-emerald-300">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          live · Binance Spot · Tokyo 19 ms
        </span>
        <h1 className="mt-6 text-4xl font-semibold tracking-tight sm:text-6xl">
          NeuCast{" "}
          <span className="bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">
            HF
          </span>
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-zinc-400">
          1-minute directional forecasts on Binance Spot L2 microstructure
          features. Walk-forward validated, conformal-calibrated,
          drift-monitored, paper-traded.
        </p>
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/forecast"
            className="group inline-flex items-center gap-2 rounded-full bg-zinc-100 px-6 py-3 text-sm font-semibold text-zinc-900 transition hover:bg-emerald-300"
          >
            Live forecast
            <span className="transition group-hover:translate-x-0.5">→</span>
          </Link>
          <Link
            href="/highfreq"
            className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/50 px-6 py-3 text-sm font-semibold text-zinc-300 transition hover:border-zinc-700 hover:text-zinc-100"
          >
            Operator dashboard
          </Link>
        </div>

        {/* Headline numbers row */}
        <div className="mt-16 grid gap-6 sm:grid-cols-3">
          <Stat
            label="Frozen-holdout dir-acc"
            value="0.584"
            sub="BTC · n = 2545 · p < 1e-17"
          />
          <Stat
            label="Edge over Klines baseline"
            value="+5 pp"
            sub="L2 microstructure value"
          />
          <Stat
            label="Test coverage"
            value="968"
            sub="passing pytest cases"
          />
        </div>
      </div>
    </section>
  );
}


function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6">
      <div className="text-xs uppercase tracking-wider text-zinc-500">
        {label}
      </div>
      <div className="mt-2 text-4xl font-semibold tabular-nums text-zinc-100">
        {value}
      </div>
      <div className="mt-1 text-xs text-zinc-500">{sub}</div>
    </div>
  );
}


function Features() {
  const items = [
    {
      title: "Tokyo HFT slice",
      body:
        "Postgres + WebSocket ingest in ap-northeast-1, ~19 ms TCP RTT to Binance. WireGuard tunnel back to the public Finland VPS.",
    },
    {
      title: "Layered safety",
      body:
        "walk-forward CV → frozen holdout → paper-trader live test → max_consecutive_losses kill-switch → tools/rollback_model. Each layer catches a different bug class.",
    },
    {
      title: "Honest negatives",
      body:
        "T.23 microstructure_v3 showed +25 pp offline; live paper-trader 0/15. Auto-halt fired in 5 min; rollback in 11. Documented as the canonical defence example.",
    },
    {
      title: "Calibrated probabilities",
      body:
        "Isotonic regression for n ≥ 1000, split-conformal prediction intervals (Vovk-Gammerman-Shafer). Brier 0.25, ECE < 0.08.",
    },
    {
      title: "Drift detection",
      body:
        "Hourly KS-test on rolling 24h reference. Severity bucket drives auto-retrain (with 6h cooldown rail) so a stuck-high reading doesn't thrash the trainer.",
    },
    {
      title: "Open metrics",
      body:
        "Reliability diagram, feature importance, conditional accuracy by confidence bucket, robustness suite (block-bootstrap + permutation + per-day) — all surfaced live.",
    },
  ];
  return (
    <section className="border-b border-zinc-900 py-20">
      <div className="mx-auto max-w-5xl px-6">
        <h2 className="text-2xl font-semibold tracking-tight">
          What&apos;s under the hood
        </h2>
        <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((it) => (
            <div
              key={it.title}
              className="card-hover rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6"
            >
              <h3 className="text-sm font-semibold text-zinc-100">
                {it.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">
                {it.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}


function Methodology() {
  return (
    <section className="border-b border-zinc-900 py-20">
      <div className="mx-auto max-w-3xl px-6">
        <h2 className="text-2xl font-semibold tracking-tight">
          Methodology
        </h2>
        <p className="mt-4 text-sm leading-relaxed text-zinc-400">
          The HF model is a CatBoost binary classifier on{" "}
          <code className="rounded bg-zinc-900 px-1.5 py-0.5 font-mono text-xs text-zinc-300">
            sign(return_1m)
          </code>
          . Features are 18 columns of order-flow imbalance (OFI),
          microprice, depth imbalance, spread, and trade flow,
          aggregated to 1-minute bars from a 1-second OFI cache. Fit
          via expanding-window walk-forward CV with rolling-origin
          folds and a 1-minute embargo (López de Prado).
        </p>
        <p className="mt-3 text-sm leading-relaxed text-zinc-400">
          The trainer&apos;s last 3 days are reserved as a frozen
          holdout — the model literally cannot see them during fit or
          CV. The headline defence number (BTC dir-acc 0.584 with p
          &lt; 1e-17 on n = 2545) is computed against that untouched
          slice.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-zinc-400">
          Probabilities are calibrated via isotonic regression
          (Niculescu-Mizil &amp; Caruana 2005) when n ≥ 1000, and
          split-conformal prediction intervals
          (Vovk-Gammerman-Shafer 2005, Angelopoulos-Bates 2023)
          provide distribution-free coverage guarantees on the live
          cards.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-zinc-400">
          Hourly KS-tests on a rolling 24-hour reference window
          monitor feature-distribution drift; severity ≥ &quot;high&quot;
          fires an auto-retrain through the daily systemd timer
          (subject to a 6h cooldown rail).
        </p>
      </div>
    </section>
  );
}


function Footer() {
  return (
    <footer className="py-12">
      <div className="mx-auto max-w-5xl px-6 text-center">
        <p className="text-xs text-zinc-600">
          Sim-only by ADR-005 · No real-money trading · Licensed MIT ·{" "}
          <a
            href="https://github.com/StailFX/NeuCast"
            className="text-zinc-400 hover:text-zinc-200"
          >
            github.com/StailFX/NeuCast
          </a>
        </p>
      </div>
    </footer>
  );
}
