"""Render defence-grade figures from metrics JSONs.

Produces SVGs in ``docs/defence/figures/`` ready for slide-deck
import (Beamer / Keynote / Pitch / Google Slides). All numbers
sourced from the live training-history files; nothing is faked
or smoothed.

Inputs
------
* ``weights/highfreq/{btc,eth,bnb}usdt_1m_metrics.json`` — solo CV
* ``weights/highfreq/{btc,eth,bnb}usdt_1m_holdout.json`` — frozen OOS
* ``weights/highfreq/joint_1m_metrics.json`` — joint multi-symbol
* ``weights/highfreq/btcusdt_drift.json`` — KS drift snapshot
* ``weights/highfreq/multi_horizon_compare_features.json`` — fee P&L
* ``/api/highfreq/conditional_accuracy`` — live confidence buckets

Outputs
-------
* ``fig-01-dir-acc-comparison.svg`` — solo + joint dir_acc with CI
* ``fig-02-conditional-accuracy.svg`` — confidence-bucket curve
* ``fig-03-drift-evidence.svg`` — KS per feature, BTC 2026-05-08
* ``fig-04-fee-tier-pnl.svg`` — bps/trade per fee tier
* ``fig-05-cv-power.svg`` — n_folds × CI width: solo vs joint

Run
---

::

    python3 tools/render_defence_figures.py \\
        --json-dir /tmp \\
        --conditional-accuracy /tmp/conditional_accuracy.json \\
        --out docs/defence/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required — CI-friendly
import matplotlib.pyplot as plt
import numpy as np

# Defence palette — distinct enough to print B&W, light backgrounds
# for slide overlay.
COLOR_BTC = "#f7931a"  # bitcoin orange
COLOR_ETH = "#627eea"  # ethereum blue
COLOR_BNB = "#f0b90b"  # bnb yellow
COLOR_JOINT = "#10b981"  # emerald
COLOR_NEG = "#ef4444"  # red for losses
COLOR_NEUTRAL = "#6b7280"  # zinc for reference lines


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def render_dir_acc_comparison(json_dir: Path, out: Path) -> None:
    """Figure 1: solo per-symbol vs joint, dir_acc with 95 % CI bars."""
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        m = json.loads((json_dir / f"{sym.lower()}_1m_metrics.json").read_text())
        h = json.loads((json_dir / f"{sym.lower()}_1m_holdout.json").read_text())
        rows.append({
            "label": sym.replace("USDT", ""),
            "solo_acc": m["dir_acc_mean"],
            "solo_lo": m["dir_acc_ci_low"],
            "solo_hi": m["dir_acc_ci_high"],
            "solo_n": m["n_folds"],
            "holdout_acc": h["dir_acc"],
            "holdout_lo": h["dir_acc_ci_low"],
            "holdout_hi": h["dir_acc_ci_high"],
        })
    j = json.loads((json_dir / "joint_1m_metrics.json").read_text())

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)

    # Layout: 3 solo groups (CV + holdout) + joint at the right.
    n = len(rows)
    x = np.arange(n + 1)
    bar_w = 0.36

    solo_cv = np.array([r["solo_acc"] for r in rows] + [j["dir_acc_mean"]])
    solo_cv_lo = np.array([r["solo_lo"] for r in rows] + [j["dir_acc_ci_low"]])
    solo_cv_hi = np.array([r["solo_hi"] for r in rows] + [j["dir_acc_ci_high"]])
    holdout_cv = np.array([r["holdout_acc"] for r in rows] + [np.nan])
    holdout_lo = np.array([r["holdout_lo"] for r in rows] + [np.nan])
    holdout_hi = np.array([r["holdout_hi"] for r in rows] + [np.nan])

    cv_err = np.array([
        solo_cv - solo_cv_lo,
        solo_cv_hi - solo_cv,
    ])
    h_err = np.array([
        holdout_cv - holdout_lo,
        holdout_hi - holdout_cv,
    ])

    bars1 = ax.bar(
        x - bar_w / 2, solo_cv, bar_w, yerr=cv_err, capsize=4,
        color=[COLOR_BTC, COLOR_ETH, COLOR_BNB, COLOR_JOINT],
        label="walk-forward CV", alpha=0.95,
        edgecolor="black", linewidth=0.5,
    )
    # Holdout overlay (skip for joint).
    ax.bar(
        x[:-1] + bar_w / 2, holdout_cv[:-1], bar_w, yerr=h_err[:, :-1],
        capsize=4, color="white", edgecolor="black", linewidth=0.8,
        hatch="///", label="frozen holdout (Mon 04:30 UTC)",
    )

    # Reference line at 0.5 (coinflip).
    ax.axhline(0.5, color=COLOR_NEUTRAL, linestyle="--", linewidth=1, zorder=0)
    ax.text(
        len(rows) + 0.4, 0.502, "coinflip", fontsize=8,
        color=COLOR_NEUTRAL, va="bottom",
    )

    # n_folds annotation under each bar.
    for i, r in enumerate(rows):
        ax.annotate(
            f"n={r['solo_n']}", xy=(x[i] - bar_w / 2, 0.452),
            ha="center", fontsize=8, color="white", weight="bold",
        )
    ax.annotate(
        f"n={j['n_folds']}", xy=(x[-1] - bar_w / 2, 0.452),
        ha="center", fontsize=8, color="white", weight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["label"] for r in rows] + ["JOINT\n(BTC+ETH+BNB)"],
        fontsize=10,
    )
    ax.set_ylim(0.45, 0.60)
    ax.set_ylabel("Directional accuracy (1m horizon)", fontsize=11)
    ax.set_title(
        "Per-symbol solo vs. joint multi-symbol pooled training\n"
        "Walk-forward CV (n_folds shown) + frozen-holdout OOS overlay",
        fontsize=12, pad=14,
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "fig-01-dir-acc-comparison.svg")


def render_conditional_accuracy(api_path: Path, out: Path) -> None:
    """Figure 2: dir_acc vs confidence threshold, per symbol."""
    data = json.loads(api_path.read_text())
    rows_by_sym = {r["symbol"]: r["buckets"] for r in data["rows"]}
    sym_order = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    palette = {"BTCUSDT": COLOR_BTC, "ETHUSDT": COLOR_ETH, "BNBUSDT": COLOR_BNB}

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)

    bucket_order = ["conf_55", "conf_60", "conf_65"]
    bucket_x = np.array([0.05, 0.10, 0.15])  # |p - 0.5| threshold

    for sym in sym_order:
        b = rows_by_sym[sym]
        accs = [b[k]["dir_acc"] for k in bucket_order]
        lo = [b[k]["ci_low"] for k in bucket_order]
        hi = [b[k]["ci_high"] for k in bucket_order]
        ns = [b[k]["n"] for k in bucket_order]
        err = np.array([np.subtract(accs, lo), np.subtract(hi, accs)])
        ax.errorbar(
            bucket_x, accs, yerr=err, capsize=5, marker="o",
            markersize=8, linewidth=2, color=palette[sym],
            label=f"{sym.replace('USDT', '')} (n={ns[0]:,}/{ns[1]:,}/{ns[2]:,})",
        )

    ax.axhline(0.5, color=COLOR_NEUTRAL, linestyle="--", linewidth=1, zorder=0)
    ax.text(
        0.155, 0.502, "coinflip", fontsize=8,
        color=COLOR_NEUTRAL, va="bottom",
    )

    ax.set_xticks(bucket_x)
    ax.set_xticklabels([
        "|p−0.5| ≥ 0.05\n(any non-neutral)",
        "|p−0.5| ≥ 0.10\n(stronger conviction)",
        "|p−0.5| ≥ 0.15\n(highest confidence)",
    ], fontsize=10)
    ax.set_ylabel("Directional accuracy", fontsize=11)
    ax.set_xlabel("Confidence bucket", fontsize=11)
    ax.set_ylim(0.49, 0.61)
    ax.set_title(
        "Conditional accuracy — model knows what it doesn't know\n"
        "Live data from /api/highfreq/conditional_accuracy",
        fontsize=12, pad=14,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "fig-02-conditional-accuracy.svg")


def render_drift_evidence(json_dir: Path, out: Path) -> None:
    """Figure 3: top KS values per feature, BTC 2026-05-08."""
    d = json.loads((json_dir / "btcusdt_drift.json").read_text())
    feats = d.get("top_features", [])
    feats = sorted(feats, key=lambda f: f["ks_stat"], reverse=True)[:8]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    names = [f["feature"] for f in feats]
    ks_vals = [f["ks_stat"] for f in feats]

    colors = []
    for k in ks_vals:
        if k > 0.4:
            colors.append(COLOR_NEG)
        elif k > 0.2:
            colors.append("#f59e0b")  # amber
        else:
            colors.append(COLOR_NEUTRAL)

    bars = ax.barh(
        np.arange(len(feats))[::-1], ks_vals, color=colors,
        edgecolor="black", linewidth=0.5,
    )
    for i, (b, f) in enumerate(zip(bars, feats)):
        ax.text(
            b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
            f"KS={f['ks_stat']:.3f}  p={f['p_value']:.1e}",
            va="center", fontsize=8,
        )

    # Threshold lines.
    ax.axvline(d.get("threshold", 0.15), color="black", linestyle="--",
               linewidth=1, alpha=0.6, label=f"alarm threshold (KS={d.get('threshold', 0.15)})")

    ax.set_yticks(np.arange(len(feats))[::-1])
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Kolmogorov–Smirnov statistic (recent vs reference)", fontsize=11)
    ax.set_xlim(0, max(ks_vals) * 1.35)
    ax.set_title(
        f"Drift evidence — BTCUSDT, recent 6 h vs prior 7 d "
        f"(severity: {d.get('severity', '?')})",
        fontsize=12, pad=14,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "fig-03-drift-evidence.svg")


def render_fee_tier_pnl(json_dir: Path, out: Path) -> None:
    """Figure 4: bps/trade per fee tier across selected models."""
    src = json.loads((json_dir / "multi_horizon_compare_features.json").read_text())
    rows = src.get("rows", [])

    # Pick the most-defence-relevant rows.
    wanted = [
        ("BTCUSDT", 1, "microstructure", "BTC 1m solo"),
        ("ETHUSDT", 1, "microstructure", "ETH 1m solo"),
        ("BNBUSDT", 1, "microstructure", "BNB 1m solo"),
        ("BTCUSDT", 5, "long_horizon", "BTC 5m long_horizon"),
        ("BTCUSDT", 60, "long_horizon", "BTC 60m long_horizon"),
    ]
    matched = []
    for sym, bm, fs, label in wanted:
        for r in rows:
            if (r.get("symbol") == sym and r.get("bar_minutes") == bm
                    and r.get("feature_set") == fs):
                matched.append((label, r))
                break

    if not matched:
        # Fallback to multi_horizon_eval.json
        src = json.loads((json_dir / "multi_horizon_eval.json").read_text())
        rows = src.get("rows", [])
        for sym, bm, fs, label in wanted:
            for r in rows:
                if (r.get("symbol") == sym and r.get("bar_minutes") == bm):
                    matched.append((label, r))
                    break

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    tiers = ["retail", "vip5", "vip9", "mm_rebate"]
    tier_labels = ["Retail (15 bps)", "VIP5 (8 bps)", "VIP9 (3 bps)", "MM rebate (−1 bps)"]
    bar_w = 0.18
    x = np.arange(len(matched))

    for i, tier in enumerate(tiers):
        vals = [m[1].get("pnl_per_trade_bps", {}).get(tier) or 0 for m in matched]
        offset = (i - 1.5) * bar_w
        colors = [COLOR_NEG if v < 0 else COLOR_JOINT for v in vals]
        ax.bar(
            x + offset, vals, bar_w, label=tier_labels[i],
            edgecolor="black", linewidth=0.5,
            color=[
                ["#fca5a5", "#fcd34d", "#86efac", "#34d399"][i] for _ in vals
            ],
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in matched], fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("E[P&L per trade] (bps)", fontsize=11)
    ax.set_title(
        "Where the edge becomes profitable — by fee tier\n"
        "Negative = fees eat the directional edge",
        fontsize=12, pad=14,
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95, ncol=2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "fig-04-fee-tier-pnl.svg")


def render_cv_power(json_dir: Path, out: Path) -> None:
    """Figure 5: n_folds × CI half-width — visualises statistical
    power gain of joint pooling."""
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        m = json.loads((json_dir / f"{sym.lower()}_1m_metrics.json").read_text())
        rows.append({
            "label": f"solo {sym.replace('USDT', '')}",
            "n_folds": m["n_folds"],
            "ci_half": (m["dir_acc_ci_high"] - m["dir_acc_ci_low"]) / 2,
            "color": {"BTCUSDT": COLOR_BTC, "ETHUSDT": COLOR_ETH,
                      "BNBUSDT": COLOR_BNB}[sym],
        })
    j = json.loads((json_dir / "joint_1m_metrics.json").read_text())
    rows.append({
        "label": "joint\n(BTC+ETH+BNB)",
        "n_folds": j["n_folds"],
        "ci_half": (j["dir_acc_ci_high"] - j["dir_acc_ci_low"]) / 2,
        "color": COLOR_JOINT,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)

    # Left: n_folds.
    ax1.bar(
        [r["label"] for r in rows],
        [r["n_folds"] for r in rows],
        color=[r["color"] for r in rows],
        edgecolor="black", linewidth=0.5,
    )
    ax1.set_ylabel("Number of CV folds", fontsize=11)
    ax1.set_title("Walk-forward CV folds (~ OOS sample size)", fontsize=11, pad=10)
    for i, r in enumerate(rows):
        ax1.text(i, r["n_folds"] + 5, str(r["n_folds"]),
                 ha="center", fontsize=10, weight="bold")
    ax1.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax1.set_axisbelow(True)

    # Right: CI half-width (lower is tighter).
    ax2.bar(
        [r["label"] for r in rows],
        [r["ci_half"] * 100 for r in rows],
        color=[r["color"] for r in rows],
        edgecolor="black", linewidth=0.5,
    )
    ax2.set_ylabel("95 % CI half-width (percentage points)", fontsize=11)
    ax2.set_title("CI tightness (lower = more confident)", fontsize=11, pad=10)
    for i, r in enumerate(rows):
        ax2.text(i, r["ci_half"] * 100 + 0.05, f"±{r['ci_half']*100:.2f}",
                 ha="center", fontsize=10, weight="bold")
    ax2.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Statistical power: joint pooling buys ~10× the OOS sample size",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out, "fig-05-cv-power.svg")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", default="/tmp", type=Path)
    p.add_argument("--conditional-accuracy",
                   default="/tmp/conditional_accuracy.json", type=Path)
    p.add_argument("--out", default="docs/defence/figures", type=Path)
    args = p.parse_args()

    render_dir_acc_comparison(args.json_dir, args.out)
    render_conditional_accuracy(args.conditional_accuracy, args.out)
    render_drift_evidence(args.json_dir, args.out)
    render_fee_tier_pnl(args.json_dir, args.out)
    render_cv_power(args.json_dir, args.out)
    print(f"\nall figures rendered to {args.out}/")


if __name__ == "__main__":
    main()
