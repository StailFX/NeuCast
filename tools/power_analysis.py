"""Power analysis — given target dir_acc lift over chance, how many
samples needed for statistical significance?

Why this exists
===============

Operator question: "I want to detect a true 0.55 dir_acc with 95%
confidence. How many OOS predictions do I need?". Without this tool,
the answer is hand-waved "more is better"; with it, we get an exact
number — useful for deciding whether to wait for more data vs
continue feature engineering.

This is a STANDARD frequentist power analysis for one-sided binomial
test (H₀: p=0.5, H_a: p>0.5).

Formula:
    n_required ≈ ((z_α + z_β)² × p × (1-p)) / (p - 0.5)²

where:
    z_α = critical value for desired type-I error (α=0.05 → 1.645)
    z_β = critical value for desired type-II error / power
          (1-β=0.80 → 0.842)
    p   = target true dir_acc to detect

Defence narrative add:
    "Power analysis confirms that to detect a true 0.55 dir_acc at
     α=0.05 with 80% power, we need n ≈ 615 OOS predictions.
     Our current n=2356 on BTC has power > 99% to detect this lift,
     so the observed p<10⁻⁷ is not a sample-size accident."

Usage
-----

    python -m tools.power_analysis
    python -m tools.power_analysis --target-acc 0.53 --power 0.80
    python -m tools.power_analysis --grid

The --grid flag prints a table of n_required for a range of
target accuracies × powers — defence-grade visual.
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable

# scipy.stats.norm percentile for one-sided z. We use scipy if
# available; fallback table covers the common cases.
_NORM_PPF: dict[float, float] = {
    # alpha levels (right-tail)
    0.005: 2.576,
    0.01:  2.326,
    0.025: 1.960,
    0.05:  1.645,
    0.10:  1.282,
    # power levels (= 1-beta)
    0.80:  0.842,
    0.85:  1.036,
    0.90:  1.282,
    0.95:  1.645,
    0.99:  2.326,
}


def _norm_ppf_one_sided(quantile: float) -> float:
    """Return inverse CDF for a quantile. Uses scipy if available;
    otherwise falls back to the lookup table for common values.

    ``quantile`` should be the right-tail probability for the alpha
    side (e.g. 0.05) or the cumulative probability for the power side
    (e.g. 0.80 = z_β where β=0.20 type-II).
    """
    try:
        from scipy.stats import norm
        # For alpha (right-tail prob), z_α = norm.ppf(1 - α).
        # For power (cumulative prob = 1 - β), z_β = norm.ppf(power).
        # Caller is responsible for passing the right form.
        return float(norm.ppf(quantile))
    except ImportError:
        # Try lookup table — for alpha we need 1-α so map both forms.
        for k, v in _NORM_PPF.items():
            if abs(k - quantile) < 1e-6:
                return v
            if abs((1 - k) - quantile) < 1e-6:
                return v  # right-tail = lookup
        raise ValueError(
            f"no scipy and quantile {quantile} not in fallback table; "
            f"add to _NORM_PPF or install scipy"
        )


def required_n(
    *,
    target_acc: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Compute n required to detect ``target_acc`` (true rate) at
    one-sided binomial test with given alpha + power.

    Returns ceil-rounded int.
    """
    if not (0.5 < target_acc <= 1.0):
        raise ValueError(
            f"target_acc must be > 0.5 (we test for skill above chance), "
            f"got {target_acc}"
        )
    if not (0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if not (0.5 < power < 1.0):
        raise ValueError(f"power must be in (0.5, 1.0), got {power}")

    # z_alpha: right-tail at alpha. norm.ppf(1 - alpha).
    z_alpha = _norm_ppf_one_sided(1.0 - alpha)
    # z_beta: cumulative at power. norm.ppf(power).
    z_beta = _norm_ppf_one_sided(power)

    # Variance under H0 vs H_a (using p × (1-p) under H_a is standard
    # — slightly conservative when p close to 0.5).
    var_h0 = 0.5 * 0.5
    var_ha = target_acc * (1.0 - target_acc)
    diff = target_acc - 0.5

    # n ≥ ((z_alpha * sqrt(var_h0) + z_beta * sqrt(var_ha)) / diff)²
    n = ((z_alpha * math.sqrt(var_h0) + z_beta * math.sqrt(var_ha)) / diff) ** 2
    return int(math.ceil(n))


def power_achieved(
    *,
    n: int,
    target_acc: float,
    alpha: float = 0.05,
) -> float:
    """Inverse: given a sample size n, what power do we have to detect
    ``target_acc``? Useful for "we have n=2356, what skill level can
    we conclusively detect?".

    Returns power in [0, 1].
    """
    if n <= 0:
        return 0.0
    if not (0.5 < target_acc <= 1.0):
        raise ValueError(
            f"target_acc must be > 0.5, got {target_acc}"
        )
    z_alpha = _norm_ppf_one_sided(1.0 - alpha)
    var_h0 = 0.5 * 0.5
    var_ha = target_acc * (1.0 - target_acc)
    diff = target_acc - 0.5

    # z_beta = (sqrt(n) * diff - z_alpha * sqrt(var_h0)) / sqrt(var_ha)
    z_beta = (math.sqrt(n) * diff - z_alpha * math.sqrt(var_h0)) / math.sqrt(var_ha)

    # Power = Φ(z_beta).
    try:
        from scipy.stats import norm
        return float(norm.cdf(z_beta))
    except ImportError:
        # Approximate via error function — Python stdlib has it.
        return 0.5 * (1 + math.erf(z_beta / math.sqrt(2)))


def grid_table(
    target_accs: Iterable[float] = (0.51, 0.52, 0.53, 0.55, 0.58, 0.60),
    powers: Iterable[float] = (0.80, 0.90, 0.95, 0.99),
    alpha: float = 0.05,
) -> str:
    """Render a markdown grid of n_required for various target × power
    combinations. Defence-grade table for the slide deck."""
    target_accs = list(target_accs)
    powers = list(powers)
    lines = []
    lines.append(
        f"# Power analysis grid (α={alpha}, one-sided binomial)\n"
    )
    lines.append(
        "Each cell = minimum N required to detect the row's true accuracy "
        "at the column's statistical power.\n"
    )
    header = "| target_acc \\ power | " + " | ".join(f"{p:.0%}" for p in powers) + " |"
    sep = "|" + "|".join(["---:"] * (len(powers) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for ta in target_accs:
        cells = []
        for pw in powers:
            try:
                n = required_n(target_acc=ta, alpha=alpha, power=pw)
                cells.append(f"{n}")
            except ValueError:
                cells.append("—")
        lines.append(f"| **{ta:.3f}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.power_analysis",
                                description=__doc__)
    p.add_argument("--target-acc", type=float, default=0.55,
                   help="target true dir_acc to detect (default 0.55)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--grid", action="store_true",
                   help="print full grid of (target × power) → n_required")
    p.add_argument("--current-n", type=int, default=None,
                   help="given current sample size, compute power for "
                        "the target accuracy (defence-grade reverse query)")
    args = p.parse_args(argv)

    if args.grid:
        print(grid_table(alpha=args.alpha))
        return 0

    n = required_n(
        target_acc=args.target_acc, alpha=args.alpha, power=args.power,
    )
    print(f"To detect dir_acc = {args.target_acc:.4f} at α={args.alpha:.2f} "
          f"with {args.power:.0%} power: **n ≥ {n}** OOS predictions.")

    if args.current_n is not None:
        achieved = power_achieved(
            n=args.current_n, target_acc=args.target_acc, alpha=args.alpha,
        )
        print(f"\nWith current n = {args.current_n}, we have "
              f"**{achieved:.1%}** power to detect dir_acc = "
              f"{args.target_acc:.4f}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
