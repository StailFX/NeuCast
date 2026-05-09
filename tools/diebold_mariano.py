"""Тест Дибольда-Мариано (Diebold–Mariano, 1995) для попарного
сравнения точности прогнозов двух моделей.

Зачем нужен тест
================

Сравнение двух моделей по средней метрике (MAPE, RMSE) не даёт ответа
на статистически значимый вопрос: «является ли наблюдаемая разница в
точности результатом действительно лучшей модели или просто
артефактом конкретной реализации шума?». Тест Дибольда-Мариано
(далее DM) формализует эту задачу: проверяет нулевую гипотезу о
равенстве ожидаемых функций потерь двух прогнозов.

Формулировка
============

Пусть {ê¹ₜ}ₜ₌₁..n и {ê²ₜ}ₜ₌₁..n — out-of-sample ошибки прогноза
двух конкурирующих моделей на одной и той же выборке. Зафиксируем
функцию потерь L (обычно квадратичную L(e) = e², но возможна и
абсолютная L(e) = |e| или асимметричная). Определим разностный ряд
loss-differential:

    dₜ = L(ê¹ₜ) − L(ê²ₜ).

Нулевая гипотеза DM:  H₀: E[dₜ] = 0  (модели одинаково точны).
Альтернатива:          H₁: E[dₜ] ≠ 0  (одна точнее другой).

Статистика теста:

    DM = (1/n) Σdₜ / √(σ̂_d² / n),

где σ̂_d² — оценка асимптотической долгосрочной дисперсии {dₜ}.
При корректной оценке σ̂_d² (учитывающей возможную автокорреляцию
ряда dₜ через HAC-оценку Newey-West) статистика DM асимптотически
распределена как N(0, 1) при H₀.

Поправка Харви-Лейборна-Ньюбольда
=================================

Для конечной выборки Харви, Лейборн и Ньюбольд (1997) предложили
коррекцию: вместо стандартной нормали использовать t-распределение
со n−1 степенями свободы и умножить статистику на корректирующий
множитель √((n + 1 − 2h + h(h−1)/n) / n), где h — горизонт прогноза.
Это делает тест более консервативным на коротких выборках.

Применение в проекте
====================

Используется в дипломе для построения матрицы попарных p-values
DM-теста между всеми моделями ансамбля (TCN, CatBoost, XGBoost,
LightGBM, foundation), а также для сравнения ансамбля целиком с
каждой компонентой. Результат — формальное статистическое
обоснование выбора финальной композиции.

Run example
-----------

::

    python3 -m tools.diebold_mariano \\
        --errors-a /path/tcn_oos_errors.npy \\
        --errors-b /path/catboost_oos_errors.npy \\
        --horizon 1 --loss squared
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger("diebold_mariano")


LossType = Literal["squared", "absolute", "linex"]


def _loss(errors: np.ndarray, kind: LossType, alpha: float = 0.5) -> np.ndarray:
    """Compute pointwise loss values for an array of errors."""
    if kind == "squared":
        return errors ** 2
    if kind == "absolute":
        return np.abs(errors)
    if kind == "linex":
        # Asymmetric Linex loss: L(e) = exp(αe) − αe − 1.
        # Penalises positive vs negative errors asymmetrically.
        return np.exp(alpha * errors) - alpha * errors - 1.0
    raise ValueError(f"unknown loss kind: {kind}")


def _newey_west_long_run_variance(d: np.ndarray, max_lag: int) -> float:
    """Heteroskedasticity- and autocorrelation-consistent (HAC) estimator
    of the long-run variance of {dₜ} via Newey-West (1987).

    σ̂² = γ₀ + 2 Σₖ₌₁^L wₖ γₖ ,   wₖ = 1 − k/(L+1) ,

    where γₖ — sample autocovariance at lag k.
    """
    n = len(d)
    if n < 2:
        return float("nan")
    d_centered = d - d.mean()
    # γ₀ — variance.
    gamma_0 = float(np.dot(d_centered, d_centered) / n)
    if max_lag <= 0:
        return gamma_0
    s = gamma_0
    for k in range(1, min(max_lag, n - 1) + 1):
        gamma_k = float(np.dot(d_centered[k:], d_centered[:-k]) / n)
        weight = 1.0 - k / (max_lag + 1)
        s += 2.0 * weight * gamma_k
    return max(s, 1e-12)  # guard against numerical underflow


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    horizon: int = 1,
    loss: LossType = "squared",
    alpha_linex: float = 0.5,
    use_hln_correction: bool = True,
) -> dict:
    """Run the Diebold-Mariano test on two error series.

    Parameters
    ----------
    errors_a, errors_b
        Equal-length arrays of out-of-sample errors (y − ŷ) for two
        competing models on the same evaluation set.
    horizon
        Forecast horizon h. Used both to set Newey-West lag (h − 1)
        and for the Harvey-Leybourne-Newbold small-sample correction.
    loss
        Loss function: 'squared', 'absolute', or 'linex'.
    alpha_linex
        Asymmetry parameter for Linex loss (ignored otherwise).
    use_hln_correction
        Apply Harvey-Leybourne-Newbold (1997) finite-sample correction
        and switch from N(0, 1) to t(n − 1) distribution for the
        p-value. Recommended for n < ~200.

    Returns
    -------
    dict with keys:
        n              — sample size
        mean_diff      — mean loss differential (positive = A worse)
        dm_stat        — Diebold-Mariano statistic (asymptotically N(0,1))
        p_value        — two-sided p-value
        long_run_var   — Newey-West long-run variance estimate
        loss_kind      — echo of input
        winner         — 'A', 'B', or 'tie' at α = 0.05
        verdict        — human-readable interpretation
    """
    a = np.asarray(errors_a, dtype=float).ravel()
    b = np.asarray(errors_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"errors_a and errors_b must have the same length, got {a.shape} vs {b.shape}",
        )
    n = len(a)
    if n < 10:
        raise ValueError(f"sample too small for DM test: n = {n} < 10")

    # Loss differential
    L_a = _loss(a, loss, alpha=alpha_linex)
    L_b = _loss(b, loss, alpha=alpha_linex)
    d = L_a - L_b
    mean_d = float(d.mean())

    # Long-run variance via Newey-West
    nw_lag = max(horizon - 1, 0)
    sigma2 = _newey_west_long_run_variance(d, max_lag=nw_lag)
    if sigma2 <= 0:
        # Degenerate case — usually means errors_a == errors_b. Two
        # identical models are by definition equally accurate, so the
        # p-value of «they differ» is 1.0 (никогда не отвергаем H₀).
        return {
            "n": n, "mean_diff": mean_d, "dm_stat": 0.0,
            "p_value": 1.0, "long_run_var": sigma2,
            "loss_kind": loss, "winner": "tie",
            "verdict": "degenerate (identical loss series → p = 1.0)",
        }

    dm = mean_d / math.sqrt(sigma2 / n)

    if use_hln_correction:
        # Harvey-Leybourne-Newbold (1997) finite-sample correction
        h = horizon
        k = math.sqrt(max(0, (n + 1 - 2 * h + h * (h - 1) / n) / n))
        dm_corrected = dm * k
        # Use t-distribution with n-1 dof (more conservative than normal)
        from scipy.stats import t as t_dist
        p = 2.0 * (1.0 - t_dist.cdf(abs(dm_corrected), df=n - 1))
        dm_final = dm_corrected
    else:
        # Asymptotic normal
        from scipy.stats import norm
        p = 2.0 * (1.0 - norm.cdf(abs(dm)))
        dm_final = dm

    if p < 0.05:
        winner = "A" if mean_d < 0 else "B"
        # mean_d < 0 ⇒ L_a < L_b on average ⇒ model A is better
        verdict = (
            f"reject H₀: model {winner} is significantly more accurate "
            f"(p = {p:.4f})"
        )
    else:
        winner = "tie"
        verdict = (
            f"fail to reject H₀: no significant difference in accuracy "
            f"(p = {p:.4f})"
        )

    return {
        "n": n,
        "mean_diff": mean_d,
        "dm_stat": float(dm_final),
        "p_value": float(p),
        "long_run_var": sigma2,
        "loss_kind": loss,
        "winner": winner,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--errors-a", required=True, help="path to .npy with errors of model A")
    p.add_argument("--errors-b", required=True, help="path to .npy with errors of model B")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--loss", default="squared",
                   choices=["squared", "absolute", "linex"])
    p.add_argument("--alpha-linex", type=float, default=0.5)
    p.add_argument("--no-hln", action="store_true",
                   help="disable Harvey-Leybourne-Newbold finite-sample correction")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper())

    a = np.load(args.errors_a)
    b = np.load(args.errors_b)
    out = diebold_mariano(
        a, b,
        horizon=args.horizon,
        loss=args.loss,
        alpha_linex=args.alpha_linex,
        use_hln_correction=not args.no_hln,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
