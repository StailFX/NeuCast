"""
Portfolio optimization module (Markowitz Mean-Variance).

Given a list of tickers and a budget, computes:
- Optimal weights (max Sharpe ratio)
- Min-variance portfolio
- Efficient frontier curve
- Per-asset stats (return, volatility, correlation)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from dataclasses import dataclass, field


@dataclass
class PortfolioResult:
    tickers: list[str]
    names: list[str]
    budget: float

    # Optimal portfolio (max Sharpe)
    opt_weights: list[float]
    opt_return: float           # annualized
    opt_volatility: float       # annualized
    opt_sharpe: float
    opt_allocations: list[dict]  # [{ticker, name, weight, amount}, ...]

    # Min-variance portfolio
    min_weights: list[float]
    min_return: float
    min_volatility: float
    min_sharpe: float

    # Per-asset stats
    asset_returns: list[float]   # annualized
    asset_volatility: list[float]
    correlation: list[list[float]]

    # Efficient frontier
    frontier_returns: list[float]
    frontier_volatilities: list[float]

    # Random portfolios (for scatter)
    rand_returns: list[float] = field(default_factory=list)
    rand_volatilities: list[float] = field(default_factory=list)
    rand_sharpes: list[float] = field(default_factory=list)

    # Historical performance
    hist_dates: list[str] = field(default_factory=list)
    hist_opt_equity: list[float] = field(default_factory=list)
    hist_equal_equity: list[float] = field(default_factory=list)


def optimize_portfolio(
    tickers: list[str],
    budget: float = 10000.0,
    period: str = "1y",
    risk_free_rate: float = 0.04,
) -> PortfolioResult:
    """
    Run Markowitz optimization on given tickers.

    Args:
        tickers: List of stock tickers (e.g. ["AAPL", "MSFT", "GOOG"])
        budget: Total investment amount
        period: Historical data period for returns estimation
        risk_free_rate: Annual risk-free rate (default 4% ~ T-bills)

    Returns:
        PortfolioResult with optimal weights, frontier, and stats
    """
    n = len(tickers)
    if n < 2:
        raise ValueError("Нужно минимум 2 тикера для оптимизации портфеля")
    if n > 15:
        raise ValueError("Максимум 15 тикеров")

    # Download data
    data = yf.download(tickers, period=period, interval="1d", progress=False)
    if data.empty:
        raise ValueError("Не удалось загрузить данные. Проверьте тикеры.")

    # Handle multi-ticker columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers

    close = close.dropna()
    if len(close) < 30:
        raise ValueError("Недостаточно данных (нужно минимум 30 торговых дней)")

    # Get company names
    names = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names.append(info.get("shortName", info.get("longName", t)))
        except Exception:
            names.append(t)

    # Daily returns
    returns = close.pct_change().dropna()

    # ── EWM (exponentially-weighted) mean & covariance ──────────────
    # Старая версия: returns.mean()/returns.cov() — равные веса всем дням истории
    # → оценка не учитывает регим-шифты (корреляции 2020 ≠ 2024).
    # Новая: half-life 63 дня (~3 месяца) — последний квартал доминирует,
    # старые данные затухают экспоненциально.
    HALF_LIFE = 63
    ewm = returns.ewm(halflife=HALF_LIFE, min_periods=20, adjust=True)
    mean_daily = ewm.mean().iloc[-1].values
    # ewm.cov() возвращает time-varying cov; последний n×n блок — самый "свежий".
    _cov_blocks = ewm.cov().dropna()
    cov_daily = _cov_blocks.iloc[-n:].values if len(_cov_blocks) >= n else returns.cov().values

    # Annualize (252 trading days)
    mean_annual = mean_daily * 252
    cov_annual = cov_daily * 252

    # --- Portfolio metrics functions ---
    def port_return(w):
        return np.dot(w, mean_annual)

    def port_volatility(w):
        return np.sqrt(np.dot(w, np.dot(cov_annual, w)))

    def port_sharpe(w):
        ret = port_return(w)
        vol = port_volatility(w)
        return (ret - risk_free_rate) / vol if vol > 0 else 0

    def neg_sharpe(w):
        return -port_sharpe(w)

    # Constraints and bounds
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))  # long-only
    init_w = np.array([1 / n] * n)

    # --- Max Sharpe portfolio ---
    opt_result = minimize(neg_sharpe, init_w, method="SLSQP",
                          bounds=bounds, constraints=constraints)
    opt_w = opt_result.x
    opt_ret = port_return(opt_w)
    opt_vol = port_volatility(opt_w)
    opt_sh = port_sharpe(opt_w)

    # --- Min-variance portfolio ---
    def port_var(w):
        return np.dot(w, np.dot(cov_annual, w))

    min_result = minimize(port_var, init_w, method="SLSQP",
                          bounds=bounds, constraints=constraints)
    min_w = min_result.x
    min_ret = port_return(min_w)
    min_vol = port_volatility(min_w)
    min_sh = port_sharpe(min_w)

    # --- Efficient frontier ---
    target_returns = np.linspace(
        max(min_ret, mean_annual.min()),
        mean_annual.max(),
        50
    )
    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: port_return(w) - t},
        ]
        res = minimize(port_var, init_w, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if res.success:
            frontier_vols.append(float(port_volatility(res.x)))
            frontier_rets.append(float(target))

    # --- Random portfolios (for visualization) ---
    n_random = 3000
    rand_rets, rand_vols, rand_shs = [], [], []
    for _ in range(n_random):
        w = np.random.dirichlet(np.ones(n))
        r = port_return(w)
        v = port_volatility(w)
        s = (r - risk_free_rate) / v if v > 0 else 0
        rand_rets.append(float(r))
        rand_vols.append(float(v))
        rand_shs.append(float(s))

    # --- Allocations ---
    allocations = []
    for i in range(n):
        w = float(opt_w[i])
        if w > 0.001:  # > 0.1%
            allocations.append({
                "ticker": tickers[i],
                "name": names[i],
                "weight": round(w * 100, 2),
                "amount": round(budget * w, 2),
            })
    allocations.sort(key=lambda x: x["weight"], reverse=True)

    # --- Historical backtest: monthly rebalance, no look-ahead ---
    # Старая версия: применяла финальные opt_w ко всей истории — но эти веса
    # рассчитывались на ВСЕЙ истории (включая будущее) → look-ahead bias,
    # equity-кривая нечестная.
    # Новая: каждые 21 торговых дня перевешиваем по данным ДО этой даты.
    # Получаем реалистичную out-of-sample кривую того, как стратегия работала
    # бы в реальном времени.
    equal_w = np.array([1 / n] * n)
    REBAL_DAYS = 21      # ~1 месяц
    MIN_HISTORY = 60     # минимум 60 дней до первой оптимизации
    returns_arr = returns.values
    n_days = len(returns_arr)

    current_w = equal_w.copy()
    hist_opt = [budget]
    hist_eq = [budget]

    for day_idx in range(n_days):
        # Ребалансируем перед применением сегодняшнего возврата
        if day_idx >= MIN_HISTORY and (day_idx - MIN_HISTORY) % REBAL_DAYS == 0:
            hist_returns = returns.iloc[:day_idx + 1]
            try:
                hist_ewm = hist_returns.ewm(halflife=HALF_LIFE, min_periods=20, adjust=True)
                hm_d = hist_ewm.mean().iloc[-1].values
                hc_blocks = hist_ewm.cov().dropna()
                hc_d = hc_blocks.iloc[-n:].values if len(hc_blocks) >= n else hist_returns.cov().values
                hm_a = hm_d * 252
                hc_a = hc_d * 252

                def _ns(w, hm=hm_a, hc=hc_a):
                    r = float(np.dot(w, hm))
                    v = float(np.sqrt(np.dot(w, np.dot(hc, w))))
                    return -((r - risk_free_rate) / v) if v > 0 else 0.0

                res = minimize(
                    _ns, current_w, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 50, "ftol": 1e-6},
                )
                if res.success and np.all(np.isfinite(res.x)):
                    current_w = res.x / res.x.sum()  # нормируем на случай численного дрейфа
            except Exception:
                pass  # держим текущие веса при ошибке

        r_eq = float(np.dot(returns_arr[day_idx], equal_w))
        r_opt = float(np.dot(returns_arr[day_idx], current_w))
        hist_opt.append(hist_opt[-1] * (1 + r_opt))
        hist_eq.append(hist_eq[-1] * (1 + r_eq))

    hist_dates = close.index.strftime("%Y-%m-%d").tolist()
    # Add one extra date for initial capital
    hist_dates = [hist_dates[0]] + hist_dates

    # Per-asset stats
    asset_rets = [round(float(r) * 100, 2) for r in mean_annual]
    asset_vols = [round(float(np.sqrt(cov_annual[i, i])) * 100, 2) for i in range(n)]
    corr = returns.corr().round(3).values.tolist()

    return PortfolioResult(
        tickers=tickers,
        names=names,
        budget=budget,
        opt_weights=[round(float(w), 4) for w in opt_w],
        opt_return=round(float(opt_ret) * 100, 2),
        opt_volatility=round(float(opt_vol) * 100, 2),
        opt_sharpe=round(float(opt_sh), 2),
        opt_allocations=allocations,
        min_weights=[round(float(w), 4) for w in min_w],
        min_return=round(float(min_ret) * 100, 2),
        min_volatility=round(float(min_vol) * 100, 2),
        min_sharpe=round(float(min_sh), 2),
        asset_returns=asset_rets,
        asset_volatility=asset_vols,
        correlation=corr,
        frontier_returns=[round(r * 100, 2) for r in frontier_rets],
        frontier_volatilities=[round(v * 100, 2) for v in frontier_vols],
        rand_returns=[round(r * 100, 2) for r in rand_rets],
        rand_volatilities=[round(v * 100, 2) for v in rand_vols],
        rand_sharpes=[round(s, 2) for s in rand_shs],
        hist_dates=hist_dates,
        hist_opt_equity=[round(e, 2) for e in hist_opt],
        hist_equal_equity=[round(e, 2) for e in hist_eq],
    )
