"""Sim-backtest engine for the 1-minute directional model.

Consumes the per-prediction DataFrame produced by
:func:`app.highfreq.trainer.walk_forward_evaluate` (joined with the
``meta`` frame to recover ``return_bps`` and ``microprice_close``) and
produces:

* **Per-trade ledger** — one row per executed signal with entry / exit
  microprice, signed return, fees, net P&L in basis points and dollars.
* **Equity curve** — cumulative P&L, drawdown, per-minute and
  annualised Sharpe.
* **Fill-rate sensitivity strip** — same back-test recomputed at maker
  fill rates :math:`f \\in \\{0, 0.3, 0.5, 0.7, 1.0\\}`.

The fee model (ADR-005) is the central artifact:

============  =========  =======================================
Side          Cost (bp)  Notes
============  =========  =======================================
Taker          +10       Standard Binance Spot tier-0 fee
Maker          −1        Rebate (Binance Spot tier-0 maker)
============  =========  =======================================

Average per-side cost at maker fill rate :math:`f` is
:math:`f \\cdot (-1) + (1 - f) \\cdot 10` bps. Roundtrip cost is twice
that. Two reports are always emitted side by side — one assuming pure
taker, one assuming the configured maker fill rate — so a reader can
see both economic regimes without us cherry-picking.

Design notes
------------

* **No slippage at this layer.** We use ``microprice_close`` as both
  entry and exit price. Microprice is depth-weighted mid, the closest
  fair value an order would actually transact at on top-of-book.
  Realistic slippage modelling (queue position, walk-the-book) is a
  Phase B concern when we have order-flow data at sub-second cadence.
* **No leverage, no compounding.** Each trade uses a fixed notional
  (default $10 000). Equity curve is the sum of per-trade dollar P&L.
  Compounding would obscure the strategy's per-bar edge with portfolio
  effects irrelevant at coursework scale.
* **Position one bar at a time.** We open at bar :math:`t` close and
  close at bar :math:`t+1` close. No multi-bar holding, no overlapping
  positions. This matches the trainer's target horizon (ADR-003).
* **Confidence threshold.** Optional gate on prediction probability:
  trade only when :math:`p \\ge \\tau` for longs or
  :math:`p \\le 1 - \\tau` for shorts. ``tau = 0.5`` trades every bar.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Fee model & strategy config
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeeModel:
    """Binance Spot tier-0 fees in basis points (1 bp = 0.01 %).

    Negative means a rebate (the exchange pays you).
    """

    maker_fee_bps: float = -1.0
    taker_fee_bps: float = 10.0

    def per_side_bps(self, maker_fill_rate: float) -> float:
        """Average cost per side given the maker-fill assumption."""
        f = float(np.clip(maker_fill_rate, 0.0, 1.0))
        return f * self.maker_fee_bps + (1.0 - f) * self.taker_fee_bps

    def roundtrip_bps(self, maker_fill_rate: float) -> float:
        """Total cost for entry + exit (2× per-side)."""
        return 2.0 * self.per_side_bps(maker_fill_rate)


@dataclass(frozen=True)
class StrategyConfig:
    """Controls signal → trade conversion."""

    confidence_threshold: float = 0.5  # 0.5 trades every bar; >0.5 filters
    notional_per_trade: float = 10_000.0
    fee: FeeModel = field(default_factory=FeeModel)


# ────────────────────────────────────────────────────────────────────────────
# Trade ledger construction
# ────────────────────────────────────────────────────────────────────────────

def _signal_to_direction(
    proba: np.ndarray, threshold: float,
) -> np.ndarray:
    """Map probabilities → {+1, 0, -1}.

    * proba ≥ threshold        → +1 (long)
    * proba ≤ 1 - threshold    → -1 (short)
    * otherwise                →  0 (flat, no trade)

    With ``threshold == 0.5`` every bar gets a non-zero direction.
    With ``threshold == 0.6`` only high-conviction signals trade.
    """
    direction = np.zeros_like(proba, dtype=np.int8)
    direction[proba >= threshold] = 1
    direction[proba <= 1.0 - threshold] = -1
    return direction


def build_trade_ledger(
    predictions: pd.DataFrame,
    *,
    strategy: StrategyConfig,
    maker_fill_rate: float,
) -> pd.DataFrame:
    """Convert a predictions frame into a per-trade ledger.

    Parameters
    ----------
    predictions
        Must contain ``minute, proba, return_bps``. ``y_true`` is
        optional but used for the win/loss column when present.
    strategy
        Confidence threshold + notional + fee model.
    maker_fill_rate
        Fraction of trades filled at the maker price (0–1).

    Returns
    -------
    pd.DataFrame
        Columns:
        ``minute, direction, gross_bps, fee_bps, net_bps, net_dollars,
        is_win, equity_dollars`` — one row per *traded* bar (flat bars
        excluded). Sorted by minute ascending.
    """
    required = {"minute", "proba", "return_bps"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")

    df = predictions.sort_values("minute").reset_index(drop=True).copy()
    direction = _signal_to_direction(
        df["proba"].to_numpy(dtype=float),
        threshold=strategy.confidence_threshold,
    )
    traded_mask = direction != 0
    if not traded_mask.any():
        # Empty ledger; return the canonical empty schema.
        return _empty_ledger()

    df = df.loc[traded_mask].reset_index(drop=True)
    direction = direction[traded_mask]
    gross_bps = direction.astype(float) * df["return_bps"].to_numpy(dtype=float)
    rt_fee_bps = strategy.fee.roundtrip_bps(maker_fill_rate)
    net_bps = gross_bps - rt_fee_bps
    net_dollars = net_bps / 1e4 * strategy.notional_per_trade

    ledger = pd.DataFrame({
        "minute": pd.to_datetime(df["minute"], utc=True),
        "direction": direction,
        "proba": df["proba"].astype(float).to_numpy(),
        "gross_bps": gross_bps,
        "fee_bps": np.full_like(gross_bps, rt_fee_bps, dtype=float),
        "net_bps": net_bps,
        "net_dollars": net_dollars,
        "is_win": net_bps > 0,
    })
    ledger["equity_dollars"] = ledger["net_dollars"].cumsum()
    return ledger


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "minute", "direction", "proba", "gross_bps", "fee_bps",
        "net_bps", "net_dollars", "is_win", "equity_dollars",
    ])


# ────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ────────────────────────────────────────────────────────────────────────────

#: Minutes per year — trading is 24/7 on crypto, so 60 × 24 × 365.
#: Used to annualise the per-minute Sharpe ratio.
MINUTES_PER_YEAR: int = 60 * 24 * 365


def max_drawdown_pct(equity: np.ndarray) -> float:
    """Largest peak-to-trough drop in the equity curve, as a fraction.

    Returns 0.0 for an empty / monotonically increasing curve. Computed
    against the running peak — a $100 → $90 drop on a $200 peak is
    reported as -0.5 (50 % drawdown), not -0.1.
    """
    if len(equity) == 0:
        return 0.0
    # Anchor to a virtual zero start so a strategy that loses on the
    # first trade still reports a sensible drawdown relative to that
    # initial "flat" baseline.
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    full = np.concatenate([[0.0], equity])
    # Avoid division by zero when peak is 0 (drawdown then is the
    # signed dollar amount expressed against a $1 base for stability).
    denom = np.where(running_peak > 0, running_peak, 1.0)
    dd = (full - running_peak) / denom
    return float(dd.min())


@dataclass
class BacktestReport:
    """Headline statistics for one (strategy, fill-rate) configuration."""

    confidence_threshold: float
    maker_fill_rate: float
    notional_per_trade: float
    fee_roundtrip_bps: float
    n_signals: int
    n_trades: int                 # signals after threshold filter
    n_wins: int
    n_losses: int
    win_rate: float               # n_wins / n_trades
    avg_win_bps: float
    avg_loss_bps: float
    gross_pnl_bps: float
    fees_bps: float
    net_pnl_bps: float
    net_pnl_dollars: float
    sharpe_per_minute: float      # mean / std of per-trade net_bps
    sharpe_annualised: float      # × sqrt(MINUTES_PER_YEAR)
    max_drawdown_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise(
    ledger: pd.DataFrame,
    *,
    strategy: StrategyConfig,
    maker_fill_rate: float,
    n_signals: int,
) -> BacktestReport:
    """Roll a trade ledger into a :class:`BacktestReport`."""
    if ledger.empty:
        return BacktestReport(
            confidence_threshold=strategy.confidence_threshold,
            maker_fill_rate=maker_fill_rate,
            notional_per_trade=strategy.notional_per_trade,
            fee_roundtrip_bps=strategy.fee.roundtrip_bps(maker_fill_rate),
            n_signals=n_signals, n_trades=0,
            n_wins=0, n_losses=0, win_rate=float("nan"),
            avg_win_bps=float("nan"), avg_loss_bps=float("nan"),
            gross_pnl_bps=0.0, fees_bps=0.0, net_pnl_bps=0.0,
            net_pnl_dollars=0.0,
            sharpe_per_minute=float("nan"),
            sharpe_annualised=float("nan"),
            max_drawdown_pct=0.0,
        )

    wins = ledger.loc[ledger["is_win"], "net_bps"]
    losses = ledger.loc[~ledger["is_win"], "net_bps"]

    net_arr = ledger["net_bps"].to_numpy()
    net_std = float(net_arr.std(ddof=1)) if len(net_arr) > 1 else float("nan")
    if net_std and not math.isnan(net_std) and net_std > 0:
        sharpe_min = float(net_arr.mean()) / net_std
    else:
        sharpe_min = float("nan")
    sharpe_ann = (
        sharpe_min * math.sqrt(MINUTES_PER_YEAR)
        if not math.isnan(sharpe_min) else float("nan")
    )

    return BacktestReport(
        confidence_threshold=strategy.confidence_threshold,
        maker_fill_rate=maker_fill_rate,
        notional_per_trade=strategy.notional_per_trade,
        fee_roundtrip_bps=strategy.fee.roundtrip_bps(maker_fill_rate),
        n_signals=n_signals,
        n_trades=int(len(ledger)),
        n_wins=int(len(wins)),
        n_losses=int(len(losses)),
        win_rate=float(len(wins) / len(ledger)),
        avg_win_bps=float(wins.mean()) if len(wins) else float("nan"),
        avg_loss_bps=float(losses.mean()) if len(losses) else float("nan"),
        gross_pnl_bps=float(ledger["gross_bps"].sum()),
        fees_bps=float(ledger["fee_bps"].sum()),
        net_pnl_bps=float(ledger["net_bps"].sum()),
        net_pnl_dollars=float(ledger["net_dollars"].sum()),
        sharpe_per_minute=sharpe_min,
        sharpe_annualised=sharpe_ann,
        max_drawdown_pct=max_drawdown_pct(
            ledger["equity_dollars"].to_numpy()
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
# Top-level entry
# ────────────────────────────────────────────────────────────────────────────

def simulate(
    predictions: pd.DataFrame,
    *,
    strategy: StrategyConfig | None = None,
    maker_fill_rate: float = 0.5,
) -> tuple[BacktestReport, pd.DataFrame]:
    """Run one back-test configuration. Returns ``(report, ledger)``."""
    cfg = strategy or StrategyConfig()
    ledger = build_trade_ledger(
        predictions, strategy=cfg, maker_fill_rate=maker_fill_rate,
    )
    report = summarise(
        ledger, strategy=cfg, maker_fill_rate=maker_fill_rate,
        n_signals=len(predictions),
    )
    return report, ledger


def fill_rate_sweep(
    predictions: pd.DataFrame,
    *,
    strategy: StrategyConfig | None = None,
    fill_rates: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0),
) -> pd.DataFrame:
    """Sensitivity sweep across maker-fill assumptions.

    Returns one DataFrame row per ``fill_rate`` containing the headline
    metrics from :class:`BacktestReport`. Indexed positionally — the
    ``maker_fill_rate`` column holds the parameter value.
    """
    cfg = strategy or StrategyConfig()
    rows: list[dict[str, Any]] = []
    for f in fill_rates:
        rep, _ = simulate(predictions, strategy=cfg, maker_fill_rate=f)
        rows.append(rep.to_dict())
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# JSON-serialisable combined report
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CombinedReport:
    """Side-by-side maker (configurable) + taker reports + fill-rate sweep.

    This is the JSON artefact the UI consumes. ADR-005 mandates that
    both economic regimes are visible to the reader — there is no
    single "true" P&L for an HFT-style strategy; it depends on what
    fraction of orders rest as makers versus chase as takers.
    """

    confidence_threshold: float
    notional_per_trade: float
    fee_model: dict[str, float]
    n_signals: int
    taker: dict[str, Any]                 # report at maker_fill_rate=0
    maker_at_50pct: dict[str, Any]        # report at maker_fill_rate=0.5
    maker_at_100pct: dict[str, Any]       # report at maker_fill_rate=1.0
    fill_rate_sweep: list[dict[str, Any]]

    def to_json(self) -> str:
        def _scrub(o: Any) -> Any:
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: _scrub(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_scrub(v) for v in o]
            return o
        return json.dumps(_scrub(asdict(self)), indent=2, default=str)


def run_backtest(
    predictions: pd.DataFrame,
    *,
    strategy: StrategyConfig | None = None,
) -> CombinedReport:
    """Produce the canonical combined report for portfolio reporting.

    Reports both the taker-only economy (maker_fill_rate=0) and two
    optimistic maker scenarios (50 % and 100 %), plus the full sweep.
    """
    cfg = strategy or StrategyConfig()
    taker_rep, _ = simulate(predictions, strategy=cfg, maker_fill_rate=0.0)
    maker50_rep, _ = simulate(predictions, strategy=cfg, maker_fill_rate=0.5)
    maker100_rep, _ = simulate(predictions, strategy=cfg, maker_fill_rate=1.0)
    sweep_df = fill_rate_sweep(predictions, strategy=cfg)
    return CombinedReport(
        confidence_threshold=cfg.confidence_threshold,
        notional_per_trade=cfg.notional_per_trade,
        fee_model={
            "maker_fee_bps": cfg.fee.maker_fee_bps,
            "taker_fee_bps": cfg.fee.taker_fee_bps,
        },
        n_signals=len(predictions),
        taker=taker_rep.to_dict(),
        maker_at_50pct=maker50_rep.to_dict(),
        maker_at_100pct=maker100_rep.to_dict(),
        fill_rate_sweep=sweep_df.to_dict(orient="records"),
    )
