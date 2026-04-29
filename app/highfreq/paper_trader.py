"""Paper-trading state machine for the 1-minute directional model (Phase B).

Phase B is **sim-only** by ADR-005 — no real orders are ever placed. This
module measures what would happen if we *did* trade on the predictor's
output, with realistic Binance Spot maker fees and explicit risk caps.
The output (a stream of :class:`PaperTrade` rows persisted to
``paper_trades``) drives:

* the live UI block on ``/highfreq`` ("paper P&L last 24h: -3.2 bps");
* the post-Tier-2 gate decision (Phase B graduates to Tier 3 only if
  cumulative paper P&L is positive after 30+ days, see ADR-005);
* threshold tuning (the ``entry_long_threshold`` / ``maker_fee_bps``
  defaults below are best guesses — once we have a few hundred trades
  we'll grid-search them on historical sims).

Why every piece is decoupled from data quantity
-----------------------------------------------

We're writing this BEFORE the trainer ships its first ``.cbm``. The code
below is pure logic + a state machine; it operates on
``(prob_up, microprice, calibrated)`` tuples that the runner produces.
At ramp-up, the runner sees ``has_model=False`` and never calls
:meth:`PaperTrader.on_bar_close` — the trader stays at zero state. When
the trainer lands, the runner starts feeding bars and the trader begins
emitting :class:`PaperTrade` rows without any code change here.

Trading semantics (sim-only)
----------------------------

* **Long/short symmetry.** Binance Spot doesn't allow shorting without
  margin, but for measuring *directional accuracy* we treat shorts as
  symbolic (compute P&L as if we sold high and bought back lower). This
  is documented prominently in the UI — it's not lying about what the
  model would do under margin.
* **Fixed-notional sizing.** Every entry uses ``max_qty_usd / microprice``
  base units (e.g. $100 / $77k ≈ 0.0013 BTC). Keeps P&L math
  interpretable across price regimes.
* **Time-stop only on first iteration.** Open at bar :math:`t`, close at
  bar :math:`t + H` where :math:`H = \\text{HORIZON\\_MIN}`. We do NOT
  exit on prob-flip mid-horizon — the trainer fits exactly the
  ``signal at t → return at t+H`` map, so flipping out early throws
  away the predictive content the model was fit on.
* **Realistic fees.** Binance Spot Standard tier with BNB is 0.075 % per
  side (7.5 bp). Round-trip cost ≈ 15 bp — meaning the model has to
  call moves of at least that magnitude correctly to be net-positive.
  This is the **bar** Tier 3 has to clear.

Risk caps (circuit breakers)
----------------------------

Three hard kill switches, all reset at UTC midnight:

1. **Max consecutive losses.** Five red trades in a row → halt. Catches
   regime shifts (the model was fit on 7d, market is now 8d in).
2. **Max daily loss.** Cumulative net P&L drops below ``-max_daily_loss_usd``
   → halt. Bounds worst-case in any 24-hour window.
3. **One position at a time.** Doubles as a position cap; a runaway loop
   can never accumulate exposure.

When halted, :meth:`on_bar_close` returns ``None`` for any new entry but
will still close an open position on time-stop (we never strand
positions in the middle of a bar).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaperTraderConfig:
    """Tunable knobs for the paper trader.

    Defaults are chosen to be **conservative** — better to under-trade
    than to inflate paper P&L with optimistic fee assumptions. All
    fields will be tuned via grid-search after Tier 2 unlocks.
    """

    #: Open long if ``prob_up >= entry_long_threshold``.
    entry_long_threshold: float = 0.55

    #: Open short if ``prob_up <= entry_short_threshold``.
    entry_short_threshold: float = 0.45

    #: Forecast horizon in minutes (must match
    #: :data:`feature_pipeline.HORIZON_MIN`). Time-stop fires after this
    #: many minutes from entry.
    horizon_minutes: int = 1

    #: Per-trade notional in USD. $100 keeps round-trip fee at $0.15
    #: even at 7.5 bp/side — comfortably below micro-noise on a fast
    #: dev VPS where every cent of latency would otherwise eat the edge.
    #:
    #: This is the BASE notional. When ``vol_adjusted_sizing`` is enabled
    #: (default, see below), the actual per-trade qty scales by
    #: ``target_vol_bps / realized_vol_bps`` so position size shrinks
    #: in volatile regimes and expands in calm ones — Sharpe-improving
    #: on most strategies, including ours.
    max_qty_usd: float = 100.0

    #: Enable inverse-volatility scaling of position size. When True,
    #: ``compute_qty_for_notional`` uses the most recent
    #: ``microprice_high_low_bps`` of the bar as the realized-vol proxy
    #: and scales notional by ``target_vol_bps / realized_vol_bps``,
    #: clipped to ``[vol_scale_min, vol_scale_max]``.
    vol_adjusted_sizing: bool = True

    #: Target intra-bar volatility (in bps of microprice). Trades sized
    #: so that ``qty × realized_vol == target_vol_bps × max_qty_usd /
    #: 1e4``. 5 bps ≈ $39 on BTC at $77k — a typical 1-minute high-low
    #: range during normal market hours.
    target_vol_bps: float = 5.0

    #: Hard clip on the volatility-adjustment factor. Without this, a
    #: dead-flat bar with realized_vol_bps → 0 would produce infinite
    #: notional (and a runaway trade). 0.25× → 4× spans most regimes.
    vol_scale_min: float = 0.25
    vol_scale_max: float = 4.0

    #: Maker fee per side in basis points. Default 7.5 bp = Binance
    #: Spot standard tier with BNB-paid discount.  Override via the
    #: ``venue`` field below for the futures venue (2 bp/side, set
    #: automatically) or by passing the field explicitly for a custom
    #: fee model.
    maker_fee_bps_per_side: float = 7.5

    #: Trading venue (release S phase 3). ``"spot"`` = Binance Spot
    #: (default, preserves backward compat); ``"futures"`` = Binance
    #: USDM Perpetual Futures. The ``__post_init__`` below auto-tunes
    #: ``maker_fee_bps_per_side`` to the venue's fee tier when the
    #: caller hasn't passed an explicit override:
    #:   spot    → 7.5 bp/side  (standard + BNB discount)
    #:   futures → 2.0 bp/side  (USDM maker, no volume-tier requirement)
    #: Round-trip cost on futures is therefore 4 bp vs spot 15 bp —
    #: the structural difference that motivates ADR-019.
    venue: str = "spot"

    #: Refuse to open new positions when the model isn't statistically
    #: calibrated (``dir_acc_ci_low <= 0.5``). UX wise this is the
    #: difference between "model is making suggestions" and "model is
    #: making suggestions backed by a positive bootstrap CI". An
    #: overridable knob in case we want to backtest the uncalibrated
    #: regime separately.
    require_calibrated: bool = True

    def __post_init__(self) -> None:  # noqa: D401
        """Validate ``venue`` and auto-tune ``maker_fee_bps_per_side``
        if the caller didn't pass an explicit override.

        We use a sentinel rather than ``replace=`` semantics: if
        ``maker_fee_bps_per_side`` is at its default 7.5 AND ``venue``
        is ``"futures"``, we reset to 2.0. A caller who explicitly
        passes ``maker_fee_bps_per_side=2.0`` (or anything else) on a
        futures config keeps that value.
        """
        if self.venue not in ("spot", "futures"):
            raise ValueError(
                f"venue must be 'spot' or 'futures', got {self.venue!r}"
            )
        # We can't reassign a frozen-dataclass field directly; use
        # object.__setattr__ as is the documented pattern for
        # post-init mutation on frozen dataclasses.
        if self.venue == "futures" and self.maker_fee_bps_per_side == 7.5:
            object.__setattr__(self, "maker_fee_bps_per_side", 2.0)


@dataclass(frozen=True)
class RiskCaps:
    """Hard kill-switches resettable only at UTC midnight."""

    #: Halt after this many consecutive losing trades.
    max_consecutive_losses: int = 5

    #: Halt when cumulative daily net P&L drops below ``-max_daily_loss_usd``.
    max_daily_loss_usd: float = 5.0


# ────────────────────────────────────────────────────────────────────────────
# Pure-function helpers (testable in isolation)
# ────────────────────────────────────────────────────────────────────────────


def fee_per_side(qty: float, price: float, maker_fee_bps: float) -> float:
    """Maker fee for one fill, in USD.

    ``qty * price`` is the notional; we charge ``maker_fee_bps / 10_000``
    of that. Returns ``0.0`` if any input is non-positive — defensive,
    matches the predictor's NaN-scrub philosophy.
    """
    if qty <= 0 or price <= 0 or maker_fee_bps < 0:
        return 0.0
    return qty * price * (maker_fee_bps / 1e4)


def compute_qty_for_notional(notional_usd: float, price: float) -> float:
    """``notional_usd / price``, with safety on bad price input."""
    if price <= 0 or notional_usd <= 0:
        return 0.0
    return notional_usd / price


def vol_adjusted_notional(
    *,
    base_notional_usd: float,
    realized_vol_bps: float,
    target_vol_bps: float,
    scale_min: float,
    scale_max: float,
) -> float:
    """Inverse-volatility scaling of notional.

    Returns ``base_notional_usd * scale``, where
    ``scale = clip(target_vol_bps / realized_vol_bps, [scale_min, scale_max])``.

    Calm bar (realized < target) → scale > 1 → bigger notional.
    Volatile bar (realized > target) → scale < 1 → smaller notional.
    Clip prevents both blow-ups (realized ≈ 0 → infinite scale) and
    over-shrinking (realized >> target → near-zero trades that get
    eaten by fees).

    Pure function — testable without a trader instance.
    """
    if base_notional_usd <= 0 or target_vol_bps <= 0:
        return 0.0
    if realized_vol_bps <= 0:
        # Dead bar — fall back to base notional rather than divide-by-zero.
        # Conservative choice: don't trade an unusual signal aggressively.
        return float(base_notional_usd)
    raw_scale = target_vol_bps / realized_vol_bps
    scale = max(scale_min, min(scale_max, raw_scale))
    return float(base_notional_usd) * scale


def compute_pnl(
    side: Literal["long", "short"],
    entry_price: float,
    exit_price: float,
    qty: float,
    maker_fee_bps_per_side: float,
) -> tuple[float, float]:
    """Net P&L for a closed paper trade.

    Returns ``(net_pnl_usd, pnl_bps)`` where:

    * ``net_pnl_usd`` is gross P&L (price move × qty, sign-corrected for
      side) minus both legs' maker fees;
    * ``pnl_bps`` normalises by the entry notional so trades across
      different price regimes are comparable. Equivalently the
      "post-fee return on the position over the bar".
    """
    if entry_price <= 0 or exit_price <= 0 or qty <= 0:
        return 0.0, 0.0

    if side == "long":
        gross_pnl_usd = qty * (exit_price - entry_price)
    elif side == "short":
        gross_pnl_usd = qty * (entry_price - exit_price)
    else:  # pragma: no cover — Literal type prevents this at type-check
        raise ValueError(f"unknown side: {side!r}")

    fee_entry = fee_per_side(qty, entry_price, maker_fee_bps_per_side)
    fee_exit = fee_per_side(qty, exit_price, maker_fee_bps_per_side)
    net_pnl_usd = gross_pnl_usd - fee_entry - fee_exit

    pnl_bps = (net_pnl_usd / (qty * entry_price)) * 1e4
    return net_pnl_usd, pnl_bps


def decide_entry_side(
    prob_up: float, config: PaperTraderConfig,
) -> Optional[Literal["long", "short"]]:
    """Pure decision: which side (if any) does this signal call for?

    * ``prob_up >= entry_long_threshold`` → ``"long"``
    * ``prob_up <= entry_short_threshold`` → ``"short"``
    * else → ``None`` (skip this bar; the model is not confident enough
      either way to overcome 15 bp of round-trip fees).

    The thresholds being symmetric around 0.5 by default
    (``0.45 / 0.55``) reflects that we have no prior on which direction
    the model will be more accurate at — a single bias term applied
    here would silently bake one in.
    """
    if not (0.0 <= prob_up <= 1.0):
        return None  # garbage input; bail
    if prob_up >= config.entry_long_threshold:
        return "long"
    if prob_up <= config.entry_short_threshold:
        return "short"
    return None


# ────────────────────────────────────────────────────────────────────────────
# State objects
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaperPosition:
    """Open paper position — one at a time; replaced on close."""

    symbol: str
    side: Literal["long", "short"]
    qty: float
    entry_ts: datetime
    entry_price: float
    entry_prob_up: float
    fee_paid_entry: float
    model_version: str          # e.g. mtime as ISO string

    def elapsed_minutes(self, now: datetime) -> float:
        """Minutes since entry — drives the time-stop check."""
        return max(0.0, (now - self.entry_ts).total_seconds() / 60.0)


ExitReason = Literal[
    "time_stop",        # most common; we held for `horizon_minutes`
    "halt_close",       # risk caps engaged after entry — close immediately
    # Future: "stop_loss" once we add price-based stops in Tier 3
]


@dataclass(frozen=True)
class PaperTrade:
    """Closed trade — one row destined for the ``paper_trades`` table.

    Designed to be JSON-serialisable for the ``/api/highfreq/paper_trades``
    endpoint and for direct insertion via :func:`asyncpg`. ``datetime``s
    are aware UTC (the schema uses ``TIMESTAMPTZ``).
    """

    symbol: str
    side: Literal["long", "short"]
    qty: float
    entry_ts: datetime
    entry_price: float
    entry_prob_up: float
    exit_ts: datetime
    exit_price: float
    exit_reason: ExitReason
    fee_paid_total_usd: float
    pnl_usd: float
    pnl_bps: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Serialize datetimes for JSON; the DB writer keeps them aware.
        d["entry_ts"] = self.entry_ts.isoformat()
        d["exit_ts"] = self.exit_ts.isoformat()
        return d


# ────────────────────────────────────────────────────────────────────────────
# State machine
# ────────────────────────────────────────────────────────────────────────────


class _NoOpReason(Enum):
    """Why ``on_bar_close`` did not open a new trade.

    Internal — the runner just sees ``None`` and moves on; this enum
    is used by :meth:`PaperTrader.last_no_op_reason` for the UI status
    block ("currently halted: max consecutive losses reached") so users
    aren't left wondering why the trader stopped.
    """

    OK_OPENED = "opened_new_position"
    OK_CLOSED = "closed_existing_position"
    SKIP_NOT_CALIBRATED = "skip_not_calibrated"
    SKIP_HALTED_DAILY_LOSS = "skip_halted_daily_loss"
    SKIP_HALTED_LOSS_STREAK = "skip_halted_loss_streak"
    SKIP_NEUTRAL_SIGNAL = "skip_neutral_signal"
    SKIP_POSITION_OPEN = "skip_position_already_open"
    SKIP_BAD_PRICE = "skip_bad_microprice"


@dataclass
class PaperTraderState:
    """Internal counters — exposed via :meth:`status` for the UI."""

    open_position: Optional[PaperPosition] = None
    closed_today: list[PaperTrade] = field(default_factory=list)
    daily_pnl_usd: float = 0.0
    consecutive_losses: int = 0
    halted_reason: Optional[str] = None  # None when not halted
    current_day_utc: Optional[datetime] = None  # midnight floor of "today"
    last_no_op: Optional[_NoOpReason] = None
    total_trades: int = 0  # lifetime, not reset daily


class PaperTrader:
    """Stateful per-symbol paper trader.

    Drive it from the runner once per minute on bar close::

        trader = PaperTrader("BTCUSDT")
        # in the loop:
        trade = trader.on_bar_close(
            ts=bar_close_ts,
            microprice=close_microprice,
            prob_up=predictor_output,
            calibrated=predictor.is_calibrated(),
            model_version=str(predictor._weights_mtime or 0.0),
        )
        if trade is not None:
            await db_writer.write_paper_trade(trade)

    The trader writes nothing itself — it returns a :class:`PaperTrade`
    when a trade closes, and the caller decides where it goes (DB, log,
    in-memory test harness). This keeps the unit-testable surface free
    of I/O.
    """

    def __init__(
        self,
        symbol: str,
        *,
        config: PaperTraderConfig | None = None,
        risk_caps: RiskCaps | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.config = config or PaperTraderConfig()
        self.risk_caps = risk_caps or RiskCaps()
        self.state = PaperTraderState()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def on_bar_close(
        self,
        *,
        ts: datetime,
        microprice: float,
        prob_up: float,
        calibrated: bool,
        model_version: str = "unknown",
        realized_vol_bps: Optional[float] = None,
    ) -> Optional[PaperTrade]:
        """Process one minute-bar close.

        ``realized_vol_bps`` (e.g. the bar's ``microprice_high_low_bps``
        feature) drives inverse-volatility position sizing when
        ``config.vol_adjusted_sizing`` is True. ``None`` (the default)
        falls back to fixed-notional sizing — preserves the old
        behaviour for callers that don't have the feature row handy.

        Returns a :class:`PaperTrade` row IF a position closed on this
        call (caller persists it). ``None`` otherwise — including when
        a new position was opened (we don't generate a trade row until
        it closes).
        """
        # 1. UTC-midnight rollover — reset daily counters and un-halt
        #    if the only halt cause was a daily-loss cap.
        self._maybe_rollover_day(ts)

        # 2. If we have an open position, check time-stop first. We want
        #    to exit even when halted (cleaner state) — but won't open
        #    a new one this call.
        closed: Optional[PaperTrade] = None
        if self.state.open_position is not None:
            elapsed = self.state.open_position.elapsed_minutes(ts)
            if elapsed >= self.config.horizon_minutes:
                closed = self._close_position(
                    ts=ts, exit_price=microprice, reason="time_stop",
                )
                self.state.last_no_op = _NoOpReason.OK_CLOSED

        # 3. Halted? Don't open new positions but DO emit the close above.
        if self.state.halted_reason is not None:
            if closed is None:
                self.state.last_no_op = (
                    _NoOpReason.SKIP_HALTED_LOSS_STREAK
                    if self.state.halted_reason == "loss_streak"
                    else _NoOpReason.SKIP_HALTED_DAILY_LOSS
                )
            return closed

        # 4. Don't open if we already have a (non-time-stopped) position.
        if self.state.open_position is not None:
            if closed is None:
                self.state.last_no_op = _NoOpReason.SKIP_POSITION_OPEN
            return closed

        # 5. Calibration gate.
        if self.config.require_calibrated and not calibrated:
            if closed is None:
                self.state.last_no_op = _NoOpReason.SKIP_NOT_CALIBRATED
            return closed

        # 6. Bad price → bail. asyncpg gives us aware timestamps and
        #    floats; if microprice is 0 / NaN something upstream is
        #    broken and we should not pretend to open a trade.
        if not (microprice > 0):
            if closed is None:
                self.state.last_no_op = _NoOpReason.SKIP_BAD_PRICE
            return closed

        # 7. Entry decision.
        side = decide_entry_side(prob_up, self.config)
        if side is None:
            if closed is None:
                self.state.last_no_op = _NoOpReason.SKIP_NEUTRAL_SIGNAL
            return closed

        self._open_position(
            ts=ts,
            entry_price=microprice,
            entry_prob_up=prob_up,
            side=side,
            model_version=model_version,
            realized_vol_bps=realized_vol_bps,
        )
        if closed is None:
            self.state.last_no_op = _NoOpReason.OK_OPENED
        return closed

    def status(self) -> dict[str, Any]:
        """JSON-friendly snapshot for the UI / endpoint.

        Reports current position, today's stats, and any active halt —
        designed to be polled at the same 2-second cadence as
        ``/api/highfreq/status``.
        """
        op = self.state.open_position
        return {
            "symbol": self.symbol,
            "open_position": (
                {
                    "side": op.side,
                    "qty": op.qty,
                    "entry_ts": op.entry_ts.isoformat(),
                    "entry_price": op.entry_price,
                    "entry_prob_up": op.entry_prob_up,
                    "model_version": op.model_version,
                }
                if op is not None
                else None
            ),
            "halted": self.state.halted_reason is not None,
            "halted_reason": self.state.halted_reason,
            "daily_pnl_usd": round(self.state.daily_pnl_usd, 4),
            "consecutive_losses": self.state.consecutive_losses,
            "trades_today": len(self.state.closed_today),
            "total_trades": self.state.total_trades,
            "current_day_utc": (
                self.state.current_day_utc.isoformat()
                if self.state.current_day_utc is not None
                else None
            ),
            "last_no_op": (
                self.state.last_no_op.value
                if self.state.last_no_op is not None
                else None
            ),
        }

    def force_close(
        self, *, ts: datetime, microprice: float, reason: ExitReason = "halt_close",
    ) -> Optional[PaperTrade]:
        """Emergency close — used by runner shutdown to flatten before exit.

        Returns the closing :class:`PaperTrade` (or ``None`` if there
        was no open position). Does not touch the halt state.
        """
        if self.state.open_position is None:
            return None
        return self._close_position(ts=ts, exit_price=microprice, reason=reason)

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _open_position(
        self,
        *,
        ts: datetime,
        entry_price: float,
        entry_prob_up: float,
        side: Literal["long", "short"],
        model_version: str,
        realized_vol_bps: Optional[float] = None,
    ) -> None:
        # Determine notional: vol-adjusted if enabled and we have a
        # realized-vol input, else fixed.
        if self.config.vol_adjusted_sizing and realized_vol_bps is not None:
            notional_usd = vol_adjusted_notional(
                base_notional_usd=self.config.max_qty_usd,
                realized_vol_bps=float(realized_vol_bps),
                target_vol_bps=self.config.target_vol_bps,
                scale_min=self.config.vol_scale_min,
                scale_max=self.config.vol_scale_max,
            )
        else:
            notional_usd = self.config.max_qty_usd

        qty = compute_qty_for_notional(notional_usd, entry_price)
        if qty <= 0:
            # Defensive — `microprice > 0` was checked but float rounding
            # could land here for crazy small prices.
            self.state.last_no_op = _NoOpReason.SKIP_BAD_PRICE
            return
        fee_entry = fee_per_side(qty, entry_price, self.config.maker_fee_bps_per_side)
        self.state.open_position = PaperPosition(
            symbol=self.symbol,
            side=side,
            qty=qty,
            entry_ts=ts,
            entry_price=entry_price,
            entry_prob_up=entry_prob_up,
            fee_paid_entry=fee_entry,
            model_version=model_version,
        )
        logger.info(
            "PaperTrader[%s]: opened %s @ %s (qty=%.6f, notional=$%.2f, "
            "vol_bps=%s, prob_up=%.3f, fee=%.4f)",
            self.symbol, side, entry_price, qty, notional_usd,
            f"{realized_vol_bps:.2f}" if realized_vol_bps is not None else "n/a",
            entry_prob_up, fee_entry,
        )

    def _close_position(
        self,
        *,
        ts: datetime,
        exit_price: float,
        reason: ExitReason,
    ) -> PaperTrade:
        op = self.state.open_position
        assert op is not None, "_close_position called with no open position"

        if exit_price <= 0:
            # Garbage exit price — don't pretend to compute P&L. Re-use
            # the entry price so the trade is recorded with zero gross
            # P&L and the operator sees the bad-data fingerprint.
            logger.warning(
                "PaperTrader[%s]: bad exit_price=%s, falling back to entry_price",
                self.symbol, exit_price,
            )
            exit_price = op.entry_price

        net_pnl_usd, pnl_bps = compute_pnl(
            side=op.side,
            entry_price=op.entry_price,
            exit_price=exit_price,
            qty=op.qty,
            maker_fee_bps_per_side=self.config.maker_fee_bps_per_side,
        )
        fee_exit = fee_per_side(op.qty, exit_price, self.config.maker_fee_bps_per_side)

        trade = PaperTrade(
            symbol=op.symbol,
            side=op.side,
            qty=op.qty,
            entry_ts=op.entry_ts,
            entry_price=op.entry_price,
            entry_prob_up=op.entry_prob_up,
            exit_ts=ts,
            exit_price=exit_price,
            exit_reason=reason,
            fee_paid_total_usd=op.fee_paid_entry + fee_exit,
            pnl_usd=net_pnl_usd,
            pnl_bps=pnl_bps,
            model_version=op.model_version,
        )

        # Update counters BEFORE clearing open_position so risk-cap
        # checks see consistent state.
        self.state.daily_pnl_usd += net_pnl_usd
        self.state.closed_today.append(trade)
        self.state.total_trades += 1

        if net_pnl_usd < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        self._maybe_engage_risk_caps()

        self.state.open_position = None
        logger.info(
            "PaperTrader[%s]: closed %s reason=%s pnl_usd=%.4f pnl_bps=%.2f",
            self.symbol, op.side, reason, net_pnl_usd, pnl_bps,
        )
        return trade

    def _maybe_engage_risk_caps(self) -> None:
        """Evaluate kill switches; flip ``halted_reason`` on first breach."""
        if self.state.consecutive_losses >= self.risk_caps.max_consecutive_losses:
            if self.state.halted_reason is None:
                self.state.halted_reason = "loss_streak"
                logger.warning(
                    "PaperTrader[%s]: HALTED on consecutive_losses=%d (cap=%d)",
                    self.symbol,
                    self.state.consecutive_losses,
                    self.risk_caps.max_consecutive_losses,
                )
            return
        if self.state.daily_pnl_usd <= -self.risk_caps.max_daily_loss_usd:
            if self.state.halted_reason is None:
                self.state.halted_reason = "daily_loss"
                logger.warning(
                    "PaperTrader[%s]: HALTED on daily_pnl_usd=%.4f (cap=-%.2f)",
                    self.symbol,
                    self.state.daily_pnl_usd,
                    self.risk_caps.max_daily_loss_usd,
                )

    def _maybe_rollover_day(self, ts: datetime) -> None:
        """Reset daily counters when the UTC date changes.

        We compare ``ts.date()`` (UTC) to the previously-stored day; on
        change, zero the counters and clear any halt that was driven by
        the daily-loss cap (loss-streak halts persist across days —
        documented choice; a long losing streak shouldn't auto-clear
        just because the clock ticked).
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)

        if self.state.current_day_utc is None:
            self.state.current_day_utc = midnight
            return

        if midnight > self.state.current_day_utc:
            logger.info(
                "PaperTrader[%s]: day rollover %s → %s, resetting daily counters",
                self.symbol, self.state.current_day_utc, midnight,
            )
            self.state.current_day_utc = midnight
            self.state.daily_pnl_usd = 0.0
            self.state.closed_today = []
            # Loss-streak halt persists; daily-loss halt clears.
            if self.state.halted_reason == "daily_loss":
                self.state.halted_reason = None
