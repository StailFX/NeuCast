"""Pluggable order-execution interface — sim, testnet, or live.

What this is for
================

Today the paper-trader closes positions by reading the next bar's
``microprice_close`` and computing the simulated fill at that price.
That's correct for a **sim-only** contract (ADR-005). But the moment
we want any kind of real order routing — Binance Spot Testnet first,
then potentially mainnet — the trader needs a place to swap
"compute fill from microprice" for "send REST POST to /api/v3/order
and listen on user-data WebSocket for FILLED".

Rather than special-casing the trader for every venue, this module
defines an :class:`OrderExecutor` :class:`Protocol`. The trader holds
a reference, calls ``open_long / open_short / close_position`` on it,
and trusts the executor to return an :class:`ExecutedFill` describing
what actually happened (price, fee, exchange order id if any).

Implementations
---------------

* :class:`SimulatedExecutor` — what the existing paper-trader does
  inline today, refactored behind the Protocol. Produces the same
  output the trader currently computes, byte-for-byte. Default for
  the paper-trader runner; preserves the sim-only contract.
* :class:`BinanceTestnetExecutor` — **stub**. Documented integration
  points; raises :class:`NotImplementedError` on every call. Filling
  it in is the next major engineering item once the model has a
  defended dir_acc CI > 0.5.
* :class:`BinanceLiveExecutor` — **stub**. Same shape as testnet, but
  hits ``api.binance.com``. Behind a hard kill-switch — never enable
  without (a) a green frozen-holdout result, (b) a successful 30-day
  testnet run, (c) an explicit ``HF_LIVE_TRADING_CONFIRMED=1`` env.

Why a Protocol, not an ABC
--------------------------

* :class:`Protocol` is structurally typed — third-party executors
  (e.g. a hypothetical Kraken executor) can plug in without inheriting.
* No runtime overhead from base-class machinery.
* Tests can stub it with any object that has the methods.

Why the trader DOESN'T use this yet
-----------------------------------

The current ``PaperTrader`` opens/closes inline using the bar's
microprice. Refactoring it to call into an executor is a non-trivial
change to the state machine — meaningful enough that it deserves its
own focused PR with full state-machine test coverage. This module
defines the seam **first** so the testnet integration work can
proceed in parallel with model calibration; the trader refactor lands
when both are ready.

Defence-grade note
------------------

This file by itself doesn't enable live trading. It only **documents
the integration path**: where the seam goes, what shape an executor
must satisfy, what happens if you try to use the un-stubbed live one.
A reviewer asking "could you go live tomorrow?" can read this and
see (a) yes, the seam exists, (b) no, the live executor isn't
written, and (c) we explicitly gate it on testnet validation.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutedFill:
    """The result of an open / close action.

    ``fill_price`` and ``fill_ts`` are the venue's report (or, in the
    sim path, the bar's microprice + the bar-close ts). ``fee_usd``
    is the *quoted* fee — for sim this is computed from the configured
    maker_fee_bps; for live it's whatever Binance reported on the
    fill. ``venue_order_id`` is exchange-side; ``None`` for sim.

    ``is_sim`` is a hard discriminator: anything downstream that
    should NEVER touch sim data (e.g. tax-reporting, exchange-state
    reconciliation) can assert on this without parsing the
    ``model_version`` string convention.
    """
    fill_price: float
    fill_ts: datetime
    fee_usd: float
    venue_order_id: str | None
    is_sim: bool


@runtime_checkable
class OrderExecutor(Protocol):
    """Executor contract.

    All methods are async to keep the door open for live executors —
    those need to await REST + WebSocket round-trips. The sim
    implementation runs sync work inside its async methods (cheap,
    ~µs of overhead per call).
    """

    async def open_long(
        self,
        *,
        symbol: str,
        qty: float,
        ts: datetime,
        microprice: float,
    ) -> ExecutedFill: ...

    async def open_short(
        self,
        *,
        symbol: str,
        qty: float,
        ts: datetime,
        microprice: float,
    ) -> ExecutedFill: ...

    async def close_position(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,            # 'long' | 'short'
        ts: datetime,
        microprice: float,
        venue_order_id: str | None,
    ) -> ExecutedFill: ...


# ──────────────────────────────────────────────────────────────────────
# Simulated executor — current behaviour, refactored behind the Protocol
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SimulatedExecutor:
    """Sim path: fill at the bar's microprice with quoted maker fee.

    Stateless except for the fee config — re-uses the trader's
    existing ``compute_pnl`` / ``fee_per_side`` math, just behind the
    executor surface. Output of ``open_long(symbol, qty, ts, microprice)``
    is byte-identical to what ``PaperTrader._open_position`` records
    today.
    """
    maker_fee_bps_per_side: float = 7.5  # mirror PaperTraderConfig default

    async def open_long(
        self, *, symbol: str, qty: float, ts: datetime, microprice: float,
    ) -> ExecutedFill:
        return self._make_fill(qty=qty, microprice=microprice, ts=ts)

    async def open_short(
        self, *, symbol: str, qty: float, ts: datetime, microprice: float,
    ) -> ExecutedFill:
        return self._make_fill(qty=qty, microprice=microprice, ts=ts)

    async def close_position(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        ts: datetime,
        microprice: float,
        venue_order_id: str | None,
    ) -> ExecutedFill:
        # Side is irrelevant for sim — the price IS the bar microprice
        # regardless of direction. Real executors will care because
        # the fill price for a market sell at the bid != market buy
        # at the ask.
        return self._make_fill(qty=qty, microprice=microprice, ts=ts)

    def _make_fill(
        self, *, qty: float, microprice: float, ts: datetime,
    ) -> ExecutedFill:
        from app.highfreq.paper_trader import fee_per_side
        fee = fee_per_side(qty, microprice, self.maker_fee_bps_per_side)
        return ExecutedFill(
            fill_price=float(microprice),
            fill_ts=ts,
            fee_usd=float(fee),
            venue_order_id=None,
            is_sim=True,
        )


# ──────────────────────────────────────────────────────────────────────
# Binance Spot Testnet — STUB. Fill in for the next phase.
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BinanceTestnetExecutor:
    """Binance Spot Testnet (https://testnet.binance.vision).

    What this needs to do
    ---------------------

    1. **Auth**: HMAC-SHA256 sign every REST call with the API secret;
       send ``X-MBX-APIKEY`` header. Testnet creds are obtained from
       the testnet UI; they are NOT shared with mainnet.
    2. **Place orders**: ``POST /api/v3/order`` with type=MARKET (or
       LIMIT for cleaner fill semantics). For live we'd want LIMIT
       maker-only; testnet is permissive so MARKET is fine for
       initial integration.
    3. **Listen for fills**: open a user-data WebSocket via
       ``POST /api/v3/userDataStream`` (returns ``listenKey``), then
       ``wss://stream.testnet.binance.vision/ws/<listenKey>``. Watch
       for ``executionReport`` events; correlate by ``clientOrderId``
       (which we set ourselves).
    4. **Reconciliation**: on startup, ``GET /api/v3/openOrders`` and
       ``GET /api/v3/myTrades`` to detect any orders that landed
       while we were down. Compare to local state in ``paper_trades``;
       reconcile mismatches (refuse to start if reconciliation fails).
    5. **Cancellation**: ``DELETE /api/v3/order``. Idempotent on the
       venue side; we always send it on shutdown.
    6. **Kill switch**: a separate sync REST endpoint
       (``POST /api/v3/closeAllOpenOrders``) is the operator's
       one-button-stop.

    Stub behaviour
    --------------

    Until we explicitly fill this in, every call raises
    :class:`NotImplementedError`. The trader's runner picks an
    executor by name (env var ``HF_EXECUTOR_KIND``); pointing at this
    one before it's ready is a loud, fail-fast error rather than a
    silent no-op.
    """
    api_key: str
    api_secret: str
    base_url: str = "https://testnet.binance.vision"

    async def open_long(self, **kwargs: Any) -> ExecutedFill:
        raise NotImplementedError(
            "BinanceTestnetExecutor.open_long is a stub. "
            "Fill in per docstring or use SimulatedExecutor."
        )

    async def open_short(self, **kwargs: Any) -> ExecutedFill:
        raise NotImplementedError(
            "BinanceTestnetExecutor.open_short is a stub. "
            "Note: Binance Spot is long-only — shorts require margin. "
            "Decide on the venue strategy (USDM futures testnet?) "
            "before filling in."
        )

    async def close_position(self, **kwargs: Any) -> ExecutedFill:
        raise NotImplementedError(
            "BinanceTestnetExecutor.close_position is a stub. "
            "Fill in per docstring or use SimulatedExecutor."
        )


# ──────────────────────────────────────────────────────────────────────
# Binance Live (mainnet) — STUB + explicit guard.
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BinanceLiveExecutor:
    """Binance Spot mainnet. **Behind a hard env-gate.**

    Activation contract
    -------------------

    Constructing this class without ``HF_LIVE_TRADING_CONFIRMED=1`` in
    the environment raises :class:`RuntimeError` immediately. This is
    intentional belt-and-suspenders: a typo in the executor-kind env
    var (``"live"`` instead of ``"testnet"``) would otherwise be the
    most expensive bug imaginable.

    Beyond the env gate, mainnet activation requires:

    * Frozen-holdout dir_acc CI lower bound > 0.5 with p < 0.05.
    * 30+ days of clean testnet operation (no reconciliation drift,
      no lost orders).
    * Documented kill switch reachable from outside the codebase
      (e.g. a Telegram-bot command, in case the host itself is
      compromised).
    * Independent review of the fee + slippage model against measured
      testnet fills.

    None of those are codified as runtime checks here — they're
    operational gates. The env-var guard catches the typo class; the
    rest is process.
    """
    api_key: str
    api_secret: str
    base_url: str = "https://api.binance.com"

    def __post_init__(self) -> None:
        if os.environ.get("HF_LIVE_TRADING_CONFIRMED") != "1":
            raise RuntimeError(
                "BinanceLiveExecutor refused to construct without "
                "HF_LIVE_TRADING_CONFIRMED=1. Mainnet trading must be "
                "explicitly opted into via env, not by argument default."
            )

    async def open_long(self, **kwargs: Any) -> ExecutedFill:
        raise NotImplementedError("BinanceLiveExecutor is a stub")

    async def open_short(self, **kwargs: Any) -> ExecutedFill:
        raise NotImplementedError("BinanceLiveExecutor is a stub")

    async def close_position(self, **kwargs: Any) -> ExecutedFill:
        raise NotImplementedError("BinanceLiveExecutor is a stub")


# ──────────────────────────────────────────────────────────────────────
# Factory — picks an executor by env-var name.
# ──────────────────────────────────────────────────────────────────────


def make_executor(*, maker_fee_bps_per_side: float = 7.5) -> OrderExecutor:
    """Construct an executor from ``HF_EXECUTOR_KIND`` (default ``sim``).

    Recognised values:

    * ``sim`` (default) — :class:`SimulatedExecutor`. Production path
      until the trader is refactored to call through executors and
      until the model is calibrated.
    * ``testnet`` — :class:`BinanceTestnetExecutor`. Requires
      ``BINANCE_TESTNET_API_KEY`` + ``BINANCE_TESTNET_API_SECRET`` env.
      Currently a stub — calls will raise.
    * ``live`` — :class:`BinanceLiveExecutor`. Additionally requires
      ``HF_LIVE_TRADING_CONFIRMED=1``. Currently a stub.

    Anything else: log a loud warning and fall back to ``sim``.
    """
    kind = os.environ.get("HF_EXECUTOR_KIND", "sim").strip().lower()
    if kind == "sim":
        return SimulatedExecutor(maker_fee_bps_per_side=maker_fee_bps_per_side)
    if kind == "testnet":
        api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        if not api_key or not api_secret:
            raise RuntimeError(
                "HF_EXECUTOR_KIND=testnet requires both "
                "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET"
            )
        return BinanceTestnetExecutor(api_key=api_key, api_secret=api_secret)
    if kind == "live":
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        if not api_key or not api_secret:
            raise RuntimeError(
                "HF_EXECUTOR_KIND=live requires both "
                "BINANCE_API_KEY and BINANCE_API_SECRET"
            )
        return BinanceLiveExecutor(api_key=api_key, api_secret=api_secret)

    logger.warning(
        "HF_EXECUTOR_KIND=%r unrecognised — falling back to sim. "
        "Recognised values: sim, testnet, live.",
        kind,
    )
    return SimulatedExecutor(maker_fee_bps_per_side=maker_fee_bps_per_side)
