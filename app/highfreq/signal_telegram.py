"""Telegram signal-flip notifications.

What this is
============

Every minute the predictor emits one of three signals (``up`` /
``down`` / ``neutral``). When the signal **changes** for a symbol
the user wants a Telegram ping with the new state. This module owns:

1. :class:`SignalFlipDetector` — pure in-memory state, "is the new
   signal different from the previous one for this symbol?" Caller
   passes (symbol, signal); detector remembers and reports.
2. :func:`send_signal_alert_async` — HTTP POST to Telegram's
   ``sendMessage`` API with an HTML-formatted body. Async so the
   paper-trader runner can ``await`` without blocking the bar loop.
3. :class:`SignalAlertConfig` — three env vars wrap config in a
   dataclass so tests can inject without monkey-patching.

Per-process state
-----------------

Each ``paper_trader_runner@{symbol}.service`` is its own process,
so each holds its own ``SignalFlipDetector``. Since each runner
handles exactly one symbol, the detector's state is trivial — one
entry. The "first observation after restart" doesn't notify (we
don't know the prior state); the second-and-subsequent flips do.

Hysteresis
----------

The MVP detects ANY signal change as a flip — including
``up → neutral → up`` bounce on a P(UP) hovering near 0.55. Real
production would add a hysteresis band (e.g. require ``prob_up`` to
cross by ≥0.03 before flipping). For now we accept the noise and
trust the user to mute notifications if it bothers them.

Why a separate bot, not the Grafana one
---------------------------------------

* Different audiences: Grafana alerts are for ops (ingest down,
  disk full); signal alerts are for the trader. Mixing them muddies
  the priority signal.
* Different mute behaviour: a user paging through signal alerts
  shouldn't accidentally mute the disk-full warning.
* Different ratelimit budgets: Telegram rate-limits per-bot; busy
  signal flips shouldn't cause Grafana alerts to be deferred.

Both bots can target the same chat_id — same Telegram chat,
different sender persona.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignalAlertConfig:
    """All config in one struct — pass into the runner explicitly,
    don't read env at the bottom of every method."""
    enabled: bool
    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> "SignalAlertConfig":
        """Read the canonical env vars. Empty/missing tokens disable
        the feature with a single source of truth."""
        token = os.environ.get("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "").strip()
        chat = os.environ.get("HF_TELEGRAM_SIGNAL_CHAT_ID", "").strip()
        flag = os.environ.get("HF_TELEGRAM_SIGNAL_ENABLED", "").strip().lower()
        enabled = (flag in ("1", "true", "yes")) and bool(token) and bool(chat)
        return cls(enabled=enabled, bot_token=token, chat_id=chat)


# ──────────────────────────────────────────────────────────────────────
# Pure: flip detection
# ──────────────────────────────────────────────────────────────────────


class SignalFlipDetector:
    """Stateful: keeps last signal per symbol, reports flips."""

    def __init__(self) -> None:
        self._last: dict[str, str] = {}

    def check_flip(self, symbol: str, new_signal: str) -> str | None:
        """Return the PREVIOUS signal if a flip occurred, else None.

        Contract:
        * First observation for a symbol → returns None (no prior
          signal to compare against; we don't notify on cold start
          because the runner restarts often and would spam).
        * Same-as-previous → returns None (no flip).
        * Different from previous → returns previous signal so the
          caller can format "flipped from X → Y".

        Always updates internal state so the NEXT call has the right
        baseline.
        """
        prev = self._last.get(symbol)
        self._last[symbol] = new_signal
        if prev is None:
            return None
        if prev == new_signal:
            return None
        return prev

    def reset(self) -> None:
        """Test helper — drops all state."""
        self._last.clear()


# ──────────────────────────────────────────────────────────────────────
# Message formatting
# ──────────────────────────────────────────────────────────────────────


_SIGNAL_EMOJI = {
    "up": "🟢",
    "down": "🔴",
    "neutral": "⚪",
}


def format_trade_closed_message(
    *,
    symbol: str,
    side: str,                       # 'long' | 'short'
    entry_price: float,
    exit_price: float,
    qty: float,
    pnl_usd: float,
    pnl_bps: float,
    exit_reason: str,                # 'time_stop' | 'halt_close'
    entry_ts: datetime,
    exit_ts: datetime,
    model_version: str = "",
) -> str:
    """HTML body for "paper trade just closed" notification.

    Why a separate format from flip_message: the audience cares
    about different things — a flip is "model just changed its
    mind", a closed trade is "here's what happened with real
    money-equivalent stakes". Mixing them in one template hides
    the most-relevant fact.

    P&L sign drives the emoji header: green for profit, red for
    loss, gray for break-even. The bps figure is the
    fee-adjusted, vol-normalised return — comparable across
    BTC / ETH / BNB at very different price levels.
    """
    pnl_emoji = "🟢" if pnl_usd > 0 else ("🔴" if pnl_usd < 0 else "⚪")
    side_arrow = "↑ LONG" if side == "long" else "↓ SHORT"
    demo_tag = ""
    if model_version == "pre-calibration-demo":
        demo_tag = "\n<i>⚠️ pre-calibration trade — for visualisation only</i>"
    elapsed_min = (exit_ts - entry_ts).total_seconds() / 60.0
    return (
        f"{pnl_emoji} <b>{symbol}</b> trade closed\n"
        f"{side_arrow}  ·  {exit_reason}\n"
        f"\n"
        f"<code>entry  = {entry_price:>12,.2f}</code>\n"
        f"<code>exit   = {exit_price:>12,.2f}</code>\n"
        f"<code>qty    = {qty:>12.6f}</code>\n"
        f"<code>pnl    = ${pnl_usd:>+11.4f}  ({pnl_bps:+.1f} bps)</code>\n"
        f"<code>held   = {elapsed_min:>11.1f} min</code>\n"
        f"<code>at     = {exit_ts.strftime('%Y-%m-%d %H:%M UTC')}</code>"
        f"{demo_tag}"
    )


def format_flip_message(
    *,
    symbol: str,
    old_signal: str,
    new_signal: str,
    prob_up: float,
    microprice: float,
    ts: datetime,
    model_version: str = "",
) -> str:
    """HTML body matching Telegram's parse_mode='HTML' subset.

    Pinned by tests so a refactor that breaks the layout (e.g. drops
    the prob_up line) fails CI rather than the user noticing visually.
    """
    arrow_old = _SIGNAL_EMOJI.get(old_signal, "?")
    arrow_new = _SIGNAL_EMOJI.get(new_signal, "?")
    demo_tag = ""
    if model_version == "pre-calibration-demo":
        demo_tag = "\n<i>⚠️ pre-calibration — for visualisation only</i>"
    return (
        f"<b>{symbol}</b> signal flipped\n"
        f"{arrow_old} <s>{old_signal}</s>  →  {arrow_new} <b>{new_signal}</b>\n"
        f"\n"
        f"<code>P(up) = {prob_up:.4f}</code>\n"
        f"<code>price  = {microprice:,.2f}</code>\n"
        f"<code>at     = {ts.strftime('%Y-%m-%d %H:%M UTC')}</code>"
        f"{demo_tag}"
    )


# ──────────────────────────────────────────────────────────────────────
# HTTP send
# ──────────────────────────────────────────────────────────────────────


async def send_signal_alert_async(
    config: SignalAlertConfig,
    *,
    body_html: str,
    timeout_seconds: float = 5.0,
) -> bool:
    """POST to Telegram sendMessage. Returns True on 2xx, False otherwise.

    Uses ``urllib`` over a thread to avoid pulling httpx into the slim
    HFT runner's deps. The request is small (≈1 KB body, ≈1 KB
    response) and fires once per signal flip — overhead is negligible.
    """
    if not config.enabled:
        return False

    import asyncio
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    url = f"https://api.telegram.org/bot{config.bot_token}/sendMessage"
    payload = {
        "chat_id": config.chat_id,
        "text": body_html,
        "parse_mode": "HTML",
        "disable_notification": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _do_request() -> tuple[int, str]:
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except URLError as exc:
            return 0, f"URLError: {exc.reason}"
        except Exception as exc:  # broad — Telegram returns 4xx with body
            return 0, f"{type(exc).__name__}: {exc}"

    code, body = await asyncio.get_running_loop().run_in_executor(
        None, _do_request,
    )
    if 200 <= code < 300:
        return True
    logger.warning(
        "telegram sendMessage failed: code=%d body=%s", code, body[:200],
    )
    return False
