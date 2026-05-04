"""Long-polling Telegram bot worker — answers /stats, /trades, /accuracy.

Architecture
============

A separate **always-on systemd service** (``neucast-signal-bot.service``)
runs this module's :func:`main`. It:

1. Calls Telegram's ``getUpdates`` long-polling endpoint, blocking up
   to 25 s for incoming messages. Cheap — Telegram holds the connection
   open server-side.
2. For each update with a recognised ``/`` command, builds the
   response by querying Postgres directly (paper_trades,
   predictions_log, training_runs).
3. Replies via ``sendMessage`` with HTML body.
4. Persists the next ``offset`` so a worker restart resumes where it
   left off (no duplicate replies, no missed commands).

Why long-polling not webhook
----------------------------

Webhook would need an HTTPS endpoint exposed publicly. Long-polling
runs entirely outbound from Tokyo through the existing internet
connection — no inbound port to firewall, no certificate to manage,
no extra nginx config. For a low-traffic ops bot (a few commands per
day) the overhead of long-polling is irrelevant.

Why a separate service from paper-trader-runner
-----------------------------------------------

* Different restart cadences — a misbehaving paper-trader shouldn't
  drop incoming bot commands; a hung HTTP poll shouldn't pause the
  trader's bar loop.
* Different lifetime — the bot runs continuously even when paper
  trading is paused; the trader needs the predictor + ingest, the
  bot only needs Postgres.
* Different deps — the bot needs no asyncpg pool (we use SQLAlchemy
  sync for simplicity), no predictor, no feature pipeline.

Authorization
-------------

ONLY messages from a chat_id matching ``HF_TELEGRAM_SIGNAL_CHAT_ID``
are processed. Random Telegram users who happen to find the bot
get a brief "not authorized" reply and their messages are
otherwise ignored. Defence-grade: no command can leak data to a
non-operator account.

Commands
--------

* ``/start``, ``/help`` — usage hint.
* ``/stats``      — overall summary: predictions, paper-trades,
  realized accuracy, P&L.
* ``/stats SYMBOL`` — same but per-symbol.
* ``/trades``     — last 5 paper-trades across all symbols.
* ``/trades SYMBOL [N]`` — last N for one symbol (default 10, max 25).
* ``/accuracy``   — directional accuracy by symbol from
  predictions_log (real-time, separates model skill from trader
  skill).
* ``/model``      — latest training_runs row per symbol (dir_acc CI,
  bars, p-value if any folds yet).

The set is small on purpose — every command answers a distinct
question. Adding more commands here forks attention from the actual
defence-grade widgets in the web UI.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# Long-poll timeout: Telegram supports up to 50 s; we use 25 s so a
# slow-restart on the bot side doesn't make Telegram's connection
# cap pile up.
LONG_POLL_TIMEOUT_SECONDS = 25

# Where the offset file lives. Owned by stailfx so the bot worker
# (running as the same user) can rewrite it atomically.
OFFSET_FILE = "/var/lib/neucast/telegram_bot_offset.txt"


# ──────────────────────────────────────────────────────────────────────
# HTTP layer
# ──────────────────────────────────────────────────────────────────────


def _api(token: str, method: str, payload: dict[str, Any] | None = None,
         timeout: float = 30.0) -> dict[str, Any]:
    """Single source of truth for Telegram API calls.

    Raises on non-2xx; the caller decides how to handle (most callers
    log + continue rather than crash the worker).
    """
    url = TELEGRAM_API.format(token=token, method=method)
    body = json.dumps(payload or {}).encode("utf-8")
    req = Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    parsed = json.loads(raw)
    if not parsed.get("ok"):
        raise RuntimeError(f"telegram {method} failed: {parsed}")
    return parsed["result"]


def _send_html(token: str, chat_id: str, html: str) -> None:
    """Best-effort send. Logs warnings on failure but never raises —
    a failed reply must not crash the worker."""
    try:
        _api(token, "sendMessage", {
            "chat_id": chat_id,
            "text": html,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=10.0)
    except Exception as exc:
        logger.warning("sendMessage failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Offset persistence
# ──────────────────────────────────────────────────────────────────────


def _load_offset() -> int:
    try:
        with open(OFFSET_FILE) as f:
            return int(f.read().strip() or "0")
    except (OSError, ValueError):
        return 0


def _save_offset(offset: int) -> None:
    try:
        os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
        with open(OFFSET_FILE, "w") as f:
            f.write(str(offset))
    except OSError as exc:
        logger.warning("offset save failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Command handlers
# ──────────────────────────────────────────────────────────────────────


HELP_TEXT = (
    "<b>NeuCast Signals bot</b>\n"
    "\n"
    "Commands:\n"
    "<code>/stats [SYMBOL]</code> — summary across symbols (or one)\n"
    "<code>/trades [SYMBOL] [N]</code> — recent paper trades (default 5)\n"
    "<code>/accuracy</code> — directional accuracy from predictions_log\n"
    "<code>/model</code> — latest trainer report per symbol\n"
    "<code>/feetiers [SYMBOL]</code> — what P&amp;L would be at retail / VIP / MM-rebate fee tiers\n"
    "<code>/antiskill [SYMBOL]</code> — gross winrate &amp; anti-skill detector status\n"
    "<code>/help</code> — this message"
)


def _format_trades_response(rows: list[dict[str, Any]]) -> str:
    """Render a list of paper_trades rows as a Telegram HTML body."""
    if not rows:
        return "<i>No paper trades yet.</i>\n\n" + (
            "Demo mode is enabled but no signals have crossed the "
            "entry thresholds in the last bar. Try again after "
            "a few minutes."
        )

    lines = [f"<b>Last {len(rows)} paper trades</b>\n"]
    for r in rows:
        pnl_emoji = "🟢" if r["pnl_usd"] > 0 else ("🔴" if r["pnl_usd"] < 0 else "⚪")
        side_str = "↑LONG" if r["side"] == "long" else "↓SHORT"
        lines.append(
            f"{pnl_emoji} <b>{r['symbol']}</b> {side_str} · {r['exit_reason']}\n"
            f"<code>{r['entry_price']:>10,.2f} → {r['exit_price']:>10,.2f}  "
            f"({(r['pnl_bps']):+.1f} bps · ${r['pnl_usd']:+.4f})</code>\n"
            f"<code>{r['exit_ts']:%Y-%m-%d %H:%M UTC}  ·  {r['model_version']}</code>"
        )
    return "\n\n".join(lines)


def _format_stats_response(stats: dict[str, Any]) -> str:
    """Render the overall summary block."""
    lines = ["<b>NeuCast HFT — live stats</b>\n"]
    for sym, s in stats.items():
        wins = s["trade_wins"]
        losses = s["trade_losses"]
        pnl = s["pnl_total_usd"]
        pnl_emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
        acc_pct = (
            f"{100.0 * s['accuracy_directional'] / s['accuracy_total']:.1f}%"
            if s["accuracy_total"] else "—"
        )
        lines.append(
            f"<b>{sym}</b>\n"
            f"<code>predictions   = {s['predictions_total']:>6d}</code>\n"
            f"<code>paper trades  = {s['trades_total']:>6d}  ({wins}W / {losses}L)</code>\n"
            f"<code>P&amp;L total    = ${pnl:>+10.4f}</code>  {pnl_emoji}\n"
            f"<code>directional   = {s['accuracy_directional']:>4d}/{s['accuracy_total']:<4d}  ({acc_pct})</code>"
        )
    return "\n\n".join(lines)


def _format_accuracy_response(rows: list[dict[str, Any]]) -> str:
    """Render realized directional accuracy with Wilson CI + one-sided
    binomial p-value (H₀: p = 0.5, H_a: p > 0.5).

    Defence-grade: a 54.8 % point estimate is just a number until
    you also state "n=2145, CI [0.527, 0.569], p < 10⁻⁵".
    Telegram is where the operator goes when /highfreq UI isn't at
    hand, so the same statistical rigour belongs here too.
    """
    if not rows:
        return "<i>No backfilled predictions yet.</i>"

    import math
    try:
        from scipy.stats import binomtest
    except ImportError:
        binomtest = None  # gracefully degrade — print acc only

    def _wilson(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return 0.0, 1.0
        p = k / n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, centre - half), min(1.0, centre + half)

    def _verdict_emoji(p_value: float | None, lo: float) -> str:
        # Definitive skill: p<0.001 AND CI lower bound > 0.5
        if p_value is not None and p_value < 0.001 and lo > 0.5:
            return "✅"
        # Borderline: CI lower above 0.5 but p > 0.001 (rare)
        if lo > 0.5:
            return "🟢"
        # Negative-edge red flag
        if p_value is not None and p_value > 0.99:
            return "🚨"
        # Noise: CI crosses 0.5
        return "⚪"

    lines = [
        "<b>Realized directional accuracy</b>",
        "<i>from predictions_log (n=all backfilled minutes; neutral excluded)</i>\n",
    ]
    for r in rows:
        n = int(r["directional"])
        hits = int(r["hits"])
        if n == 0:
            lines.append(f"<code>{r['symbol']:>8}: no data yet</code>")
            continue
        acc = hits / n
        lo, hi = _wilson(hits, n)
        if binomtest is not None:
            p_val = float(binomtest(k=hits, n=n, p=0.5, alternative="greater").pvalue)
            p_str = f"p={p_val:.2e}" if p_val < 0.01 else f"p={p_val:.3f}"
        else:
            p_val = None
            p_str = ""
        emoji = _verdict_emoji(p_val, lo)
        lines.append(
            f"{emoji} <b>{r['symbol']}</b>\n"
            f"<code>acc  = {acc:.4f}  ({hits}/{n})</code>\n"
            f"<code>CI95 = [{lo:.4f}, {hi:.4f}]</code>\n"
            f"<code>{p_str}</code>"
        )
    return "\n\n".join(lines)


def _format_fee_tiers_response(symbol: str, n_trades: int,
                                tiers: list[dict[str, Any]]) -> str:
    """Render the fee-tier P&L breakdown.

    Each tier shows the same trade sequence simulated at a
    different fee assumption — the gap between rows IS the fee
    burden. Operator can read off "edge exists, fees eat it" from
    the table.
    """
    if n_trades == 0:
        return f"<i>No paper trades for {symbol} yet.</i>"
    lines = [
        f"<b>{symbol}</b> — P&amp;L by fee tier",
        f"<i>{n_trades} trades, recomputed at different fee assumptions</i>\n",
    ]
    for t in tiers:
        pnl = t["pnl_usd"]
        emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
        bps_total = t["pnl_bps_avg"] * t["n_trades"]
        fee_str = f"{t['fee_bps_per_side']:+.1f}bp/side"
        lines.append(
            f"{emoji} <b>{t['tier']:>10}</b>  <code>{fee_str}</code>\n"
            f"   <code>pnl  = ${pnl:>+10.4f}  ({t['pnl_bps_avg']:+.1f} bps avg)</code>\n"
            f"   <code>w/l  = {t['n_wins']:>3d}W / {t['n_losses']:<3d}L</code>"
        )
    return "\n\n".join(lines)


def _format_anti_skill_response(reports: list[dict[str, Any]]) -> str:
    """Render anti-skill detector status across symbols.

    The ``note`` field contains operator-facing text including the
    literal ``<`` character (e.g. "n=21 < min_sample=30") — those
    MUST be html-escaped before embedding into the Telegram HTML
    body, otherwise Telegram's parser reads them as unclosed tags
    and rejects the whole message with HTTP 400.
    """
    import html as _html
    if not reports:
        return "<i>No symbols to evaluate.</i>"
    lines = ["<b>Anti-skill detector</b>", "<i>gross winrate over last N closed trades</i>\n"]
    for r in reports:
        note_esc = _html.escape(r.get("note") or "")
        if r["gross_winrate"] is None:
            lines.append(
                f"<b>{_html.escape(r['symbol'])}</b>\n"
                f"<code>{note_esc}</code>"
            )
            continue
        emoji = "🚨" if r["is_anti_skilled"] else (
            "⚠️" if r["gross_winrate"] < 0.50 else "✅"
        )
        ci = ""
        if r.get("gross_winrate_ci_low") is not None:
            ci = f"  CI [{r['gross_winrate_ci_low']:.3f}, {r['gross_winrate_ci_high']:.3f}]"
        lines.append(
            f"{emoji} <b>{_html.escape(r['symbol'])}</b>\n"
            f"<code>winrate = {r['gross_winrate']:.3f}  ({r['n_gross_wins']}/{r['n_trades_in_window']}){ci}</code>\n"
            f"<i>{note_esc}</i>"
        )
    return "\n\n".join(lines)


def _query_anti_skill(database_url: str, *,
                       symbol: str | None = None) -> list[dict[str, Any]]:
    """Run the anti-skill detector for one symbol or all known."""
    from sqlalchemy import create_engine, text
    from app.highfreq.anti_skill_detector import compute_anti_skill_from_rows

    eng = create_engine(database_url, future=True)
    with eng.connect() as conn:
        if symbol:
            symbols = [symbol.upper()]
        else:
            symbols = [
                r[0] for r in conn.execute(text(
                    "SELECT DISTINCT symbol FROM paper_trades ORDER BY symbol"
                ))
            ]
            if not symbols:
                return []
        reports = []
        for sym in symbols:
            res = conn.execute(text(
                "SELECT side, entry_price, exit_price, exit_reason, exit_ts "
                "  FROM paper_trades "
                " WHERE symbol = :symbol "
                " ORDER BY exit_ts DESC LIMIT 50"
            ), {"symbol": sym})
            rows = [dict(r._mapping) for r in res]
            reports.append(compute_anti_skill_from_rows(rows, symbol=sym).to_dict())
    return reports


def _query_fee_tiers(database_url: str, *,
                     symbol: str | None = None) -> tuple[int, list[dict[str, Any]]]:
    """Fetch trades + run the fee-tier sim. Returns (n_trades, tier_dicts)."""
    from sqlalchemy import create_engine, text
    from app.highfreq.fee_tiers import summarise_all_tiers

    eng = create_engine(database_url, future=True)
    sql = text(
        "SELECT side, qty, entry_price, exit_price "
        "  FROM paper_trades "
        + (" WHERE symbol = :symbol " if symbol else " ")
        + " ORDER BY exit_ts ASC"
    )
    params = {"symbol": symbol.upper()} if symbol else {}
    with eng.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql, params)]
    summaries = summarise_all_tiers(rows)
    return len(rows), [s.to_dict() for s in summaries]


def _format_model_response(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<i>No trainer runs yet.</i>"
    lines = ["<b>Latest trainer report per symbol</b>\n"]
    for r in rows:
        if r["dir_acc_mean"] is None:
            ci_str = "—"
            p_str = "—"
            edge_str = "—"
        else:
            ci_str = (
                f"{r['dir_acc_mean']:.4f} "
                f"[{r['dir_acc_ci_low']:.4f}, {r['dir_acc_ci_high']:.4f}]"
            )
            p_str = (
                f"{r['dir_acc_p_value']:.4f}"
                if r["dir_acc_p_value"] is not None else "—"
            )
            # Edge over naive majority-class classifier — POSITIVE means
            # the model captures more than the trivial "always predict
            # the dominant class" baseline. NEGATIVE = model under-
            # performs the trivial classifier (red flag at large n).
            base_rate = r.get("base_rate")
            if base_rate is not None:
                naive = max(float(base_rate), 1.0 - float(base_rate))
                edge = float(r["dir_acc_mean"]) - naive
                edge_emoji = "✅" if edge > 0 else ("⚠️" if edge < 0 else "➖")
                edge_str = f"{edge_emoji} {edge:+.4f}  (naive baseline = {naive:.4f})"
            else:
                edge_str = "—"
        lines.append(
            f"<b>{r['symbol']}</b>\n"
            f"<code>bars        = {r['n_minutes_after_neutral_drop']}</code>\n"
            f"<code>folds       = {r['n_folds']}</code>\n"
            f"<code>dir_acc     = {ci_str}</code>\n"
            f"<code>p-value     = {p_str}</code>\n"
            f"<code>edge vs naive = {edge_str}</code>\n"
            f"<code>at          = {r['run_started_at']:%Y-%m-%d %H:%M UTC}</code>"
        )
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# DB queries
# ──────────────────────────────────────────────────────────────────────


def _query_trades(database_url: str, *, symbol: str | None = None,
                  limit: int = 5) -> list[dict[str, Any]]:
    from sqlalchemy import create_engine, text
    eng = create_engine(database_url, future=True)
    sql = text(
        "SELECT symbol, side, entry_price, exit_price, qty, "
        "       pnl_usd, pnl_bps, exit_reason, exit_ts, model_version "
        "  FROM paper_trades "
        + (" WHERE symbol = :symbol " if symbol else " ")
        + " ORDER BY exit_ts DESC "
        + " LIMIT :lim"
    )
    params = {"lim": int(limit)}
    if symbol:
        params["symbol"] = symbol.upper()
    with eng.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(str(sql)), params)]


def _query_stats(database_url: str) -> dict[str, dict[str, Any]]:
    """Aggregate per-symbol stats. Single round-trip query."""
    from sqlalchemy import create_engine, text
    eng = create_engine(database_url, future=True)
    sql = text("""
        WITH symbols AS (
          SELECT DISTINCT symbol FROM (
            SELECT symbol FROM paper_trades
            UNION ALL SELECT symbol FROM predictions_log
          ) s
        )
        SELECT
          s.symbol,
          COALESCE(p.n_predictions, 0)         AS predictions_total,
          COALESCE(p.n_directional, 0)         AS accuracy_total,
          COALESCE(p.n_hits, 0)                AS accuracy_directional,
          COALESCE(t.n_trades, 0)              AS trades_total,
          COALESCE(t.n_wins, 0)                AS trade_wins,
          COALESCE(t.n_losses, 0)              AS trade_losses,
          COALESCE(t.pnl_total, 0.0)           AS pnl_total_usd
        FROM symbols s
        LEFT JOIN (
          SELECT symbol,
                 COUNT(*) AS n_predictions,
                 COUNT(realized_correct) AS n_directional,
                 COUNT(*) FILTER (WHERE realized_correct IS TRUE) AS n_hits
          FROM predictions_log
          GROUP BY symbol
        ) p ON s.symbol = p.symbol
        LEFT JOIN (
          SELECT symbol,
                 COUNT(*) AS n_trades,
                 COUNT(*) FILTER (WHERE pnl_usd > 0) AS n_wins,
                 COUNT(*) FILTER (WHERE pnl_usd < 0) AS n_losses,
                 SUM(pnl_usd) AS pnl_total
          FROM paper_trades
          GROUP BY symbol
        ) t ON s.symbol = t.symbol
        ORDER BY s.symbol
    """)
    with eng.connect() as conn:
        out = {}
        for r in conn.execute(sql):
            d = dict(r._mapping)
            out[d["symbol"]] = {k: v for k, v in d.items() if k != "symbol"}
        return out


def _query_accuracy(database_url: str) -> list[dict[str, Any]]:
    from sqlalchemy import create_engine, text
    eng = create_engine(database_url, future=True)
    sql = text("""
        SELECT symbol,
               COUNT(realized_correct)                          AS directional,
               COUNT(*) FILTER (WHERE realized_correct IS TRUE) AS hits
          FROM predictions_log
         GROUP BY symbol
         ORDER BY symbol
    """)
    with eng.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql)]


def _query_model(database_url: str) -> list[dict[str, Any]]:
    """Latest training_runs row per symbol. Includes ``base_rate`` so
    the formatter can compute edge-over-naive-baseline locally."""
    from sqlalchemy import create_engine, text
    eng = create_engine(database_url, future=True)
    sql = text("""
        SELECT DISTINCT ON (symbol)
               symbol, run_started_at, n_minutes_after_neutral_drop,
               n_folds, dir_acc_mean, dir_acc_ci_low, dir_acc_ci_high,
               dir_acc_p_value, base_rate
          FROM training_runs
         ORDER BY symbol, run_started_at DESC
    """)
    with eng.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql)]


# ──────────────────────────────────────────────────────────────────────
# Update dispatcher
# ──────────────────────────────────────────────────────────────────────


def _process_update(
    update: dict[str, Any],
    *,
    bot_token: str,
    allowed_chat_id: str,
    database_url: str,
) -> None:
    """Handle one Telegram update. Best-effort: log and continue on
    any error so a malformed command doesn't crash the worker."""
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id", ""))
    text = (msg.get("text") or "").strip()
    if not text or not text.startswith("/"):
        return  # ignore non-commands silently

    if chat_id != allowed_chat_id:
        # Don't leak data; one curt reply then ignore.
        _send_html(
            bot_token, chat_id,
            "<i>not authorized — this bot is operator-only</i>",
        )
        logger.warning(
            "rejected command from chat_id=%s text=%r", chat_id, text[:50],
        )
        return

    parts = text.split()
    cmd = parts[0].lower().split("@")[0]   # /stats@NeuCastSignals_bot → /stats
    args = parts[1:]

    try:
        if cmd in ("/start", "/help"):
            _send_html(bot_token, chat_id, HELP_TEXT)
        elif cmd == "/stats":
            stats = _query_stats(database_url)
            if args:
                wanted = args[0].upper()
                stats = {k: v for k, v in stats.items() if k == wanted}
                if not stats:
                    _send_html(
                        bot_token, chat_id,
                        f"<i>no data for {args[0]!r}</i>",
                    )
                    return
            _send_html(bot_token, chat_id, _format_stats_response(stats))
        elif cmd == "/trades":
            symbol = args[0].upper() if args else None
            try:
                limit = int(args[1]) if len(args) >= 2 else (10 if symbol else 5)
            except ValueError:
                limit = 5
            limit = max(1, min(limit, 25))
            rows = _query_trades(database_url, symbol=symbol, limit=limit)
            _send_html(bot_token, chat_id, _format_trades_response(rows))
        elif cmd == "/accuracy":
            rows = _query_accuracy(database_url)
            _send_html(bot_token, chat_id, _format_accuracy_response(rows))
        elif cmd == "/model":
            rows = _query_model(database_url)
            _send_html(bot_token, chat_id, _format_model_response(rows))
        elif cmd == "/feetiers":
            symbol_arg = args[0].upper() if args else None
            n_trades, tiers = _query_fee_tiers(database_url, symbol=symbol_arg)
            _send_html(
                bot_token, chat_id,
                _format_fee_tiers_response(
                    symbol_arg or "ALL", n_trades, tiers,
                ),
            )
        elif cmd == "/antiskill":
            symbol_arg = args[0].upper() if args else None
            reports = _query_anti_skill(database_url, symbol=symbol_arg)
            _send_html(
                bot_token, chat_id,
                _format_anti_skill_response(reports),
            )
        else:
            _send_html(
                bot_token, chat_id,
                f"<i>unknown command {cmd!r}</i>\n\n{HELP_TEXT}",
            )
    except Exception:
        # Code-review H-4sec (2026-05-04): full exception (including
        # SQLAlchemy DSN fragments + DB schema details) goes to the
        # journal, NOT to the Telegram chat. The chat reply is generic
        # so a hijacked operator session can't be used as a SQL-schema
        # exploration interface.
        logger.exception("command %s failed", cmd)
        _send_html(
            bot_token, chat_id,
            f"<i>command {cmd!r} failed; see journalctl -u "
            "neucast-highfreq-telegram-bot for details</i>",
        )


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    bot_token = os.environ.get("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("HF_TELEGRAM_SIGNAL_CHAT_ID", "").strip()
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not bot_token or not chat_id or not database_url:
        logger.error(
            "missing required env: HF_TELEGRAM_SIGNAL_BOT_TOKEN / "
            "HF_TELEGRAM_SIGNAL_CHAT_ID / DATABASE_URL"
        )
        return 2

    logger.info("telegram bot worker started — long-polling")
    offset = _load_offset()
    while True:
        try:
            updates = _api(
                bot_token, "getUpdates",
                {"offset": offset, "timeout": LONG_POLL_TIMEOUT_SECONDS},
                timeout=LONG_POLL_TIMEOUT_SECONDS + 10,
            )
        except URLError as exc:
            logger.warning("getUpdates network error: %s — sleep 5 s", exc)
            time.sleep(5)
            continue
        except Exception as exc:
            logger.warning("getUpdates failed: %s — sleep 5 s", exc)
            time.sleep(5)
            continue

        for upd in updates:
            try:
                _process_update(
                    upd,
                    bot_token=bot_token,
                    allowed_chat_id=chat_id,
                    database_url=database_url,
                )
            except Exception:
                logger.exception("update handler crashed")
            offset = max(offset, int(upd["update_id"]) + 1)
            _save_offset(offset)


if __name__ == "__main__":
    raise SystemExit(main())
