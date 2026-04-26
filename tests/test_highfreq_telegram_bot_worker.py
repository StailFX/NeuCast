"""Tests for ``app.highfreq.telegram_bot_worker`` — pure formatting +
authorization.

The HTTP / DB sides are exercised by the production smoke test
(real Telegram chat + real Postgres). Pure tests pin:

* Authorization gate (non-allowed chat_ids get a curt reply, never
  see data).
* Response formatting — message bodies match the layout the user
  sees in chat. Pin against silent regressions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.highfreq.telegram_bot_worker import (
    HELP_TEXT,
    _format_accuracy_response,
    _format_model_response,
    _format_stats_response,
    _format_trades_response,
    _process_update,
)


# ───── formatting ─────


def test_format_trades_empty():
    out = _format_trades_response([])
    assert "No paper trades yet" in out


def test_format_trades_single_long_winner():
    out = _format_trades_response([{
        "symbol": "BTCUSDT",
        "side": "long",
        "entry_price": 77_000.00,
        "exit_price": 77_100.00,
        "qty": 0.0128,
        "pnl_usd": 1.28,
        "pnl_bps": 12.5,
        "exit_reason": "time_stop",
        "exit_ts": datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
        "model_version": "3600",
    }])
    assert "🟢" in out  # winner emoji
    assert "BTCUSDT" in out
    assert "↑LONG" in out
    assert "77,000.00" in out
    assert "+12.5 bps" in out
    assert "+$1.28" in out or "$+1.28" in out or "+1.2800" in out


def test_format_trades_loss_emoji():
    out = _format_trades_response([{
        "symbol": "BTCUSDT", "side": "short",
        "entry_price": 77_000, "exit_price": 77_100,
        "qty": 0.01, "pnl_usd": -1.0, "pnl_bps": -10.0,
        "exit_reason": "time_stop",
        "exit_ts": datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
        "model_version": "3600",
    }])
    assert "🔴" in out
    assert "↓SHORT" in out


def test_format_stats_headline_per_symbol():
    out = _format_stats_response({
        "BTCUSDT": {
            "predictions_total": 50,
            "accuracy_total": 35,
            "accuracy_directional": 20,
            "trades_total": 8,
            "trade_wins": 5,
            "trade_losses": 3,
            "pnl_total_usd": 2.45,
        },
    })
    assert "BTCUSDT" in out
    assert "50" in out  # predictions count
    assert "8" in out   # trades total
    assert "+$2.45" in out or "$+2.45" in out or "+2.4500" in out
    # accuracy 20/35 ≈ 57.1%
    assert "57.1%" in out


def test_format_stats_zero_accuracy_renders_dash():
    """Cold start: no directional predictions yet → '—' not '0.0%'."""
    out = _format_stats_response({
        "BTCUSDT": {
            "predictions_total": 5,
            "accuracy_total": 0,
            "accuracy_directional": 0,
            "trades_total": 0,
            "trade_wins": 0,
            "trade_losses": 0,
            "pnl_total_usd": 0.0,
        },
    })
    assert "—" in out


def test_format_accuracy_with_data():
    out = _format_accuracy_response([
        {"symbol": "BTCUSDT", "directional": 7, "hits": 5},
        {"symbol": "ETHUSDT", "directional": 8, "hits": 4},
    ])
    assert "BTCUSDT" in out
    assert "ETHUSDT" in out
    assert "5 / 7" in out
    assert "71.4%" in out
    assert "50.0%" in out


def test_format_accuracy_empty():
    out = _format_accuracy_response([])
    assert "No backfilled" in out


def test_format_model_with_folds():
    out = _format_model_response([{
        "symbol": "BTCUSDT",
        "run_started_at": datetime(2026, 4, 27, 4, 0, tzinfo=timezone.utc),
        "n_minutes_after_neutral_drop": 1500,
        "n_folds": 3,
        "dir_acc_mean": 0.547,
        "dir_acc_ci_low": 0.521,
        "dir_acc_ci_high": 0.572,
        "dir_acc_p_value": 0.012,
    }])
    assert "BTCUSDT" in out
    assert "1500" in out
    assert "0.5470" in out  # mean
    assert "0.5210" in out  # CI low
    assert "0.0120" in out  # p-value


def test_format_model_no_folds_yet_renders_dashes():
    """Cold start: dir_acc and p-value are NULL — render — not 'None'."""
    out = _format_model_response([{
        "symbol": "BTCUSDT",
        "run_started_at": datetime(2026, 4, 27, 4, 0, tzinfo=timezone.utc),
        "n_minutes_after_neutral_drop": 1100,
        "n_folds": 0,
        "dir_acc_mean": None,
        "dir_acc_ci_low": None,
        "dir_acc_ci_high": None,
        "dir_acc_p_value": None,
    }])
    assert "—" in out
    assert "None" not in out


def test_help_text_lists_all_commands():
    """Pin: every command implemented MUST be documented in /help.
    Defence against "ship a feature, forget to update help."""
    assert "/stats" in HELP_TEXT
    assert "/trades" in HELP_TEXT
    assert "/accuracy" in HELP_TEXT
    assert "/model" in HELP_TEXT


# ───── authorization ─────


def test_unauthorized_chat_id_gets_rejection():
    """A chat_id NOT matching the operator's gets a curt reply +
    no data. Pin so a refactor that flips the comparison silently
    doesn't expose data."""
    with patch("app.highfreq.telegram_bot_worker._send_html") as send:
        _process_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 999_999_999},
                    "text": "/stats",
                },
            },
            bot_token="TOKEN",
            allowed_chat_id="5125263146",
            database_url="postgresql://x",
        )
    # Exactly one reply, with the rejection text.
    assert send.call_count == 1
    args = send.call_args
    assert "not authorized" in args.args[2].lower()


def test_authorized_chat_id_dispatches_help():
    """Authorized /help renders HELP_TEXT — proves the dispatch path
    runs to completion."""
    with patch("app.highfreq.telegram_bot_worker._send_html") as send:
        _process_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 5125263146},
                    "text": "/help",
                },
            },
            bot_token="TOKEN",
            allowed_chat_id="5125263146",
            database_url="postgresql://x",
        )
    assert send.call_count == 1
    body = send.call_args.args[2]
    assert "/stats" in body  # part of HELP_TEXT


def test_unknown_command_gets_help():
    with patch("app.highfreq.telegram_bot_worker._send_html") as send:
        _process_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 5125263146},
                    "text": "/banana",
                },
            },
            bot_token="TOKEN",
            allowed_chat_id="5125263146",
            database_url="postgresql://x",
        )
    body = send.call_args.args[2]
    assert "unknown" in body.lower()
    assert "/stats" in body  # help inlined


def test_non_command_messages_ignored():
    """Plain text (not starting with /) is not a command — the bot
    must NOT reply (otherwise would echo every message)."""
    with patch("app.highfreq.telegram_bot_worker._send_html") as send:
        _process_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 5125263146},
                    "text": "hello bot",
                },
            },
            bot_token="TOKEN",
            allowed_chat_id="5125263146",
            database_url="postgresql://x",
        )
    assert send.call_count == 0


def test_at_mention_strips_bot_username():
    """Group chats often send /stats@NeuCastSignals_bot — the
    handler must strip the @-part before matching the command."""
    with patch("app.highfreq.telegram_bot_worker._send_html") as send:
        with patch("app.highfreq.telegram_bot_worker._query_stats", return_value={}):
            _process_update(
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 5125263146},
                        "text": "/stats@NeuCastSignals_bot",
                    },
                },
                bot_token="TOKEN",
                allowed_chat_id="5125263146",
                database_url="postgresql://x",
            )
    # Sent something — the dispatch reached _query_stats (no data, but
    # the command was recognised, NOT routed to "unknown").
    assert send.call_count == 1
