"""Tests for ``tools.telegram_notify``.

A small synchronous notifier reused for ad-hoc operational alerts
(pipeline complete, deploy decision, manual emergency). Pin contract:

1. Sending requires both ``HF_TELEGRAM_SIGNAL_BOT_TOKEN`` and
   ``HF_TELEGRAM_SIGNAL_CHAT_ID`` in env (or via flags).
2. Gating by ``HF_TELEGRAM_SIGNAL_ENABLED`` — when not '1'/'true'/etc,
   the script logs a warning but exits 0 (graceful no-op).
3. ``--from-file`` is preferred for multi-line bodies.
4. ``--escape-html`` HTML-escapes the body.
5. The HTTP POST hits the right endpoint with the right JSON body.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.telegram_notify import main, send_telegram


# ──────────────────────── send_telegram pure function ────────────────────────


def test_send_telegram_success_returns_2xx_code():
    fake_resp = MagicMock()
    fake_resp.getcode.return_value = 200
    fake_resp.read.return_value = b'{"ok":true}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *a: None

    with patch("tools.telegram_notify.urlopen", return_value=fake_resp) as mock_open:
        code, body = send_telegram(
            bot_token="TOKEN", chat_id="123", body_html="hello",
        )
        assert code == 200
        assert body == '{"ok":true}'
        # Check the request payload structure.
        called_req = mock_open.call_args.args[0]
        assert "api.telegram.org/botTOKEN/sendMessage" in called_req.full_url
        sent = json.loads(called_req.data.decode("utf-8"))
        assert sent["chat_id"] == "123"
        assert sent["text"] == "hello"
        assert sent["parse_mode"] == "HTML"
        # Web preview disabled by default to avoid chat clutter.
        assert sent["disable_web_page_preview"] is True


def test_send_telegram_url_error_returns_zero_code():
    from urllib.error import URLError
    with patch("tools.telegram_notify.urlopen", side_effect=URLError("dns boom")):
        code, body = send_telegram(
            bot_token="x", chat_id="y", body_html="hi",
        )
        assert code == 0
        assert "URLError" in body


# ──────────────────────────── CLI dispatch ───────────────────────────────────


def test_cli_disabled_env_skips_send_returns_0(monkeypatch, capsys):
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "T")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "C")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "0")
    with patch("tools.telegram_notify.send_telegram") as mock_send:
        rc = main(["--text", "hello"])
    assert rc == 0
    assert not mock_send.called  # gracefully skipped


def test_cli_missing_credentials_returns_2(monkeypatch):
    monkeypatch.delenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HF_TELEGRAM_SIGNAL_CHAT_ID", raising=False)
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    rc = main(["--text", "hello"])
    assert rc == 2


def test_cli_missing_text_and_from_file_returns_2(monkeypatch):
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "T")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "C")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    rc = main([])
    assert rc == 2


def test_cli_from_file_reads_body(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "T")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "C")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    body_path = tmp_path / "msg.html"
    body_path.write_text("<b>multi</b>\nline\nbody", encoding="utf-8")
    with patch("tools.telegram_notify.send_telegram", return_value=(200, "ok")) as mock_send:
        rc = main(["--from-file", str(body_path)])
    assert rc == 0
    sent_kwargs = mock_send.call_args.kwargs
    assert sent_kwargs["body_html"] == "<b>multi</b>\nline\nbody"


def test_cli_escape_html_escapes_special_chars(monkeypatch):
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "T")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "C")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    with patch("tools.telegram_notify.send_telegram", return_value=(200, "ok")) as mock_send:
        rc = main(["--text", "<script>x</script>", "--escape-html"])
    assert rc == 0
    assert (
        mock_send.call_args.kwargs["body_html"]
        == "&lt;script&gt;x&lt;/script&gt;"
    )


def test_cli_send_failure_returns_1(monkeypatch):
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "T")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "C")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    with patch("tools.telegram_notify.send_telegram", return_value=(429, "rate limit")):
        rc = main(["--text", "hello"])
    assert rc == 1


def test_cli_silent_flag_sets_disable_notification(monkeypatch):
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "T")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "C")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    with patch("tools.telegram_notify.send_telegram", return_value=(200, "ok")) as mock_send:
        main(["--text", "x", "--silent"])
    assert mock_send.call_args.kwargs["disable_notification"] is True
