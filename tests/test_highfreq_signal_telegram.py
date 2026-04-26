"""Tests for ``app.highfreq.signal_telegram`` — flip detection +
message formatting + config.

The HTTP-send path is exercised by the production smoke test (one
real signal flip → one Telegram message); we don't mock urllib here
because the body+code logic is trivial and a mocked test would just
re-encode the same JSON.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.highfreq.signal_telegram import (
    SignalAlertConfig,
    SignalFlipDetector,
    format_flip_message,
)


# ───── flip detector ─────


def test_first_observation_does_not_flip():
    """Cold start (after process restart) — must NOT notify on the
    first signal seen, otherwise systemd-driven restarts would spam
    every time the runner came back."""
    d = SignalFlipDetector()
    assert d.check_flip("BTCUSDT", "up") is None


def test_same_signal_does_not_flip():
    """Steady state — repeated identical signals are no-ops."""
    d = SignalFlipDetector()
    d.check_flip("BTCUSDT", "up")  # establish baseline
    assert d.check_flip("BTCUSDT", "up") is None
    assert d.check_flip("BTCUSDT", "up") is None


def test_change_returns_previous_signal():
    """The flip return value tells the caller what the OLD signal was —
    so the message can read 'flipped from up → down'. Pin the
    direction so a future refactor that returns the new instead of
    the old breaks this test."""
    d = SignalFlipDetector()
    d.check_flip("BTCUSDT", "up")
    assert d.check_flip("BTCUSDT", "down") == "up"


def test_flips_per_symbol_independent():
    """Per-symbol state isolation — a flip on BTC must not be
    counted as a flip on ETH."""
    d = SignalFlipDetector()
    d.check_flip("BTCUSDT", "up")
    d.check_flip("ETHUSDT", "down")
    # ETHUSDT first observation, no flip
    assert d.check_flip("ETHUSDT", "down") is None
    # BTCUSDT goes up → neutral, that's a flip
    assert d.check_flip("BTCUSDT", "neutral") == "up"
    # ETHUSDT still down, no flip
    assert d.check_flip("ETHUSDT", "down") is None


def test_flip_chain_up_down_up():
    """Sequence: up → down → up. Both transitions are flips. Pin
    so a "tracker only fires once per direction" bug fails CI."""
    d = SignalFlipDetector()
    d.check_flip("BTCUSDT", "up")  # baseline
    assert d.check_flip("BTCUSDT", "down") == "up"
    assert d.check_flip("BTCUSDT", "up") == "down"


def test_neutral_in_the_middle_counts_as_two_flips():
    """up → neutral → up triggers two notifications. This is a
    documented rough edge — hysteresis would prevent the noise but
    is deferred to a future release."""
    d = SignalFlipDetector()
    d.check_flip("BTCUSDT", "up")
    assert d.check_flip("BTCUSDT", "neutral") == "up"
    assert d.check_flip("BTCUSDT", "up") == "neutral"


def test_reset_clears_state():
    d = SignalFlipDetector()
    d.check_flip("BTCUSDT", "up")
    d.reset()
    # After reset, BTC is back to first-observation behavior.
    assert d.check_flip("BTCUSDT", "down") is None


# ───── config ─────


def test_config_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HF_TELEGRAM_SIGNAL_CHAT_ID", raising=False)
    monkeypatch.delenv("HF_TELEGRAM_SIGNAL_ENABLED", raising=False)
    cfg = SignalAlertConfig.from_env()
    assert cfg.enabled is False


def test_config_disabled_when_only_flag_set(monkeypatch):
    """Setting the flag without creds is operator error — must
    disable, NOT silently 401 from Telegram every minute."""
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    monkeypatch.delenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HF_TELEGRAM_SIGNAL_CHAT_ID", raising=False)
    cfg = SignalAlertConfig.from_env()
    assert cfg.enabled is False


def test_config_disabled_when_flag_off_even_with_creds(monkeypatch):
    """Belt-and-suspenders: explicitly disable the feature even if
    creds remain in the env. Operator can disable without removing
    secrets."""
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "x")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "y")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "0")
    cfg = SignalAlertConfig.from_env()
    assert cfg.enabled is False


def test_config_enabled_with_full_setup(monkeypatch):
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_BOT_TOKEN", "TOKEN")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_CHAT_ID", "12345")
    monkeypatch.setenv("HF_TELEGRAM_SIGNAL_ENABLED", "1")
    cfg = SignalAlertConfig.from_env()
    assert cfg.enabled is True
    assert cfg.bot_token == "TOKEN"
    assert cfg.chat_id == "12345"


# ───── message formatting ─────


def test_format_flip_message_includes_symbol_and_signals():
    """Pin the layout: symbol header, old → new, prob_up, price, ts."""
    msg = format_flip_message(
        symbol="BTCUSDT",
        old_signal="up", new_signal="down",
        prob_up=0.4227, microprice=77123.45,
        ts=datetime(2026, 4, 27, 12, 34, tzinfo=timezone.utc),
    )
    assert "BTCUSDT" in msg
    assert "up" in msg and "down" in msg
    assert "0.4227" in msg
    assert "77,123.45" in msg
    assert "2026-04-27 12:34 UTC" in msg


def test_format_flip_message_uses_html_safe_tags():
    """HTML parse_mode subset only allows <b>, <i>, <s>, <code>,
    <pre> etc. Pin so a refactor doesn't introduce <span> or other
    rejected tags that'd cause Telegram to 400."""
    msg = format_flip_message(
        symbol="BTCUSDT",
        old_signal="up", new_signal="down",
        prob_up=0.4227, microprice=77123.45,
        ts=datetime(2026, 4, 27, 12, 34, tzinfo=timezone.utc),
    )
    # All used tags must be in the Telegram HTML allowlist.
    allowed = {"b", "i", "s", "code", "u", "pre", "a"}
    import re
    found_tags = set(re.findall(r"</?(\w+)", msg))
    assert found_tags <= allowed, f"unsupported HTML tags: {found_tags - allowed}"


def test_format_flip_message_demo_tag_appended():
    """When model_version is 'pre-calibration-demo', the message
    must include a warning so notification recipient knows the
    signal isn't backed by calibration."""
    msg = format_flip_message(
        symbol="BTCUSDT",
        old_signal="up", new_signal="down",
        prob_up=0.42, microprice=77000.0,
        ts=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
        model_version="pre-calibration-demo",
    )
    assert "pre-calibration" in msg
    assert "visualisation" in msg.lower()


def test_format_flip_message_no_demo_tag_for_real_model():
    msg = format_flip_message(
        symbol="BTCUSDT",
        old_signal="up", new_signal="down",
        prob_up=0.42, microprice=77000.0,
        ts=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
        model_version="3600",  # numeric (mtime) — real model
    )
    assert "pre-calibration" not in msg
