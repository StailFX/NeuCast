"""Tests for ``tools.poll_funding_rate`` — the REST-based funding-rate
poller (release S phase 3).  We don't hit live Binance in unit tests;
``parse_premium_index`` is a pure transform and ``fetch_premium_index``
is exercised by an integration smoke test that's marked slow."""
from __future__ import annotations

import json

import pytest

from tools.poll_funding_rate import parse_premium_index


# ─── parse_premium_index ───


def _well_formed_payload(**overrides) -> dict:
    """Sample REST payload from /fapi/v1/premiumIndex — values from a
    real Binance docs example, slightly anonymised."""
    base = {
        "symbol": "BTCUSDT",
        "markPrice": "65430.10",
        "indexPrice": "65420.50",
        "estimatedSettlePrice": "65425.00",
        "lastFundingRate": "0.00012345",
        "interestRate": "0.0001",
        "nextFundingTime": 1711238400000,
        "time": 1711234567890,
    }
    base.update(overrides)
    return base


def test_parse_premium_index_well_formed():
    out = parse_premium_index(_well_formed_payload())
    assert out is not None
    symbol, mark, funding, next_ms = out
    assert symbol == "BTCUSDT"
    assert mark == pytest.approx(65430.10)
    assert funding == pytest.approx(0.00012345)
    assert next_ms == 1711238400000


def test_parse_premium_index_uppercases_symbol():
    """REST always returns uppercase but defensively ensure we do too —
    keeps the SQL primary key (symbol, ts) consistent."""
    out = parse_premium_index(_well_formed_payload(symbol="btcusdt"))
    assert out is not None
    assert out[0] == "BTCUSDT"


def test_parse_premium_index_negative_funding_preserved():
    """Negative funding (shorts paying longs) is signed-meaningful for
    feature input + paper-trader cost calc.  Pin: no abs()."""
    out = parse_premium_index(
        _well_formed_payload(lastFundingRate="-0.0005"),
    )
    assert out is not None
    assert out[2] == pytest.approx(-0.0005)


def test_parse_premium_index_zero_funding():
    """0.0 funding rate is a legitimate value (occurs near settlement
    boundaries) — not a sentinel for missing data.  Pin: returns 0.0,
    NOT None."""
    out = parse_premium_index(
        _well_formed_payload(lastFundingRate="0"),
    )
    assert out is not None
    assert out[2] == 0.0


def test_parse_premium_index_returns_none_on_missing_fields():
    """Each required field individually missing → None.  Pin all four
    so a refactor doesn't silently let a None field through and crash
    the SQL UPDATE."""
    for missing in ("symbol", "markPrice", "lastFundingRate", "nextFundingTime"):
        payload = _well_formed_payload()
        payload.pop(missing)
        out = parse_premium_index(payload)
        assert out is None, (
            f"expected None when {missing!r} is missing, got {out!r}"
        )


def test_parse_premium_index_returns_none_on_bad_numerics():
    """A non-numeric markPrice / funding rate must produce None rather
    than NaN/inf leaking into the UPDATE.  Pin so an upstream API quirk
    doesn't corrupt the table."""
    out = parse_premium_index(
        _well_formed_payload(markPrice="N/A"),
    )
    assert out is None
    out = parse_premium_index(
        _well_formed_payload(lastFundingRate="not-a-number"),
    )
    assert out is None


def test_parse_premium_index_returns_none_on_empty_dict():
    assert parse_premium_index({}) is None


def test_parse_premium_index_handles_string_int_for_next_funding():
    """Some API versions return ``nextFundingTime`` as a string. Coerce
    via ``int()`` already does the right thing — pin this contract."""
    out = parse_premium_index(
        _well_formed_payload(nextFundingTime="1711238400000"),
    )
    assert out is not None
    assert out[3] == 1711238400000


# ─── Integration smoke (deferred — requires network access) ───


@pytest.mark.skip(reason="hits live Binance; run manually")
def test_fetch_premium_index_smoke():
    """Hit the real REST endpoint and ensure the return shape parses.
    Skipped by default — runs in CI/dev only when network is available."""
    from tools.poll_funding_rate import fetch_premium_index
    raw = fetch_premium_index("BTCUSDT", timeout_seconds=10)
    assert raw is not None
    parsed = parse_premium_index(raw)
    assert parsed is not None
    symbol, mark, funding, next_ms = parsed
    assert symbol == "BTCUSDT"
    # Sanity: BTC mark is in a reasonable range (anywhere 5k - 200k).
    assert 5000 < mark < 200000
    # Funding is a fraction (-1 to 1) — usually within ±0.01.
    assert -1.0 <= funding <= 1.0
    # next_funding_ms is in milliseconds, in the future-ish range.
    import time
    now_ms = int(time.time() * 1000)
    assert now_ms - 8 * 3600 * 1000 <= next_ms <= now_ms + 8 * 3600 * 1000
