"""Pure-function tests for ``app.highfreq.l2_consumer``.

Что покрыто:
  * парсеры (``_parse_snapshot``, ``_parse_trade``) — границы входа JSON
    Binance → наши типизированные dataclass-ы;
  * ``_build_url`` — кодирует наш контракт подписки (стрим depth + trade);
  * ``__init__`` — валидация параметров (depth_levels, update_speed_ms,
    пустые символы) — раннее падение лучше тихого недо-стрима.

Что НЕ покрыто (и почему):
  * ``run_forever`` / реконнект-цикл — это сетевая интеграция; адекватно
    проверяется только через live VPS smoke (``reconnects=0`` в journalctl
    24-48 часов). Юнит-тесты с фейковым websocket-сервером дают мало
    уверенности и много чурбана.
  * ``websockets.connect`` — внешняя либа.

Зачем парсеры тестируются как unit
----------------------------------
1. **Тихая смена формы dict-а на стороне Binance** ("e" vs "E", float vs
   string price) — точно такого рода баг невидим до тех пор, пока строки
   просто перестанут долетать в ``highfreq_ofi_1s``.
2. **None-ветки на malformed-фрейме** — сервис ``always-on``, padает
   = простой; тесты заякоряют контракт «битый кадр → None, не
   exception».
3. **URL-builder** — неверное имя стрима даст пустой канал тихо.
"""
from __future__ import annotations

import pytest

from app.highfreq.l2_consumer import (
    BINANCE_WSS_BASE,
    L2Consumer,
)


# ───────────────────────── _parse_snapshot ───────────────────────────


def test_parse_snapshot_valid_partial_book():
    # Стандартный partial-book frame (нет "E", нет "s" — берём всё из
    # имени стрима + local_recv_ms).
    data = {
        "lastUpdateId": 12345,
        "bids": [["77000.10", "0.5"], ["76999.90", "1.2"]],
        "asks": [["77000.20", "0.6"], ["77000.40", "0.8"]],
    }
    snap = L2Consumer._parse_snapshot(data, "btcusdt@depth20@100ms", 1700000000000)
    assert snap is not None
    assert snap.symbol == "BTCUSDT"
    assert snap.local_recv_ms == 1700000000000
    # partial-book НЕ присылает "E" → fallback на local_recv_ms (а не падать)
    assert snap.event_time_ms == 1700000000000
    # строки → float, стабильные tuple-ы
    assert snap.bids == ((77000.10, 0.5), (76999.90, 1.2))
    assert snap.asks == ((77000.20, 0.6), (77000.40, 0.8))


def test_parse_snapshot_uses_event_time_when_present():
    # Diff-stream и иногда rich combined-stream envelope содержат "E"
    # (engine event time) и "s" (symbol) — должны иметь приоритет.
    data = {
        "E": 1700000123456,
        "s": "ETHUSDT",
        "bids": [["3500.0", "1.0"]],
        "asks": [["3501.0", "1.0"]],
    }
    snap = L2Consumer._parse_snapshot(data, "ethusdt@depth20@100ms", 1700000000000)
    assert snap is not None
    assert snap.event_time_ms == 1700000123456
    assert snap.local_recv_ms == 1700000000000


def test_parse_snapshot_falls_back_to_data_symbol_when_no_stream():
    # Defensive: если стрим-имя пустое (форма envelope-а Binance меняется),
    # пробуем достать "s" из payload-а.
    data = {
        "s": "BTCUSDT",
        "bids": [["77000.0", "1.0"]],
        "asks": [["77001.0", "1.0"]],
    }
    snap = L2Consumer._parse_snapshot(data, "", 1700000000000)
    assert snap is not None
    assert snap.symbol == "BTCUSDT"


def test_parse_snapshot_returns_none_on_empty_book():
    # Любая сторона пустая → не из чего считать microprice/spread → дроп.
    data_no_bids = {"bids": [], "asks": [["77001.0", "1.0"]]}
    assert L2Consumer._parse_snapshot(data_no_bids, "btcusdt@depth20@100ms", 0) is None
    data_no_asks = {"bids": [["77000.0", "1.0"]], "asks": []}
    assert L2Consumer._parse_snapshot(data_no_asks, "btcusdt@depth20@100ms", 0) is None


def test_parse_snapshot_returns_none_on_unparseable_numeric():
    # Иногда Binance в illiquid-парах эмитит мусор. Лучше дроп кадра, чем
    # упасть и потерять весь стрим (always-on сервис, кадр в секунду).
    data = {
        "bids": [["not-a-number", "1.0"]],
        "asks": [["77001.0", "1.0"]],
    }
    assert L2Consumer._parse_snapshot(data, "btcusdt@depth20@100ms", 0) is None


def test_parse_snapshot_returns_none_when_no_symbol_anywhere():
    # Нет "@" в стриме И нет "s" в payload — некуда атрибутировать строку.
    data = {"bids": [["77000.0", "1.0"]], "asks": [["77001.0", "1.0"]]}
    assert L2Consumer._parse_snapshot(data, "", 0) is None


# ───────────────────────── _parse_trade ──────────────────────────────


def test_parse_trade_valid_aggressive_buy():
    # is_buyer_maker=False → buyer is taker → агрессивный buy лифтнул ask.
    # Знак трейда +qty (см. aggregator._SecondBucket.add_trade).
    data = {
        "e": "trade",
        "E": 1700000000123,
        "s": "btcusdt",  # на проводе lower-case → мы upper-case
        "p": "77000.50",
        "q": "0.123",
        "m": False,
    }
    t = L2Consumer._parse_trade(data, 1700000000200)
    assert t is not None
    assert t.symbol == "BTCUSDT"
    assert t.event_time_ms == 1700000000123
    assert t.local_recv_ms == 1700000000200
    assert t.price == pytest.approx(77000.50)
    assert t.qty == pytest.approx(0.123)
    assert t.is_buyer_maker is False


def test_parse_trade_valid_aggressive_sell():
    # is_buyer_maker=True → buyer пассивный (стоял на bid) → агрессивный
    # sell его взял. Знак трейда -qty.
    data = {
        "E": 1700000000200,
        "s": "ETHUSDT",
        "p": "3500.0",
        "q": "0.5",
        "m": True,
    }
    t = L2Consumer._parse_trade(data, 0)
    assert t is not None
    assert t.is_buyer_maker is True


def test_parse_trade_returns_none_on_missing_fields():
    # Нет цены — невозможно использовать.
    no_price = {"E": 0, "s": "BTCUSDT", "q": "0.1", "m": False}
    assert L2Consumer._parse_trade(no_price, 0) is None
    # Нет event time — теряем привязку ко времени.
    no_ts = {"s": "BTCUSDT", "p": "1.0", "q": "1.0", "m": False}
    assert L2Consumer._parse_trade(no_ts, 0) is None


def test_parse_trade_returns_none_on_bad_numeric():
    bad = {"E": 0, "s": "BTCUSDT", "p": "not-a-number", "q": "1.0", "m": False}
    assert L2Consumer._parse_trade(bad, 0) is None


def test_parse_trade_defaults_buyer_maker_to_false():
    # "m" критичен для знака — но если Binance его не пришлёт, мы НЕ
    # должны падать. Defaul False = аггрессивный buy. Если когда-нибудь
    # увидим такой кадр — это будет повод для алерта, но не для краша.
    data = {"E": 0, "s": "BTCUSDT", "p": "1.0", "q": "1.0"}
    t = L2Consumer._parse_trade(data, 0)
    assert t is not None
    assert t.is_buyer_maker is False


# ───────────────────────── _build_url ────────────────────────────────


def test_build_url_single_symbol():
    c = L2Consumer(["BTCUSDT"], depth_levels=20, update_speed_ms=100)
    url = c._build_url()
    assert url.startswith(BINANCE_WSS_BASE)
    # Контракт: depth20@100ms + trade.
    assert "btcusdt@depth20@100ms" in url
    assert "btcusdt@trade" in url


def test_build_url_multiple_symbols_and_lowercases():
    c = L2Consumer(["BTCUSDT", "ETHUSDT"], depth_levels=10, update_speed_ms=1000)
    url = c._build_url()
    assert "btcusdt@depth10@1000ms" in url
    assert "ethusdt@depth10@1000ms" in url
    assert "btcusdt@trade" in url
    assert "ethusdt@trade" in url
    # Стримы джойнятся через "/", 2 символа × 2 типа = 4.
    streams = url.split("?streams=")[1].split("/")
    assert len(streams) == 4


# ───────────────────────── __init__ validation ───────────────────────


def test_consumer_rejects_empty_symbols():
    with pytest.raises(ValueError, match="at least one symbol"):
        L2Consumer([])


def test_consumer_rejects_invalid_depth():
    # Binance partial-book принимает 5 / 10 / 20.
    with pytest.raises(ValueError, match="depth_levels"):
        L2Consumer(["BTCUSDT"], depth_levels=15)


def test_consumer_rejects_invalid_speed():
    # Binance partial-book принимает 100 / 1000 ms.
    with pytest.raises(ValueError, match="update_speed_ms"):
        L2Consumer(["BTCUSDT"], update_speed_ms=500)


def test_consumer_uppercases_symbols():
    c = L2Consumer(["btcusdt", "EthUsdt"])
    assert c.symbols == ["BTCUSDT", "ETHUSDT"]


def test_consumer_init_counters_are_zero():
    # Мониторинг (web-роут /api/highfreq/status) читает эти счётчики —
    # они должны стартовать с честных нулей, а не None.
    c = L2Consumer(["BTCUSDT"])
    assert c.frames_received == 0
    assert c.snapshots_dispatched == 0
    assert c.trades_dispatched == 0
    assert c.reconnect_count == 0
    assert c.last_event_time_ms == {}


def test_stop_signal_sets_event():
    # stop() — единственный публичный способ корректно завершить
    # run_forever() без отмены задачи. Семантика: стоп-флаг ставится
    # синхронно, реальный exit происходит в цикле.
    c = L2Consumer(["BTCUSDT"])
    assert not c._stop.is_set()
    c.stop()
    assert c._stop.is_set()
