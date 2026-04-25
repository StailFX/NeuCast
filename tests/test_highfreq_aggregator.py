"""Tests for ``app.highfreq.aggregator``.

Что покрыто:

* ``_safe`` — скрабинг NaN/inf/None для Postgres ``DOUBLE PRECISION``.
  Postgres явно отвергает NaN в числовых колонках, и одна-единственная
  такая строка валит весь батч insert. Тесты заякоряют контракт.
* ``_SecondBucket.add_trade`` — sign convention. Знак трейда в OFI/trade
  imbalance — самая частая ошибка в order-flow коде ("я перепутал
  buyer_is_maker"); один inverted-test и пол-фичи едет в обратную сторону.
* ``Aggregator._finalize`` — bucket → AggregatedRow. Здесь жёсткий
  контракт: OFI **сумма** (поток накапливается за секунду), microprice/
  depth_imb/spread — **среднее** (level-quantities, представительная точка).
* ``on_snapshot`` / ``_emit_completed`` — секундный rollover (бакет
  закрывается, когда приходит кадр следующей секунды), без базы
  (``_pool=None`` — код это явно поддерживает как no-op).

Что НЕ покрыто (умышленно):

* Реальный Postgres write path — это integration-тест, проверяется
  через health-чек ``/api/highfreq/status`` (rows arriving) + trainer
  dry-run на проде.
* asyncpg pool error recovery — внешняя либа.

Стиль: синхронные ``def test_*``, async-пути обёрнуты в ``asyncio.run()``
(в репо нет pytest-asyncio, не хочу тащить новую dep ради 5 тестов).
"""
from __future__ import annotations

import asyncio

import pytest

from app.highfreq.aggregator import (
    AggregatedRow,
    Aggregator,
    _SecondBucket,
    _safe,
)
from app.highfreq.l2_consumer import L2Snapshot, Trade
from app.highfreq.ofi_features import FrameFeatures


# ───────────────────────── _safe ─────────────────────────────


def test_safe_passes_finite_floats():
    assert _safe(0.0) == 0.0
    assert _safe(-1.5) == -1.5
    assert _safe(1e9) == 1e9


def test_safe_replaces_nan_with_zero():
    # NaN может прилететь из spread_bps когда mid=0 (degenerate book) —
    # и Postgres отвергнет всю транзакцию вставки. Лучше «секунда без
    # spread-сигнала» (0.0), чем дроп пяти строк батча.
    assert _safe(float("nan")) == 0.0


def test_safe_replaces_inf_with_zero():
    assert _safe(float("inf")) == 0.0
    assert _safe(float("-inf")) == 0.0


def test_safe_replaces_none_with_zero():
    # Defensive: пайплайн не должен передавать None — но если когда-нибудь
    # передаст, Postgres крашнется на NOT NULL колонке. Скрабим в 0.
    assert _safe(None) == 0.0


# ─────────────────── _SecondBucket: sign convention ──────────────


def test_bucket_buy_lift_increases_trade_imb():
    # is_buyer_maker=False → buyer is taker → агрессивный buy лифтнул ask.
    # Контракт: знак +qty (положительный поток buy-side давления).
    b = _SecondBucket(symbol="BTCUSDT", second_ms=0)
    b.add_trade(Trade(0, 0, "BTCUSDT", price=77000.0, qty=0.5, is_buyer_maker=False))
    b.add_trade(Trade(0, 0, "BTCUSDT", price=77001.0, qty=0.3, is_buyer_maker=False))
    assert b.trade_imb == pytest.approx(0.8)


def test_bucket_sell_hit_decreases_trade_imb():
    # is_buyer_maker=True → buyer пассивный (стоял на bid) → агрессивный
    # sell его взял. Знак -qty.
    b = _SecondBucket(symbol="BTCUSDT", second_ms=0)
    b.add_trade(Trade(0, 0, "BTCUSDT", price=77000.0, qty=0.5, is_buyer_maker=True))
    b.add_trade(Trade(0, 0, "BTCUSDT", price=76999.0, qty=0.2, is_buyer_maker=True))
    assert b.trade_imb == pytest.approx(-0.7)


def test_bucket_mixed_trades_net_correctly():
    # Регрессия на «забыл сбросить аккумулятор»: чистый поток за секунду
    # = сумма знаковых qty.
    b = _SecondBucket(symbol="BTCUSDT", second_ms=0)
    b.add_trade(Trade(0, 0, "BTCUSDT", price=1, qty=1.0, is_buyer_maker=False))  # +1.0
    b.add_trade(Trade(0, 0, "BTCUSDT", price=1, qty=0.4, is_buyer_maker=True))   # -0.4
    b.add_trade(Trade(0, 0, "BTCUSDT", price=1, qty=0.1, is_buyer_maker=False))  # +0.1
    assert b.trade_imb == pytest.approx(0.7)


def test_bucket_add_frame_collects_features_per_call():
    b = _SecondBucket(symbol="BTCUSDT", second_ms=0)
    feat = FrameFeatures(
        event_time_ms=0,
        symbol="BTCUSDT",
        ofi=1.0,
        microprice=77000.5,
        depth_imb=0.1,
        spread_bps=0.5,
        mid=77000.5,
    )
    b.add_frame(feat)
    b.add_frame(feat)
    b.add_frame(feat)
    assert b.n_updates == 3
    assert len(b.ofi_values) == 3
    assert len(b.microprice_values) == 3
    assert len(b.depth_imb_values) == 3
    assert len(b.spread_bps_values) == 3


# ───────────────────── Aggregator._finalize ──────────────────────


def _agg() -> Aggregator:
    """Aggregator без open-pool. start() не вызывается, _pool остаётся None.
    Все DB-paths (`_flush_locked`) корректно деградируют в no-op — это
    задокументированный контракт (см. start of `_flush_locked`)."""
    return Aggregator(database_url="postgresql://unused", symbols=["BTCUSDT"])


def test_finalize_sums_ofi_averages_levels():
    # CONTRACT: OFI — это flow (per-frame delta), за секунду берём СУММУ.
    # microprice/depth_imb/spread_bps — level-quantities, берём СРЕДНЕЕ.
    # Эта асимметрия фундаментальна: OFI «накапливается», цена «есть».
    b = _SecondBucket(symbol="BTCUSDT", second_ms=1000)
    b.ofi_values = [1.0, -0.5, 2.0]               # сумма 2.5
    b.microprice_values = [77000.0, 77001.0, 77002.0]  # среднее 77001.0
    b.depth_imb_values = [0.1, -0.1, 0.0]         # среднее 0.0
    b.spread_bps_values = [0.5, 0.7, 0.6]         # среднее 0.6
    b.trade_imb = 0.42
    b.n_updates = 3

    row = Aggregator._finalize(b, jitter=(100, 350))

    assert row.symbol == "BTCUSDT"
    assert row.second_ms == 1000
    assert row.ofi == pytest.approx(2.5)
    assert row.microprice == pytest.approx(77001.0)
    assert row.depth_imb == pytest.approx(0.0)
    assert row.spread_bps == pytest.approx(0.6)
    assert row.trade_imb == pytest.approx(0.42)
    assert row.n_updates == 3
    assert row.local_recv_ms_jitter == 250  # max - min
    # VPIN — Phase B, в A.* эмитим 0.0 как placeholder.
    assert row.vpin == 0.0


def test_finalize_handles_trade_only_bucket():
    # Бакет, в который попал трейд, но НЕ попал ни один snapshot
    # (snapshot до/после, трейд в gap). Не должен крашить — должен
    # эмитить нули по level-фичам и реальный trade_imb.
    b = _SecondBucket(symbol="BTCUSDT", second_ms=1000)
    b.trade_imb = 0.1

    row = Aggregator._finalize(b, jitter=(0, 0))

    assert row.ofi == 0.0
    assert row.microprice == 0.0
    assert row.depth_imb == 0.0
    assert row.spread_bps == 0.0
    assert row.trade_imb == pytest.approx(0.1)
    assert row.n_updates == 0
    assert row.local_recv_ms_jitter == 0


def test_finalize_scrubs_nan_and_inf_before_write():
    # NaN/inf могут прийти от degenerate book (mid=0 → 0/0 в spread_bps).
    # _finalize прогоняет через _safe → 0.0 → Postgres примет.
    b = _SecondBucket(symbol="BTCUSDT", second_ms=0)
    b.ofi_values = [float("nan")]
    b.microprice_values = [float("inf")]
    b.depth_imb_values = [float("-inf")]
    b.spread_bps_values = [1.0]

    row = Aggregator._finalize(b, jitter=(0, 0))

    assert row.ofi == 0.0
    assert row.microprice == 0.0
    assert row.depth_imb == 0.0
    assert row.spread_bps == pytest.approx(1.0)


# ───────── on_snapshot / _emit_completed (async, no DB) ──────────


def _snap(symbol: str, event_time_ms: int, mid: float = 77000.0) -> L2Snapshot:
    """Минимальный валидный snapshot: один уровень bid/ask вокруг mid."""
    return L2Snapshot(
        event_time_ms=event_time_ms,
        local_recv_ms=event_time_ms,
        symbol=symbol,
        bids=((mid - 0.5, 1.0),),
        asks=((mid + 0.5, 1.0),),
    )


def test_on_snapshot_emits_previous_second_bucket_on_rollover():
    """Кадр секунды N+1 закрывает бакет секунды N. Это сердце пайплайна."""
    async def run():
        agg = _agg()
        # Кадр в 1000 ms — бакет (BTCUSDT, 1000), ничего не эмитится.
        await agg.on_snapshot(_snap("BTCUSDT", event_time_ms=1000))
        assert agg.rows_emitted == 0
        # Кадр в 2050 ms — бакет (BTCUSDT, 2000), бакет 1000 теперь
        # «строго старше» → эмит.
        await agg.on_snapshot(_snap("BTCUSDT", event_time_ms=2050))
        assert agg.rows_emitted == 1
        # Один pending row (бакет 1000), бакет 2000 в полёте.
        assert len(agg._pending_writes) == 1
        assert agg._pending_writes[0].second_ms == 1000
        assert ("BTCUSDT", 2000) in agg._buckets
        assert ("BTCUSDT", 1000) not in agg._buckets

    asyncio.run(run())


def test_on_trade_creates_bucket_when_no_snapshot_yet():
    """Трейд может прилететь раньше snapshot для своей секунды.
    Бакет должен создаться, чтобы trade_imb не потерялся."""
    async def run():
        agg = _agg()
        await agg.on_trade(Trade(
            event_time_ms=5500,
            local_recv_ms=5500,
            symbol="BTCUSDT",
            price=77000.0,
            qty=0.3,
            is_buyer_maker=False,  # +0.3
        ))
        key = ("BTCUSDT", 5000)  # floor до секунды
        assert key in agg._buckets
        assert agg._buckets[key].trade_imb == pytest.approx(0.3)
        # Без snapshot-кадров эмиссия не триггерится — бакет ждёт.
        assert agg.rows_emitted == 0

    asyncio.run(run())


def test_emit_does_not_cross_symbols():
    """Контракт: BTC-кадр НЕ эмитит ETH-бакет другого времени.
    Это критично для multi-symbol будущего расширения."""
    async def run():
        agg = Aggregator(database_url="postgresql://unused", symbols=["BTCUSDT", "ETHUSDT"])
        # ETH в секунду 1, BTC в секунду 2 — ETH-бакет НЕ эмитится BTC-кадром.
        await agg.on_snapshot(_snap("ETHUSDT", event_time_ms=1000, mid=3500.0))
        await agg.on_snapshot(_snap("BTCUSDT", event_time_ms=2050, mid=77000.0))

        assert agg.rows_emitted == 0
        assert ("ETHUSDT", 1000) in agg._buckets
        assert ("BTCUSDT", 2000) in agg._buckets

    asyncio.run(run())


def test_flush_batch_size_triggers_no_op_when_pool_is_none():
    """Когда `_pool is None` (тест-режим, ранний shutdown), `_flush_locked`
    — корректный no-op. Без этого guard-а always-on сервис крашится в
    момент закрытия пула при graceful shutdown."""
    async def run():
        agg = Aggregator(
            database_url="postgresql://unused",
            symbols=["BTCUSDT"],
            flush_batch_size=2,  # триггерим flush на 2-м pending-роу
        )
        # 3 кадра → 2 завершённых бакета (1000, 2000) → flush triggered.
        await agg.on_snapshot(_snap("BTCUSDT", event_time_ms=1000))
        await agg.on_snapshot(_snap("BTCUSDT", event_time_ms=2000))
        await agg.on_snapshot(_snap("BTCUSDT", event_time_ms=3000))

        assert agg.rows_emitted == 2
        # `_flush_locked` сработал, но pool=None → no-op → rows_written=0.
        assert agg.rows_written == 0
        # И rows СТАТУС-КВО в pending (не теряем, не падаем).
        assert len(agg._pending_writes) == 2

    asyncio.run(run())


def test_flush_when_no_pool_is_safe():
    """Прямой вызов `flush()` — для shutdown-handler-а. Безопасно при
    любом состоянии (нет pool, нет pending)."""
    async def run():
        agg = _agg()
        # Никаких pending — должен молча завершиться.
        await agg.flush()
        # Один pending — pool всё ещё None — должен остаться в очереди.
        agg._pending_writes.append(AggregatedRow(
            second_ms=1000, symbol="BTCUSDT",
            ofi=0.0, microprice=0.0, depth_imb=0.0, spread_bps=0.0,
            trade_imb=0.0, vpin=0.0, n_updates=0, local_recv_ms_jitter=0,
        ))
        await agg.flush()
        assert len(agg._pending_writes) == 1

    asyncio.run(run())


def test_close_is_idempotent_with_no_pool():
    """`close()` зовётся из finally-блока shutdown-а; если уже закрыт
    или ни разу не открыт — не должен ничего сломать."""
    async def run():
        agg = _agg()
        await agg.close()  # pool никогда не открывался
        await agg.close()  # ещё раз — всё равно ок

    asyncio.run(run())
