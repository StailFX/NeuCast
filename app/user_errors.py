"""
User-facing error sanitizer.

Сырые Python-tracebacks (UnboundLocalError, KeyError, ZeroDivisionError, ...)
не должны утекать в UI: для пользователя это шум и feeling "сломалось", тогда
как реальная причина — bug в нашем коде или временная инфраструктурная
проблема. Здесь центральный mapper, который превращает исключение/строку с
исключением в человеческое русскоязычное сообщение.

Принципы:
  • НИКАКИХ внутренностей Python (имена переменных, файлы, строки, типы) —
    они уходят в логи через `logger.exception(...)`, не пользователю.
  • Конкретные категории, которые знаем (network, data quality, capacity,
    timeout, rate limit) → специфичные подсказки + что пользователь может
    сделать.
  • Всё остальное → generic fallback "временная неполадка, попробуйте позже".

Использование:
    from app.user_errors import friendly_error
    try:
        ...
    except Exception as e:
        logger.exception("prediction failed")  # full traceback в логи
        return {"error": friendly_error(e)}    # friendly текст в API/UI
"""
from __future__ import annotations

# Generic fallback. Лучше быть аккуратными: не обещаем "повторите" если
# проблема воспроизводимая — пользователь может задолбаться. Но в большинстве
# случаев retry помогает (yfinance, OOM, transient network).
_GENERIC = (
    "Не удалось выполнить расчёт. Попробуйте через минуту или сообщите в "
    "поддержку, если ошибка повторяется."
)

# (substring, friendly message) — порядок важен (more specific сверху).
# Substring matching — простой и достаточный (исключения часто содержат
# характерные подстроки: "timeout", "memory", "no data" итп).
_PATTERNS: list[tuple[str, str]] = [
    # ── Yahoo Finance / data fetching ────────────────────────────────
    # ВАЖНО: ловим и русские, и английские варианты — fetch_and_preprocess
    # raise'ит русский ValueError ("Нет данных..."), yfinance — английский.
    (
        "нет данных по указанному тикеру",
        "Тикер не найден или нет данных за выбранный период. Проверьте символ"
        " (например, AAPL, BTC-USD) и попробуйте более широкий диапазон дат.",
    ),
    (
        "no data found for ticker",
        "Тикер не найден или нет данных за выбранный период. Проверьте символ"
        " (например, AAPL, BTC-USD) и попробуйте более широкий диапазон дат.",
    ),
    (
        "possibly delisted",
        "Тикер не найден или удалён. Проверьте символ и попробуйте другой.",
    ),
    (
        "недостаточно данных",
        "Слишком короткий период для обучения модели. Возьмите окно от 1 года.",
    ),
    (
        "too many requests",
        "Yahoo Finance временно ограничил запросы. Подождите 1–2 минуты и"
        " попробуйте снова.",
    ),
    (
        "rate limit",
        "Yahoo Finance временно ограничил запросы. Подождите 1–2 минуты и"
        " попробуйте снова.",
    ),
    ("yfinance", "Не удалось загрузить котировки. Попробуйте через минуту."),
    # ── Network ─────────────────────────────────────────────────────
    (
        "connection",
        "Проблема с сетью при загрузке данных. Попробуйте ещё раз.",
    ),
    ("timeout", "Расчёт занял слишком много времени. Попробуйте ещё раз."),
    # ── Resource limits ─────────────────────────────────────────────
    (
        "memory",
        "Не хватило памяти на расчёт. Попробуйте более короткий период или"
        " без Foundation-модели.",
    ),
    (
        "soft time limit",
        "Расчёт превысил лимит по времени. Попробуйте более короткий период.",
    ),
    # ── Auth / config ───────────────────────────────────────────────
    (
        "database",
        "Временная проблема с БД. Попробуйте через минуту.",
    ),
]


def friendly_error(exc: BaseException | str) -> str:
    """Превратить исключение или его строковое представление в user-friendly
    русскоязычное сообщение.

    Args:
        exc: Exception object или уже str(exception). Принимаем оба, чтобы
            работать и в celery worker (есть exc), и в main.py FAILURE
            branch (есть только str(result.info)).

    Returns:
        Короткая русская фраза для показа в UI. Никогда не возвращает
        пустую строку или None.
    """
    if isinstance(exc, BaseException):
        text = f"{type(exc).__name__}: {exc}"
    else:
        text = str(exc) if exc else ""
    if not text:
        return _GENERIC
    text_low = text.lower()
    for needle, friendly in _PATTERNS:
        if needle in text_low:
            return friendly
    return _GENERIC
