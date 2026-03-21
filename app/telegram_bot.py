"""
Telegram Bot integration for NeuCast.

Features:
- Users link their Telegram account via /start <token> in bot
- When prediction completes (Celery), bot sends PDF report + link
- Uses Telegram Bot API directly (no extra library needed)
"""

import os
import io
import json
import logging
import hashlib
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://neucast.stailfx.ru")

# In-memory store: user_id (int) -> telegram_chat_id (int)
# For persistence, we also use a simple JSON file
_LINKS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "telegram_links.json")
_links: dict[int, int] = {}       # user_id -> chat_id
_pending: dict[str, int] = {}     # link_token -> user_id


_links_mtime = 0.0


def _load_links():
    global _links, _links_mtime
    try:
        if os.path.exists(_LINKS_FILE):
            mtime = os.path.getmtime(_LINKS_FILE)
            if mtime != _links_mtime:
                with open(_LINKS_FILE) as f:
                    raw = json.load(f)
                    _links = {int(k): int(v) for k, v in raw.items()}
                _links_mtime = mtime
    except Exception as e:
        logger.warning(f"Failed to load telegram links: {e}")


def _save_links():
    global _links_mtime
    try:
        with open(_LINKS_FILE, "w") as f:
            json.dump({str(k): v for k, v in _links.items()}, f)
        _links_mtime = os.path.getmtime(_LINKS_FILE)
    except Exception as e:
        logger.warning(f"Failed to save telegram links: {e}")


_load_links()


def is_configured() -> bool:
    return bool(BOT_TOKEN)


def get_chat_id(user_id: int) -> int | None:
    _load_links()  # re-read if file changed (cross-process sync)
    return _links.get(user_id)


def is_linked(user_id: int) -> bool:
    _load_links()
    return user_id in _links


def generate_link_token(user_id: int) -> str:
    """Generate a one-time token for linking Telegram account."""
    token = hashlib.sha256(f"{user_id}:{os.urandom(16).hex()}".encode()).hexdigest()[:16]
    _pending[token] = user_id
    return token


def complete_link(token: str, chat_id: int) -> bool:
    """Complete the linking process (called when user sends /start <token> to bot)."""
    user_id = _pending.pop(token, None)
    if user_id is None:
        return False
    _links[user_id] = chat_id
    _save_links()
    return True


def unlink(user_id: int):
    _links.pop(user_id, None)
    _save_links()


def _api_call(method: str, data: dict = None, files: dict = None) -> dict | None:
    """Call Telegram Bot API."""
    if not BOT_TOKEN:
        return None

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"

    try:
        if files:
            # Multipart form upload
            import mimetypes
            boundary = "----NeuCastBoundary"
            body = b""
            for key, val in (data or {}).items():
                body += f"--{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
                body += f"{val}\r\n".encode()
            for key, (filename, filedata, mimetype) in files.items():
                body += f"--{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
                body += f"Content-Type: {mimetype}\r\n\r\n".encode()
                body += filedata + b"\r\n"
            body += f"--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
        else:
            payload = json.dumps(data or {}).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )

        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Telegram API error ({method}): {e}")
        return None


def send_message(chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
    """Send a text message to a Telegram chat."""
    result = _api_call("sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    })
    return result is not None and result.get("ok", False)


def send_document(chat_id: int, file_bytes: bytes, filename: str, caption: str = "") -> bool:
    """Send a document (PDF) to a Telegram chat."""
    result = _api_call("sendDocument", {
        "chat_id": chat_id,
        "caption": caption,
        "parse_mode": "HTML",
    }, files={
        "document": (filename, file_bytes, "application/pdf"),
    })
    return result is not None and result.get("ok", False)


def send_prediction_result(
    user_id: int,
    ticker: str,
    prediction_id: int | None,
    pdf_bytes: bytes | None = None,
    mae: str = "",
    mape: str = "",
    r2: str = "",
    dir_acc: str = "",
    model_name: str = "",
):
    """Send prediction notification with PDF to linked Telegram account."""
    chat_id = get_chat_id(user_id)
    if not chat_id:
        return False

    link = f"{APP_BASE_URL}/prediction/{prediction_id}" if prediction_id else APP_BASE_URL

    text = (
        f"<b>NeuCast — Прогноз готов</b>\n\n"
        f"<b>Тикер:</b> {ticker}\n"
        f"<b>Модель:</b> {model_name}\n"
        f"<b>MAPE:</b> {mape}%\n"
        f"<b>MAE:</b> ${mae}\n"
        f"<b>R²:</b> {r2}\n"
        f"<b>Direction Acc:</b> {dir_acc}%\n\n"
        f"<a href=\"{link}\">Открыть полный прогноз</a>"
    )

    if pdf_bytes:
        filename = f"NeuCast_{ticker}_{model_name.replace(' ', '_')}.pdf"
        send_document(chat_id, pdf_bytes, filename, caption=text)
    else:
        send_message(chat_id, text)

    return True


def process_update(update: dict) -> str | None:
    """
    Process incoming webhook update from Telegram.
    Returns response text or None.
    """
    message = update.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    if not chat_id or not text:
        return None

    if text.startswith("/start"):
        parts = text.split()
        if len(parts) == 2:
            token = parts[1]
            if complete_link(token, chat_id):
                send_message(chat_id,
                    "<b>Telegram привязан к NeuCast!</b>\n\n"
                    "Теперь вы будете получать уведомления когда прогноз будет готов.\n"
                    "PDF-отчёт будет прикреплён к сообщению.\n\n"
                    "Для отвязки: /unlink"
                )
                return "linked"
            else:
                send_message(chat_id,
                    "Токен недействителен или истёк.\n"
                    "Получите новый на сайте NeuCast."
                )
                return "invalid_token"
        else:
            send_message(chat_id,
                "<b>NeuCast</b> — AI-прогнозирование финансовых рынков\n\n"

                "Стек из 4 моделей машинного обучения анализирует рынок "
                "и даёт прогноз цены акций:\n\n"

                "  <b>TCN</b> — Temporal Convolutional Network\n"
                "  <b>CatBoost</b> — градиентный бустинг (Яндекс)\n"
                "  <b>XGBoost</b> — экстремальный бустинг\n"
                "  <b>LightGBM</b> — быстрый бустинг (Microsoft)\n\n"

                "Все 4 модели объединяются в <b>Stacking Ensemble</b> "
                "через Ridge-мета-модель. Прогноз на будущее строится "
                "методом <b>Monte Carlo GBM</b> (1000 симуляций).\n\n"

                "Дополнительно: <b>FinBERT NLP</b> анализирует тональность "
                "последних новостей по тикеру.\n\n"

                "<b>Как привязать Telegram:</b>\n"
                "1. Откройте <a href=\"https://neucast.ru\">neucast.ru</a>\n"
                "2. Запустите прогноз по любому тикеру\n"
                "3. На странице ожидания нажмите «Уведомить в Telegram»\n"
                "4. Когда прогноз будет готов — получите сюда отчёт\n\n"

                "<b>Команды:</b>\n"
                "/start — это сообщение\n"
                "/unlink — отвязать аккаунт"
            )
            return "start"

    elif text.startswith("/unlink"):
        # Find user by chat_id and unlink
        for uid, cid in list(_links.items()):
            if cid == chat_id:
                unlink(uid)
                send_message(chat_id, "Аккаунт отвязан от NeuCast.")
                return "unlinked"
        send_message(chat_id, "Аккаунт не был привязан.")
        return "not_linked"

    elif text.startswith("/status"):
        send_message(chat_id,
            "<b>NeuCast Bot</b>\n"
            f"Привязанных аккаунтов: {len(_links)}\n"
            "Бот активен и готов отправлять уведомления."
        )
        return "status"

    return None
