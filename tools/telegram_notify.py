"""Lightweight synchronous Telegram notifier.

Used for ad-hoc operational notifications (training pipeline finished,
robustness suite verdict, manual emergency alerts) — distinct from the
production signal-bot which is async + always-on.

Reads ``HF_TELEGRAM_SIGNAL_*`` env vars (same as the signal-bot, so we
don't carry two sets of credentials). Returns 0 on success, non-zero
on send failure or missing config.

Run from anywhere with the env vars
-----------------------------------

::

    HF_TELEGRAM_SIGNAL_ENABLED=1 \\
    HF_TELEGRAM_SIGNAL_BOT_TOKEN=... \\
    HF_TELEGRAM_SIGNAL_CHAT_ID=... \\
        python -m tools.telegram_notify --text "Pretrain done"

For multi-line / formatted bodies pass ``--from-file`` to read the
body from a file (so shell escaping doesn't murder backticks etc.).

The body is sent as ``parse_mode=HTML`` — escape user-supplied
strings via ``--escape-html`` if they could contain ``< > &``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from html import escape as _html_escape
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def send_telegram(
    *,
    bot_token: str,
    chat_id: str,
    body_html: str,
    timeout_seconds: float = 5.0,
    disable_notification: bool = False,
) -> tuple[int, str]:
    """Synchronous POST to Telegram sendMessage.

    Returns ``(http_code, response_body)``. ``http_code == 0`` means
    URLError or other transport failure (response body carries the
    Python-side error description).
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": body_html,
        "parse_mode": "HTML",
        "disable_notification": disable_notification,
        # Disable web-page-preview so URLs in the body don't auto-expand
        # into a fat preview card.
        "disable_web_page_preview": True,
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return 0, f"URLError: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default=None,
                   help="message body (HTML allowed). Either --text "
                        "or --from-file is required.")
    p.add_argument("--from-file", default=None, type=Path,
                   help="read message body from file (UTF-8).")
    p.add_argument("--escape-html", action="store_true",
                   help="HTML-escape the body (for un-trusted text "
                        "with literal < > & characters).")
    p.add_argument("--silent", action="store_true",
                   help="send with disable_notification=True (no "
                        "phone vibration / sound, just log it).")
    p.add_argument("--bot-token", default=os.environ.get(
        "HF_TELEGRAM_SIGNAL_BOT_TOKEN", "",
    ).strip())
    p.add_argument("--chat-id", default=os.environ.get(
        "HF_TELEGRAM_SIGNAL_CHAT_ID", "",
    ).strip())
    p.add_argument("--enabled-env", default="HF_TELEGRAM_SIGNAL_ENABLED",
                   help="env var that gates sending (must be '1' / "
                        "'true' / 'yes' to enable). Default "
                        "HF_TELEGRAM_SIGNAL_ENABLED.")
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not args.bot_token or not args.chat_id:
        logger.error(
            "missing HF_TELEGRAM_SIGNAL_BOT_TOKEN or "
            "HF_TELEGRAM_SIGNAL_CHAT_ID env"
        )
        return 2

    enabled_raw = os.environ.get(args.enabled_env, "").strip().lower()
    if enabled_raw not in ("1", "true", "yes", "on"):
        logger.warning(
            "%s != 1 — would send: %s",
            args.enabled_env,
            (args.text or "[from-file]")[:100],
        )
        return 0  # graceful no-op when disabled

    if args.from_file is not None:
        body = args.from_file.read_text(encoding="utf-8")
    elif args.text is not None:
        body = args.text
    else:
        logger.error("either --text or --from-file is required")
        return 2

    if args.escape_html:
        body = _html_escape(body)

    code, resp = send_telegram(
        bot_token=args.bot_token,
        chat_id=args.chat_id,
        body_html=body,
        disable_notification=args.silent,
    )
    if 200 <= code < 300:
        logger.info("telegram OK (code=%d)", code)
        return 0
    logger.error("telegram FAILED code=%d body=%s", code, resp[:200])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
