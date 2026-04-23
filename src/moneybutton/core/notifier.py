"""Operator notifications via Discord/Slack webhooks, with log fallback.

Usage:
    from moneybutton.core.notifier import notify
    notify("kill_switch tripped", level="critical")

Behavior:
    - If DISCORD_WEBHOOK_URL is set, POST a compact Discord message.
    - Else if SLACK_WEBHOOK_URL is set, POST to Slack incoming webhook.
    - Else log to logs/moneybutton.log (always, regardless of webhook).
    - Network failures never propagate; they degrade to log-only.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Literal

import httpx

from moneybutton.core.config import get_settings

Level = Literal["info", "warning", "critical"]

_LOGGER_NAME = "moneybutton"
_LOGGER_CONFIGURED = False


def _ensure_logger() -> logging.Logger:
    global _LOGGER_CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if _LOGGER_CONFIGURED:
        return logger
    settings = get_settings()
    log_dir = Path(settings.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "moneybutton.log"
    handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=14, utc=True, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)sZ %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _LOGGER_CONFIGURED = True
    return logger


def _post_discord(url: str, message: str, level: Level) -> bool:
    icon = {"info": ":blue_circle:", "warning": ":warning:", "critical": ":red_circle:"}[level]
    body = {"content": f"{icon} **moneybutton** `{level}` :: {message}"}
    try:
        r = httpx.post(url, json=body, timeout=5.0)
        return 200 <= r.status_code < 300
    except httpx.HTTPError:
        return False


def _post_slack(url: str, message: str, level: Level) -> bool:
    body = {"text": f"[moneybutton/{level}] {message}"}
    try:
        r = httpx.post(url, json=body, timeout=5.0)
        return 200 <= r.status_code < 300
    except httpx.HTTPError:
        return False


def notify(message: str, *, level: Level = "info", extra: dict | None = None) -> None:
    """Fan out a notification. Always logs; posts to webhook if configured."""
    logger = _ensure_logger()
    line = message if extra is None else f"{message} :: {json.dumps(extra, default=str)}"
    log_fn = {"info": logger.info, "warning": logger.warning, "critical": logger.error}[level]
    log_fn(line)

    settings = get_settings()
    discord = settings.discord_webhook_url.get_secret_value() if settings.discord_webhook_url else ""
    slack = settings.slack_webhook_url.get_secret_value() if settings.slack_webhook_url else ""
    if discord:
        _post_discord(discord, line, level)
    elif slack:
        _post_slack(slack, line, level)
