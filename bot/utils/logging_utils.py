"""Logging utilities for deepiri-sorge"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    format_string: str | None = None,
) -> None:
    logger.remove()

    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=True,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level="DEBUG",
            format=format_string,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    logger.enable("bot")


def get_logger(name: str = "bot"):
    return logger.bind(name=name)


class LogCapture:

    def __init__(self):
        self.records = []
        self.output = ""

    def __enter__(self):
        self.handler_id = logger.add(
            self._capture,
            format="{message}",
        )
        return self

    def __exit__(self, *args):
        logger.remove(self.handler_id)

    def _capture(self, message):
        self.records.append(message.record)
        self.output += str(message) + "\n"
