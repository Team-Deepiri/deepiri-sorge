"""Shared test helpers."""

from __future__ import annotations

import sys
import types
from typing import Any


class _LoggerStub:
    def warning(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def remove(self, *args, **kwargs):
        return None

    def add(self, *args, **kwargs):
        return None


def install_loguru_stub() -> None:
    loguru_stub: Any = types.ModuleType("loguru")
    loguru_stub.logger = _LoggerStub()
    sys.modules.setdefault("loguru", loguru_stub)
