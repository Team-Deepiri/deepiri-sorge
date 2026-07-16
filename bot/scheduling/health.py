"""Continuous provider health scoring."""

from __future__ import annotations

import threading
import time


class HealthTracker:
    def __init__(self, initial: float = 100.0):
        self._score = initial
        self._lock = threading.Lock()
        self._last_event = time.monotonic()
        self.cooling_until: float = 0.0

    @property
    def score(self) -> float:
        with self._lock:
            self._maybe_recover()
            return self._score

    def _maybe_recover(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_event
        ticks = int(elapsed // 10)
        if ticks > 0:
            self._score = min(100.0, self._score + ticks)
            self._last_event += ticks * 10

    def record_success(self, latency_ms: float = 0.0) -> None:
        with self._lock:
            self._maybe_recover()
            self._score = min(100.0, self._score + 2.0)
            if latency_ms > 0:
                self._score = max(0.0, self._score - (latency_ms / 500.0))
            self._last_event = time.monotonic()

    def record_rate_limit(self, retry_after: float | None = None) -> None:
        with self._lock:
            self._maybe_recover()
            self._score = max(0.0, self._score - 40.0)
            wait = retry_after if retry_after is not None else 45.0
            wait = min(max(wait, 1.0), 120.0)
            self.cooling_until = time.monotonic() + wait
            self._last_event = time.monotonic()

    def record_server_error(self) -> None:
        with self._lock:
            self._maybe_recover()
            self._score = max(0.0, self._score - 20.0)
            self._last_event = time.monotonic()

    def record_timeout(self) -> None:
        with self._lock:
            self._maybe_recover()
            self._score = max(0.0, self._score - 15.0)
            self._last_event = time.monotonic()

    def record_payload_too_large(self) -> None:
        with self._lock:
            self._maybe_recover()
            self._score = max(0.0, self._score - 5.0)
            self._last_event = time.monotonic()

    def is_cooling(self) -> bool:
        with self._lock:
            return time.monotonic() < self.cooling_until

    def cooling_remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cooling_until - time.monotonic())

    def seconds_until_score(self, target: float) -> float:
        """Estimated wait until score recovers to ``target`` (+1 health per 10s)."""
        with self._lock:
            self._maybe_recover()
            if self._score >= target:
                return 0.0
            return (target - self._score) * 10.0
