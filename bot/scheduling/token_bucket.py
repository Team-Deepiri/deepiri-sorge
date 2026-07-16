"""Thread-safe token bucket for provider RPM capacity."""

from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_minute: float, capacity: float | None = None):
        self.rate_per_sec = max(rate_per_minute, 0.0) / 60.0
        self.capacity = float(capacity if capacity is not None else max(rate_per_minute, 1.0))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        if self.rate_per_sec > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)

    def try_consume(self, amount: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    def remaining(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def time_until(self, amount: float = 1.0) -> float:
        """Seconds until ``amount`` tokens are available (0 if already)."""
        with self._lock:
            self._refill()
            if self._tokens >= amount:
                return 0.0
            if self.rate_per_sec <= 0:
                return float("inf")
            return (amount - self._tokens) / self.rate_per_sec
