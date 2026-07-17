"""Shared mutable state for one review run."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from bot.quota_tracker import QuotaTracker
from bot.scheduling.health import HealthTracker
from bot.scheduling.semaphore import ProviderSemaphore
from bot.scheduling.token_bucket import TokenBucket
from bot.scheduling.types import ProviderStatus


@dataclass
class ProviderRuntime:
    name: str
    bucket: TokenBucket
    health: HealthTracker
    max_context_tokens: int
    max_inflight: int
    nominal_latency_ms: float
    quality_prior: float
    in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class RunContext:
    providers: dict[str, ProviderRuntime]
    quota: QuotaTracker
    deadline: float
    repo_context: str | None = None
    context_fingerprint: str = ""
    health_threshold: float = 25.0
    cache_enabled: bool = False
    cache_ttl_hours: int = 24
    history: object | None = None  # ProviderHistory; typed loosely to avoid cycles
    # Diff-token estimate ignores template + repo context; add this for eligibility.
    prompt_overhead_tokens: int = 0
    # Optional identity for escalate ledger / drain upgrades.
    repo: str = ""
    pr_number: int = 0
    installation_id: int | None = None
    head_sha: str = ""
    # Soft cap on time spent sleeping for capacity (not too early; default 120s).
    max_capacity_wait_sec: float = 120.0
    capacity_waited_sec: float = 0.0
    semaphore: ProviderSemaphore | None = None

    def alive(self) -> bool:
        return time.monotonic() < self.deadline

    def remaining_sec(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def capacity_budget_remaining(self) -> float:
        return max(0.0, self.max_capacity_wait_sec - self.capacity_waited_sec)

    def status(self, name: str) -> ProviderStatus | None:
        rt = self.providers.get(name)
        if not rt:
            return None
        return ProviderStatus(
            name=name,
            health=rt.health.score,
            rpm_remaining=rt.bucket.remaining(),
            max_context_tokens=rt.max_context_tokens,
            nominal_latency_ms=rt.nominal_latency_ms,
            quality_prior=rt.quality_prior,
            cooling_until=rt.health.cooling_until,
            in_flight=rt.in_flight,
            max_inflight=rt.max_inflight,
        )

    def try_acquire(self, name: str) -> bool:
        rt = self.providers.get(name)
        if not rt:
            return False
        with rt.lock:
            if rt.health.is_cooling():
                return False
            if rt.health.score < self.health_threshold:
                return False
            if rt.in_flight >= rt.max_inflight:
                return False
            if not self.quota.can_use(name):
                return False
            # Peek local RPM without consuming yet.
            if rt.bucket.remaining() < 1.0:
                return False
        # Cross-run KV semaphore outside local lock (network I/O).
        if self.semaphore is not None:
            if not self.semaphore.try_acquire(
                name, max_inflight=rt.max_inflight, ttl_sec=180.0
            ):
                return False
        with rt.lock:
            # Re-check after remote acquire.
            if rt.in_flight >= rt.max_inflight:
                if self.semaphore is not None:
                    self.semaphore.release(name)
                return False
            if not rt.bucket.try_consume(1.0):
                if self.semaphore is not None:
                    self.semaphore.release(name)
                return False
            rt.in_flight += 1
            return True

    def release(self, name: str) -> None:
        rt = self.providers.get(name)
        if not rt:
            return
        with rt.lock:
            rt.in_flight = max(0, rt.in_flight - 1)
        if self.semaphore is not None:
            self.semaphore.release(name)
