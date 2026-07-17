"""Cross-run provider concurrency slots via Worker KV (soft semaphore)."""

from __future__ import annotations

import os
import time
import uuid

from loguru import logger

from bot.escalate_ledger import EscalateLedger


def _holder_id() -> str:
    run = os.getenv("GITHUB_RUN_ID") or ""
    job = os.getenv("GITHUB_JOB") or ""
    suffix = uuid.uuid4().hex[:8]
    return f"{run}:{job}:{suffix}" if run else f"local:{suffix}"


class ProviderSemaphore:
    """Acquire/release provider slots shared across Actions runs."""

    def __init__(self, ledger: EscalateLedger | None = None):
        self.ledger = ledger or EscalateLedger()
        self.holder_id = _holder_id()
        self._held: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.ledger.remote

    def try_acquire(self, provider: str, *, max_inflight: int, ttl_sec: float = 180.0) -> bool:
        if not self.enabled:
            return True
        ok = self.ledger.acquire_slot(
            provider,
            holder_id=self.holder_id,
            max_inflight=max(1, int(max_inflight)),
            ttl_sec=ttl_sec,
        )
        if ok:
            self._held.add(provider)
        return ok

    def release(self, provider: str) -> None:
        if provider not in self._held:
            return
        self._held.discard(provider)
        if not self.enabled:
            return
        self.ledger.release_slot(provider, holder_id=self.holder_id)

    def release_all(self) -> None:
        for name in list(self._held):
            self.release(name)
