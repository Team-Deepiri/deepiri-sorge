"""Cross-run shared RPM budget (Worker KV) — semaphore, ledger, and run_context wiring."""

import time

from bot.escalate_ledger import EscalateLedger
from bot.quota_tracker import QuotaTracker
from bot.scheduling.health import HealthTracker
from bot.scheduling.run_context import ProviderRuntime, RunContext
from bot.scheduling.semaphore import ProviderSemaphore
from bot.scheduling.token_bucket import TokenBucket


class _FakeLedger:
    def __init__(self, *, remote: bool, rpm_ok: bool = True, slot_ok: bool = True):
        self.remote = remote
        self._rpm_ok = rpm_ok
        self._slot_ok = slot_ok
        self.rpm_calls: list[tuple[str, float]] = []
        self.released: list[str] = []

    def try_consume_rpm(self, provider: str, *, rpm: float) -> bool:
        self.rpm_calls.append((provider, rpm))
        return self._rpm_ok

    def acquire_slot(self, provider, *, holder_id, max_inflight, ttl_sec=180.0):
        return self._slot_ok

    def release_slot(self, provider, *, holder_id):
        self.released.append(provider)


def _quota():
    return QuotaTracker(
        limits={"gpt": 100, "gemini": 100, "openrouter": 100},
        used={"gpt": 0, "gemini": 0, "openrouter": 0},
    )


def _run_with_semaphore(semaphore: ProviderSemaphore) -> RunContext:
    return RunContext(
        providers={
            "groq": ProviderRuntime(
                name="groq",
                bucket=TokenBucket(30),
                health=HealthTracker(100),
                max_context_tokens=100000,
                max_inflight=5,
                nominal_latency_ms=200,
                quality_prior=0.9,
            ),
        },
        quota=_quota(),
        deadline=time.monotonic() + 30,
        semaphore=semaphore,
    )


def test_semaphore_try_consume_rpm_disabled_passes_through():
    sem = ProviderSemaphore(ledger=_FakeLedger(remote=False))
    assert sem.try_consume_rpm("groq", rpm=30) is True


def test_semaphore_try_consume_rpm_delegates_to_ledger():
    ledger = _FakeLedger(remote=True, rpm_ok=False)
    sem = ProviderSemaphore(ledger=ledger)
    assert sem.try_consume_rpm("groq", rpm=30) is False
    assert ledger.rpm_calls == [("groq", 30)]


def test_try_acquire_blocked_when_shared_rpm_exhausted():
    ledger = _FakeLedger(remote=True, rpm_ok=False, slot_ok=True)
    sem = ProviderSemaphore(ledger=ledger)
    run = _run_with_semaphore(sem)

    assert run.try_acquire("groq") is False
    # Slot was acquired then released — must not leak a held slot.
    assert ledger.released == ["groq"]


def test_try_acquire_succeeds_when_shared_rpm_has_headroom():
    ledger = _FakeLedger(remote=True, rpm_ok=True, slot_ok=True)
    sem = ProviderSemaphore(ledger=ledger)
    run = _run_with_semaphore(sem)

    assert run.try_acquire("groq") is True


def test_ledger_try_consume_rpm_fails_open_without_remote_config():
    ledger = EscalateLedger(base_url=None, secret=None)
    assert ledger.remote is False
    assert ledger.try_consume_rpm("groq", rpm=30) is True
