"""Phase 1 scheduling primitive tests."""

import time
from unittest.mock import MagicMock

from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.quota_tracker import QuotaTracker
from bot.scheduling.health import HealthTracker
from bot.scheduling.market_score import score_provider
from bot.scheduling.scheduler import ReviewScheduler
from bot.scheduling.token_bucket import TokenBucket
from bot.scheduling.types import ProviderResult, ProviderStatus, ScheduledChunk
from bot.schemas import ReviewResult
from bot.config import Config


def test_token_bucket_refuses_burst():
    bucket = TokenBucket(rate_per_minute=60, capacity=2)
    assert bucket.try_consume()
    assert bucket.try_consume()
    assert not bucket.try_consume()


def test_health_drops_on_429_and_cools():
    h = HealthTracker(100)
    h.record_rate_limit(retry_after=30)
    assert h.score <= 60
    assert h.is_cooling()


def test_market_score_rejects_oversized_context():
    status = ProviderStatus(
        name="groq",
        health=100,
        rpm_remaining=10,
        max_context_tokens=1000,
        nominal_latency_ms=200,
        quality_prior=0.9,
    )
    chunk = ReviewChunk(
        files=["a.py"],
        parsed_diff=ParsedDiff(raw="x" * 100),
        estimated_tokens=5000,
    )
    scheduled = ScheduledChunk(chunk=chunk)
    assert score_provider(status, scheduled) == 0.0


def test_scheduler_does_not_stampede_dead_provider():
    """When groq always 429s, scheduler moves to gemini and doesn't loop forever."""
    calls = {"groq": 0, "gemini": 0}

    class FakeGroq:
        name = "groq"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 200
        quality_prior = 0.9

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run):
            calls["groq"] += 1
            return ProviderResult(
                ok=False,
                provider="groq",
                status_code=429,
                retry_after=60,
                error="rate limited",
            )

    class FakeGemini:
        name = "gemini"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 800
        quality_prior = 0.8

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run):
            calls["gemini"] += 1
            return ProviderResult(
                ok=True,
                provider="gemini",
                status_code=200,
                result=ReviewResult(
                    summary="ok",
                    issues=[],
                    recommendations=[],
                    score=9.0,
                    latency_ms=10,
                    model="fake",
                    review_type="gemini",
                ),
            )

    config = Config()
    config.scheduler.wall_clock_sec = 30
    config.providers.groq.rpm = 30
    config.providers.gemini.rpm = 30
    # Shrink context filter not needed

    providers = [FakeGroq(), FakeGemini()]
    # Build runtimes only for these two — patch from_providers manually
    from bot.scheduling.health import HealthTracker
    from bot.scheduling.run_context import ProviderRuntime, RunContext
    from bot.scheduling.token_bucket import TokenBucket

    quota = QuotaTracker(limits={"gpt": 100, "gemini": 100, "openrouter": 100}, used={"gpt": 0, "gemini": 0, "openrouter": 0})
    run = RunContext(
        providers={
            "groq": ProviderRuntime(
                name="groq",
                bucket=TokenBucket(30),
                health=HealthTracker(100),
                max_context_tokens=100000,
                max_inflight=1,
                nominal_latency_ms=200,
                quality_prior=0.9,
            ),
            "gemini": ProviderRuntime(
                name="gemini",
                bucket=TokenBucket(30),
                health=HealthTracker(100),
                max_context_tokens=100000,
                max_inflight=1,
                nominal_latency_ms=800,
                quality_prior=0.8,
            ),
        },
        quota=quota,
        deadline=time.monotonic() + 30,
    )
    scheduler = ReviewScheduler(providers, run, max_workers=2)
    chunk = ReviewChunk(
        files=["a.py"],
        parsed_diff=ParsedDiff(raw="+print(1)\n"),
        estimated_tokens=100,
    )
    results, skipped, meta = scheduler.run([chunk])

    assert len(results) == 1
    assert calls["groq"] == 1  # cooled after one 429 — not stampeded
    assert calls["gemini"] == 1
    assert meta.dispatches == 2
