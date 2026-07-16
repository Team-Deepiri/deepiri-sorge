"""Phase 1+2 scheduling primitive tests."""

import time

from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.quota_tracker import QuotaTracker
from bot.scheduling.health import HealthTracker
from bot.scheduling.market_score import score_provider
from bot.scheduling.scheduler import ReviewScheduler
from bot.scheduling.token_bucket import TokenBucket
from bot.scheduling.types import ProviderResult, ProviderStatus, ScheduledChunk
from bot.schemas import ReviewResult
from bot.utils import cache as review_cache


def test_token_bucket_refuses_burst():
    bucket = TokenBucket(rate_per_minute=60, capacity=2)
    assert bucket.try_consume()
    assert bucket.try_consume()
    assert not bucket.try_consume()


def test_token_bucket_time_until():
    bucket = TokenBucket(rate_per_minute=60, capacity=1)
    assert bucket.try_consume()
    wait = bucket.time_until(1.0)
    assert 0.5 < wait < 2.0


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


def test_market_score_overhead_excludes_tight_provider():
    """Diff fits window but prompt overhead must exclude the provider (prod 413 case)."""
    status = ProviderStatus(
        name="groq",
        health=100,
        rpm_remaining=10,
        max_context_tokens=8000,
        nominal_latency_ms=200,
        quality_prior=0.9,
    )
    chunk = ReviewChunk(
        files=["a.py"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=5339,
    )
    scheduled = ScheduledChunk(chunk=chunk)
    assert score_provider(status, scheduled, prompt_overhead_tokens=0) > 0
    assert score_provider(status, scheduled, prompt_overhead_tokens=3000) == 0.0


def test_lane_affinity_prefers_groq_when_it_fits():
    """Small effective prompts must rank Groq above OpenRouter (stop OR pecking)."""
    from bot.scheduling.market_score import score_provider as score

    chunk = ReviewChunk(
        files=["src/a.py"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=1000,
    )
    scheduled = ScheduledChunk(chunk=chunk, priority=80)
    overhead = 2000  # effective 3000 < groq lane
    groq = ProviderStatus(
        name="groq",
        health=100,
        rpm_remaining=10,
        max_context_tokens=4500,
        nominal_latency_ms=400,
        quality_prior=0.9,
    )
    openrouter = ProviderStatus(
        name="openrouter",
        health=100,
        rpm_remaining=10,
        max_context_tokens=100000,
        nominal_latency_ms=800,
        quality_prior=0.95,  # even with better prior, lane should win
    )
    sg = score(groq, scheduled, prompt_overhead_tokens=overhead, historical_quality=0.5)
    so = score(
        openrouter, scheduled, prompt_overhead_tokens=overhead, historical_quality=0.95
    )
    assert sg > so


def _ok_result(summary="ok") -> ReviewResult:
    return ReviewResult(
        summary=summary,
        issues=[],
        recommendations=[],
        score=9.0,
        latency_ms=10,
        model="fake",
        review_type="gemini",
    )


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
                result=_ok_result(),
            )

    from bot.scheduling.run_context import ProviderRuntime, RunContext

    quota = QuotaTracker(
        limits={"gpt": 100, "gemini": 100, "openrouter": 100},
        used={"gpt": 0, "gemini": 0, "openrouter": 0},
    )
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
    scheduler = ReviewScheduler([FakeGroq(), FakeGemini()], run, max_workers=2)
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


def test_scheduler_cache_hit_skips_acquire(tmp_path, monkeypatch):
    monkeypatch.setattr(review_cache, "CACHE_DIR", tmp_path)
    calls = {"gemini": 0}

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
            return ProviderResult(ok=True, provider="gemini", status_code=200, result=_ok_result())

    from bot.scheduling.run_context import ProviderRuntime, RunContext

    diff = ParsedDiff(raw="+cached()\n")
    review_cache.set_chunk(
        diff.raw,
        _ok_result("from-cache").to_dict(),
        context_fingerprint="fp1",
    )

    quota = QuotaTracker(
        limits={"gpt": 100, "gemini": 100, "openrouter": 100},
        used={"gpt": 0, "gemini": 0, "openrouter": 0},
    )
    run = RunContext(
        providers={
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
        context_fingerprint="fp1",
        cache_enabled=True,
        cache_ttl_hours=24,
    )
    scheduler = ReviewScheduler([FakeGemini()], run, max_workers=2)
    chunk = ReviewChunk(files=["a.py"], parsed_diff=diff, estimated_tokens=50)
    results, skipped, meta = scheduler.run([chunk])

    assert len(results) == 1
    assert results[0].summary == "from-cache"
    assert calls["gemini"] == 0
    assert meta.cache_hits == 1
    assert meta.dispatches == 0


def test_scheduler_priority_under_deadline():
    """High-priority auth chunk reviews before docs; deadline skips the rest."""
    reviewed: list[str] = []

    class SlowProvider:
        name = "gemini"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 800
        quality_prior = 0.8

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run):
            reviewed.append(chunk.files[0])
            time.sleep(0.15)
            return ProviderResult(
                ok=True,
                provider="gemini",
                status_code=200,
                result=_ok_result(chunk.files[0]),
            )

    from bot.scheduling.run_context import ProviderRuntime, RunContext

    quota = QuotaTracker(
        limits={"gpt": 100, "gemini": 100, "openrouter": 100},
        used={"gpt": 0, "gemini": 0, "openrouter": 0},
    )
    run = RunContext(
        providers={
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
        deadline=time.monotonic() + 0.12,
    )
    scheduler = ReviewScheduler([SlowProvider()], run, max_workers=1)
    chunks = [
        ReviewChunk(
            files=["docs/README.md"],
            parsed_diff=ParsedDiff(raw="+doc\n"),
            estimated_tokens=100,
        ),
        ReviewChunk(
            files=["src/auth/login.py"],
            parsed_diff=ParsedDiff(raw="+auth\n"),
            estimated_tokens=100,
        ),
    ]
    results, skipped, meta = scheduler.run(chunks)

    assert reviewed[0] == "src/auth/login.py"
    assert meta.stop_reason == "deadline" or any("deadline" in s.reason for s in skipped)
    assert any("priority=" in s.reason for s in skipped)


def test_desired_workers_tracks_inflight_slots():
    from bot.scheduling.run_context import ProviderRuntime, RunContext

    quota = QuotaTracker(
        limits={"gpt": 100, "gemini": 100, "openrouter": 100},
        used={"gpt": 0, "gemini": 0, "openrouter": 0},
    )
    run = RunContext(
        providers={
            "groq": ProviderRuntime(
                name="groq",
                bucket=TokenBucket(30),
                health=HealthTracker(100),
                max_context_tokens=8000,
                max_inflight=2,
                nominal_latency_ms=200,
                quality_prior=0.9,
                in_flight=1,
            ),
            "gemini": ProviderRuntime(
                name="gemini",
                bucket=TokenBucket(30),
                health=HealthTracker(100),
                max_context_tokens=100000,
                max_inflight=1,
                nominal_latency_ms=800,
                quality_prior=0.8,
                in_flight=0,
            ),
        },
        quota=quota,
        deadline=time.monotonic() + 30,
    )

    class P:
        def __init__(self, name):
            self.name = name

    sch = ReviewScheduler([P("groq"), P("gemini")], run, max_workers=4)
    # groq has 1 free slot + gemini 1 = 2
    assert sch._desired_workers() == 2
