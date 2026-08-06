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


def test_health_seconds_until_score_after_penalty():
    h = HealthTracker(100)
    h.record_rate_limit(retry_after=1)
    # 100 - 40 = 60; threshold 25 is already met
    assert h.seconds_until_score(25) == 0.0
    h.record_rate_limit(retry_after=1)
    # 60 - 40 = 20; need ~50s to reach 25
    wait = h.seconds_until_score(25)
    assert 40 <= wait <= 60


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
        max_context_tokens=7000,
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


def test_lane_affinity_prefers_groq_for_emotion_sized_prompt():
    """~3461 diff + ~2600 overhead (emotion#81) must still home on Groq."""
    from bot.scheduling.market_score import score_provider as score

    chunk = ReviewChunk(
        files=["src/a.py"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=3461,
    )
    scheduled = ScheduledChunk(chunk=chunk, priority=80)
    overhead = 2604
    groq = ProviderStatus(
        name="groq",
        health=100,
        rpm_remaining=10,
        max_context_tokens=7000,
        nominal_latency_ms=400,
        quality_prior=0.9,
    )
    openrouter = ProviderStatus(
        name="openrouter",
        health=100,
        rpm_remaining=10,
        max_context_tokens=100000,
        nominal_latency_ms=800,
        quality_prior=0.95,
    )
    sg = score(groq, scheduled, prompt_overhead_tokens=overhead, historical_quality=0.5)
    so = score(
        openrouter, scheduled, prompt_overhead_tokens=overhead, historical_quality=0.95
    )
    assert sg > 0
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

        def review(self, chunk, run, *, prior_partial=None):
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

        def review(self, chunk, run, *, prior_partial=None):
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

        def review(self, chunk, run, *, prior_partial=None):
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

        def review(self, chunk, run, *, prior_partial=None):
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


def test_capacity_wait_budget_defers_without_burning_wall_clock():
    """Long cooldowns should early-defer after max_capacity_wait_sec, not sit for 10m."""
    from bot.scheduling.run_context import ProviderRuntime, RunContext
    from bot.scheduling.history import ProviderHistory

    class AlwaysCool:
        name = "groq"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 200
        quality_prior = 0.9

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run, *, prior_partial=None):
            raise AssertionError("should not dispatch while cooling")

    history = ProviderHistory(sync_remote=False)
    history.mark_rate_limited("groq", retry_after=600)

    quota = QuotaTracker(
        limits={"gpt": 100, "gemini": 100, "openrouter": 100},
        used={"gpt": 0, "gemini": 0, "openrouter": 0},
        sync_remote=False,
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
        },
        quota=quota,
        deadline=time.monotonic() + 600,
        history=history,
        max_capacity_wait_sec=25.0,
    )
    # Pretend we already waited most of the budget.
    run.capacity_waited_sec = 22.0

    sch = ReviewScheduler([AlwaysCool()], run, max_workers=1)
    chunk = ReviewChunk(
        files=["a.py"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=100,
    )
    results, skipped, meta = sch.run([chunk])
    assert results == []
    assert skipped
    assert meta.stop_reason == "providers_exhausted"
    assert any("capacity_waited" in s.reason or "no eligible" in s.reason for s in skipped)

def test_capacity_wait_budget_scales_up_for_large_queues():
    """A fixed 120s budget shouldn't nuke every chunk in a 100+ file PR after
    a couple of early rate-limit cooldowns — the budget should scale with how
    much work is actually queued, bounded by wall-clock time remaining."""
    from bot.scheduling.run_context import RunContext

    quota = QuotaTracker(limits={}, used={}, sync_remote=False)
    run = RunContext(
        providers={},
        quota=quota,
        deadline=time.monotonic() + 600,
        max_capacity_wait_sec=120.0,
    )
    sch = ReviewScheduler([], run, max_workers=1)

    # A small PR keeps the default budget.
    sch._scale_capacity_wait(3)
    assert run.max_capacity_wait_sec == 120.0

    # A huge PR (many chunks) gets a much larger patience budget instead of
    # abandoning most of the queue after the first ~2 minutes of cooldowns.
    sch._scale_capacity_wait(118)
    assert run.max_capacity_wait_sec > 120.0
    # ...but never past wall-clock remaining minus the posting margin.
    assert run.max_capacity_wait_sec <= run.remaining_sec()


def test_capacity_wait_budget_never_exceeds_wall_clock():
    """Even a massive queue can't push the budget past the run's own deadline."""
    from bot.scheduling.run_context import RunContext

    quota = QuotaTracker(limits={}, used={}, sync_remote=False)
    run = RunContext(
        providers={},
        quota=quota,
        deadline=time.monotonic() + 60,
        max_capacity_wait_sec=120.0,
    )
    sch = ReviewScheduler([], run, max_workers=1)
    sch._scale_capacity_wait(500)
    assert run.max_capacity_wait_sec <= run.remaining_sec()


def test_large_pr_survives_early_rate_limits_end_to_end(monkeypatch):
    """Reproduces the real incident: a 118-chunk PR, 3 early 429s that used
    to burn the whole 120s capacity-wait budget and skip everything else.
    Runs the actual scheduler loop (not just the budget helper) on a fake
    clock so cooldowns "pass" without real sleeping, and proves that once
    the provider recovers, the scaled budget lets the rest of the queue get
    reviewed instead of being abandoned in one shot."""
    from bot.scheduling.run_context import ProviderRuntime, RunContext

    clock = {"t": 0.0}

    def fake_monotonic():
        return clock["t"]

    def fake_sleep(seconds):
        clock["t"] += seconds

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    class FlakyThenRecovers:
        name = "groq"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 200
        quality_prior = 0.9

        def __init__(self):
            self.calls = 0

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run, *, prior_partial=None):
            self.calls += 1
            if self.calls <= 3:
                return ProviderResult(
                    ok=False,
                    provider="groq",
                    status_code=429,
                    retry_after=45.0,
                    latency_ms=50.0,
                    error="http_429",
                )
            return ProviderResult(
                ok=True,
                provider="groq",
                status_code=200,
                latency_ms=50.0,
                result=ReviewResult(
                    summary="ok",
                    issues=[],
                    recommendations=[],
                    score=8.0,
                    latency_ms=50.0,
                    model="groq",
                ),
            )

    quota = QuotaTracker(limits={"gpt": 10000}, used={"gpt": 0}, sync_remote=False)
    run = RunContext(
        providers={
            "groq": ProviderRuntime(
                name="groq",
                bucket=TokenBucket(3000, capacity=3000),
                health=HealthTracker(100),
                max_context_tokens=100000,
                max_inflight=2,
                nominal_latency_ms=200,
                quality_prior=0.9,
            ),
        },
        quota=quota,
        deadline=fake_monotonic() + 720.0,
        max_capacity_wait_sec=120.0,
    )
    provider = FlakyThenRecovers()
    sch = ReviewScheduler([provider], run, max_workers=1)
    chunks = [
        ReviewChunk(
            files=[f"file_{i}.ts"],
            parsed_diff=ParsedDiff(raw="+x\n"),
            estimated_tokens=100,
        )
        for i in range(118)
    ]
    results, skipped, meta = sch.run(chunks)

    # Old fixed 120s budget would've been exhausted by the 3 early 429
    # cooldowns (~2 min) and skipped ~112 of these in one shot. The scaled
    # budget should let almost the entire queue get reviewed once the
    # provider recovers.
    assert len(results) > 100
    assert len(skipped) < 18


def test_scheduler_forwards_partial_output_to_fallback_provider():
    """A truncated Groq response primes the next provider instead of being dropped."""
    seen = {"gemini_prior": "unset"}
    partial = (
        '{"summary": "Reviewed mmap.cpp", "issues": [{"severity": "high", '
        '"file": "src/mmap.cpp", "line": 42, "message": "munmap called on a '
        'region that was already reset, so the second call double-unmaps"'
    )

    class FakeGroq:
        name = "groq"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 200
        quality_prior = 0.9

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run, *, prior_partial=None):
            # HTTP 200 but cut off mid-generation (finish_reason=length).
            return ProviderResult(
                ok=False,
                provider="groq",
                status_code=200,
                error="empty_or_invalid_review",
                partial_output=partial,
            )

    class FakeGemini:
        name = "gemini"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 800
        quality_prior = 0.8

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run, *, prior_partial=None):
            seen["gemini_prior"] = prior_partial
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
    scheduler = ReviewScheduler([FakeGroq(), FakeGemini()], run, max_workers=1)
    chunk = ReviewChunk(
        files=["src/mmap.cpp"],
        parsed_diff=ParsedDiff(raw="+void reset();\n"),
        estimated_tokens=100,
    )
    results, _skipped, _meta = scheduler.run([chunk])

    assert len(results) == 1
    assert seen["gemini_prior"] == partial, "fallback should resume from Groq's partial"


def test_scheduler_does_not_prime_when_no_partial_salvaged():
    seen = {"gemini_prior": "unset"}

    class FakeGroq:
        name = "groq"
        max_context_tokens = 100000
        cost_tier = "free"
        nominal_latency_ms = 200
        quality_prior = 0.9

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run, *, prior_partial=None):
            return ProviderResult(
                ok=False, provider="groq", status_code=429, retry_after=60,
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

        def review(self, chunk, run, *, prior_partial=None):
            seen["gemini_prior"] = prior_partial
            return ProviderResult(
                ok=True, provider="gemini", status_code=200, result=_ok_result(),
            )

    from bot.scheduling.run_context import ProviderRuntime, RunContext

    quota = QuotaTracker(
        limits={"gpt": 100, "gemini": 100, "openrouter": 100},
        used={"gpt": 0, "gemini": 0, "openrouter": 0},
    )
    run = RunContext(
        providers={
            "groq": ProviderRuntime(
                name="groq", bucket=TokenBucket(30), health=HealthTracker(100),
                max_context_tokens=100000, max_inflight=1,
                nominal_latency_ms=200, quality_prior=0.9,
            ),
            "gemini": ProviderRuntime(
                name="gemini", bucket=TokenBucket(30), health=HealthTracker(100),
                max_context_tokens=100000, max_inflight=1,
                nominal_latency_ms=800, quality_prior=0.8,
            ),
        },
        quota=quota,
        deadline=time.monotonic() + 30,
    )
    scheduler = ReviewScheduler([FakeGroq(), FakeGemini()], run, max_workers=1)
    chunk = ReviewChunk(
        files=["a.py"],
        parsed_diff=ParsedDiff(raw="+print(1)\n"),
        estimated_tokens=100,
    )
    scheduler.run([chunk])

    assert seen["gemini_prior"] is None
