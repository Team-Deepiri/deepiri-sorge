"""Quality-aware routing regression pack (emotion#81 + complexity lanes)."""

from __future__ import annotations

from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.runners.groq_runner import GroqRunner
from bot.scheduling.complexity import complexity_score, is_high_complexity
from bot.scheduling.market_score import (
    LANE_GROQ_MAX,
    home_lane,
    score_provider,
)
from bot.scheduling.types import ProviderStatus, ScheduledChunk


def _chunk(
    *,
    files: list[str],
    tokens: int,
    lines_added: int = 10,
    lines_deleted: int = 0,
) -> ReviewChunk:
    return ReviewChunk(
        files=files,
        parsed_diff=ParsedDiff(
            raw="+x\n" * max(1, lines_added),
            files=files,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            files_changed=len(files),
        ),
        estimated_tokens=tokens,
    )


def test_complexity_escalate_threshold_is_068():
    from bot.scheduling.complexity import COMPLEXITY_ESCALATE, is_high_complexity

    assert COMPLEXITY_ESCALATE == 0.68
    assert is_high_complexity(0.67) is False
    assert is_high_complexity(0.68) is True


def test_emotion81_effective_tokens_fit_groq_lane():
    """~3461 + ~2604 must stay in Groq lane when complexity is low."""
    chunk = _chunk(files=["cli/agent/AgentWorker.js"], tokens=3461, lines_added=80)
    overhead = 2604
    scheduled = ScheduledChunk(
        chunk=chunk,
        priority=60,
        complexity=complexity_score(chunk, prompt_overhead_tokens=overhead),
    )
    assert scheduled.complexity < 0.68
    assert home_lane(3461 + overhead, complexity=scheduled.complexity, path_priority=60) == "groq"


def test_groq_max_tokens_never_exceeds_context_with_safety():
    """0.85 headroom: input 6065 → max_tokens well under remaining window."""
    capped = GroqRunner._cap_max_tokens(6065, GroqRunner.DESIRED_MAX_TOKENS)
    assert capped >= 256
    assert 6065 + capped <= GroqRunner.CONTEXT_TOKEN_LIMIT
    # Safety factor applied
    raw = GroqRunner.CONTEXT_TOKEN_LIMIT - 6065 - GroqRunner.CONTEXT_SAFETY_BUFFER
    assert capped <= int(raw * GroqRunner.HEADROOM_SAFETY) + 1


def test_auth_path_homes_on_gemini_even_when_tokens_fit_groq():
    chunk = _chunk(files=["src/auth/oauth.py"], tokens=1200, lines_added=40)
    cx = complexity_score(chunk, prompt_overhead_tokens=2000)
    assert home_lane(3200, complexity=cx, path_priority=100) == "gemini"


def test_simple_chunk_prefers_groq_over_gemini_in_market_score():
    chunk = _chunk(files=["src/util/helpers.py"], tokens=800, lines_added=20)
    scheduled = ScheduledChunk(
        chunk=chunk,
        priority=75,
        complexity=complexity_score(chunk, prompt_overhead_tokens=2000),
    )
    groq = ProviderStatus(
        name="groq",
        health=100,
        rpm_remaining=10,
        max_context_tokens=7000,
        nominal_latency_ms=400,
        quality_prior=0.9,
    )
    gemini = ProviderStatus(
        name="gemini",
        health=100,
        rpm_remaining=10,
        max_context_tokens=200000,
        nominal_latency_ms=1200,
        quality_prior=0.95,
    )
    sg = score_provider(groq, scheduled, prompt_overhead_tokens=2000, historical_quality=0.5)
    sm = score_provider(gemini, scheduled, prompt_overhead_tokens=2000, historical_quality=0.95)
    assert sg > sm


def test_security_chunk_prefers_gemini_in_market_score():
    chunk = _chunk(files=["src/auth/jwt.py"], tokens=800, lines_added=30)
    scheduled = ScheduledChunk(
        chunk=chunk,
        priority=100,
        complexity=complexity_score(chunk, prompt_overhead_tokens=2000),
    )
    groq = ProviderStatus(
        name="groq",
        health=100,
        rpm_remaining=10,
        max_context_tokens=7000,
        nominal_latency_ms=400,
        quality_prior=0.9,
    )
    gemini = ProviderStatus(
        name="gemini",
        health=100,
        rpm_remaining=10,
        max_context_tokens=200000,
        nominal_latency_ms=1200,
        quality_prior=0.85,
    )
    sg = score_provider(groq, scheduled, prompt_overhead_tokens=2000, historical_quality=0.5)
    sm = score_provider(gemini, scheduled, prompt_overhead_tokens=2000, historical_quality=0.5)
    assert sm > sg


def test_vacuous_high_score_needs_escalate_signal():
    from types import SimpleNamespace

    from bot.scheduling.scheduler import ReviewScheduler
    from bot.schemas import ReviewResult

    # Non-trivial source file: clean 10.0 should still look suspiciously empty.
    chunk = _chunk(files=["cli/agent/AgentWorker.js"], tokens=3461, lines_added=80)
    scheduled = ScheduledChunk(
        chunk=chunk,
        priority=60,
        complexity=complexity_score(chunk, prompt_overhead_tokens=2000),
    )
    result = ReviewResult(
        summary="No critical issues",
        issues=[],
        recommendations=[],
        score=10.0,
        latency_ms=100,
        model="groq",
        review_type="groq",
    )
    fake = SimpleNamespace(providers={"groq": object(), "gemini": object()})
    assert ReviewScheduler._is_vacuous_review(scheduled, result) is True
    assert ReviewScheduler._needs_escalate(fake, scheduled, "groq", result) is True


def test_docs_clean_review_does_not_vacuous_escalate():
    from types import SimpleNamespace

    from bot.scheduling.complexity import expected_review_difficulty
    from bot.scheduling.priority import prioritize_chunk
    from bot.scheduling.scheduler import ReviewScheduler
    from bot.schemas import ReviewResult

    chunk = _chunk(
        files=["docs/NEXT_PHASE.md", "README.md", "docs/ROADMAP.md"],
        tokens=2570,
        lines_added=174,
        lines_deleted=10,
    )
    scheduled = ScheduledChunk(
        chunk=chunk,
        priority=prioritize_chunk(chunk),
        complexity=complexity_score(chunk, prompt_overhead_tokens=2500),
    )
    result = ReviewResult(
        summary="Docs look good",
        issues=[],
        recommendations=[],
        score=10.0,
        latency_ms=100,
        model="groq",
        review_type="groq",
    )
    fake = SimpleNamespace(providers={"groq": object(), "gemini": object()})
    difficulty = expected_review_difficulty(
        chunk, complexity=scheduled.complexity, priority=scheduled.priority
    )
    assert difficulty < 0.42
    assert ReviewScheduler._is_vacuous_review(scheduled, result) is False
    assert ReviewScheduler._needs_escalate(fake, scheduled, "groq", result) is False


def test_auth_refactor_zero_issues_still_vacuous():
    from types import SimpleNamespace

    from bot.scheduling.complexity import expected_review_difficulty, is_surprisingly_empty_review
    from bot.scheduling.priority import prioritize_chunk
    from bot.scheduling.scheduler import ReviewScheduler
    from bot.schemas import ReviewResult

    chunk = _chunk(
        files=["src/auth/middleware.py", "src/auth/jwt.py"],
        tokens=900,
        lines_added=120,
    )
    pri = prioritize_chunk(chunk)
    cx = complexity_score(chunk, prompt_overhead_tokens=2000)
    difficulty = expected_review_difficulty(chunk, complexity=cx, priority=pri)
    assert difficulty >= 0.42
    assert is_surprisingly_empty_review(
        chunk, score=10.0, issue_count=0, complexity=cx, priority=pri
    )
    scheduled = ScheduledChunk(chunk=chunk, priority=pri, complexity=cx)
    result = ReviewResult(
        summary="clean",
        issues=[],
        recommendations=[],
        score=10.0,
        latency_ms=1,
        model="groq",
        review_type="groq",
    )
    fake = SimpleNamespace(providers={"groq": object(), "gemini": object()})
    assert ReviewScheduler._needs_escalate(fake, scheduled, "groq", result) is True
    # Auth paths escalate as security first; still must not skip deeper review.
    assert ReviewScheduler._escalate_reason(fake, scheduled, result) in {
        "security",
        "vacuous",
        "complexity",
    }


def test_soften_incomplete_triage_caps_fake_perfect_score():
    from bot.scheduling.scheduler import ReviewScheduler
    from bot.schemas import ReviewResult

    result = ReviewResult(
        summary="All good",
        issues=[],
        recommendations=[],
        score=10.0,
        latency_ms=100,
        model="groq",
        review_type="groq",
    )
    softened = ReviewScheduler._soften_incomplete_triage(result)
    assert softened.score <= 7.0
    assert softened.parse_warning and "escalation_unavailable" in softened.parse_warning
    assert any("/sorge" in r for r in softened.recommendations)


def _hard_code_chunk() -> ReviewChunk:
    """Non-security source chunk with expected difficulty above vacuous threshold.

    Complexity stays below COMPLEXITY_ESCALATE so escalate reason is ``vacuous``,
    not ``complexity`` / ``security``.
    """
    lines = [
        "+++ b/cli/agent/AgentWorker.js",
        "@@ -1,0 +1,40 @@",
    ]
    for _ in range(40):
        lines.extend(
            [
                "+async function runAgent(job) {",
                "+  const tools = job.tools || [];",
                "+  for (const t of tools) { await invoke(t); }",
                "+  return { ok: true };",
                "+}",
            ]
        )
    raw = "\n".join(lines)
    return ReviewChunk(
        files=["cli/agent/AgentWorker.js"],
        parsed_diff=ParsedDiff(
            raw=raw,
            files=["cli/agent/AgentWorker.js"],
            lines_added=200,
            lines_deleted=0,
            files_changed=1,
        ),
        estimated_tokens=1200,
    )


def test_vacuous_clean_hard_review_escalates_via_gemini_multiplex():
    """Positive path: Groq 10.0 / 0 issues on hard code → vacuous → Gemini multiplex."""
    import time

    from bot.quota_tracker import QuotaTracker
    from bot.scheduling.health import HealthTracker
    from bot.scheduling.run_context import ProviderRuntime, RunContext
    from bot.scheduling.scheduler import ReviewScheduler
    from bot.scheduling.token_bucket import TokenBucket
    from bot.scheduling.types import ProviderResult
    from bot.schemas import ReviewResult

    escalate_tickets: list = []
    calls = {"groq": 0, "gemini_review": 0, "gemini_multiplex": 0}

    class FakeGroq:
        name = "groq"
        max_context_tokens = 7000
        cost_tier = "free"
        nominal_latency_ms = 200
        quality_prior = 0.9

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run):
            calls["groq"] += 1
            return ProviderResult(
                ok=True,
                provider="groq",
                status_code=200,
                result=ReviewResult(
                    summary="No issues found",
                    issues=[],
                    recommendations=[],
                    score=10.0,
                    latency_ms=50,
                    model="openai/gpt-oss-120b",
                    review_type="groq",
                ),
            )

    class FakeGeminiRunner:
        def review_escalate_batch(self, tickets):
            calls["gemini_multiplex"] += 1
            escalate_tickets.extend(tickets)
            return {
                t.ticket_id: ReviewResult(
                    summary="Deeper review found a concurrency concern",
                    issues=[],
                    recommendations=["add locking"],
                    score=8.5,
                    latency_ms=120,
                    model="gemini-2.5-flash",
                    review_type="gemini",
                )
                for t in tickets
            }

    class FakeGemini:
        name = "gemini"
        max_context_tokens = 200000
        cost_tier = "free"
        nominal_latency_ms = 800
        quality_prior = 0.95
        _runner = FakeGeminiRunner()

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run):
            calls["gemini_review"] += 1
            return ProviderResult(
                ok=True,
                provider="gemini",
                status_code=200,
                result=ReviewResult(
                    summary="should not be primary",
                    issues=[],
                    recommendations=[],
                    score=9.0,
                    latency_ms=80,
                    model="gemini-2.5-flash",
                    review_type="gemini",
                ),
            )

    chunk = _hard_code_chunk()
    cx = complexity_score(chunk, prompt_overhead_tokens=2500)
    assert cx < 0.68
    assert ReviewScheduler._is_vacuous_review(
        ScheduledChunk(chunk=chunk, priority=60, complexity=cx),
        ReviewResult(
            summary="x",
            issues=[],
            recommendations=[],
            score=10.0,
            latency_ms=1,
            model="groq",
            review_type="groq",
        ),
    )

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
                max_context_tokens=7000,
                max_inflight=1,
                nominal_latency_ms=200,
                quality_prior=0.9,
            ),
            "gemini": ProviderRuntime(
                name="gemini",
                bucket=TokenBucket(30),
                health=HealthTracker(100),
                max_context_tokens=200000,
                max_inflight=1,
                nominal_latency_ms=800,
                quality_prior=0.95,
            ),
        },
        quota=quota,
        deadline=time.monotonic() + 30,
        prompt_overhead_tokens=2500,
    )
    scheduler = ReviewScheduler([FakeGroq(), FakeGemini()], run, max_workers=1)
    results, skipped, meta = scheduler.run([chunk])

    assert not skipped
    assert calls["groq"] == 1
    assert calls["gemini_review"] == 0
    assert calls["gemini_multiplex"] == 1
    assert len(escalate_tickets) == 1
    assert escalate_tickets[0].reason == "vacuous"
    assert escalate_tickets[0].groq_score == 10.0
    assert meta.escalations == 1
    assert meta.escalate_multiplex_tickets == 1
    assert len(results) == 1
    assert results[0].review_type == "gemini"
    assert results[0].score == 8.5


def test_docs_clean_review_does_not_call_gemini_multiplex():
    """Regression path through scheduler.run: docs stay on Groq, no multiplex."""
    import time

    from bot.quota_tracker import QuotaTracker
    from bot.scheduling.health import HealthTracker
    from bot.scheduling.priority import prioritize_chunk
    from bot.scheduling.run_context import ProviderRuntime, RunContext
    from bot.scheduling.scheduler import ReviewScheduler
    from bot.scheduling.token_bucket import TokenBucket
    from bot.scheduling.types import ProviderResult
    from bot.schemas import ReviewResult

    calls = {"groq": 0, "gemini_multiplex": 0}

    class FakeGroq:
        name = "groq"
        max_context_tokens = 7000
        cost_tier = "free"
        nominal_latency_ms = 200
        quality_prior = 0.9

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run):
            calls["groq"] += 1
            return ProviderResult(
                ok=True,
                provider="groq",
                status_code=200,
                result=ReviewResult(
                    summary="Docs only",
                    issues=[],
                    recommendations=[],
                    score=10.0,
                    latency_ms=40,
                    model="openai/gpt-oss-120b",
                    review_type="groq",
                ),
            )

    class FakeGeminiRunner:
        def review_escalate_batch(self, tickets):
            calls["gemini_multiplex"] += 1
            return {}

    class FakeGemini:
        name = "gemini"
        max_context_tokens = 200000
        cost_tier = "free"
        nominal_latency_ms = 800
        quality_prior = 0.95
        _runner = FakeGeminiRunner()

        def advertise(self, run):
            return run.status(self.name)

        def review(self, chunk, run):
            raise AssertionError("docs must not primary-route to Gemini")

    chunk = _chunk(
        files=["docs/NEXT_PHASE.md", "README.md"],
        tokens=2570,
        lines_added=174,
        lines_deleted=10,
    )
    # Sanity: this shape is not surprisingly empty.
    pri = prioritize_chunk(chunk)
    cx = complexity_score(chunk, prompt_overhead_tokens=2500)
    assert not ReviewScheduler._is_vacuous_review(
        ScheduledChunk(chunk=chunk, priority=pri, complexity=cx),
        ReviewResult(
            summary="x",
            issues=[],
            recommendations=[],
            score=10.0,
            latency_ms=1,
            model="groq",
            review_type="groq",
        ),
    )

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
                max_context_tokens=7000,
                max_inflight=1,
                nominal_latency_ms=200,
                quality_prior=0.9,
            ),
            "gemini": ProviderRuntime(
                name="gemini",
                bucket=TokenBucket(30),
                health=HealthTracker(100),
                max_context_tokens=200000,
                max_inflight=1,
                nominal_latency_ms=800,
                quality_prior=0.95,
            ),
        },
        quota=quota,
        deadline=time.monotonic() + 30,
        prompt_overhead_tokens=2500,
    )
    scheduler = ReviewScheduler([FakeGroq(), FakeGemini()], run, max_workers=1)
    results, skipped, meta = scheduler.run([chunk])

    assert not skipped
    assert calls["groq"] == 1
    assert calls["gemini_multiplex"] == 0
    assert meta.escalations == 0
    assert len(results) == 1
    assert results[0].review_type == "groq"
    assert results[0].score == 10.0
