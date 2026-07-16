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


def test_emotion81_effective_tokens_fit_groq_lane():
    """~3461 + ~2604 must stay in Groq lane when complexity is low."""
    chunk = _chunk(files=["cli/agent/AgentWorker.js"], tokens=3461, lines_added=80)
    overhead = 2604
    scheduled = ScheduledChunk(
        chunk=chunk,
        priority=60,
        complexity=complexity_score(chunk, prompt_overhead_tokens=overhead),
    )
    assert scheduled.complexity < 0.6
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

    chunk = _chunk(files=["cli/agent/AgentWorker.js"], tokens=3461, lines_added=80)
    scheduled = ScheduledChunk(chunk=chunk, priority=60, complexity=0.43)
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
