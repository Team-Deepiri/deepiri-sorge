"""Market-style provider scoring for a chunk.

Includes soft **lane affinity**: size + complexity/security bias so simple PRs
prefer Groq and auth/complex PRs prefer Gemini without hardcoding a winner.
"""

from __future__ import annotations

from bot.scheduling.complexity import (
    COMPLEXITY_ESCALATE,
    SECURITY_PATH_PRIORITY,
    complexity_score,
)
from bot.scheduling.types import ProviderStatus, ScheduledChunk


DEFAULT_WEIGHTS = {
    "health": 0.28,
    "latency": 0.18,
    "rpm": 0.12,
    "context": 0.12,
    "history": 0.10,
    "lane": 0.20,
}

# Review template + JSON schema instructions roughly consume this many tokens.
TEMPLATE_OVERHEAD_TOKENS = 2500

# Soft home lanes by effective prompt size (diff + overhead).
# Groq free tier is ~8k total; leave headroom for tokenizer variance / system bits.
LANE_GROQ_MAX = 7000
LANE_OPENROUTER_MAX = 100_000


def effective_tokens(scheduled: ScheduledChunk, prompt_overhead_tokens: int = 0) -> int:
    """Diff estimate + prompt/context overhead (what the provider actually receives)."""
    return max(0, scheduled.chunk.estimated_tokens) + max(0, prompt_overhead_tokens)


def home_lane(
    tokens: int,
    *,
    complexity: float = 0.0,
    path_priority: int = 50,
) -> str:
    """Preferred provider family for this effective size + complexity."""
    high = complexity >= COMPLEXITY_ESCALATE or path_priority >= SECURITY_PATH_PRIORITY
    if high:
        return "gemini"
    if tokens <= LANE_GROQ_MAX:
        return "groq"
    if tokens <= LANE_OPENROUTER_MAX:
        return "openrouter"
    return "gemini"


def lane_affinity(
    provider: str,
    tokens: int,
    *,
    complexity: float = 0.0,
    path_priority: int = 50,
) -> float:
    """1.0 on home lane; soft adjacent failover."""
    home = home_lane(tokens, complexity=complexity, path_priority=path_priority)
    if provider == home:
        return 1.0
    if home == "groq" and provider in ("openrouter", "gemini"):
        return 0.25
    if home == "openrouter" and provider == "gemini":
        return 0.45
    if home == "openrouter" and provider == "groq":
        return 0.2
    if home == "gemini" and provider == "openrouter":
        return 0.55
    if home == "gemini" and provider == "groq":
        # Allow Groq only if it clearly fits and complexity is mid
        if tokens <= LANE_GROQ_MAX and complexity < COMPLEXITY_ESCALATE:
            return 0.35
        return 0.05
    return 0.3


def context_fit(status: ProviderStatus, tokens: int) -> float:
    if tokens <= 0:
        return 1.0
    if tokens > status.max_context_tokens:
        return 0.0
    ratio = tokens / max(status.max_context_tokens, 1)
    return max(0.0, 1.0 - ratio * 0.5)


def latency_score(nominal_ms: float) -> float:
    return max(0.0, min(1.0, 1.0 - (nominal_ms / 2500.0)))


def pick_reason(
    provider: str,
    tokens: int,
    *,
    complexity: float = 0.0,
    path_priority: int = 50,
) -> str:
    home = home_lane(tokens, complexity=complexity, path_priority=path_priority)
    if path_priority >= SECURITY_PATH_PRIORITY:
        return "security"
    if complexity >= COMPLEXITY_ESCALATE:
        return "complexity"
    if provider == home:
        return "lane"
    return "failover"


def score_provider(
    status: ProviderStatus,
    scheduled: ScheduledChunk,
    *,
    historical_quality: float = 0.5,
    prompt_overhead_tokens: int = 0,
    weights: dict[str, float] | None = None,
) -> float:
    w = weights or DEFAULT_WEIGHTS
    tokens = effective_tokens(scheduled, prompt_overhead_tokens)
    fit = context_fit(status, tokens)
    if fit <= 0.0:
        return 0.0

    cx = scheduled.complexity
    if cx <= 0.0:
        cx = complexity_score(scheduled.chunk, prompt_overhead_tokens=prompt_overhead_tokens)

    health_n = max(0.0, min(1.0, status.health / 100.0))
    rpm_n = 0.0
    if status.rpm_remaining >= 1.0:
        rpm_n = min(1.0, 0.5 + 0.5 * min(status.rpm_remaining / 10.0, 1.0))
        if status.in_flight >= status.max_inflight:
            rpm_n *= 0.25

    lane = lane_affinity(
        status.name,
        tokens,
        complexity=cx,
        path_priority=scheduled.priority,
    )

    return (
        w.get("health", 0.3) * health_n
        + w.get("latency", 0.2) * latency_score(status.nominal_latency_ms)
        + w.get("rpm", 0.15) * rpm_n
        + w.get("context", 0.15) * fit
        + w.get("history", 0.05) * max(0.0, min(1.0, historical_quality))
        + w.get("lane", 0.15) * lane
    )
