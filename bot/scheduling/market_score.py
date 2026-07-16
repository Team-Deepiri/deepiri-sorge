"""Market-style provider scoring for a chunk.

Includes soft **lane affinity**: when Groq can fit the prompt, prefer it over
OpenRouter/Gemini so free-tier OR/Gemini RPM is not pecked on small PRs.
"""

from __future__ import annotations

from bot.scheduling.types import ProviderStatus, ScheduledChunk


DEFAULT_WEIGHTS = {
    "health": 0.30,
    "latency": 0.20,
    "rpm": 0.15,
    "context": 0.15,
    "history": 0.05,
    "lane": 0.15,
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


def home_lane(tokens: int) -> str:
    """Preferred provider family for this effective size."""
    if tokens <= LANE_GROQ_MAX:
        return "groq"
    if tokens <= LANE_OPENROUTER_MAX:
        return "openrouter"
    return "gemini"


def lane_affinity(provider: str, tokens: int) -> float:
    """1.0 on home lane, 0.35 on adjacent, 0.0 when clearly wrong."""
    home = home_lane(tokens)
    if provider == home:
        return 1.0
    # Adjacent: allow failover without preferring them first
    if home == "groq" and provider in ("openrouter", "gemini"):
        return 0.25
    if home == "openrouter" and provider == "gemini":
        return 0.45
    # context_fit already zeros Groq when over max_context; soft affinity for edge cases
    if home == "openrouter" and provider == "groq":
        return 0.2
    if home == "gemini" and provider == "openrouter":
        return 0.4
    if home == "gemini" and provider == "groq":
        return 0.0
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

    health_n = max(0.0, min(1.0, status.health / 100.0))
    rpm_n = 0.0
    if status.rpm_remaining >= 1.0:
        rpm_n = min(1.0, 0.5 + 0.5 * min(status.rpm_remaining / 10.0, 1.0))
        if status.in_flight >= status.max_inflight:
            rpm_n *= 0.25

    lane = lane_affinity(status.name, tokens)

    return (
        w.get("health", 0.3) * health_n
        + w.get("latency", 0.2) * latency_score(status.nominal_latency_ms)
        + w.get("rpm", 0.15) * rpm_n
        + w.get("context", 0.15) * fit
        + w.get("history", 0.05) * max(0.0, min(1.0, historical_quality))
        + w.get("lane", 0.15) * lane
    )
