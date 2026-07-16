"""Market-style provider scoring for a chunk."""

from __future__ import annotations

from bot.scheduling.types import ProviderStatus, ScheduledChunk


DEFAULT_WEIGHTS = {
    "health": 0.35,
    "latency": 0.25,
    "rpm": 0.20,
    "context": 0.15,
    "history": 0.05,
}

# Review template + JSON schema instructions roughly consume this many tokens.
TEMPLATE_OVERHEAD_TOKENS = 2500


def effective_tokens(scheduled: ScheduledChunk, prompt_overhead_tokens: int = 0) -> int:
    """Diff estimate + prompt/context overhead (what the provider actually receives)."""
    return max(0, scheduled.chunk.estimated_tokens) + max(0, prompt_overhead_tokens)


def context_fit(status: ProviderStatus, tokens: int) -> float:
    if tokens <= 0:
        return 1.0
    if tokens > status.max_context_tokens:
        return 0.0
    # Prefer room to spare lightly
    ratio = tokens / max(status.max_context_tokens, 1)
    return max(0.0, 1.0 - ratio * 0.5)


def latency_score(nominal_ms: float) -> float:
    # Map ~200ms → ~0.9, ~2000ms → ~0.2
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
    # RPM: soft-cap remaining; penalize when already at max inflight
    rpm_n = 0.0
    if status.rpm_remaining >= 1.0:
        rpm_n = min(1.0, 0.5 + 0.5 * min(status.rpm_remaining / 10.0, 1.0))
        if status.in_flight >= status.max_inflight:
            rpm_n *= 0.25

    return (
        w["health"] * health_n
        + w["latency"] * latency_score(status.nominal_latency_ms)
        + w["rpm"] * rpm_n
        + w["context"] * fit
        + w["history"] * max(0.0, min(1.0, historical_quality))
    )
