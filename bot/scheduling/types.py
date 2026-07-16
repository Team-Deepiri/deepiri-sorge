"""Structured results for provider-backed review execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bot.file_splitter import ReviewChunk
from bot.schemas import ReviewResult


@dataclass
class ProviderResult:
    """Outcome of a single provider.review() call — no cross-provider fallback."""

    ok: bool
    provider: str
    status_code: int | None = None
    retry_after: float | None = None
    latency_ms: float = 0.0
    result: ReviewResult | None = None
    error: str | None = None
    timed_out: bool = False

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429

    @property
    def is_payload_too_large(self) -> bool:
        return self.status_code == 413


@dataclass
class ProviderStatus:
    name: str
    health: float
    rpm_remaining: float
    max_context_tokens: int
    nominal_latency_ms: float
    quality_prior: float
    cooling_until: float = 0.0
    in_flight: int = 0
    max_inflight: int = 1


@dataclass
class ScheduledChunk:
    chunk: ReviewChunk
    priority: int = 50
    attempted_providers: set[str] = field(default_factory=set)
    # After all providers 429, clear attempts and wait once (bounded).
    rate_limit_rounds: int = 0


MAX_RATE_LIMIT_ROUNDS = 3


@dataclass
class SkipRecord:
    chunk: ReviewChunk
    reason: str


@dataclass
class SchedulerMeta:
    rung: str = "scheduled"
    dispatches: int = 0
    cache_hits: int = 0
    provider_picks: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0
    stop_reason: str | None = None
    health_snapshot: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "dispatches": self.dispatches,
            "cache_hits": self.cache_hits,
            "provider_picks": self.provider_picks,
            "skipped": self.skipped,
            "stop_reason": self.stop_reason,
            "health": self.health_snapshot,
        }
