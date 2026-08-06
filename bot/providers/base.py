"""Provider Protocol — execution backends for the review scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bot.file_splitter import ReviewChunk
from bot.scheduling.types import ProviderResult, ProviderStatus

if TYPE_CHECKING:
    from bot.scheduling.run_context import RunContext


@runtime_checkable
class Provider(Protocol):
    name: str
    max_context_tokens: int
    cost_tier: str
    nominal_latency_ms: float
    quality_prior: float

    def advertise(self, run: RunContext) -> ProviderStatus: ...

    def review(
        self,
        chunk: ReviewChunk,
        run: RunContext,
        *,
        prior_partial: str | None = None,
    ) -> ProviderResult: ...
