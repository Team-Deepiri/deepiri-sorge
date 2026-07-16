"""Groq provider adapter."""

from __future__ import annotations

from bot.config import CacheConfig, Config
from bot.file_splitter import ReviewChunk
from bot.providers._runner_adapter import run_runner_review
from bot.runners.groq_runner import GroqRunner
from bot.scheduling.run_context import RunContext
from bot.scheduling.types import ProviderResult, ProviderStatus


class GroqProvider:
    name = "groq"
    cost_tier = "free"
    nominal_latency_ms = 400.0
    quality_prior = 0.9

    def __init__(self, config: Config, cache_config: CacheConfig | None = None):
        runtime = config.providers.groq
        self.max_context_tokens = runtime.max_context_tokens
        self.quality_prior = runtime.quality_prior
        self.nominal_latency_ms = runtime.nominal_latency_ms
        self._runner = GroqRunner(
            api_key=config.groq.api_key,
            model=config.groq.model,
            endpoint=config.groq.endpoint,
            cache_config=None,  # scheduler owns caching later; Phase 1: no double cache fan-out
        )

    def advertise(self, run: RunContext) -> ProviderStatus:
        status = run.status(self.name)
        if status:
            return status
        return ProviderStatus(
            name=self.name,
            health=100.0,
            rpm_remaining=0.0,
            max_context_tokens=self.max_context_tokens,
            nominal_latency_ms=self.nominal_latency_ms,
            quality_prior=self.quality_prior,
        )

    def review(self, chunk: ReviewChunk, run: RunContext) -> ProviderResult:
        self._runner._last_http_status = None
        self._runner._last_retry_after = None
        result = run_runner_review(
            provider_name=self.name,
            runner=self._runner,
            chunk=chunk,
            run=run,
        )
        if not result.ok and result.status_code is None:
            status = getattr(self._runner, "_last_http_status", None)
            retry_after = getattr(self._runner, "_last_retry_after", None)
            timed_out = getattr(self._runner, "_last_timed_out", False)
            return ProviderResult(
                ok=False,
                provider=self.name,
                status_code=status,
                retry_after=retry_after,
                latency_ms=result.latency_ms,
                result=result.result,
                error=result.error,
                timed_out=bool(timed_out),
            )
        return result
