"""Gemini provider adapter."""

from __future__ import annotations

from bot.config import CacheConfig, Config
from bot.file_splitter import ReviewChunk
from bot.providers._runner_adapter import run_runner_review
from bot.runners.gemini_runner import GeminiRunner
from bot.scheduling.run_context import RunContext
from bot.scheduling.types import ProviderResult, ProviderStatus


class GeminiProvider:
    name = "gemini"
    cost_tier = "free"
    nominal_latency_ms = 1200.0
    quality_prior = 0.85

    def __init__(self, config: Config, cache_config: CacheConfig | None = None):
        runtime = config.providers.gemini
        self.max_context_tokens = runtime.max_context_tokens
        self.quality_prior = runtime.quality_prior
        self.nominal_latency_ms = runtime.nominal_latency_ms
        self._runner = GeminiRunner(
            api_key=config.gemini.api_key,
            model=config.gemini.model,
            cache_config=None,
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
        self._runner._last_timed_out = False
        result = run_runner_review(
            provider_name=self.name,
            runner=self._runner,
            chunk=chunk,
            run=run,
        )
        if not result.ok and result.status_code is None:
            return ProviderResult(
                ok=False,
                provider=self.name,
                status_code=getattr(self._runner, "_last_http_status", None),
                retry_after=getattr(self._runner, "_last_retry_after", None),
                latency_ms=result.latency_ms,
                result=result.result,
                error=result.error,
                timed_out=bool(getattr(self._runner, "_last_timed_out", False)),
            )
        return result
