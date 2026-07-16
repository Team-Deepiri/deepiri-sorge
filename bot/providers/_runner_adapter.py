"""Helpers shared by provider adapters."""

from __future__ import annotations

import time

import requests

from bot.file_splitter import ReviewChunk
from bot.scheduling.run_context import RunContext
from bot.scheduling.types import ProviderResult
from bot.schemas import ReviewResult


def run_runner_review(
    *,
    provider_name: str,
    runner,
    chunk: ReviewChunk,
    run: RunContext,
) -> ProviderResult:
    start = time.monotonic()
    try:
        result: ReviewResult | None = runner.review(
            chunk.parsed_diff,
            repo_context=run.repo_context,
            context_fingerprint=run.context_fingerprint,
        )
        latency = (time.monotonic() - start) * 1000
        if result and not result.parse_warning:
            return ProviderResult(
                ok=True,
                provider=provider_name,
                status_code=200,
                latency_ms=latency,
                result=result,
            )
        return ProviderResult(
            ok=False,
            provider=provider_name,
            status_code=200,
            latency_ms=latency,
            result=result,
            error=getattr(result, "parse_warning", None) or "empty_or_invalid_review",
        )
    except requests.Timeout:
        return ProviderResult(
            ok=False,
            provider=provider_name,
            timed_out=True,
            latency_ms=(time.monotonic() - start) * 1000,
            error="timeout",
        )
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        retry_after = None
        if exc.response is not None:
            raw = exc.response.headers.get("Retry-After")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
        return ProviderResult(
            ok=False,
            provider=provider_name,
            status_code=status,
            retry_after=retry_after,
            latency_ms=(time.monotonic() - start) * 1000,
            error=str(exc),
        )
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return ProviderResult(
            ok=False,
            provider=provider_name,
            status_code=status,
            latency_ms=(time.monotonic() - start) * 1000,
            error=str(exc),
        )
