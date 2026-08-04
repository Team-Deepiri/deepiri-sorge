"""Helpers shared by provider adapters."""

from __future__ import annotations

import time

import requests

from bot.file_splitter import ReviewChunk
from bot.scheduling.run_context import RunContext
from bot.scheduling.types import MAX_PARTIAL_CHARS, ProviderResult
from bot.schemas import ReviewResult


# A fragment shorter than this carries no findings worth forwarding.
_MIN_PARTIAL_CHARS = 200


def _salvage_partial(runner) -> str | None:
    """Return the runner's last raw response if it's a usable cut-off fragment."""
    raw = getattr(runner, "_last_raw_response", None)
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if len(text) < _MIN_PARTIAL_CHARS:
        return None
    return text[:MAX_PARTIAL_CHARS]


def run_runner_review(
    *,
    provider_name: str,
    runner,
    chunk: ReviewChunk,
    run: RunContext,
    prior_partial: str | None = None,
) -> ProviderResult:
    start = time.monotonic()
    try:
        result: ReviewResult | None = runner.review(
            chunk.parsed_diff,
            repo_context=run.repo_context,
            context_fingerprint=run.context_fingerprint,
            prior_partial=prior_partial,
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
        # Runners often swallow HTTP errors and return None while stashing status
        # on the runner — surface that so the scheduler can act on 413/429.
        last_status = getattr(runner, "_last_http_status", None)
        last_retry = getattr(runner, "_last_retry_after", None)
        if result is None and last_status is not None:
            return ProviderResult(
                ok=False,
                provider=provider_name,
                status_code=last_status,
                retry_after=last_retry,
                latency_ms=latency,
                error=f"http_{last_status}",
                timed_out=bool(getattr(runner, "_last_timed_out", False)),
            )
        # HTTP 200 but unusable — typically finish_reason=length. The text the
        # model did produce is the whole point of forwarding.
        return ProviderResult(
            ok=False,
            provider=provider_name,
            status_code=200,
            latency_ms=latency,
            result=result,
            error=getattr(result, "parse_warning", None) or "empty_or_invalid_review",
            partial_output=_salvage_partial(runner),
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
