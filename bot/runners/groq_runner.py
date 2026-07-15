"""Groq runner using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import os
import time

import requests
from loguru import logger

from bot.config import CacheConfig
from bot.diff_parser import ParsedDiff
from bot.runners.base import BaseRunner, ReviewResult
from bot.schemas import ReviewIssue
from bot.utils.http_retry import post_with_retry


class GroqRunner(BaseRunner):
    """Runner for Groq-hosted models (currently GPT OSS 120B)."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        cache_config: CacheConfig | None = None,
    ):
        super().__init__(api_key or os.getenv("GROQ_API_KEY"), cache_config)
        self.model = model or self.DEFAULT_MODEL
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self._last_raw_response: str | None = None
        self._last_http_status: int | None = None
        self._last_retry_after: float | None = None
        self._last_timed_out: bool = False

    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No Groq API key configured")
            return None

        start_time = time.time()
        self._last_http_status = None
        self._last_retry_after = None
        self._last_timed_out = False

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("Groq request timed out")
            self._last_timed_out = True
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"Groq request failed: {e}")
            response = getattr(e, "response", None)
            self._last_http_status = getattr(response, "status_code", None)
            if response is not None:
                raw = response.headers.get("Retry-After")
                if raw:
                    try:
                        self._last_retry_after = float(raw)
                    except ValueError:
                        self._last_retry_after = None
            return None

    def _call_api(self, diff: ParsedDiff, start_time: float) -> ReviewResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Prompt already requires raw JSON — skip response_format (often rejected).
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a code review bot. Respond with a single raw JSON object only. "
                        "No markdown fences, no prose before or after the JSON."
                    ),
                },
                {"role": "user", "content": self._build_prompt(diff)},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        logger.debug(f"Calling Groq with model: {self.model}")

        response = post_with_retry(self.endpoint, json=payload, headers=headers, timeout=120)
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        self._last_raw_response = content  # store for salvage on parse failure
        if choice.get("finish_reason") == "length":
            logger.warning("Groq response truncated (finish_reason=length)")
        tokens_used = data.get("usage", {}).get("total_tokens")

        parsed = self._parse_response(content)

        return self._build_result(
            parsed,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            review_type="groq",
        )

    def _timeout_result(self, start_time: float) -> ReviewResult:
        return ReviewResult(
            summary="Groq request timed out - consider using a smaller diff",
            issues=[
                ReviewIssue(
                    severity="low",
                    file=None,
                    message="Groq request timed out",
                    suggestion="Split large PRs into smaller changes",
                )
            ],
            recommendations=["Split large PRs into smaller changes", "Use a smaller model for quicker feedback"],
            score=5.0,
            latency_ms=(time.time() - start_time) * 1000,
            model=self.model,
            tokens_used=None,
        )