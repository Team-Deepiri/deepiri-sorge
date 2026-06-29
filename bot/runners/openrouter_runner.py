"""OpenRouter runner using the OpenAI-compatible chat completions API."""

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


class OpenRouterRunner(BaseRunner):
    """Runner for OpenRouter-hosted models."""

    DEFAULT_MODEL = "google/gemma-4-31b-it:free"
    DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        cache_config: CacheConfig | None = None,
    ):
        super().__init__(api_key or os.getenv("OPENROUTER_API_KEY"), cache_config)
        self.model = model or self.DEFAULT_MODEL
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT

    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No OpenRouter API key configured")
            return None

        start_time = time.time()

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("OpenRouter request timed out")
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"OpenRouter request failed: {e}")
            return None

    def _call_api(self, diff: ParsedDiff, start_time: float) -> ReviewResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/deepiri/deepiri-sorge",
            "X-Title": "deepiri-sorge",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self._build_prompt(diff)}],
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        logger.debug(f"Calling OpenRouter with model: {self.model}")

        response = post_with_retry(self.endpoint, json=payload, headers=headers, timeout=120)
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens_used = data.get("usage", {}).get("total_tokens")

        parsed = self._parse_response(content)

        return self._build_result(
            parsed,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            review_type="openrouter",
        )

    def _timeout_result(self, start_time: float) -> ReviewResult:
        return ReviewResult(
            summary="OpenRouter request timed out - consider using a smaller diff",
            issues=[
                ReviewIssue(
                    severity="low",
                    file=None,
                    message="OpenRouter request timed out",
                    suggestion="Split large PRs into smaller changes",
                )
            ],
            recommendations=["Split large PRs into smaller changes", "Use a smaller model for quicker feedback"],
            score=5.0,
            latency_ms=(time.time() - start_time) * 1000,
            model=self.model,
            tokens_used=None,
        )