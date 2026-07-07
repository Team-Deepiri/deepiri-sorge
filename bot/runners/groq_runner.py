"""Groq runner using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import os
import time

import requests
from loguru import logger

from bot.config import CacheConfig
from bot.diff_parser import ParsedDiff
from bot.runners.base import BaseRunner, ReviewResult
from bot.runners.json_schema import SchemaEncoder
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

    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No Groq API key configured")
            return None

        start_time = time.time()

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("Groq request timed out")
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"Groq request failed: {e}")
            return None

    def _call_api(self, diff: ParsedDiff, start_time: float) -> ReviewResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

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
            "response_format": SchemaEncoder.for_openai(),
        }

        logger.debug(f"Calling Groq with model: {self.model}")

        try:
            response = post_with_retry(self.endpoint, json=payload, headers=headers, timeout=120)
        except requests.HTTPError as exc:
            if payload.get("response_format") and self._is_json_mode_rejected(exc):
                logger.warning("Groq rejected json_object mode; retrying without it")
                payload = dict(payload)
                payload.pop("response_format", None)
                response = post_with_retry(self.endpoint, json=payload, headers=headers, timeout=120)
            else:
                raise
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
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

    @staticmethod
    def _is_json_mode_rejected(exc: requests.HTTPError) -> bool:
        response = exc.response
        if response is None:
            return False
        if response.status_code not in {400, 422}:
            return False
        body = (response.text or "").lower()
        return "response_format" in body or "json" in body

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