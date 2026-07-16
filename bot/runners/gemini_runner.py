"""Gemini 2.5 Flash runner - uses Google AI Studio API"""

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


class GeminiRunner(BaseRunner):
    """Runner for Gemini 2.5 Flash via Google AI Studio API"""

    DEFAULT_MODEL = "gemini-2.5-flash"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cache_config: CacheConfig | None = None,
    ):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"), cache_config)
        self.model = model or self.DEFAULT_MODEL
        self._last_raw_response: str | None = None
        self._last_http_status: int | None = None
        self._last_retry_after: float | None = None
        self._last_timed_out: bool = False

    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No Google API key configured")
            return None

        start_time = time.time()
        self._last_http_status = None
        self._last_retry_after = None
        self._last_timed_out = False

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("Gemini request timed out")
            self._last_timed_out = True
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"Gemini request failed: {e}")
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
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": self._build_prompt(diff)}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 16384,
                "topP": 0.95,
                "topK": 40,
                "responseMimeType": "application/json",
                "responseSchema": SchemaEncoder.for_gemini(),
            },
        }

        logger.debug(f"Calling Gemini with model: {self.model}")

        response = post_with_retry(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180,
            max_retries=1,
        )
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()
        content = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        self._last_raw_response = content  # store for salvage on parse failure
        tokens_used = data.get("usageMetadata", {}).get("totalTokenCount")

        parsed = self._parse_response(content)

        return self._build_result(
            parsed,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            review_type="gemini",
        )

    def _timeout_result(self, start_time: float) -> ReviewResult:
        return ReviewResult(
            summary="Gemini request timed out - the diff may be too large",
            issues=[
                ReviewIssue(
                    severity="medium",
                    file=None,
                    message="Request timed out - consider splitting the PR",
                    suggestion="Break large PRs into smaller, focused changes",
                )
            ],
            recommendations=["Split large PRs into smaller changes", "Review files individually for very large changes"],
            score=4.0,
            latency_ms=(time.time() - start_time) * 1000,
            model=self.model,
            tokens_used=None,
        )