"""OpenRouter runner using the OpenAI-compatible chat completions API."""

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
        http_retries: int = 3,
        http_timeout: int = 120,
        use_structured_output: bool = True,
    ):
        super().__init__(api_key or os.getenv("OPENROUTER_API_KEY"), cache_config)
        self.model = model or self.DEFAULT_MODEL
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.http_retries = http_retries
        self.http_timeout = http_timeout
        self.use_structured_output = use_structured_output
        self._last_raw_response: str | None = None
        self._last_http_status: int | None = None
        self._last_retry_after: float | None = None
        self._last_timed_out: bool = False

    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No OpenRouter API key configured")
            return None

        start_time = time.time()
        self._last_http_status = None
        self._last_retry_after = None
        self._last_timed_out = False

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("OpenRouter request timed out")
            self._last_timed_out = True
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"OpenRouter request failed: {e}")
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
            "HTTP-Referer": "https://github.com/deepiri/deepiri-sorge",
            "X-Title": "deepiri-sorge",
        }

        if self.use_structured_output:
            system_msg = (
                "You are a code review bot. Respond with a single raw JSON object only. "
                "No markdown fences, no prose before or after the JSON."
            )
            response_format = SchemaEncoder.for_openai()
        else:
            # Inject schema as a prompt hint for models without native structured output
            schema_hint = f"\n\nYou MUST respond with valid JSON matching this schema:\n{SchemaEncoder.for_prompt_injection()}"
            system_msg = (
                "You are a code review bot. Respond with a single raw JSON object only. "
                "No markdown fences, no prose before or after the JSON."
                + schema_hint
            )
            response_format = None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": self._build_prompt(diff)},
            ],
            "temperature": 0.2,
            "max_tokens": 16384,
        }
        if response_format:
            payload["response_format"] = response_format

        logger.debug(f"Calling OpenRouter with model: {self.model}")

        response = self._post_openrouter(payload, headers)
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        self._last_raw_response = content  # store for salvage on parse failure
        finish_reason = choice.get("finish_reason")
        tokens_used = data.get("usage", {}).get("total_tokens")

        if finish_reason == "length":
            logger.warning("OpenRouter response truncated (finish_reason=length)")

        parsed = self._parse_response(content)

        return self._build_result(
            parsed,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            review_type="openrouter",
        )

    def _post_openrouter(self, payload: dict, headers: dict) -> requests.Response:
        try:
            return post_with_retry(
                self.endpoint, json=payload, headers=headers, timeout=self.http_timeout,
                max_retries=self.http_retries,
            )
        except requests.HTTPError as exc:
            if payload.get("response_format") and self._is_json_mode_rejected(exc):
                logger.warning("OpenRouter rejected json_object mode; retrying without it")
                retry_payload = dict(payload)
                retry_payload.pop("response_format", None)
                return post_with_retry(
                    self.endpoint, json=retry_payload, headers=headers, timeout=self.http_timeout,
                    max_retries=self.http_retries,
                )
            raise

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