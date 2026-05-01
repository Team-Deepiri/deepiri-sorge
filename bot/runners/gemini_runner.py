"""Gemini 2.5 Pro runner - uses Google AI Studio API"""

import os
import time

import requests
from loguru import logger

from bot.diff_parser import ParsedDiff
from bot.runners.base import BaseRunner, ReviewResult


class GeminiRunner(BaseRunner):
    """Runner for Gemini 2.5 Pro via Google AI Studio API"""

    DEFAULT_MODEL = "gemini-2.5-flash"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = model or self.DEFAULT_MODEL

    def review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No Google API key configured")
            return None

        start_time = time.time()

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("Gemini request timed out")
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"Gemini request failed: {e}")
            return None

    def _call_api(self, diff: ParsedDiff, start_time: float) -> ReviewResult:
        prompt = self._build_prompt(diff)

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 8192,
                "topP": 0.95,
                "topK": 40,
            },
        }

        logger.debug(f"Calling Gemini with model: {self.model}")

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=180,
        )

        response.raise_for_status()
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()

        content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[
            0
        ].get("text", "")

        usage = data.get("usageMetadata", {})
        tokens_used = usage.get("totalTokenCount", None)

        parsed = self._parse_response(content)

        from bot.cpu_reviewer import ReviewIssue

        issues = [
            ReviewIssue(
                severity=i.get("severity", "medium"),
                file=i.get("file"),
                line=i.get("line"),
                message=i.get("message", ""),
                rule=i.get("rule"),
                suggestion=i.get("suggestion"),
            )
            for i in parsed.get("issues", [])
        ]

        return ReviewResult(
            summary=parsed.get("summary", "Review complete"),
            issues=issues,
            recommendations=parsed.get("recommendations", []),
            score=parsed.get("score", 7.0),
            latency_ms=latency_ms,
            model=self.model,
            tokens_used=tokens_used,
            review_type="gemini",
        )

    def _timeout_result(self, start_time: float) -> ReviewResult:
        from bot.cpu_reviewer import ReviewIssue

        latency_ms = (time.time() - start_time) * 1000

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
            recommendations=[
                "Split large PRs into smaller, focused changes",
                "Review files individually for very large changes",
            ],
            score=4.0,
            latency_ms=latency_ms,
            model=self.model,
            tokens_used=None,
        )