"""GitHub Models runner - uses Azure AI Inference API"""

import os
import time

import requests
from loguru import logger

from bot.diff_parser import ParsedDiff
from bot.runners.base import BaseRunner, ReviewResult


class GitHubModelsRunner(BaseRunner):
    """Runner for GitHub Models (Azure AI Inference)"""

    ENDPOINT = "https://models.inference.ai.azure.com/chat/completions"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key or os.getenv("GITHUB_TOKEN"))
        self.model = model or self.DEFAULT_MODEL

    def review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No GitHub token configured")
            return None

        start_time = time.time()

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("GitHub Models request timed out")
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"GitHub Models request failed: {e}")
            return None

    def _call_api(self, diff: ParsedDiff, start_time: float) -> ReviewResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        prompt = self._build_prompt(diff)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        logger.debug(f"Calling GitHub Models with model: {self.model}")

        response = requests.post(
            self.ENDPOINT,
            json=payload,
            headers=headers,
            timeout=120,
        )

        response.raise_for_status()
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", None)

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
            review_type="github_models",
        )

    def _timeout_result(self, start_time: float) -> ReviewResult:
        from bot.cpu_reviewer import ReviewIssue

        latency_ms = (time.time() - start_time) * 1000

        return ReviewResult(
            summary="Request timed out - consider using Gemini for larger PRs",
            issues=[
                ReviewIssue(
                    severity="low",
                    file=None,
                    message="GitHub Models request timed out",
                    suggestion="Large PRs may benefit from Gemini's larger context",
                )
            ],
            recommendations=[
                "Split large PRs into smaller changes",
                "Use Gemini for complex, large diffs",
            ],
            score=5.0,
            latency_ms=latency_ms,
            model=self.model,
            tokens_used=None,
        )