"""Base runner class for model runners"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from bot.diff_parser import ParsedDiff


@dataclass
class ReviewResult:
    summary: str
    issues: list
    recommendations: list
    score: float
    latency_ms: float
    model: str
    tokens_used: int | None = None
    review_type: str = "api"

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "issues": [
                {
                    "severity": i.severity,
                    "file": i.file,
                    "line": i.line,
                    "message": i.message,
                    "rule": i.rule,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "recommendations": self.recommendations,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "review_type": self.review_type,
        }


class BaseRunner(ABC):
    """Abstract base class for model runners, with optional caching."""

    def __init__(self, api_key: str | None = None, cache_config=None):
        self.api_key = api_key
        self._cache_config = cache_config  # bot.config.CacheConfig or None

    # ------------------------------------------------------------------
    # Public entry point — handles cache read/write around _call_api
    # ------------------------------------------------------------------

    def review(self, diff: ParsedDiff) -> ReviewResult | None:
        if self._cache_config and self._cache_config.enabled:
            from bot.utils import cache as _cache
            cached = _cache.get(diff.raw, self.model, self._cache_config.ttl_hours)
            if cached is not None:
                return self._result_from_dict(cached)

        result = self._run_review(diff)

        if result is not None and self._cache_config and self._cache_config.enabled:
            from bot.utils import cache as _cache
            _cache.set(diff.raw, self.model, result.to_dict())

        return result

    @abstractmethod
    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        """Subclasses implement the actual API call here."""
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, diff: ParsedDiff) -> str:
        return f"""You are an expert code reviewer. Analyze the following code diff and provide a detailed review.

Provide your response in JSON format with the following structure:
{{
    "summary": "Brief overview of changes",
    "issues": [
        {{
            "severity": "critical|high|medium|low",
            "file": "filename",
            "line": line_number_or_null,
            "message": "issue description",
            "rule": "security|performance|style|best_practice",
            "suggestion": "how to fix"
        }}
    ],
    "recommendations": ["recommendation1", "recommendation2"],
    "score": 1-10
}}

DIFF:
{diff.raw}

Files changed: {", ".join(diff.files)}
Total lines: +{diff.lines_added} -{diff.lines_deleted}"""

    def _parse_response(self, response_text: str) -> dict:
        import json
        import re

        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "summary": response_text[:500],
            "issues": [],
            "recommendations": [response_text[:500]],
            "score": 5.0,
        }

    def _result_from_dict(self, data: dict) -> ReviewResult:
        """Reconstruct a ReviewResult from a cached to_dict() payload."""
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
            for i in data.get("issues", [])
        ]
        return ReviewResult(
            summary=data.get("summary", ""),
            issues=issues,
            recommendations=data.get("recommendations", []),
            score=data.get("score", 7.0),
            latency_ms=data.get("latency_ms", 0.0),
            model=data.get("model", self.model),
            tokens_used=data.get("tokens_used"),
            review_type=data.get("review_type", "api"),
        )