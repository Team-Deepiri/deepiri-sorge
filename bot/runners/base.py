"""Base runner class for model runners"""

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
    """Abstract base class for model runners"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @abstractmethod
    def review(self, diff: ParsedDiff) -> ReviewResult | None:
        """Run review on the diff"""
        pass

    def _build_prompt(self, diff: ParsedDiff) -> str:
        """Build the review prompt from diff"""
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
        """Parse JSON response from model"""
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