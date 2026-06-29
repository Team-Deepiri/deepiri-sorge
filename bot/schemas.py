"""Shared review result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReviewIssue:
    severity: str
    file: str | None = None
    line: int | None = None
    message: str = ""
    rule: str | None = None
    suggestion: str | None = None


@dataclass
class ReviewResult:
    summary: str
    issues: list[ReviewIssue]
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


def issues_from_parsed(parsed: dict) -> list[ReviewIssue]:
    return [
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


def result_from_parsed(
    parsed: dict,
    *,
    latency_ms: float,
    model: str,
    tokens_used: int | None,
    review_type: str,
) -> ReviewResult:
    return ReviewResult(
        summary=parsed.get("summary", "Review complete"),
        issues=issues_from_parsed(parsed),
        recommendations=parsed.get("recommendations", []),
        score=parsed.get("score", 7.0),
        latency_ms=latency_ms,
        model=model,
        tokens_used=tokens_used,
        review_type=review_type,
    )
