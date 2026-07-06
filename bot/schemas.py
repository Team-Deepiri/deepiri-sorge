"""Shared review result types."""

from __future__ import annotations

from dataclasses import dataclass

from bot.utils.response_parser import normalize_review_payload


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
    parse_warning: str | None = None
    routing_meta: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "summary": self.summary,
            "parse_warning": self.parse_warning,
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
        if self.routing_meta:
            d["routing_meta"] = self.routing_meta
        return d


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
    data = dict(parsed)
    parse_warning = data.pop("_parse_warning", None) or data.get("parse_warning")
    data = normalize_review_payload(data)

    score = data.get("score", 7.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 7.0

    return ReviewResult(
        summary=data.get("summary", "Review complete"),
        issues=issues_from_parsed(data),
        recommendations=data.get("recommendations", []),
        score=score,
        latency_ms=latency_ms,
        model=model,
        tokens_used=tokens_used,
        review_type=review_type,
        parse_warning=parse_warning,
    )
