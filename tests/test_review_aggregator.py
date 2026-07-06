"""Tests for review aggregator."""

from bot.review_aggregator import ReviewAggregator
from bot.schemas import ReviewIssue, ReviewResult
from bot.file_splitter import ReviewChunk
from bot.diff_parser import ParsedDiff


def _result(score: float, review_type: str) -> ReviewResult:
    return ReviewResult(
        summary=f"summary {review_type}",
        issues=[ReviewIssue(severity="low", file="a.py", message=f"issue-{review_type}")],
        recommendations=[f"rec-{review_type}"],
        score=score,
        latency_ms=100.0,
        model="m",
        tokens_used=100,
        review_type=review_type,
    )


def test_merge_single_result():
    r = _result(8.0, "openrouter")
    merged = ReviewAggregator.merge([r], rung="standard")
    assert merged.summary == r.summary
    assert len(merged.issues) == 1


def test_merge_dedupes_issues():
    a = _result(8.0, "groq")
    b = ReviewResult(
        summary="b",
        issues=[ReviewIssue(severity="low", file="a.py", message="issue-groq")],
        recommendations=[],
        score=6.0,
        latency_ms=50.0,
        model="m2",
        tokens_used=50,
        review_type="openrouter",
    )
    merged = ReviewAggregator.merge([a, b], rung="oversized")
    assert len(merged.issues) == 1
    assert merged.routing_meta["chunks"] == 2


def test_unreviewable_adds_info_issue():
    chunk = ReviewChunk(
        files=["huge.py"],
        parsed_diff=ParsedDiff(raw=""),
        estimated_tokens=999_999,
        unreviewable=True,
        unreviewable_reason="too big",
    )
    merged = ReviewAggregator.merge([], rung="oversized", unreviewable=[chunk])
    assert any("too big" in i.message for i in merged.issues)
