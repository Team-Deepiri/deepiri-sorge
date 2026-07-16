"""Tests for review aggregator."""

from bot.review_aggregator import ReviewAggregator
from bot.schemas import ReviewIssue, ReviewResult
from bot.file_splitter import ReviewChunk
from bot.diff_parser import ParsedDiff
from bot.scheduling.types import SkipRecord


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


def test_rate_limit_only_skips_are_not_quality_zero():
    chunk = ReviewChunk(
        files=["src/a.py"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=1000,
    )
    skipped = [SkipRecord(chunk, "http_429")]
    merged = ReviewAggregator.merge([], rung="scheduled", skipped=skipped)
    assert merged.review_type == "rate_limited"
    assert merged.issues == []
    assert "not a code-quality score" in merged.summary
    assert any("/sorge" in r for r in merged.recommendations)
    assert merged.routing_meta.get("final_state") == "NO_PROVIDER_AVAILABLE"


def test_empty_response_after_stampede_is_not_quality_zero():
    """emotion#81 failure mode: truncated/empty after 429s must not invent 0.0 redesign."""
    chunk = ReviewChunk(
        files=["cli/agent/AgentWorker.js"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=3461,
    )
    skipped = [SkipRecord(chunk, "capacity:empty_response")]
    merged = ReviewAggregator.merge(
        [],
        rung="scheduled",
        skipped=skipped,
        scheduler_meta={"retry_after_sec": 90, "stop_reason": None},
    )
    assert merged.review_type == "rate_limited"
    assert merged.issues == []
    assert "No automated review was generated" in merged.summary
    assert merged.routing_meta.get("final_state") == "NO_PROVIDER_AVAILABLE"
