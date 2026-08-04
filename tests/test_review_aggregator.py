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


def test_non_json_response_is_not_quality_zero():
    chunk = ReviewChunk(
        files=["src/App.tsx"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=1000,
    )
    skipped = [SkipRecord(chunk, "non_json_response")]
    merged = ReviewAggregator.merge([], rung="scheduled", skipped=skipped)
    assert merged.review_type == "rate_limited"
    assert merged.routing_meta.get("final_state") == "NO_PROVIDER_AVAILABLE"


def _chunk(path="big.bin", reason=None) -> ReviewChunk:
    return ReviewChunk(
        files=[path],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=10,
        unreviewable_reason=reason,
    )


def test_deadline_skip_never_yields_a_scored_review():
    """deepiri-crankl#28: 'deadline' wasn't in the capacity reason list, so a
    run where zero chunks succeeded produced review_type='aggregated' and the
    adjudicator rescored the empty issue list to a perfect 10.0."""
    skipped = [
        SkipRecord(_chunk("src/mmap.cpp"), "wall-clock deadline (priority=50 not reached)")
    ]
    merged = ReviewAggregator.merge(
        [], rung="scheduled", skipped=skipped, scheduler_meta={"stop_reason": "deadline"}
    )

    assert merged.review_type == "rate_limited"
    assert merged.score == 0.0
    assert "not a code-quality score" in merged.summary


def test_unknown_skip_reason_still_yields_no_score():
    """The gate is 'did any chunk succeed', not 'does the reason string match'."""
    skipped = [SkipRecord(_chunk("a.py"), "something nobody has seen before")]
    merged = ReviewAggregator.merge([], rung="scheduled", skipped=skipped)

    from bot.schemas import is_no_score

    assert is_no_score(merged.review_type)
    assert merged.score == 0.0


def test_unreviewable_files_yield_no_result_not_a_score():
    unreviewable = [_chunk("generated.pb.go", reason="File too large for automated review")]
    merged = ReviewAggregator.merge([], rung="scheduled", unreviewable=unreviewable)

    assert merged.review_type == "no_result"
    assert merged.score == 0.0
    # Distinct from rate_limited: retrying will not make the file smaller.
    assert merged.routing_meta["final_state"] == "NO_CHUNK_REVIEWED"


def test_adjudicator_does_not_rescore_a_run_that_reviewed_nothing():
    """The other half of #28: the adjudicator dropped the lone skip-record and
    then recomputed a score from the resulting empty list, yielding 10.0."""
    from bot.finding_adjudicator import FindingAdjudicator

    unreviewable = [_chunk("generated.pb.go", reason="File too large")]
    merged = ReviewAggregator.merge([], rung="scheduled", unreviewable=unreviewable)
    assert merged.issues, "aggregator should emit a placeholder issue"
    assert merged.score == 0.0

    adjudicator = FindingAdjudicator.__new__(FindingAdjudicator)
    # Adjudicator drops the placeholder as not-a-real-finding, emptying the list.
    adjudicator._complete = lambda prompt: {
        "decisions": [{"index": 0, "action": "drop", "reason": "scheduler artifact"}]
    }

    out = adjudicator.adjudicate_result(merged, deploy_facts={})

    assert out.issues == []
    assert out.score == 0.0, "an empty kept list must not score 10.0"
    assert out.review_type == "no_result"


def test_adjudicator_still_rescores_a_real_review():
    from bot.finding_adjudicator import FindingAdjudicator

    real = _result(4.0, "groq")
    real.issues = [
        ReviewIssue(severity="critical", file="a.py", message="null deref"),
        ReviewIssue(severity="low", file="a.py", message="nit"),
    ]
    adjudicator = FindingAdjudicator.__new__(FindingAdjudicator)
    adjudicator._complete = lambda prompt: {
        "decisions": [{"index": 1, "action": "drop", "reason": "style nit"}]
    }

    out = adjudicator.adjudicate_result(real, deploy_facts={})

    assert len(out.issues) == 1
    # 10.0 - 2.5 for the remaining critical.
    assert out.score == 7.5
