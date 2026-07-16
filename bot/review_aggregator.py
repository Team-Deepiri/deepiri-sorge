"""Merge multi-chunk review results into one PR comment payload."""

from __future__ import annotations

from bot.runners.base import ReviewResult
from bot.schemas import ReviewIssue
from bot.file_splitter import ReviewChunk
from bot.scheduling.types import SchedulerMeta, SkipRecord


def _is_capacity_skip(reason: str) -> bool:
    """True when the skip means providers were unavailable — not a code verdict."""
    r = (reason or "").lower()
    return any(
        token in r
        for token in (
            "http_429",
            "429",
            "rate limit",
            "rate_limited",
            "rate limited",
            "no eligible provider",
            "providers_exhausted",
            "capacity",
            "empty_response",
            "empty_or_invalid",
            "truncated",
            "timeout",
            "retry_in_approx",
            "no_provider",
        )
    )


def _skip_issue(skip: SkipRecord) -> ReviewIssue:
    return ReviewIssue(
        severity="low",
        file=skip.chunk.files[0] if skip.chunk.files else None,
        message=f"Chunk skipped by scheduler: {skip.reason}",
        suggestion="Re-run /sorge when provider rate limits recover",
    )


def _no_provider_result(
    *,
    rung: str,
    quota_snapshot: dict | None,
    skipped: list[SkipRecord],
    unreviewable: list[ReviewChunk] | None,
    scheduler_meta: SchedulerMeta | dict | None,
) -> ReviewResult:
    meta = {
        "rung": rung,
        "chunks": 0,
        "quota": quota_snapshot,
        "unreviewable": len(unreviewable or []),
        "skipped": len(skipped or []),
        "final_state": "NO_PROVIDER_AVAILABLE",
    }
    if scheduler_meta:
        meta["scheduler"] = (
            scheduler_meta.to_dict()
            if hasattr(scheduler_meta, "to_dict")
            else scheduler_meta
        )

    retry_mins = None
    if scheduler_meta is not None:
        sched_dict = (
            scheduler_meta.to_dict()
            if hasattr(scheduler_meta, "to_dict")
            else scheduler_meta
        )
        retry_sec = (sched_dict or {}).get("retry_after_sec")
        if retry_sec and float(retry_sec) > 0:
            retry_mins = max(1, int((float(retry_sec) + 59) // 60))

    rec = (
        f"Wait ~{retry_mins} minute(s) for provider capacity to recover, "
        "then comment `/sorge` again"
        if retry_mins
        else "Wait a few minutes for provider capacity to recover, then comment `/sorge` again"
    )
    summary = (
        "All review providers were temporarily unavailable (rate limits / empty "
        "capacity responses). No automated review was generated — this is not a "
        "code-quality score."
    )
    if retry_mins:
        summary += f" Approximate retry window: ~{retry_mins} minute(s)."

    return ReviewResult(
        summary=summary,
        issues=[],
        recommendations=[rec],
        score=0.0,
        latency_ms=0.0,
        model="none",
        review_type="rate_limited",
        routing_meta=meta,
    )


class ReviewAggregator:

    @staticmethod
    def merge(
        results: list[ReviewResult],
        *,
        rung: str,
        quota_snapshot: dict | None = None,
        unreviewable: list[ReviewChunk] | None = None,
        skipped: list[SkipRecord] | None = None,
        scheduler_meta: SchedulerMeta | dict | None = None,
    ) -> ReviewResult:
        if not results:
            issues: list[ReviewIssue] = []
            for chunk in unreviewable or []:
                issues.append(
                    ReviewIssue(
                        severity="low",
                        file=chunk.files[0] if chunk.files else None,
                        message=chunk.unreviewable_reason or "File too large for automated review",
                        suggestion="Split this file into a smaller PR or review manually",
                    )
                )
            for skip in skipped or []:
                issues.append(_skip_issue(skip))

            # Capacity / rate-limit exhaustion → never invent a quality score.
            capacity_only = (
                bool(skipped)
                and not (unreviewable or [])
                and all(_is_capacity_skip(s.reason) for s in skipped)
            )
            if capacity_only or (
                not issues
                and scheduler_meta is not None
                and (
                    getattr(scheduler_meta, "stop_reason", None) == "providers_exhausted"
                    or (
                        isinstance(scheduler_meta, dict)
                        and scheduler_meta.get("stop_reason") == "providers_exhausted"
                    )
                )
            ):
                return _no_provider_result(
                    rung=rung,
                    quota_snapshot=quota_snapshot,
                    skipped=skipped or [],
                    unreviewable=unreviewable,
                    scheduler_meta=scheduler_meta,
                )

            if issues:
                return ReviewResult(
                    summary="Partial review — some files could not be processed.",
                    issues=issues,
                    recommendations=["Reduce PR size or split large generated files"],
                    score=0.0,
                    latency_ms=0.0,
                    model="none",
                    review_type="aggregated",
                    routing_meta={
                        "rung": rung,
                        "chunks": 0,
                        "quota": quota_snapshot,
                        "unreviewable": len(unreviewable or []),
                        "skipped": len(skipped or []),
                        **(
                            {
                                "scheduler": (
                                    scheduler_meta.to_dict()
                                    if hasattr(scheduler_meta, "to_dict")
                                    else scheduler_meta
                                )
                            }
                            if scheduler_meta
                            else {}
                        ),
                    },
                )
            return ReviewResult(
                summary="No review results produced.",
                issues=[],
                recommendations=["Re-run the review or reduce PR size."],
                score=0.0,
                latency_ms=0.0,
                model="none",
                review_type="aggregated",
                routing_meta={
                    "rung": rung,
                    "chunks": 0,
                    "quota": quota_snapshot,
                },
            )

        if len(results) == 1 and not unreviewable and not skipped:
            r = results[0]
            r.routing_meta = {
                "rung": rung,
                "chunks": 1,
                "quota": quota_snapshot,
            }
            if scheduler_meta:
                r.routing_meta["scheduler"] = (
                    scheduler_meta.to_dict()
                    if hasattr(scheduler_meta, "to_dict")
                    else scheduler_meta
                )
            return r

        all_issues: list[ReviewIssue] = []
        seen: set[tuple] = set()
        total_tokens = 0
        total_latency = 0.0
        weight_score = 0.0
        weight_sum = 0.0
        models: list[str] = []
        recommendations: list[str] = []
        summaries: list[str] = []

        for r in results:
            models.append(f"{r.review_type}:{r.model}")
            total_latency += r.latency_ms
            tok = r.tokens_used or 1
            total_tokens += tok
            weight_score += r.score * tok
            weight_sum += tok
            summaries.append(r.summary)
            recommendations.extend(r.recommendations or [])

            for issue in r.issues:
                key = (issue.file, issue.line, issue.message)
                if key in seen:
                    continue
                seen.add(key)
                all_issues.append(issue)

        for chunk in unreviewable or []:
            all_issues.append(
                ReviewIssue(
                    severity="low",
                    file=chunk.files[0] if chunk.files else None,
                    message=chunk.unreviewable_reason or "File too large for automated review",
                    suggestion="Split this file into a smaller PR or review manually",
                )
            )

        for skip in skipped or []:
            all_issues.append(_skip_issue(skip))

        score = weight_score / weight_sum if weight_sum else 7.0
        summary = (
            f"Multi-chunk review ({len(results)} parts, rung={rung}).\n\n"
            + "\n\n".join(f"- {s}" for s in summaries[:5])
        )

        rec_dedup = list(dict.fromkeys(recommendations))

        result = ReviewResult(
            summary=summary,
            issues=all_issues,
            recommendations=rec_dedup,
            score=round(score, 1),
            latency_ms=total_latency,
            model=", ".join(dict.fromkeys(models)),
            tokens_used=total_tokens,
            review_type="aggregated",
        )
        result.routing_meta = {
            "rung": rung,
            "chunks": len(results),
            "quota": quota_snapshot,
            "unreviewable": len(unreviewable or []),
            "skipped": len(skipped or []),
        }
        if scheduler_meta:
            result.routing_meta["scheduler"] = (
                scheduler_meta.to_dict()
                if hasattr(scheduler_meta, "to_dict")
                else scheduler_meta
            )
        return result
