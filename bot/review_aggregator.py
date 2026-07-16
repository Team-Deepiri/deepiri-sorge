"""Merge multi-chunk review results into one PR comment payload."""

from __future__ import annotations

from bot.runners.base import ReviewResult
from bot.schemas import ReviewIssue
from bot.file_splitter import ReviewChunk
from bot.scheduling.types import SchedulerMeta, SkipRecord


def _is_rate_limit_skip(reason: str) -> bool:
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
        )
    )


def _skip_issue(skip: SkipRecord) -> ReviewIssue:
    return ReviewIssue(
        severity="low",
        file=skip.chunk.files[0] if skip.chunk.files else None,
        message=f"Chunk skipped by scheduler: {skip.reason}",
        suggestion="Re-run /sorge when provider rate limits recover",
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

            meta = {
                "rung": rung,
                "chunks": 0,
                "quota": quota_snapshot,
                "unreviewable": len(unreviewable or []),
                "skipped": len(skipped or []),
            }
            if scheduler_meta:
                meta["scheduler"] = (
                    scheduler_meta.to_dict()
                    if hasattr(scheduler_meta, "to_dict")
                    else scheduler_meta
                )

            rate_only = (
                bool(skipped)
                and not (unreviewable or [])
                and all(_is_rate_limit_skip(s.reason) for s in skipped)
            )
            if rate_only:
                return ReviewResult(
                    summary=(
                        "Review deferred — free-tier provider rate limits were hit before "
                        "any chunk could be reviewed. This is not a code-quality score."
                    ),
                    issues=[],
                    recommendations=[
                        "Wait a few minutes for RPM quotas to recover, then comment `/sorge` again",
                    ],
                    score=0.0,
                    latency_ms=0.0,
                    model="none",
                    review_type="rate_limited",
                    routing_meta=meta,
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
                    routing_meta=meta,
                )
            return ReviewResult(
                summary="No review results produced.",
                issues=[],
                recommendations=["Re-run the review or reduce PR size."],
                score=0.0,
                latency_ms=0.0,
                model="none",
                review_type="aggregated",
                routing_meta=meta,
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
