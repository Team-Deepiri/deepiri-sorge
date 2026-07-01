"""Merge multi-chunk review results into one PR comment payload."""

from __future__ import annotations

from bot.runners.base import ReviewResult
from bot.schemas import ReviewIssue
from bot.file_splitter import ReviewChunk


class ReviewAggregator:

    @staticmethod
    def merge(
        results: list[ReviewResult],
        *,
        rung: str,
        quota_snapshot: dict | None = None,
        unreviewable: list[ReviewChunk] | None = None,
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
            )

        if len(results) == 1 and not unreviewable:
            r = results[0]
            r.routing_meta = {
                "rung": rung,
                "chunks": 1,
                "quota": quota_snapshot,
            }
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
        }
        return result
