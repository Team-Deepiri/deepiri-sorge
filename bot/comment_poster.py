"""Post review comments to GitHub PRs"""

import requests
from loguru import logger

from bot.runners.base import ReviewResult
from bot.utils.formatting import (
    clean_multiline_text,
    format_blockquote,
    format_issue_location,
    normalize_whitespace,
)

# Stable marker so drain can find & edit the provisional review comment.
SORGE_COMMENT_ANCHOR = "<!-- sorge-review-anchor -->"


def score_to_label(score: float) -> str:
    """Map a numeric score to a human-readable label."""
    if score >= 9:
        return "Production-ready, minimal issues"
    elif score >= 7:
        return "Good quality, minor issues worth addressing"
    elif score >= 5:
        return "Adequate, notable issues to fix before merge"
    elif score >= 3:
        return "Significant problems requiring changes"
    else:
        return "Major issues, needs redesign or re-architecture"


class CommentPoster:

    def __init__(self, token: str = ""):
        self.token = token or ""
        self.base_url = "https://api.github.com"

    def post_review(
        self,
        repo: str,
        pr_number: int,
        review: ReviewResult,  # type: ignore
        commit_id: str | None = None,
        *,
        edit_existing: bool = True,
        preferred_comment_id: int | None = None,
    ) -> int | None:
        """Post a review comment, or edit this run's provisional comment.

        Per-run hybrid: each `/sorge` gets a new comment. Within a run we edit
        ``preferred_comment_id`` (Starting… → final). We never rewrite an older
        run's review via ``get_previous_review``.
        """
        comment_body = self._format_review_comment(review)
        if edit_existing and preferred_comment_id is not None:
            return self.upsert_comment(
                repo,
                pr_number,
                comment_body,
                preferred_comment_id=preferred_comment_id,
                reuse_previous=False,
            )
        return self.post_comment(repo, pr_number, comment_body)

    def post_comment(
        self,
        repo: str,
        pr_number: int,
        body: str,
        commit_id: str | None = None,
    ) -> int | None:
        if not self.token:
            logger.warning("No GitHub token - cannot post comment")
            logger.info(f"Would post:\n{body}")
            return None

        if SORGE_COMMENT_ANCHOR not in body and "Sorge AI Code Review" in body:
            body = f"{SORGE_COMMENT_ANCHOR}\n{body}"

        url = f"{self.base_url}/repos/{repo}/issues/{pr_number}/comments"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        payload = {
            "body": body,
        }

        if commit_id:
            payload["commit_id"] = commit_id

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            comment_id = response.json().get("id")
            logger.info(f"Posted review comment to {repo}#{pr_number} id={comment_id}")
            return int(comment_id) if comment_id is not None else None

        except requests.RequestException as e:
            logger.error(f"Failed to post comment: {e}")
            return None

    def upsert_comment(
        self,
        repo: str,
        pr_number: int,
        body: str,
        *,
        preferred_comment_id: int | None = None,
        reuse_previous: bool = False,
    ) -> int | None:
        """Edit a known comment if present; otherwise post new.

        ``preferred_comment_id`` is this run's provisional (Starting…) comment.
        ``reuse_previous`` (default False) may also edit the latest anchored
        Sorge comment on the PR — avoid for normal `/sorge` so prior runs stay
        as history.
        """
        if SORGE_COMMENT_ANCHOR not in body and "Sorge AI Code Review" in body:
            body = f"{SORGE_COMMENT_ANCHOR}\n{body}"
        candidates: list[int] = []
        if preferred_comment_id is not None:
            candidates.append(int(preferred_comment_id))
        if reuse_previous:
            existing = self.get_previous_review(repo, pr_number)
            if existing is not None and existing not in candidates:
                candidates.append(existing)
        for cid in candidates:
            if self.update_comment(repo, cid, body):
                logger.info(f"Edited Sorge review comment {cid} on {repo}#{pr_number}")
                return cid
            logger.warning(f"Edit failed for comment {cid}; trying next candidate")
        return self.post_comment(repo, pr_number, body)

    def _format_rate_limited_comment(self, review: ReviewResult) -> str:  # type: ignore
        lines = [
            SORGE_COMMENT_ANCHOR,
            "## Sorge AI Code Review",
            "",
            "**Status:** Temporarily unavailable — provider rate limits",
            "",
            "---",
            "",
            "### What happened",
            clean_multiline_text(review.summary),
            "",
        ]
        if review.recommendations:
            lines.extend(["### What to do", ""])
            for rec in review.recommendations:
                lines.append(f"- {normalize_whitespace(rec)}")
            lines.append("")

        routing_meta = getattr(review, "routing_meta", None)
        if routing_meta:
            sched = routing_meta.get("scheduler") or {}
            retry_sec = sched.get("retry_after_sec")
            if retry_sec:
                mins = max(1, int((float(retry_sec) + 59) // 60))
                lines.extend([f"**Retry in ~{mins} minute(s)**", ""])
            picks = sched.get("provider_picks") or []
            detail_lines = []
            if picks:
                summary = ", ".join(
                    f"{p.get('provider')}({'ok' if p.get('ok') else 'fail'}"
                    f"|{p.get('reason', '?')})"
                    for p in picks[:8]
                )
                more = f" +{len(picks) - 8} more" if len(picks) > 8 else ""
                detail_lines.append(f"- Picks: {summary}{more}")
            if sched.get("stop_reason"):
                detail_lines.append(f"- Stop: `{sched.get('stop_reason')}`")
            if detail_lines:
                lines.extend(
                    [
                        "<details>",
                        "<summary>Routing details</summary>",
                        "",
                        *detail_lines,
                        "",
                        "</details>",
                        "",
                    ]
                )

        lines.extend(
            [
                "---",
                "",
                "*No quality score was computed — this is not a code review result.*",
                "",
                "*Review deferred by [deepiri-sorge](https://github.com/deepiri/deepiri-sorge)*",
            ]
        )
        return "\n".join(lines)

    def _format_review_comment(self, review: ReviewResult) -> str:  # type: ignore
        if getattr(review, "review_type", "") == "rate_limited":
            return self._format_rate_limited_comment(review)

        lines = [
            SORGE_COMMENT_ANCHOR,
            "## Sorge AI Code Review",
            "",
            f"**Model:** {normalize_whitespace(review.model)} "
            f"({normalize_whitespace(review.review_type)})",
            f"**Quality Score:** {review.score:.1f}/10 — *{score_to_label(review.score)}*",
            "",
        ]

        if review.parse_warning:
            lines.extend([
                "> ⚠️ **Review incomplete:** the model response could not be fully parsed "
                f"(`{review.parse_warning}`). Re-run the workflow or switch provider if this persists.",
                "",
            ])

        lines.extend([
            "---",
            "",
            "### Summary",
            clean_multiline_text(review.summary),
            "",
        ])

        if review.issues:
            lines.extend([
                "---",
                "",
                "### Issues Found",
                "",
            ])

            severity_icons = {
                "high": ":x:",
                "medium": ":warning:",
                "low": ":information_source:",
            }

            for issue in review.issues:
                icon = severity_icons.get(issue.severity, ":memo:")
                location = format_issue_location(issue.file, issue.line)

                lines.append(f"{icon} {location}")
                lines.append(format_blockquote(issue.message))

                if issue.suggestion:
                    lines.append(
                        format_blockquote(
                            f"_Suggestion: {normalize_whitespace(issue.suggestion)}_"
                        )
                    )

                lines.append("")

        if review.recommendations:
            lines.extend([
                "---",
                "",
                "### Recommendations",
                "",
            ])

            for rec in review.recommendations:
                lines.append(f"- {normalize_whitespace(rec)}")

            lines.append("")

        routing_meta = getattr(review, "routing_meta", None)
        if routing_meta:
            rung = routing_meta.get("rung", "?")
            chunks = routing_meta.get("chunks", 1)
            quota = routing_meta.get("quota") or {}
            quota_lines = [
                f"{k}: {v.get('used', 0)}/{v.get('limit', 0)}"
                for k, v in quota.items()
            ]
            lines.extend([
                "<details>",
                f"<summary>Routing ({rung}, {chunks} chunk(s))</summary>",
                "",
                f"- Rung: `{rung}`",
                f"- Chunks: {chunks}",
            ])
            if quota_lines:
                lines.append(f"- Quota this run: {', '.join(quota_lines)}")
            claim_meta = routing_meta.get("claim_verifier") or {}
            suppressed = claim_meta.get("suppressed", 0)
            if suppressed:
                lines.append(
                    f"- Claim verifier: suppressed {suppressed} structural "
                    "false-positive(s) against PR-head symbol map"
                )
            adj_meta = routing_meta.get("finding_adjudicator") or {}
            adj_dropped = adj_meta.get("dropped", 0)
            adj_demoted = adj_meta.get("demoted", 0)
            if adj_dropped or adj_demoted:
                lines.append(
                    f"- Finding adjudicator: dropped {adj_dropped}, demoted {adj_demoted}"
                )
            sched = routing_meta.get("scheduler") or {}
            if sched:
                cache_hits = sched.get("cache_hits", 0)
                cache_bit = f", {cache_hits} cache hit(s)" if cache_hits else ""
                lines.append(
                    f"- Scheduler: {sched.get('dispatches', 0)} dispatch(es), "
                    f"{sched.get('skipped', 0)} skipped{cache_bit}"
                    + (f", stop={sched.get('stop_reason')}" if sched.get("stop_reason") else "")
                )
                if sched.get("avg_complexity") is not None:
                    lines.append(f"- Avg complexity: `{sched.get('avg_complexity'):.2f}`")
                if sched.get("escalations"):
                    lines.append(f"- Escalations: {sched.get('escalations')}")
                if sched.get("escalation_blocked"):
                    lines.append(
                        f"- Escalation blocked: {sched.get('escalation_blocked')} "
                        "(vacuous triage softened)"
                    )
                if sched.get("retry_after_sec"):
                    mins = max(1, int((float(sched["retry_after_sec"]) + 59) // 60))
                    lines.append(f"- Retry after: ~{mins} minute(s)")
                health = sched.get("health") or {}
                if health:
                    parts = [f"{k}={v:.0f}" for k, v in sorted(health.items())]
                    lines.append(f"- Provider health: {', '.join(parts)}")
                picks = sched.get("provider_picks") or []
                if picks:
                    summary = ", ".join(
                        f"{p.get('provider')}({'ok' if p.get('ok') else 'fail'}"
                        f"{',esc' if p.get('escalation') else ''}"
                        f"|{p.get('reason', '?')}"
                        f"|eff={p.get('effective_tokens', p.get('tokens', '?'))}"
                        f"{'|mt=' + str(p['max_tokens']) if p.get('max_tokens') is not None else ''}"
                        f"{'|sc=' + str(p['pick_score']) if p.get('pick_score') is not None else ''})"
                        for p in picks[:8]
                    )
                    more = f" +{len(picks) - 8} more" if len(picks) > 8 else ""
                    lines.append(f"- Picks: {summary}{more}")
            lines.extend(["", "</details>", ""])

        lines.extend([
            "---",
            "",
            "*Review generated by [deepiri-sorge](https://github.com/deepiri/deepiri-sorge)*",
        ])

        return "\n".join(lines)

    def update_comment(
        self,
        repo: str,
        comment_id: int,
        body: str,
    ) -> bool:
        if not self.token:
            logger.warning("No GitHub token - cannot update comment")
            return False

        url = f"{self.base_url}/repos/{repo}/issues/comments/{comment_id}"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        payload = {"body": body}

        try:
            response = requests.patch(url, json=payload, headers=headers)
            response.raise_for_status()

            logger.info(f"Updated comment {comment_id}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to update comment: {e}")
            return False

    def delete_comment(self, repo: str, comment_id: int) -> bool:
        if not self.token:
            logger.warning("No GitHub token - cannot delete comment")
            return False

        url = f"{self.base_url}/repos/{repo}/issues/comments/{comment_id}"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            response = requests.delete(url, headers=headers)
            response.raise_for_status()

            logger.info(f"Deleted comment {comment_id}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to delete comment: {e}")
            return False

    def get_previous_review(
        self,
        repo: str,
        pr_number: int,
    ) -> int | None:
        if not self.token:
            return None

        url = f"{self.base_url}/repos/{repo}/issues/{pr_number}/comments"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            comments = response.json()
            # Prefer newest matching Sorge review (skip "Starting AI review...")
            for comment in reversed(comments):
                body = comment.get("body", "") or ""
                if SORGE_COMMENT_ANCHOR in body:
                    return comment.get("id")
            for comment in reversed(comments):
                body = comment.get("body", "") or ""
                if "Starting AI review" in body:
                    continue
                if "Sorge AI Code Review" in body or "deepiri-sorge" in body:
                    return comment.get("id")

            return None

        except requests.RequestException as e:
            logger.error(f"Failed to get previous reviews: {e}")
            return None
