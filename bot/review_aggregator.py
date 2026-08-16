"""Merge multi-chunk review results into one PR comment payload."""

from __future__ import annotations

from bot.runners.base import ReviewResult
from bot.schemas import ReviewIssue
from bot.file_splitter import ReviewChunk
from bot.scheduling.types import SchedulerMeta, SkipRecord


def _classify_deferral(
    *,
    skipped: list[SkipRecord],
    quota_snapshot: dict | None,
    scheduler_meta: SchedulerMeta | dict | None,
) -> str:
    """Return a precise deferral class for PR comments (not a quality score)."""
    sched = (
        scheduler_meta.to_dict()
        if hasattr(scheduler_meta, "to_dict")
        else (scheduler_meta or {})
    )
    picks = list((sched or {}).get("provider_picks") or [])
    skip_text = " ".join(s.reason for s in skipped).lower()

    soft_gemini = False
    if quota_snapshot:
        gem = quota_snapshot.get("gemini") or {}
        try:
            soft_gemini = int(gem.get("remaining", 1)) <= 0
        except (TypeError, ValueError):
            soft_gemini = False
    if "deferred_gemini_quota_exhausted" in skip_text:
        soft_gemini = True

    http_429 = any(
        (not p.get("ok"))
        and (
            p.get("status") == 429
            or "429" in str(p.get("error") or "").lower()
            or "http_429" in str(p.get("error") or "").lower()
        )
        for p in picks
    ) or ("http_429" in skip_text or "429" in skip_text)

    vacuous = any(
        (not p.get("ok"))
        and any(
            tok in str(p.get("error") or "").lower()
            for tok in (
                "empty_response",
                "truncated",
                "non_json",
                "empty_or_invalid",
                "truncated_vacuous",
            )
        )
        for p in picks
    ) or any(
        tok in skip_text
        for tok in ("empty_response", "truncated", "non_json", "truncated_vacuous")
    )

    # Authoritative: the scheduler never reached a provider because a
    # concurrent run held every slot. Not a capacity or quota problem.
    if (sched or {}).get("stop_reason") == "lock_contention" or (
        "lock_contention" in skip_text
    ):
        return "lock_contention"

    if soft_gemini and not http_429 and not vacuous:
        return "soft_quota_exhausted"
    if http_429 and not vacuous and not soft_gemini:
        return "http_429"
    if vacuous and not http_429 and not soft_gemini:
        return "vacuous_or_truncated"
    if soft_gemini and (http_429 or vacuous):
        return "mixed_capacity"
    if http_429 and vacuous:
        return "mixed_capacity"
    return "providers_exhausted"


def _skip_issue(skip: SkipRecord) -> ReviewIssue:
    suggestion = "Re-run /sorge when provider capacity recovers"
    reason = (skip.reason or "").lower()
    if "deferred_gemini_quota_exhausted" in reason:
        suggestion = (
            "Gemini soft daily quota is exhausted — top-priority chunks were "
            "reviewed (or attempted) on free-tier providers; re-run tomorrow "
            "or after Gemini RPD resets (UTC day)"
        )
    elif "429" in reason or "rate" in reason:
        suggestion = "Re-run /sorge after HTTP 429 cooldowns clear (~1–3 minutes)"
    elif any(t in reason for t in ("empty", "truncated", "non_json")):
        suggestion = (
            "Provider returned truncated/empty JSON — re-run /sorge; "
            "Sorge will shrink Groq prompts / failover free OpenRouter models"
        )
    return ReviewIssue(
        severity="low",
        file=skip.chunk.files[0] if skip.chunk.files else None,
        message=f"Chunk skipped by scheduler: {skip.reason}",
        suggestion=suggestion,
    )


def _no_provider_result(
    *,
    rung: str,
    quota_snapshot: dict | None,
    skipped: list[SkipRecord],
    unreviewable: list[ReviewChunk] | None,
    scheduler_meta: SchedulerMeta | dict | None,
) -> ReviewResult:
    deferral = _classify_deferral(
        skipped=skipped or [],
        quota_snapshot=quota_snapshot,
        scheduler_meta=scheduler_meta,
    )
    meta = {
        "rung": rung,
        "chunks": 0,
        "quota": quota_snapshot,
        "unreviewable": len(unreviewable or []),
        "skipped": len(skipped or []),
        "final_state": "NO_PROVIDER_AVAILABLE",
        "deferral_class": deferral,
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

    if deferral == "soft_quota_exhausted":
        summary = (
            "Gemini soft daily quota is exhausted (configured free-tier RPD). "
            "No full automated review was generated — this is not a code-quality score."
        )
        recs = [
            "Wait for the UTC day to roll (Gemini soft RPD resets), then comment `/sorge`",
            "Large PRs are capped to top-priority chunks while Gemini is exhausted",
        ]
        review_type = "soft_quota_exhausted"
        status_hint = "soft_quota"
    elif deferral == "http_429":
        summary = (
            "Provider HTTP 429 rate limits blocked the review (often OpenRouter free "
            "models and/or Groq RPM). No automated review was generated — this is not "
            "a code-quality score."
        )
        recs = [
            (
                f"Wait ~{retry_mins} minute(s) for 429 cooldowns, then comment `/sorge`"
                if retry_mins
                else "Wait a few minutes for 429 cooldowns, then comment `/sorge`"
            ),
            "Sorge will failover across free OpenRouter models on 429 when possible",
        ]
        review_type = "http_429"
        status_hint = "http_429"
    elif deferral == "vacuous_or_truncated":
        summary = (
            "Providers returned truncated or empty review JSON (usually Groq output "
            "budget vs prompt size), not an HTTP rate-limit. No trustworthy review "
            "was generated — this is not a code-quality score."
        )
        recs = [
            "Re-run `/sorge` — prompts are shrunk to keep Groq max_tokens ≥ 1024 within the 8k window",
            "If this persists, split the PR or reduce generated file noise",
        ]
        review_type = "vacuous_or_truncated"
        status_hint = "vacuous_or_truncated"
    elif deferral == "lock_contention":
        summary = (
            "Review deferred — another Sorge run held every provider slot for this "
            "run's whole wait budget. No provider was called and no quota was spent; "
            "this is not a rate limit and not a code-quality score."
        )
        recs = [
            "This clears on its own — the retry below usually succeeds",
            "If it repeats, several reviews are being triggered at once on the same PR",
        ]
        review_type = "lock_contention"
        status_hint = "lock_contention"
    else:
        summary = (
            "Review deferred — no provider accepted the work before the wait budget "
            "ran out. No automated review was generated — this is not a code-quality "
            "score. See the routing details below for the per-provider outcome."
        )
        if retry_mins:
            summary += f" Approximate retry window: ~{retry_mins} minute(s)."
        recs = [
            (
                f"Wait ~{retry_mins} minute(s), then comment `/sorge`"
                if retry_mins
                else "Wait a few minutes, then comment `/sorge` again"
            ),
            "Check routing details below for the per-provider status and error",
        ]
        review_type = "rate_limited"
        status_hint = "mixed"

    meta["status_hint"] = status_hint
    if retry_mins and deferral != "soft_quota_exhausted":
        # Soft quota needs a day boundary, not a 2-minute retry.
        pass

    return ReviewResult(
        summary=summary,
        issues=[],
        recommendations=recs,
        score=0.0,
        latency_ms=0.0,
        model="none",
        review_type=review_type,
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

            # Zero chunks succeeded. No quality score is defensible here no
            # matter *why* — previously this branch enumerated known failure
            # reasons, and any reason outside that list ("deadline") fell
            # through to a review_type the adjudicator would happily rescore
            # from an empty issue list, i.e. a fabricated 10.0/10.
            #
            # The only remaining question is whether a retry could help:
            # provider capacity recovers, an oversized file does not.
            if not (unreviewable or []):
                return _no_provider_result(
                    rung=rung,
                    quota_snapshot=quota_snapshot,
                    skipped=skipped or [],
                    unreviewable=unreviewable,
                    scheduler_meta=scheduler_meta,
                )

            return ReviewResult(
                summary=(
                    "No automated review was generated — some files could not be "
                    "processed. This is not a code-quality score."
                ),
                issues=issues,
                recommendations=["Reduce PR size or split large generated files"],
                score=0.0,
                latency_ms=0.0,
                model="none",
                review_type="no_result",
                routing_meta={
                    "rung": rung,
                    "chunks": 0,
                    "quota": quota_snapshot,
                    "unreviewable": len(unreviewable or []),
                    "skipped": len(skipped or []),
                    "final_state": "NO_CHUNK_REVIEWED",
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
