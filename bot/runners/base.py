"""Base runner class for model runners"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger

from bot.diff_parser import ParsedDiff
from bot.prompts import load_review_template
from bot.schemas import ReviewIssue, ReviewResult, issues_from_parsed, result_from_parsed
from bot.utils.response_parser import parse_review_response

# Re-export for backward compatibility with existing imports.
__all__ = ["BaseRunner", "ReviewIssue", "ReviewResult"]


class BaseRunner(ABC):
    """Abstract base class for model runners, with optional caching."""

    def __init__(self, api_key: str | None = None, cache_config=None):
        self.api_key = api_key
        self._cache_config = cache_config  # bot.config.CacheConfig or None
        self._repo_context: str | None = None
        self._prior_partial: str | None = None

    def review(
        self,
        diff: ParsedDiff,
        *,
        repo_context: str | None = None,
        context_fingerprint: str = "",
        prior_partial: str | None = None,
    ) -> ReviewResult | None:
        if self._cache_config and self._cache_config.enabled:
            from bot.utils import cache as _cache
            cached = _cache.get(
                diff.raw,
                self.model,
                self._cache_config.ttl_hours,
                context_fingerprint=context_fingerprint,
            )
            if cached is not None:
                # Don't serve cached results that had parse warnings — a retry
                # should get a fresh API call, not the same broken response.
                if cached.get("parse_warning"):
                    logger.warning(f"Cache hit for {self.model} ignored (has parse_warning)")
                else:
                    return self._result_from_dict(cached)

        self._repo_context = repo_context
        self._prior_partial = prior_partial
        result = self._run_review(diff)

        if result is not None and not result.parse_warning and self._cache_config and self._cache_config.enabled:
            from bot.utils import cache as _cache
            _cache.set(
                diff.raw,
                self.model,
                result.to_dict(),
                context_fingerprint=context_fingerprint,
            )

        return result

    @abstractmethod
    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        """Subclasses implement the actual API call here."""
        pass

    @staticmethod
    def _partial_block(partial: str | None) -> str:
        """Render a cut-off response from a previous provider as a primer."""
        if not partial or not partial.strip():
            return ""
        return f"""
---

## PRIOR PARTIAL ANALYSIS (incomplete)
A more capable model started this review and was cut off mid-response. Its
partial output is below. Treat it as a lead, not as truth: keep the findings
you can confirm against the DIFF, discard the ones you cannot, and finish the
review. Respond with a complete JSON object of your own — do not echo this
fragment.

```
{partial.strip()}
```
"""

    def _build_prompt(self, diff: ParsedDiff) -> str:
        template = load_review_template()
        context_block = self._repo_context or (
            "(No repository context — note reuse risks if the PR adds helpers already in repo.)"
        )

        return f"""{template}

---

## DIFF (primary review target)
Files changed: {", ".join(diff.files)}
Total lines: +{diff.lines_added} -{diff.lines_deleted}

```diff
{diff.raw}
```

---

## REPOSITORY CONTEXT
{context_block}
{self._partial_block(self._prior_partial)}"""

    def _parse_response(self, response_text: str) -> dict:
        return parse_review_response(response_text)

    def _build_result(
        self,
        parsed: dict,
        *,
        latency_ms: float,
        tokens_used: int | None,
        review_type: str,
    ) -> ReviewResult:
        return result_from_parsed(
            parsed,
            latency_ms=latency_ms,
            model=self.model,
            tokens_used=tokens_used,
            review_type=review_type,
        )

    def _result_from_dict(self, data: dict) -> ReviewResult:
        """Reconstruct a ReviewResult from a cached to_dict() payload."""
        issues = issues_from_parsed({"issues": data.get("issues", [])})
        return ReviewResult(
            summary=data.get("summary", ""),
            issues=issues,
            recommendations=data.get("recommendations", []),
            score=data.get("score", 7.0),
            latency_ms=data.get("latency_ms", 0.0),
            model=data.get("model", self.model),
            tokens_used=data.get("tokens_used"),
            review_type=data.get("review_type", "api"),
            parse_warning=data.get("parse_warning"),
        )
