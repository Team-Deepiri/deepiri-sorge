"""Groq runner using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import os
import time

import requests
from loguru import logger

from bot.config import CacheConfig
from bot.diff_parser import ParsedDiff
from bot.prompts import load_groq_bug_detector_template
from bot.runners.base import BaseRunner, ReviewResult
from bot.schemas import ReviewIssue
from bot.utils.http_retry import post_with_retry

# Cap repo context chars so input+output stay inside Groq's ~8k window.
_GROQ_CONTEXT_CHAR_CAP = 6_000
_GROQ_DIFF_CHAR_FLOOR = 2_000


class GroqRunner(BaseRunner):
    """Runner for Groq-hosted models (currently GPT OSS 120B)."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    # Groq free tier validates input + max_tokens against ~8k total context.
    CONTEXT_TOKEN_LIMIT = 8192
    CONTEXT_SAFETY_BUFFER = 192
    HEADROOM_SAFETY = 0.85
    MIN_OUTPUT_TOKENS = 1024
    DESIRED_MAX_TOKENS = 4096
    RETRY_MAX_TOKENS = 6144
    # Scheduler token estimates are often high; never let them starve output below this.
    SCHEDULER_ESTIMATE_SLACK = 512

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        cache_config: CacheConfig | None = None,
    ):
        super().__init__(api_key or os.getenv("GROQ_API_KEY"), cache_config)
        self.model = model or self.DEFAULT_MODEL
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self._last_raw_response: str | None = None
        self._last_http_status: int | None = None
        self._last_retry_after: float | None = None
        self._last_timed_out: bool = False
        self._effective_input_tokens: int | None = None
        self._last_max_tokens: int | None = None

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, str]]) -> int:
        chars = sum(len(m.get("content") or "") for m in messages)
        return max(1, chars // 4)

    def _input_token_estimate(self, messages: list[dict[str, str]]) -> int:
        """Budget max_tokens from real prompt size.

        Scheduler estimates (diff tokens + template overhead) often overshoot the
        actual message, which used to force max_tokens≈1300 and vacuous truncations.
        Prefer the character estimate; only nudge upward slightly toward the
        scheduler hint so we do not under-budget the window.
        """
        char_est = self._estimate_message_tokens(messages)
        sched = self._effective_input_tokens
        if sched is None or sched <= 0:
            return char_est
        # Keep within ~SCHEDULER_ESTIMATE_SLACK of the real prompt.
        return max(char_est, min(int(sched), char_est + self.SCHEDULER_ESTIMATE_SLACK))

    @classmethod
    def _cap_max_tokens(cls, est_input: int, desired: int) -> int:
        """Cap output so input + max_tokens stays within Groq context (* 0.85 safety)."""
        raw_headroom = cls.CONTEXT_TOKEN_LIMIT - est_input - cls.CONTEXT_SAFETY_BUFFER
        headroom = int(raw_headroom * cls.HEADROOM_SAFETY)
        if headroom < cls.MIN_OUTPUT_TOKENS:
            return max(256, headroom)
        return max(cls.MIN_OUTPUT_TOKENS, min(desired, headroom))

    def _headroom_for(self, messages: list[dict[str, str]]) -> int:
        est = self._input_token_estimate(messages)
        raw = self.CONTEXT_TOKEN_LIMIT - est - self.CONTEXT_SAFETY_BUFFER
        return int(raw * self.HEADROOM_SAFETY)

    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        if not self.api_key:
            logger.error("No Groq API key configured")
            return None

        start_time = time.time()
        self._last_http_status = None
        self._last_retry_after = None
        self._last_timed_out = False
        self._last_max_tokens = None

        try:
            return self._call_api(diff, start_time)
        except requests.Timeout:
            logger.error("Groq request timed out")
            self._last_timed_out = True
            return self._timeout_result(start_time)
        except requests.RequestException as e:
            logger.error(f"Groq request failed: {e}")
            response = getattr(e, "response", None)
            self._last_http_status = getattr(response, "status_code", None)
            if response is not None:
                raw = response.headers.get("Retry-After")
                if raw:
                    try:
                        self._last_retry_after = float(raw)
                    except ValueError:
                        self._last_retry_after = None
            return None

    def _build_prompt(
        self,
        diff: ParsedDiff,
        *,
        context_block: str | None = None,
        diff_raw: str | None = None,
    ) -> str:
        template = load_groq_bug_detector_template()
        if context_block is None:
            context_block = self._repo_context or (
                "(No repository context — note reuse risks if the PR adds helpers already in repo.)"
            )
        if len(context_block) > _GROQ_CONTEXT_CHAR_CAP:
            context_block = (
                context_block[:_GROQ_CONTEXT_CHAR_CAP]
                + "\n\n… _(repo context truncated for Groq context window)_"
            )
        raw = diff.raw if diff_raw is None else diff_raw

        return f"""{template}

---

## DIFF (primary review target)
Files changed: {", ".join(diff.files)}
Total lines: +{diff.lines_added} -{diff.lines_deleted}

```diff
{raw}
```

---

## REPOSITORY CONTEXT
{context_block}
"""

    def _system_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are a fast production bug detector. Respond with a single raw JSON "
                "object only. No markdown fences. Max 5 hard findings; ignore style nits."
            ),
        }

    def _build_messages(
        self,
        diff: ParsedDiff,
        *,
        context_block: str | None = None,
        diff_raw: str | None = None,
    ) -> list[dict[str, str]]:
        return [
            self._system_message(),
            {
                "role": "user",
                "content": self._build_prompt(
                    diff, context_block=context_block, diff_raw=diff_raw
                ),
            },
        ]

    def _fit_messages_for_output(self, diff: ParsedDiff) -> list[dict[str, str]]:
        """Shrink context/diff until max_tokens can be at least MIN_OUTPUT_TOKENS.

        Stays inside the same Groq 8k window — one request budget, no extra RPM
        beyond the optional truncated retry already in _call_api.
        """
        context = self._repo_context or (
            "(No repository context — note reuse risks if the PR adds helpers already in repo.)"
        )
        raw = diff.raw
        messages = self._build_messages(diff, context_block=context, diff_raw=raw)
        if self._headroom_for(messages) >= self.MIN_OUTPUT_TOKENS:
            return messages

        # 1) Drop repo context aggressively.
        context = "(Repo context omitted to fit Groq 8k window.)"
        messages = self._build_messages(diff, context_block=context, diff_raw=raw)
        if self._headroom_for(messages) >= self.MIN_OUTPUT_TOKENS:
            logger.info("Groq prompt fit: omitted repo context for output headroom")
            return messages

        # 2) Binary-shrink the diff body until headroom is enough (keep a floor).
        lo, hi = _GROQ_DIFF_CHAR_FLOOR, len(raw)
        best = raw[: max(_GROQ_DIFF_CHAR_FLOOR, min(len(raw), 4000))]
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = raw[:mid] + ("\n… _(diff truncated for Groq context window)_" if mid < len(raw) else "")
            trial_msgs = self._build_messages(diff, context_block=context, diff_raw=trial)
            if self._headroom_for(trial_msgs) >= self.MIN_OUTPUT_TOKENS:
                best = trial
                messages = trial_msgs
                lo = mid + 1
            else:
                hi = mid - 1
        logger.info(
            f"Groq prompt fit: truncated diff to ~{len(best)} chars for "
            f">={self.MIN_OUTPUT_TOKENS} output tokens"
        )
        return messages

    def _call_api(self, diff: ParsedDiff, start_time: float) -> ReviewResult:
        messages = self._fit_messages_for_output(diff)
        est_input = self._input_token_estimate(messages)
        first_cap = self._cap_max_tokens(est_input, self.DESIRED_MAX_TOKENS)
        result, truncated = self._complete(messages, start_time, first_cap)
        if truncated and result.parse_warning:
            retry_cap = self._cap_max_tokens(est_input, self.RETRY_MAX_TOKENS)
            if retry_cap > first_cap:
                logger.warning(
                    f"Groq truncated at max_tokens={first_cap} with parse_warning; "
                    f"retrying once with max_tokens={retry_cap}"
                )
                result, _ = self._complete(messages, start_time, retry_cap)
        return result

    def _complete(
        self,
        messages: list[dict[str, str]],
        start_time: float,
        max_tokens: int,
    ) -> tuple[ReviewResult, bool]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self._last_max_tokens = max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }

        logger.debug(f"Calling Groq with model: {self.model} (max_tokens={max_tokens})")

        response = post_with_retry(
            self.endpoint, json=payload, headers=headers, timeout=120, max_retries=1
        )
        latency_ms = (time.time() - start_time) * 1000

        data = response.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        self._last_raw_response = content
        truncated = choice.get("finish_reason") == "length"
        if truncated:
            logger.warning("Groq response truncated (finish_reason=length)")
        tokens_used = data.get("usage", {}).get("total_tokens")

        parsed = self._parse_response(content)
        result = self._build_result(
            parsed,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            review_type="groq",
        )
        # Truncated + empty/high score is not a trustworthy all-clear.
        if truncated and not result.issues and result.score >= 8.5:
            result.parse_warning = result.parse_warning or "truncated_vacuous_review"
            result.score = min(float(result.score), 7.0)
        return result, truncated

    def complete_user_prompt(self, prompt: str, *, max_tokens: int = 512) -> str | None:
        """Lightweight completion for context-shave extracts (no review schema)."""
        if not self.api_key:
            return None
        messages = [
            {
                "role": "system",
                "content": "Return JSON only. No markdown fences.",
            },
            {"role": "user", "content": prompt},
        ]
        est = self._estimate_message_tokens(messages)
        cap = self._cap_max_tokens(est, max_tokens)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": cap,
        }
        response = post_with_retry(
            self.endpoint, json=payload, headers=headers, timeout=60, max_retries=1
        )
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _timeout_result(self, start_time: float) -> ReviewResult:
        return ReviewResult(
            summary="Groq request timed out - consider using a smaller diff",
            issues=[
                ReviewIssue(
                    severity="low",
                    file=None,
                    message="Groq request timed out",
                    suggestion="Split large PRs into smaller changes",
                )
            ],
            recommendations=[
                "Split large PRs into smaller changes",
                "Use a smaller model for quicker feedback",
            ],
            score=5.0,
            latency_ms=(time.time() - start_time) * 1000,
            model=self.model,
            tokens_used=None,
        )
