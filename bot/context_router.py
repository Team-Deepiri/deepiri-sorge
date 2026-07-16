"""Context-first router: size + quota + complexity-aware preference chains.

Live dispatch is owned by ReviewScheduler; this module remains the size/quota
planning layer (PR #23) and now shares complexity scoring for preference order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bot.config import RoutingConfig
from bot.decision_engine import Action
from bot.decision_engine import PRMetrics
from bot.file_splitter import ReviewChunk
from bot.quota_tracker import QuotaTracker
from bot.scheduling.complexity import complexity_score, is_high_complexity, is_security_sensitive


@dataclass
class ChunkAssignment:
    chunk: ReviewChunk
    action: Action
    reason: str


@dataclass
class RoutingPlan:
    rung: str
    assignments: list[ChunkAssignment] = field(default_factory=list)
    quota_adjustments: list[str] = field(default_factory=list)


class ContextRouter:
    def __init__(
        self,
        *,
        routing: RoutingConfig,
        groq_enabled: bool = True,
        openrouter_enabled: bool = True,
        gemini_enabled: bool = True,
    ):
        self.small_threshold = routing.small_pr_threshold
        self.max_chunk_tokens = routing.medium_pr_threshold
        self.groq_enabled = groq_enabled
        self.openrouter_enabled = openrouter_enabled
        self.gemini_enabled = gemini_enabled

    def route(
        self,
        metrics: PRMetrics,
        chunks: list[ReviewChunk],
        quota: QuotaTracker,
    ) -> RoutingPlan:
        adjustments: list[str] = list(quota.adjustments)

        if metrics.total_tokens <= self.small_threshold and len(chunks) == 1:
            chunk = chunks[0]
            chain = self._preference_for_chunk(chunk, tiny=True)
            action = self._pick(*chain, quota=quota, adjustments=adjustments)
            if action:
                return RoutingPlan(
                    rung="tiny",
                    assignments=[
                        ChunkAssignment(
                            chunk,
                            action,
                            f"tiny PR (~{metrics.total_tokens} tokens; "
                            f"complexity={complexity_score(chunk):.2f})",
                        )
                    ],
                    quota_adjustments=adjustments,
                )

        if metrics.total_tokens <= self.max_chunk_tokens and len(chunks) == 1:
            chunk = chunks[0]
            chain = self._preference_for_chunk(chunk, tiny=False)
            action = self._pick(*chain, quota=quota, adjustments=adjustments)
            if action:
                return RoutingPlan(
                    rung="standard",
                    assignments=[
                        ChunkAssignment(
                            chunk,
                            action,
                            f"standard PR (~{metrics.total_tokens} tokens; "
                            f"complexity={complexity_score(chunk):.2f})",
                        )
                    ],
                    quota_adjustments=adjustments,
                )

        assignments: list[ChunkAssignment] = []
        for chunk in chunks:
            if chunk.unreviewable:
                continue
            action, reason = self._route_chunk(chunk, quota, adjustments)
            if action:
                assignments.append(ChunkAssignment(chunk, action, reason))

        return RoutingPlan(
            rung="oversized",
            assignments=assignments,
            quota_adjustments=adjustments,
        )

    def _preference_for_chunk(
        self,
        chunk: ReviewChunk,
        *,
        tiny: bool,
    ) -> tuple[Action | None, Action | None, Action | None]:
        cx = complexity_score(chunk)
        if is_security_sensitive(chunk) or is_high_complexity(cx):
            return (Action.GEMINI, Action.OPENROUTER, Action.GROQ if tiny else None)
        if tiny:
            return (Action.GROQ, Action.OPENROUTER, Action.GEMINI)
        return (Action.OPENROUTER, Action.GEMINI, Action.GROQ)

    def _route_chunk(
        self,
        chunk: ReviewChunk,
        quota: QuotaTracker,
        adjustments: list[str],
    ) -> tuple[Action | None, str]:
        tokens = chunk.estimated_tokens
        cx = complexity_score(chunk)

        if tokens <= self.small_threshold:
            chain = self._preference_for_chunk(chunk, tiny=True)
            action = self._pick(*chain, quota=quota, adjustments=adjustments)
            if action:
                return action, f"tiny chunk ({tokens} tokens; complexity={cx:.2f})"

        if tokens <= self.max_chunk_tokens:
            chain = self._preference_for_chunk(chunk, tiny=False)
            action = self._pick(*chain, quota=quota, adjustments=adjustments)
            if action:
                return action, f"standard chunk ({tokens} tokens; complexity={cx:.2f})"

        action = self._pick(
            Action.GEMINI, Action.OPENROUTER, None, quota=quota, adjustments=adjustments
        )
        if action:
            return action, f"large chunk ({tokens} tokens; complexity={cx:.2f})"

        return None, "no provider available"

    def _pick(
        self,
        first: Action | None,
        second: Action | None,
        third: Action | None,
        *,
        quota: QuotaTracker,
        adjustments: list[str],
    ) -> Action | None:
        for action in (first, second, third):
            if action is None:
                continue
            if not self._enabled(action):
                continue
            if not quota.can_use(action.value):
                adjustments.append(f"{action.value} quota exhausted, trying fallback")
                continue
            return action
        return None

    def _enabled(self, action: Action) -> bool:
        if action == Action.GROQ:
            return self.groq_enabled
        if action == Action.OPENROUTER:
            return self.openrouter_enabled
        if action == Action.GEMINI:
            return self.gemini_enabled
        return False
