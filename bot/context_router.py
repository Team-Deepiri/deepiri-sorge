"""Context-first router: tiny → Groq, standard → Gemma, oversized → split."""

from __future__ import annotations

from dataclasses import dataclass, field

from bot.decision_engine import Action
from bot.decision_engine import PRMetrics
from bot.file_splitter import ReviewChunk
from bot.quota_tracker import QuotaTracker


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
        small_threshold: int = 3_700,
        max_chunk_tokens: int = 200_000,
        groq_enabled: bool = True,
        openrouter_enabled: bool = True,
        gemini_enabled: bool = True,
    ):
        self.small_threshold = small_threshold
        self.max_chunk_tokens = max_chunk_tokens
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
            action = self._pick(Action.GROQ, Action.OPENROUTER, Action.GEMINI, quota, adjustments)
            if action:
                return RoutingPlan(
                    rung="tiny",
                    assignments=[
                        ChunkAssignment(chunks[0], action, f"tiny PR (~{metrics.total_tokens} tokens)")
                    ],
                    quota_adjustments=adjustments,
                )

        if metrics.total_tokens <= self.max_chunk_tokens and len(chunks) == 1:
            action = self._pick(Action.OPENROUTER, Action.GEMINI, Action.GROQ, quota, adjustments)
            if action:
                return RoutingPlan(
                    rung="standard",
                    assignments=[
                        ChunkAssignment(
                            chunks[0],
                            action,
                            f"standard PR (~{metrics.total_tokens} tokens)",
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

    def _route_chunk(
        self,
        chunk: ReviewChunk,
        quota: QuotaTracker,
        adjustments: list[str],
    ) -> tuple[Action | None, str]:
        tokens = chunk.estimated_tokens

        if tokens <= self.small_threshold:
            action = self._pick(Action.GROQ, Action.OPENROUTER, None, quota, adjustments)
            if action:
                return action, f"tiny chunk ({tokens} tokens)"

        if tokens <= self.max_chunk_tokens:
            action = self._pick(Action.OPENROUTER, Action.GEMINI, None, quota, adjustments)
            if action:
                return action, f"standard chunk ({tokens} tokens)"

        action = self._pick(Action.GEMINI, Action.OPENROUTER, None, quota, adjustments)
        if action:
            return action, f"large chunk ({tokens} tokens)"

        return None, "no provider available"

    def _pick(
        self,
        first: Action,
        second: Action | None,
        third: Action | None,
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
