"""Review complexity scoring for quality-aware provider routing.

Returns a score in [0, 1]. Used by the scheduler market score (live path)
and ContextRouter (size/quota plan) so both share one definition.
"""

from __future__ import annotations

from bot.file_splitter import ReviewChunk
from bot.scheduling.priority import prioritize_chunk

# Soft threshold: at/above this, prefer deeper reasoning (Gemini) when healthy.
COMPLEXITY_ESCALATE = 0.6
# Path priority from prioritize_chunk — auth/security rules are 90+.
SECURITY_PATH_PRIORITY = 90
# Mirror Groq lane ceiling (local constant avoids circular import with market_score).
_GROQ_LANE_MAX = 7000

_COMPLEXITY_PATH_HINTS = (
    "migration",
    "migrate",
    "schema",
    "auth",
    "oauth",
    "jwt",
    "security",
    "crypto",
    "permission",
    "rbac",
    "middleware",
    "architecture",
)


def complexity_score(
    chunk: ReviewChunk,
    *,
    prompt_overhead_tokens: int = 0,
) -> float:
    """Estimate review difficulty in [0, 1]."""
    diff = chunk.parsed_diff
    lines = max(0, diff.lines_added) + max(0, diff.lines_deleted)
    files = max(1, len(chunk.files) or diff.files_changed or 1)
    tokens = max(0, chunk.estimated_tokens) + max(0, prompt_overhead_tokens)
    priority = prioritize_chunk(chunk)

    line_n = min(1.0, lines / 500.0)
    file_n = min(1.0, (files - 1) / 15.0)
    fill_n = min(1.0, tokens / max(1, _GROQ_LANE_MAX))
    path_n = max(0.0, min(1.0, (priority - 50) / 50.0))

    hint_bonus = 0.0
    for path in chunk.files:
        lower = path.lower()
        if any(h in lower for h in _COMPLEXITY_PATH_HINTS):
            hint_bonus = 0.15
            break

    raw = (
        0.30 * line_n
        + 0.15 * file_n
        + 0.25 * fill_n
        + 0.25 * path_n
        + hint_bonus
    )
    return max(0.0, min(1.0, raw))


def is_high_complexity(score: float) -> bool:
    return score >= COMPLEXITY_ESCALATE


def is_security_sensitive(chunk: ReviewChunk) -> bool:
    return prioritize_chunk(chunk) >= SECURITY_PATH_PRIORITY
