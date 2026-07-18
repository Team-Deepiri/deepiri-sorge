"""Review complexity scoring for quality-aware provider routing.

Returns a score in [0, 1]. Used by the scheduler market score (live path)
and ContextRouter (size/quota plan) so both share one definition.
"""

from __future__ import annotations

import re
from pathlib import Path

from bot.file_splitter import ReviewChunk
from bot.scheduling.priority import prioritize_chunk

# Soft threshold: at/above this, prefer deeper reasoning (Gemini) when healthy.
COMPLEXITY_ESCALATE = 0.68
# Path priority from prioritize_chunk — auth/security rules are 90+.
SECURITY_PATH_PRIORITY = 90
# Mirror Groq lane ceiling (local constant avoids circular import with market_score).
_GROQ_LANE_MAX = 7000

# Vacuous escalate: clean high-score review is surprising only when expected
# difficulty is high — not merely because the diff is long.
VACUOUS_DIFFICULTY_THRESHOLD = 0.42
VACUOUS_SCORE_FLOOR = 8.5

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

_PROSE_EXTS = frozenset({".md", ".rst", ".txt", ".adoc", ".markdown"})
_PROSE_PATH_HINTS = (
    "readme",
    "docs/",
    "doc/",
    "documentation/",
    "changelog",
    "license",
    "contributing",
    "authors",
)
_LOCK_HINTS = (
    "package-lock",
    "yarn.lock",
    "pnpm-lock",
    "poetry.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
)
_CODE_LINE_RE = re.compile(
    r"^\+\s*(?:"
    r"(?:async\s+)?(?:def|class|function|fn|func|pub|private|public|protected)\b|"
    r"(?:import|from|require|use|using|include|#include)\b|"
    r"(?:if|for|while|switch|return|throw|await|yield|const|let|var|type|interface|struct|enum)\b"
    r")"
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


def _is_prose_path(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    ext = Path(lower).suffix
    if ext in _PROSE_EXTS:
        return True
    return any(h in lower for h in _PROSE_PATH_HINTS)


def _is_lockfile_path(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    return any(h in lower for h in _LOCK_HINTS) or lower.endswith(".lock")


def code_ratio(chunk: ReviewChunk) -> float:
    """Fraction of chunk files that look like reviewable source (not prose/locks)."""
    files = list(chunk.files or [])
    if not files:
        return 0.5
    code = 0
    for path in files:
        if _is_prose_path(path) or _is_lockfile_path(path):
            continue
        code += 1
    return code / len(files)


def semantic_change_score(chunk: ReviewChunk) -> float:
    """Density of added lines that look like executable/structural code [0, 1]."""
    raw = chunk.parsed_diff.raw or ""
    added = 0
    codeish = 0
    for line in raw.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added += 1
        if _CODE_LINE_RE.match(line):
            codeish += 1
    if added <= 0:
        return 0.0
    return min(1.0, codeish / added)


def expected_review_difficulty(
    chunk: ReviewChunk,
    *,
    complexity: float | None = None,
    priority: int | None = None,
    prompt_overhead_tokens: int = 0,
) -> float:
    """How surprising a clean (0-issue, high-score) review would be [0, 1].

    Combines scheduler signals — complexity, code vs prose mix, path priority,
    cross-file coupling, and code-like change density — instead of raw token count.
    """
    cx = (
        float(complexity)
        if complexity is not None
        else complexity_score(chunk, prompt_overhead_tokens=prompt_overhead_tokens)
    )
    pri = int(priority) if priority is not None else prioritize_chunk(chunk)
    # Docs lane (~25) → ~0; default (~50) → ~0.33; auth (~100) → 1.0
    pri_n = max(0.0, min(1.0, (pri - 25) / 75.0))
    cr = code_ratio(chunk)
    files = len(chunk.files or [])
    couple_n = min(1.0, max(0, files - 1) / 8.0)
    sem = semantic_change_score(chunk)

    raw = (
        0.35 * max(0.0, min(1.0, cx))
        + 0.30 * cr
        + 0.20 * pri_n
        + 0.10 * couple_n
        + 0.05 * sem
    )
    return max(0.0, min(1.0, raw))


def is_surprisingly_empty_review(
    chunk: ReviewChunk,
    *,
    score: float,
    issue_count: int,
    complexity: float | None = None,
    priority: int | None = None,
    prompt_overhead_tokens: int = 0,
    difficulty_threshold: float = VACUOUS_DIFFICULTY_THRESHOLD,
) -> bool:
    """True when a clean high score is unlikely given expected review difficulty."""
    if issue_count > 0:
        return False
    if score < VACUOUS_SCORE_FLOOR:
        return False
    difficulty = expected_review_difficulty(
        chunk,
        complexity=complexity,
        priority=priority,
        prompt_overhead_tokens=prompt_overhead_tokens,
    )
    return difficulty >= difficulty_threshold
