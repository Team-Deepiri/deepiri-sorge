"""Assign review-chunk priority from path keywords (higher = review first)."""

from __future__ import annotations

from bot.file_splitter import ReviewChunk

# (substrings matched against lowercased paths, priority weight)
# First matching rule wins per file; chunk takes the max across its files.
_PATH_RULES: tuple[tuple[tuple[str, ...], int], ...] = (
    (("auth", "oauth", "jwt", "password", "credential", "secret"), 100),
    (("security", "crypto", "crypt", "encrypt", "tls", "ssl", "permission"), 95),
    (("middleware", "session", "login", "rbac", "acl"), 90),
    (("api/", "/api", "handler", "service", "controller"), 80),
    (("src/", "lib/", "app/", "pkg/", "internal/"), 75),
    (("test", "spec", "_test.", ".test.", "/tests/"), 60),
    (("readme", "docs/", "doc/", "changelog", "license"), 25),
    (("lock", "package-lock", "yarn.lock", "pnpm-lock", "poetry.lock", "Cargo.lock"), 20),
)

DEFAULT_PRIORITY = 50


def priority_for_path(path: str) -> int:
    """Return priority weight for a single file path."""
    lower = path.lower().replace("\\", "/")
    for needles, weight in _PATH_RULES:
        if any(n in lower for n in needles):
            return weight
    return DEFAULT_PRIORITY


def prioritize_chunk(chunk: ReviewChunk) -> int:
    """Chunk priority = max path weight among its files."""
    if not chunk.files:
        return DEFAULT_PRIORITY
    return max(priority_for_path(f) for f in chunk.files)


def sort_key(scheduled) -> tuple:
    """Highest priority first; tie-break larger tokens (more valuable) first."""
    return (-scheduled.priority, -scheduled.chunk.estimated_tokens)
