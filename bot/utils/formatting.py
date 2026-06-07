"""Formatting helpers for review comments and related text output."""

from __future__ import annotations


def normalize_whitespace(value: str | None) -> str:
    """Collapse repeated whitespace and trim the result."""
    if not value:
        return ""
    return " ".join(value.split())


def clean_multiline_text(value: str | None) -> str:
    """Trim surrounding whitespace while preserving line breaks."""
    if not value:
        return ""

    lines = [line.rstrip() for line in value.strip().splitlines()]
    return "\n".join(lines)


def format_issue_location(file: str | None, line: int | None = None) -> str:
    """Render a markdown label for an issue location."""
    location = f"**{normalize_whitespace(file)}**" if file else "General"
    if line is not None:
        location += f":{line}"
    return location


def format_blockquote(text: str | None) -> str:
    """Render text as a markdown blockquote."""
    cleaned = clean_multiline_text(text)
    if not cleaned:
        return ">"

    return "\n".join(f"> {line}" if line else ">" for line in cleaned.splitlines())


def chunk_text(text: str, max_length: int) -> list[str]:
    """Split text into whitespace-aware chunks under max_length."""
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    cleaned = text.strip()
    if not cleaned:
        return []

    words = cleaned.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        word_length = len(word)

        if not current:
            current.append(word)
            current_length = word_length
            continue

        projected_length = current_length + 1 + word_length
        if projected_length <= max_length:
            current.append(word)
            current_length = projected_length
            continue

        chunks.append(" ".join(current))
        current = [word]
        current_length = word_length

    if current:
        chunks.append(" ".join(current))

    return chunks
