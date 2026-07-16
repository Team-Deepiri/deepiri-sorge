"""Tests for path-based chunk priority."""

from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.scheduling.priority import prioritize_chunk, priority_for_path, sort_key
from bot.scheduling.types import ScheduledChunk


def _chunk(files: list[str], tokens: int = 100) -> ReviewChunk:
    return ReviewChunk(
        files=files,
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=tokens,
    )


def test_security_paths_outrank_docs():
    assert priority_for_path("src/auth/login.py") >= 95
    assert priority_for_path("docs/README.md") <= 25
    assert prioritize_chunk(_chunk(["src/auth/login.py", "docs/README.md"])) >= 95


def test_tests_and_locks():
    assert prioritize_chunk(_chunk(["tests/test_foo.py"])) == 60
    assert prioritize_chunk(_chunk(["poetry.lock"])) == 20


def test_default_priority():
    assert priority_for_path("random.xyz") == 50


def test_sort_key_priority_then_tokens():
    low = ScheduledChunk(chunk=_chunk(["docs/a.md"], tokens=9000), priority=25)
    high_small = ScheduledChunk(chunk=_chunk(["auth/x.py"], tokens=100), priority=100)
    high_large = ScheduledChunk(chunk=_chunk(["auth/y.py"], tokens=500), priority=100)
    ordered = sorted([low, high_small, high_large], key=sort_key)
    assert ordered[0] is high_large
    assert ordered[1] is high_small
    assert ordered[2] is low
