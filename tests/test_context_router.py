"""Tests for context-first router."""

import pytest

from bot.context_router import ContextRouter
from bot.decision_engine import PRMetrics
from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.quota_tracker import QuotaTracker


def _chunk(tokens: int, raw: str = "diff") -> ReviewChunk:
    diff = ParsedDiff(raw=raw, files=["a.py"], lines_added=10, lines_deleted=0, files_changed=1)
    return ReviewChunk(files=["a.py"], parsed_diff=diff, estimated_tokens=tokens)


@pytest.fixture
def quota():
    return QuotaTracker(limits={"gemini": 20, "gpt": 1000, "openrouter": 50}, used={"gemini": 0, "gpt": 0, "openrouter": 0})


class TestContextRouter:
    def test_tiny_rung_uses_groq(self, quota):
        router = ContextRouter(groq_enabled=True, openrouter_enabled=True, gemini_enabled=True)
        metrics = PRMetrics(
            total_tokens=2000,
            max_file_tokens=2000,
            file_count=1,
            file_tokens={"a.py": 2000},
        )
        plan = router.route(metrics, [_chunk(2000)], quota)
        assert plan.rung == "tiny"
        assert plan.assignments[0].action.value == "groq"

    def test_standard_rung_uses_openrouter(self, quota):
        router = ContextRouter(groq_enabled=True, openrouter_enabled=True, gemini_enabled=True)
        metrics = PRMetrics(
            total_tokens=50_000,
            max_file_tokens=50_000,
            file_count=1,
            file_tokens={"a.py": 50_000},
        )
        plan = router.route(metrics, [_chunk(50_000, raw="x" * 200_000)], quota)
        assert plan.rung == "standard"
        assert plan.assignments[0].action.value == "openrouter"

    def test_oversized_splits_to_chunks(self, quota):
        router = ContextRouter(groq_enabled=False, openrouter_enabled=True, gemini_enabled=True)
        metrics = PRMetrics(
            total_tokens=300_000,
            max_file_tokens=150_000,
            file_count=2,
            file_tokens={"a.py": 150_000, "b.py": 150_000},
        )
        chunks = [_chunk(150_000), _chunk(150_000)]
        plan = router.route(metrics, chunks, quota)
        assert plan.rung == "oversized"
        assert len(plan.assignments) == 2

    def test_gemini_quota_fallback(self, quota):
        quota.used["openrouter"] = 50
        router = ContextRouter(groq_enabled=False, openrouter_enabled=True, gemini_enabled=True)
        metrics = PRMetrics(
            total_tokens=50_000,
            max_file_tokens=50_000,
            file_count=1,
            file_tokens={"a.py": 50_000},
        )
        plan = router.route(metrics, [_chunk(50_000)], quota)
        assert plan.assignments[0].action.value == "gemini"
