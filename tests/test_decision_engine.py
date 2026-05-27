"""Tests for decision engine"""

import pytest

from bot.config import Config
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import ParsedDiff


@pytest.fixture
def config():
    cfg = Config()
    cfg.gemini.enabled = False
    cfg.groq.enabled = False
    cfg.openrouter.enabled = False
    return cfg


@pytest.fixture
def engine(config):
    return DecisionEngine(config)


@pytest.fixture
def empty_diff():
    return ParsedDiff(
        raw="",
        files=[],
        file_changes={},
        lines_added=0,
        lines_deleted=0,
        files_changed=0,
    )


@pytest.fixture
def small_diff():
    return ParsedDiff(
        raw="diff --git a/test.py b/test.py",
        files=["test.py"],
        file_changes={},
        lines_added=10,
        lines_deleted=5,
        files_changed=1,
    )


@pytest.fixture
def medium_diff():
    return ParsedDiff(
        raw="diff --git a/main.py b/main.py\n+new code",
        files=["main.py"],
        file_changes={},
        lines_added=100,
        lines_deleted=50,
        files_changed=1,
    )


@pytest.fixture
def large_diff():
    return ParsedDiff(
        raw="diff",
        files=["a.py", "b.py", "c.py"],
        file_changes={},
        lines_added=1500,
        lines_deleted=500,
        files_changed=3,
    )


@pytest.fixture
def docs_diff():
    return ParsedDiff(
        raw="diff --git a/README.md b/README.md",
        files=["README.md"],
        file_changes={},
        lines_added=10,
        lines_deleted=2,
        files_changed=1,
    )


@pytest.fixture
def deps_diff():
    return ParsedDiff(
        raw="diff --git a/package-lock.json b/package-lock.json",
        files=["package-lock.json"],
        file_changes={},
        lines_added=1000,
        lines_deleted=990,
        files_changed=1,
    )


class TestDecisionEngine:
    def test_disabled_config(self, engine, small_diff):
        engine.config.sorge["enabled"] = False

        decision = engine.decide(small_diff)

        assert decision.action == Action.SKIP
        assert decision.skip_category == "disabled"

    def test_empty_diff(self, engine, empty_diff):
        decision = engine.decide(empty_diff)

        assert decision.action == Action.SKIP

    def test_small_diff_skipped(self, engine, small_diff):
        decision = engine.decide(small_diff)

        assert decision.action == Action.SKIP
        assert decision.skip_category == "too_small"

    def test_docs_only_skipped(self, engine, docs_diff):
        decision = engine.decide(docs_diff)

        assert decision.action == Action.SKIP
        assert decision.skip_category == "docs"

    def test_deps_only_skipped(self, engine, deps_diff):
        decision = engine.decide(deps_diff)

        assert decision.action == Action.SKIP
        assert decision.skip_category == "deps"

    def test_medium_diff_cpu(self, engine, medium_diff):
        decision = engine.decide(medium_diff)

        assert decision.action == Action.CPU_REVIEW

    def test_large_diff_gpu_when_enabled(self, engine, large_diff):
        engine.config.gpu.enabled = True

        decision = engine.decide(large_diff)

        assert decision.action == Action.GPU_REVIEW

    def test_large_diff_cpu_when_gpu_disabled(self, engine, large_diff):
        engine.config.gpu.enabled = False

        decision = engine.decide(large_diff)

        assert decision.action == Action.CPU_REVIEW

    def test_routing_prefers_groq_for_small_tokens(self, engine, medium_diff):
        engine.config.groq.enabled = True
        engine.config.openrouter.enabled = True
        engine.config.gemini.enabled = True

        decision = engine.decide(medium_diff)

        assert decision.action == Action.GROQ

    def test_routing_prefers_openrouter_when_groq_disabled(self, engine, medium_diff):
        engine.config.groq.enabled = False
        engine.config.openrouter.enabled = True
        engine.config.gemini.enabled = True

        decision = engine.decide(medium_diff)

        assert decision.action == Action.OPENROUTER

    def test_routing_prefers_gemini_for_large_tokens(self, engine, large_diff):
        engine.config.groq.enabled = False
        engine.config.openrouter.enabled = False
        engine.config.gemini.enabled = True

        decision = engine.decide(large_diff)

        assert decision.action == Action.GEMINI


class TestComplexityScoring:
    def test_empty_diff_score(self, engine, empty_diff):
        score = engine.get_complexity_score(empty_diff)
        assert score == 0.0

    def test_small_diff_score(self, engine, small_diff):
        score = engine.get_complexity_score(small_diff)
        assert score > 0
        assert score < 10.0

    def test_large_diff_score(self, engine, large_diff):
        score = engine.get_complexity_score(large_diff)
        assert score > 0
        assert score <= 10.0

    def test_core_file_bonus(self, engine):
        diff = ParsedDiff(
            raw="diff",
            files=["src/core/api_handler.py"],
            file_changes={},
            lines_added=100,
            lines_deleted=50,
            files_changed=1,
        )

        score = engine.get_complexity_score(diff)

        assert score > 0
