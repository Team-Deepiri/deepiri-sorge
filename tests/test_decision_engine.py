"""Tests for decision engine"""

import pytest

from bot.config import Config
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser, ParsedDiff


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

    def test_medium_diff_no_provider(self, engine, medium_diff):
        decision = engine.decide(medium_diff)

        assert decision.action == Action.SKIP
        assert decision.skip_category == "no_provider"

    def test_large_diff_no_provider(self, engine, large_diff):
        decision = engine.decide(large_diff)

        assert decision.action == Action.SKIP
        assert decision.skip_category == "no_provider"

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

    def test_large_tier_prefers_openrouter_when_gemini_disabled(self, engine):
        engine.config.groq.enabled = True
        engine.config.openrouter.enabled = True
        engine.config.gemini.enabled = False
        engine.config.routing.small_pr_threshold = 50
        engine.config.routing.medium_pr_threshold = 80
        engine.config.routing.large_pr_threshold = 100

        huge = ParsedDiff(
            raw="x" * 500,
            files=["big.py"],
            file_changes={},
            lines_added=200,
            lines_deleted=0,
            files_changed=1,
        )

        decision = engine.decide(huge)

        assert decision.action == Action.OPENROUTER
        assert "large" in decision.reason

    def test_extra_chars_can_promote_to_larger_tier(self, engine, medium_diff):
        engine.config.groq.enabled = True
        engine.config.openrouter.enabled = True
        engine.config.gemini.enabled = True
        engine.config.routing.small_pr_threshold = 10
        engine.config.routing.medium_pr_threshold = 50
        engine.config.routing.large_pr_threshold = 100

        small_route = engine.decide(medium_diff)
        assert small_route.action == Action.GROQ

        large_route = engine.decide(medium_diff, extra_chars=10_000)
        assert large_route.action == Action.GEMINI


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


def _mixed_lockfile_diff() -> str:
    return """diff --git a/poetry.lock b/poetry.lock
--- a/poetry.lock
+++ b/poetry.lock
@@ -1,4 +1,4 @@
-# This file is automatically @generated by Poetry 2.4.1 and should not be changed by hand.
+# This file is automatically @generated by Poetry 2.2.1 and should not be changed by hand.
diff --git a/app/storage.py b/app/storage.py
--- a/app/storage.py
+++ b/app/storage.py
@@ -1,3 +1,3 @@
-    backend = "s3"
+    backend = "local"
"""


def test_strip_generated_files_drops_lockfile_from_mixed_diff():
    """Regression for prismpipe: a lockfile rode into review beside real source.

    _is_deps_only only fires when every file is a dependency file, so the
    generator stamp on poetry.lock line 1 reached the model as reviewable code
    and was read as a dependency downgrade.
    """
    engine = DecisionEngine(Config())
    diff = DiffParser().parse(_mixed_lockfile_diff())

    stripped, generated = engine.strip_generated_files(diff)

    assert generated == ["poetry.lock"]
    assert stripped.files == ["app/storage.py"]
    assert "Poetry 2.2.1" not in stripped.raw


def test_strip_generated_files_keeps_hand_edited_manifests():
    """requirements.txt is authored by a person; a change there is real."""
    engine = DecisionEngine(Config())
    diff = DiffParser().parse(
        """diff --git a/requirements.txt b/requirements.txt
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,2 +1,2 @@
-requests>=2.31
+requests>=2.32
diff --git a/app/main.py b/app/main.py
--- a/app/main.py
+++ b/app/main.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
"""
    )

    stripped, generated = engine.strip_generated_files(diff)

    assert generated == []
    assert set(stripped.files) == {"requirements.txt", "app/main.py"}


def test_strip_generated_files_noop_when_all_generated():
    """Never leave the caller with an empty diff to review."""
    engine = DecisionEngine(Config())
    diff = DiffParser().parse(
        """diff --git a/poetry.lock b/poetry.lock
--- a/poetry.lock
+++ b/poetry.lock
@@ -1,2 +1,2 @@
-a
+b
"""
    )

    stripped, generated = engine.strip_generated_files(diff)

    assert generated == []
    assert stripped.files == ["poetry.lock"]
