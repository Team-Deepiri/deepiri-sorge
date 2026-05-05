"""Regression tests for diff parsing and skip-classification behavior."""

from __future__ import annotations

from tests.helpers import install_loguru_stub

install_loguru_stub()

from bot.config import Config
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser, ParsedDiff


class TestDiffParserRegressions:
    def test_parse_added_file_marks_status_and_counts_additions(self):
        diff = """diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,2 @@
+print("hello")
+print("world")"""

        result = DiffParser().parse(diff)

        assert result.files == ["new_file.py"]
        assert result.file_changes["new_file.py"].status == "added"
        assert result.file_changes["new_file.py"].additions == 2
        assert result.file_changes["new_file.py"].deletions == 0

    def test_parse_deleted_file_marks_status_and_counts_deletions(self):
        diff = """diff --git a/old_file.py b/old_file.py
deleted file mode 100644
index 1111111..0000000
--- a/old_file.py
+++ /dev/null
@@ -1,2 +0,0 @@
-print("old")
-print("code")"""

        result = DiffParser().parse(diff)

        assert result.files == ["old_file.py"]
        assert result.file_changes["old_file.py"].status == "deleted"
        assert result.file_changes["old_file.py"].additions == 0
        assert result.file_changes["old_file.py"].deletions == 2


class TestDecisionEngineRegressions:
    def test_mixed_docs_and_code_diff_is_not_skipped_as_docs_only(self):
        diff = ParsedDiff(
            raw="diff",
            files=["README.md", "src/app.py"],
            file_changes={},
            lines_added=40,
            lines_deleted=5,
            files_changed=2,
        )

        decision = DecisionEngine(Config()).decide(diff)

        assert decision.action == Action.CPU_REVIEW
        assert decision.skip_category is None

    def test_mixed_dependency_and_code_diff_is_not_skipped_as_deps_only(self):
        diff = ParsedDiff(
            raw="diff",
            files=["package-lock.json", "src/app.js"],
            file_changes={},
            lines_added=50,
            lines_deleted=10,
            files_changed=2,
        )

        decision = DecisionEngine(Config()).decide(diff)

        assert decision.action == Action.CPU_REVIEW
        assert decision.skip_category is None

    def test_tests_only_diff_is_skipped_when_test_skipping_is_enabled(self):
        config = Config()
        config.filters.skip_tests = True
        diff = ParsedDiff(
            raw="diff",
            files=["test_app.py", "src/tests/api_test.py"],
            file_changes={},
            lines_added=60,
            lines_deleted=5,
            files_changed=2,
        )

        decision = DecisionEngine(config).decide(diff)

        assert decision.action == Action.SKIP
        assert decision.skip_category == "tests"

    def test_mixed_test_and_code_diff_is_not_skipped_as_tests_only(self):
        config = Config()
        config.filters.skip_tests = True
        diff = ParsedDiff(
            raw="diff",
            files=["test_app.py", "src/app.py"],
            file_changes={},
            lines_added=60,
            lines_deleted=5,
            files_changed=2,
        )

        decision = DecisionEngine(config).decide(diff)

        assert decision.action == Action.CPU_REVIEW
        assert decision.skip_category is None
