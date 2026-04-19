"""Tests for CPU reviewer"""

import json
from pathlib import Path

import pytest

from bot.config import Config
from bot.cpu_reviewer import CPUReviewer, ReviewIssue, ReviewResult
from bot.diff_parser import FileChange, ParsedDiff

_LLM_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "llm_review_outputs"


def _llm_fixture_text(name: str) -> str:
    return (_LLM_FIXTURE_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def reviewer(config):
    return CPUReviewer(config)


@pytest.fixture
def sample_diff():
    return ParsedDiff(
        raw="diff --git a/main.py b/main.py\n+def new_function():\n+    pass",
        files=["main.py"],
        file_changes={
            "main.py": FileChange(
                path="main.py",
                status="modified",
                additions=150,
                deletions=10,
            )
        },
        lines_added=150,
        lines_deleted=10,
        files_changed=1,
    )


class TestCPUReviewer:
    def test_heuristic_review_returns_result(self, reviewer, sample_diff):
        result = reviewer._heuristic_review(sample_diff)

        assert isinstance(result, ReviewResult)
        assert result.review_type == "cpu"
        assert result.model == "heuristic"

    def test_heuristic_catches_large_additions(self, reviewer, sample_diff):
        result = reviewer._heuristic_review(sample_diff)

        assert len(result.issues) > 0
        assert any("Large addition" in i.message for i in result.issues)

    def test_heuristic_catches_many_files(self, reviewer):
        diff = ParsedDiff(
            raw="diff",
            files=[f"file{i}.py" for i in range(15)],
            file_changes={},
            lines_added=50,
            lines_deleted=50,
            files_changed=15,
        )

        result = reviewer._heuristic_review(diff)

        assert any("many files" in i.message.lower() for i in result.issues)

    def test_review_heuristic_fallback(self, reviewer, sample_diff):
        result = reviewer.review(sample_diff)

        assert isinstance(result, ReviewResult)
        assert result.summary is not None

    def test_score_in_range(self, reviewer, sample_diff):
        result = reviewer.review(sample_diff)

        assert 0 <= result.score <= 10


class TestReviewIssue:
    def test_issue_creation(self):
        issue = ReviewIssue(
            severity="high",
            file="test.py",
            line=42,
            message="Bug found",
            suggestion="Fix it",
        )

        assert issue.severity == "high"
        assert issue.file == "test.py"
        assert issue.line == 42
        assert issue.message == "Bug found"
        assert issue.suggestion == "Fix it"


class TestReviewResult:
    def test_result_to_dict(self):
        result = ReviewResult(
            summary="Test summary",
            issues=[
                ReviewIssue(
                    severity="medium",
                    file="test.py",
                    message="Issue 1",
                )
            ],
            recommendations=["Rec 1", "Rec 2"],
            score=7.5,
            model="test-model",
        )

        data = result.to_dict()

        assert data["summary"] == "Test summary"
        assert len(data["issues"]) == 1
        assert len(data["recommendations"]) == 2
        assert data["score"] == 7.5
        assert data["model"] == "test-model"

    def test_empty_issues(self):
        result = ReviewResult(
            summary="Clean code",
            issues=[],
            recommendations=[],
            score=10.0,
            model="test",
        )

        data = result.to_dict()

        assert data["issues"] == []
        assert data["score"] == 10.0


class TestParseLlamaOutput:
    """Tests for cpu_reviewer._parse_llama_output (no GGUF / Llama)

    Example files in tests/fixtures/llm_review_outputs/
    """

    def test_valid_json_fixture_parses(self, reviewer, sample_diff):
        raw = _llm_fixture_text("valid_review.json").strip()
        expected = json.loads(raw)

        result = reviewer._parse_llama_output(raw, sample_diff)

        assert result.review_type == "cpu"
        assert result.summary == expected["summary"]
        assert result.score == expected["score"]
        assert result.model == reviewer.config.model.name
        assert len(result.issues) == len(expected["issues"])
        assert result.recommendations == expected["recommendations"]

        if expected["issues"]:
            ri = result.issues[0]
            ei = expected["issues"][0]
            assert ri.severity == ei["severity"]
            assert ri.file == ei.get("file")
            assert ri.line == ei.get("line")
            assert ri.message == ei["message"]
            assert ri.suggestion == ei.get("suggestion")

    def test_minimal_json_fixture_empty_issues(self, reviewer, sample_diff):
        raw = _llm_fixture_text("minimal_review.json").strip()
        expected = json.loads(raw)

        result = reviewer._parse_llama_output(raw, sample_diff)

        assert result.review_type == "cpu"
        assert result.summary == expected["summary"]
        assert result.issues == []
        assert result.score == expected["score"]

    def test_wrapped_model_response_fixture_matches_valid_summary(self, reviewer, sample_diff):
        wrapped = _llm_fixture_text("wrapped_model_response.txt")
        expected = json.loads(_llm_fixture_text("valid_review.json"))

        result = reviewer._parse_llama_output(wrapped, sample_diff)

        assert result.review_type == "cpu"
        assert result.summary == expected["summary"]
        assert len(result.issues) == len(expected["issues"])

    def test_invalid_json_fixture_returns_cpu_error(self, reviewer, sample_diff):
        garbage = _llm_fixture_text("garbage_not_json.txt")

        result = reviewer._parse_llama_output(garbage, sample_diff)

        assert result.review_type == "cpu-error"
        assert result.issues == []
        assert result.score == 7.0
        assert "corrupted" in result.summary.lower()
        assert len(result.recommendations) == 1

    def test_invalid_severity_fixture_returns_cpu_error(self, reviewer, sample_diff):
        raw = _llm_fixture_text("invalid_bad_severity.json").strip()

        result = reviewer._parse_llama_output(raw, sample_diff)

        assert result.review_type == "cpu-error"
        assert result.issues == []

    def test_invalid_score_fixture_returns_cpu_error(self, reviewer, sample_diff):
        raw = _llm_fixture_text("invalid_score.json").strip()

        result = reviewer._parse_llama_output(raw, sample_diff)

        assert result.review_type == "cpu-error"
        assert result.issues == []
