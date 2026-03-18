"""Tests for CPU reviewer"""

import pytest

from bot.cpu_reviewer import CPUReviewer, ReviewResult, ReviewIssue
from bot.diff_parser import ParsedDiff, FileChange
from bot.config import Config


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
