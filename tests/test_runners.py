"""Tests for bot/runners — base cache wiring, GeminiRunner, GroqRunner, OpenRouterRunner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import install_loguru_stub

install_loguru_stub()

from bot.config import CacheConfig
from bot.runners.base import ReviewIssue
from bot.diff_parser import DiffParser, ParsedDiff
from bot.runners.base import BaseRunner, ReviewResult
from bot.runners.gemini_runner import GeminiRunner
from bot.runners.groq_runner import GroqRunner
from bot.runners.json_schema import REVIEW_JSON_SCHEMA, SchemaEncoder
from bot.runners.openrouter_runner import OpenRouterRunner
from bot.utils import cache as _cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_diff() -> ParsedDiff:
    return DiffParser().parse((FIXTURES / "sample.diff").read_text())


@pytest.fixture
def docs_diff() -> ParsedDiff:
    return DiffParser().parse((FIXTURES / "docs.diff").read_text())


@pytest.fixture
def cache_config(tmp_path, monkeypatch) -> CacheConfig:
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    return CacheConfig(enabled=True, ttl_hours=24)


@pytest.fixture
def no_cache_config() -> CacheConfig:
    return CacheConfig(enabled=False)


def _mock_response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


GITHUB_API_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": json.dumps({
                    "summary": "Added validation logic",
                    "issues": [
                        {
                            "severity": "high",
                            "file": "src/main.py",
                            "line": 10,
                            "message": "Hardcoded debug=True",
                            "rule": "best_practice",
                            "suggestion": "Read from env var",
                        }
                    ],
                    "recommendations": ["Add unit tests"],
                    "score": 7.5,
                })
            }
        }
    ],
    "usage": {"total_tokens": 512},
}

GEMINI_API_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": json.dumps({
                            "summary": "Config refactor",
                            "issues": [],
                            "recommendations": ["Consider dataclass"],
                            "score": 9.0,
                        })
                    }
                ]
            }
        }
    ],
    "usageMetadata": {"totalTokenCount": 800},
}

OPENAI_COMPAT_CHAT_COMPLETIONS_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": json.dumps({
                    "summary": "Added input validation",
                    "issues": [],
                    "recommendations": ["Avoid adding unnecessary public configuration or parameters"],
                    "score": 8.5,
                })
            }
        }
    ],
    "usage": {"total_tokens": 256},
}


# ---------------------------------------------------------------------------
# BaseRunner cache wiring
# ---------------------------------------------------------------------------

class ConcreteRunner(BaseRunner):
    """Minimal concrete runner for testing BaseRunner cache logic."""

    model = "test-model"
    call_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def _run_review(self, diff: ParsedDiff) -> ReviewResult | None:
        self.call_count += 1
        return ReviewResult(
            summary="fresh result",
            issues=[],
            recommendations=[],
            score=8.0,
            latency_ms=100.0,
            model=self.model,
            review_type="test",
        )


class TestBaseRunnerCache:
    def test_cache_disabled_always_calls_api(self, sample_diff, no_cache_config):
        runner = ConcreteRunner(cache_config=no_cache_config)
        runner.review(sample_diff)
        runner.review(sample_diff)
        assert runner.call_count == 2

    def test_cache_enabled_second_call_is_cache_hit(self, sample_diff, cache_config):
        runner = ConcreteRunner(cache_config=cache_config)
        r1 = runner.review(sample_diff)
        r2 = runner.review(sample_diff)
        assert runner.call_count == 1
        assert r1.summary == r2.summary

    def test_cache_hit_reconstructs_result_correctly(self, sample_diff, cache_config):
        runner = ConcreteRunner(cache_config=cache_config)
        original = runner.review(sample_diff)
        cached = runner.review(sample_diff)
        assert cached.score == original.score
        assert cached.model == original.model
        assert cached.review_type == original.review_type

    def test_none_result_is_not_cached(self, sample_diff, cache_config):
        runner = ConcreteRunner(cache_config=cache_config)
        runner._run_review = MagicMock(return_value=None)
        runner.review(sample_diff)
        runner.review(sample_diff)
        assert runner._run_review.call_count == 2


# ---------------------------------------------------------------------------
# GeminiRunner
# ---------------------------------------------------------------------------

class TestGeminiRunner:
    def test_returns_none_without_api_key(self, sample_diff):
        runner = GeminiRunner(api_key=None)
        runner.api_key = None
        result = runner._run_review(sample_diff)
        assert result is None

    def test_successful_review(self, sample_diff):
        runner = GeminiRunner(api_key="fake-gemini-key")
        with patch("requests.post", return_value=_mock_response(GEMINI_API_RESPONSE)):
            result = runner._run_review(sample_diff)

        assert result is not None
        assert result.review_type == "gemini"
        assert result.summary == "Config refactor"
        assert result.issues == []
        assert result.score == 9.0
        assert result.tokens_used == 800

    def test_recommendations_passed_through(self, sample_diff):
        runner = GeminiRunner(api_key="fake-gemini-key")
        with patch("requests.post", return_value=_mock_response(GEMINI_API_RESPONSE)):
            result = runner._run_review(sample_diff)
        assert "Consider dataclass" in result.recommendations

    def test_timeout_returns_timeout_result(self, sample_diff):
        import requests as req
        runner = GeminiRunner(api_key="fake-gemini-key")
        with patch("requests.post", side_effect=req.Timeout):
            result = runner._run_review(sample_diff)
        assert result is not None
        assert result.score == 4.0
        assert any("timed out" in i.message.lower() for i in result.issues)

    def test_request_error_returns_none(self, sample_diff):
        import requests as req
        runner = GeminiRunner(api_key="fake-gemini-key")
        with patch("requests.post", side_effect=req.RequestException("dns error")):
            result = runner._run_review(sample_diff)
        assert result is None

    def test_uses_google_api_key_env_var(self, monkeypatch, sample_diff):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-gemini-key")
        runner = GeminiRunner()
        assert runner.api_key == "env-gemini-key"

    def test_cache_prevents_duplicate_api_call(self, sample_diff, cache_config):
        runner = GeminiRunner(api_key="fake-gemini-key", cache_config=cache_config)
        with patch("requests.post", return_value=_mock_response(GEMINI_API_RESPONSE)) as mock_post:
            runner.review(sample_diff)
            runner.review(sample_diff)
        assert mock_post.call_count == 1

    def test_docs_diff_still_reviewed(self, docs_diff):
        """Runners don't filter — that's the decision engine's job."""
        runner = GeminiRunner(api_key="fake-gemini-key")
        with patch("requests.post", return_value=_mock_response(GEMINI_API_RESPONSE)):
            result = runner._run_review(docs_diff)
        assert result is not None


# ---------------------------------------------------------------------------
# OpenRouterRunner / GroqRunner
# ---------------------------------------------------------------------------

class TestOpenRouterRunner:
    def test_returns_none_without_api_key(self, sample_diff):
        runner = OpenRouterRunner(api_key=None)
        runner.api_key = None
        result = runner._run_review(sample_diff)
        assert result is None

    def test_successful_review(self, sample_diff):
        runner = OpenRouterRunner(api_key="fake-openrouter-key")
        with patch("requests.post", return_value=_mock_response(OPENAI_COMPAT_CHAT_COMPLETIONS_RESPONSE)):
            result = runner._run_review(sample_diff)

        assert result is not None
        assert result.review_type == "openrouter"
        assert result.summary == "Added input validation"
        assert result.issues == []
        assert result.score == 8.5
        assert result.tokens_used == 256

    def test_uses_openrouter_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-key")
        runner = OpenRouterRunner()
        assert runner.api_key == "env-openrouter-key"

    def test_timeout_returns_timeout_result(self, sample_diff):
        import requests as req

        runner = OpenRouterRunner(api_key="fake-openrouter-key")
        with patch("requests.post", side_effect=req.Timeout):
            result = runner._run_review(sample_diff)

        assert result is not None
        assert result.score == 5.0
        assert any("timed out" in i.message.lower() for i in result.issues)

    def test_request_error_returns_none(self, sample_diff):
        import requests as req

        runner = OpenRouterRunner(api_key="fake-openrouter-key")
        with patch("requests.post", side_effect=req.RequestException("connection error")):
            result = runner._run_review(sample_diff)

        assert result is None


class TestGroqRunner:
    def test_returns_none_without_api_key(self, sample_diff):
        runner = GroqRunner(api_key=None)
        runner.api_key = None
        result = runner._run_review(sample_diff)
        assert result is None

    def test_successful_review(self, sample_diff):
        runner = GroqRunner(api_key="fake-groq-key")
        with patch("requests.post", return_value=_mock_response(OPENAI_COMPAT_CHAT_COMPLETIONS_RESPONSE)):
            result = runner._run_review(sample_diff)

        assert result is not None
        assert result.review_type == "groq"
        assert result.summary == "Added input validation"
        assert result.issues == []
        assert result.score == 8.5
        assert result.tokens_used == 256

    def test_uses_groq_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env-groq-key")
        runner = GroqRunner()
        assert runner.api_key == "env-groq-key"

    def test_timeout_returns_timeout_result(self, sample_diff):
        import requests as req

        runner = GroqRunner(api_key="fake-groq-key")
        with patch("requests.post", side_effect=req.Timeout):
            result = runner._run_review(sample_diff)

        assert result is not None
        assert result.score == 5.0
        assert any("timed out" in i.message.lower() for i in result.issues)

    def test_request_error_returns_none(self, sample_diff):
        import requests as req

        runner = GroqRunner(api_key="fake-groq-key")
        with patch("requests.post", side_effect=req.RequestException("connection error")):
            result = runner._run_review(sample_diff)

        assert result is None


# ---------------------------------------------------------------------------
# SchemaEncoder
# ---------------------------------------------------------------------------

class TestSchemaEncoder:
    """SchemaEncoder should produce correct provider-specific formats from a single JSON Schema."""

    def test_for_openai_returns_correct_wrapper(self):
        result = SchemaEncoder.for_openai()
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "code_review"
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["schema"] is REVIEW_JSON_SCHEMA

    def test_for_gemini_returns_raw_schema(self):
        result = SchemaEncoder.for_gemini()
        assert result is REVIEW_JSON_SCHEMA  # same object identity
        assert result["type"] == "object"
        assert "properties" in result

    def test_for_gemini_has_correct_structure(self):
        result = SchemaEncoder.for_gemini()
        assert "summary" in result["properties"]
        assert "metrics" in result["properties"]
        assert "issues" in result["properties"]
        assert "best_practice_notes" in result["properties"]

    def test_for_prompt_injection_returns_json_string(self):
        result = SchemaEncoder.for_prompt_injection()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["type"] == "object"
        assert "summary" in parsed["properties"]

    def test_for_openai_payload_is_serializable(self):
        result = SchemaEncoder.for_openai()
        # Should not raise
        serialized = json.dumps(result)
        assert '"strict": true' in serialized

    def test_schema_name_is_consistent(self):
        assert SchemaEncoder.SCHEMA_NAME == "code_review"