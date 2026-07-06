"""Tests for LLM response parsing and normalization."""

from bot.schemas import result_from_parsed
from bot.utils.response_parser import normalize_review_payload, parse_review_response


VALID_JSON = """
```json
{
  "summary": "Solid refactor.",
  "metrics": {"score": 8.5, "requires_human_review": false},
  "issues": [
    {
      "severity": "medium",
      "file": "bot/main.py",
      "line": 12,
      "category": "Reuse",
      "message": "Duplicates cache helper",
      "mentoring": "Prefer existing utils",
      "suggestion": "Import bot.utils.cache"
    }
  ],
  "best_practice_notes": ["Add integration test for routing"]
}
```
"""

MARKDOWN_ONLY = """
### Summary
This is a comprehensive update to the bot's infrastructure, transitioning from a single-provider model.

### Key Architectural Changes
1. Multi-Provider Routing
The bot now routes by token size.
"""

# A response that has key-value pairs in markdown but no valid JSON structure
MARKDOWN_WITH_STRUCTURED_DATA = """
## Summary
This PR adds a caching mechanism. The implementation needs improvements.

## Issues Found

- severity: medium | file: src/cache.py | line: 45 | category: Reuse | message: Duplicates existing LRU cache | suggestion: Use existing module
- severity: high | file: src/main.py | line: 23 | category: Security | message: SQL injection risk | suggestion: Use parameterized queries

## Score
Score: 6.5

## Recommendations
- Add integration tests for cache
- Add input validation middleware
"""

# Response with partial JSON-like key-value pairs but no valid JSON object
DICT_LIKE_MARKDOWN = """
summary: This PR introduces a new authentication module.

issues:
- severity: medium
  file: auth/login.py
  line: 42
  category: Security
  message: Hardcoded secret key
  suggestion: Use environment variable

score: 7.0
"""


def test_parse_fenced_json_and_normalize_schema():
    parsed = parse_review_response(VALID_JSON)

    assert parsed["summary"] == "Solid refactor."
    assert parsed["score"] == 8.5
    assert parsed["recommendations"] == ["Add integration test for routing"]
    assert parsed["issues"][0]["rule"] == "Reuse"
    assert "Prefer existing utils" in parsed["issues"][0]["suggestion"]


def test_markdown_fallback_does_not_duplicate_recommendations():
    parsed = parse_review_response(MARKDOWN_ONLY)

    assert parsed["_parse_warning"] == "non_json_response"
    assert parsed["recommendations"] == []
    assert "comprehensive update" in parsed["summary"]
    assert parsed["recommendations"] != [parsed["summary"]]


def test_markdown_with_structured_data_extracts_issues():
    """When model returns markdown with severity/file/line key-value pairs
    but no valid JSON, we should still extract the structured data."""
    parsed = parse_review_response(MARKDOWN_WITH_STRUCTURED_DATA)

    # Should have extracted issues even though no JSON
    assert len(parsed["issues"]) >= 2

    # Check first issue values were extracted
    issue = parsed["issues"][0]
    assert issue["severity"] == "medium"
    assert issue["file"] == "src/cache.py"
    assert issue["line"] == 45
    assert "LRU" in issue["message"]

    # Check second issue
    issue2 = parsed["issues"][1]
    assert issue2["severity"] == "high"
    assert "injection" in issue2["message"]

    # Score should have been extracted
    assert parsed["score"] == 6.5

    # Should have a parse warning since it wasn't valid JSON
    assert parsed["_parse_warning"] == "non_json_response"


def test_markdown_with_dict_like_data_extracts_issues():
    """Indented dict-like markdown should still extract structured data."""
    parsed = parse_review_response(DICT_LIKE_MARKDOWN)

    assert len(parsed["issues"]) >= 1
    issue = parsed["issues"][0]
    assert issue["severity"] == "medium"
    assert issue["file"] == "auth/login.py"
    assert issue["line"] == 42
    assert "secret" in issue["message"].lower()

    # Score should be extracted
    assert parsed["score"] == 7.0


def test_result_from_parsed_reads_metrics_score():
    result = result_from_parsed(
        {
            "summary": "ok",
            "metrics": {"score": 9.0},
            "issues": [],
            "best_practice_notes": ["ship it"],
        },
        latency_ms=1.0,
        model="test",
        tokens_used=10,
        review_type="openrouter",
    )

    assert result.score == 9.0
    assert result.recommendations == ["ship it"]


def test_normalize_maps_category_and_mentoring():
    data = normalize_review_payload(
        {
            "summary": "x",
            "issues": [
                {
                    "severity": "low",
                    "category": "Architecture",
                    "message": "Mismatch",
                    "mentoring": "Stay consistent",
                }
            ],
        }
    )

    assert data["issues"][0]["rule"] == "Architecture"
    assert data["issues"][0]["suggestion"] == "Stay consistent"


def test_empty_response_returns_fallback():
    parsed = parse_review_response("")

    assert parsed["_parse_warning"] == "empty_response"
    assert parsed["score"] == 5.0
    assert parsed["issues"] == []


def test_markdown_with_only_text_returns_fallback():
    """Plain markdown without structured key-value pairs should still fall back
    gracefully, but no structured extraction should happen."""
    parsed = parse_review_response(MARKDOWN_ONLY)

    assert parsed["_parse_warning"] == "non_json_response"
    assert parsed["issues"] == []
    assert "comprehensive update" in parsed["summary"]


def test_no_summary_header_fallback():
    """If there's no ## Summary header, the fallback should still extract
    the first meaningful lines."""
    text = """
This is a code review.

Some issues were found but they're not structured.

Severity: medium
File: foo.py
    """
    parsed = parse_review_response(text)

    assert parsed["_parse_warning"] == "non_json_response"
    # The fallback should pick up the first lines
    assert "code review" in parsed["summary"]
