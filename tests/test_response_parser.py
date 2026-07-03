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
