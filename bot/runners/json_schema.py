"""Shared JSON Schema definitions for structured output enforcement across all runners.

Both Gemini (responseSchema) and OpenAI-compatible APIs (response_format: json_schema)
support structured output schemas. This module provides a single source of truth
for the review output schema to keep all runners in sync.
"""

REVIEW_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Overall review summary under 400 words"},
        "metrics": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "Quality score 0-10"},
                "requires_human_review": {
                    "type": "boolean",
                    "description": "Whether this PR needs human attention",
                },
            },
            "required": ["score", "requires_human_review"],
        },
        "issues": {
            "type": "array",
            "description": "Code issues found. Max 8 items. No filler.",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Impact level",
                    },
                    "file": {
                        "type": "string",
                        "description": "Changed file path, or null for system-wide notes",
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number if anchored to a change",
                    },
                    "category": {
                        "type": "string",
                        "description": "Issue category (e.g. Reuse, Security, Architecture)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Clear explanation of the issue",
                    },
                    "mentoring": {
                        "type": "string",
                        "description": "Learning/mentoring context for the developer",
                    },
                    "suggestion": {
                        "type": "string",
                        "description": "Specific actionable fix recommendation",
                    },
                },
                "required": ["severity", "message"],
            },
        },
        "best_practice_notes": {
            "type": "array",
            "items": {"type": "string", "description": "General best practice recommendations"},
        },
    },
    "required": ["summary", "metrics", "issues", "best_practice_notes"],
}

# OpenAI-compatible json_schema wrapper (for OpenRouter, Groq, and similar)
REVIEW_OPENAI_JSON_SCHEMA_WRAPPER = {
    "type": "json_schema",
    "json_schema": {
        "name": "code_review",
        "strict": True,
        "schema": REVIEW_JSON_SCHEMA,
    },
}