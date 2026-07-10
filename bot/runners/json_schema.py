"""Shared JSON Schema definitions for structured output enforcement across all runners.

Both Gemini (responseSchema) and OpenAI-compatible APIs (response_format: json_schema)
support structured output schemas. This module provides a single source of truth
for the review output schema to keep all runners in sync.

The SchemaEncoder class converts the generic REVIEW_JSON_SCHEMA into provider-specific
formats, so adding a new provider never requires changing the schema itself.
"""

from __future__ import annotations

REVIEW_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Overall review summary under 400 words"},
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
    "required": ["summary", "issues", "best_practice_notes"],
}


class SchemaEncoder:
    """Encode the generic REVIEW_JSON_SCHEMA into provider-specific formats.

    Each static method returns the correct API-level payload for structured
    output enforcement on a given provider.  To add a new provider, simply
    add a ``for_<provider>()`` method here — no schema constants change.
    """

    SCHEMA_NAME = "code_review"

    @staticmethod
    def for_openai() -> dict:
        """OpenAI-compatible ``response_format: json_schema`` envelope.

        Used by OpenAI, OpenRouter, Groq, and any provider that speaks the
        OpenAI chat completions protocol with ``response_format`` support.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": SchemaEncoder.SCHEMA_NAME,
                "strict": True,
                "schema": REVIEW_JSON_SCHEMA,
            },
        }

    @staticmethod
    def for_gemini() -> dict:
        """Gemini ``responseSchema`` / ``responseMimeType: application/json`` format.

        Returns the raw JSON Schema dict — Gemini's API accepts it directly
        inside ``generationConfig.responseSchema``.
        """
        return REVIEW_JSON_SCHEMA

    @staticmethod
    def for_prompt_injection() -> str:
        """Schema as JSON text for providers that lack native structured output.

        The caller should inject this string into the system prompt so the
        model knows the expected JSON shape.  Falls back to text-only parsing
        with ``_parse_response`` validation on the caller side.
        """
        import json
        return json.dumps(REVIEW_JSON_SCHEMA, indent=2)