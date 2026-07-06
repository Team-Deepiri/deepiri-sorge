"""Parse and normalize LLM review responses into a consistent dict shape."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

_CODE_FENCE_RE = re.compile(
    r"^(?:```(?:json)?)\s*$|^\s*(?:```)\s*$", re.IGNORECASE | re.MULTILINE
)
# Also strip inline code fence remnants
_INLINE_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)

# Regex patterns for extracting review data from markdown
# These handle:
#   - severity: medium | file: foo.py | line: 42
#   - **severity:** medium **file:** foo.py
#   - "severity": "medium", "file": "foo.py"
#   - severity: medium\n  file: foo.py\n  line: 42
_SEVERITY_RE = re.compile(
    r'(?:\*\*)?["\']?severity["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?(high|medium|low|critical)["\']?(?:\*\*)?',
    re.IGNORECASE,
)
_FILE_RE = re.compile(
    r'(?:\*\*)?["\']?file["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?([a-zA-Z0-9_./\\-]+(?:\.[a-zA-Z0-9]+)?)["\']?(?:\*\*)?',
    re.IGNORECASE,
)
_LINE_RE = re.compile(
    r'(?:\*\*)?["\']?line["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?(\d+)["\']?(?:\*\*)?',
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(
    r'(?:\*\*)?["\']?category["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?([a-zA-Z]\w*(?:\s+\w+)*)["\']?(?:\*\*)?',
    re.IGNORECASE,
)
_MESSAGE_RE = re.compile(
    r'(?:\*\*)?["\']?message["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?([a-zA-Z][^"\'|)\n]{5,})["\']?(?:\*\*)?',
    re.IGNORECASE,
)
_SUGGESTION_RE = re.compile(
    r'(?:\*\*)?["\']?suggestion["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?([a-zA-Z][^"\'|)\n]{5,})["\']?(?:\*\*)?',
    re.IGNORECASE,
)
_SCORE_RE = re.compile(
    r'(?:\*\*)?["\']?score["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?(\d+(?:\.\d+)?)["\']?(?:\*\*)?',
    re.IGNORECASE,
)

# Match markdown list items like "- Add integration tests" or "1. do something"
# But NOT headings like "### Key Architectural Changes" or subsection markers
_RECOMMENDATION_RE = re.compile(
    r'(?:^|\n)\s*[-*]\s+([A-Z][a-zA-Z]{4,}(?:\s+[a-z]{2,}){2,})',
)
_SUMMARY_RE = re.compile(
    r'(?:#{1,3}\s*)?[Ss]ummary\s*:?\s*(.*?)(?=\n#{1,3}\s|\Z)',
    re.DOTALL,
)


# Known markdown heading patterns to skip during issue/recommendation extraction
_HEADING_RE = re.compile(r'^#{1,6}\s+\w+')


def _build_issue_from_field_data(
    field_data: dict[str, Any], original_block: str
) -> dict[str, Any]:
    """Build a normalized issue dict from extracted field data.

    Handles message extraction fallback when no explicit message field was found.
    """
    issue: dict[str, Any] = {
        "severity": field_data.get("severity", "medium"),
        "file": field_data.get("file"),
        "line": field_data.get("line"),
        "category": field_data.get("category"),
        "message": field_data.get("message", ""),
        "suggestion": field_data.get("suggestion") or field_data.get("mentoring"),
    }

    # If no explicit message, try to extract surrounding text
    if not issue["message"]:
        cleaned = original_block
        for field_name in ("severity", "file", "line", "category", "suggestion", "mentoring"):
            pat = re.compile(
                r'(?:\*\*)?["\']?' + re.escape(field_name) + r'["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?[^"\'|)]+["\']?(?:\*\*)?,?\s*',
                re.IGNORECASE,
            )
            cleaned = pat.sub("", cleaned).strip()
        cleaned = re.sub(r'^\s*[-*]\s+', '', cleaned).strip()
        cleaned = cleaned.strip("- \t|*,").strip()
        if cleaned and len(cleaned) > 10:
            issue["message"] = cleaned

    return issue


def parse_review_response(response_text: str) -> dict[str, Any]:
    """Extract review JSON from model output; degrade gracefully on failure."""
    text = (response_text or "").strip()
    if not text:
        return _fallback_payload("", "empty_response")

    parsed = _extract_json_object(text)
    if parsed is not None:
        normalized = normalize_review_payload(parsed)
        normalized.pop("_parse_warning", None)
        return normalized

    logger.warning("Model response was not valid JSON; trying structured markdown fallback")

    # Second pass: try to extract structured data from markdown
    fallback = _fallback_payload(text, "non_json_response")
    enhanced = _extract_structured_from_markdown(text, fallback)
    if enhanced is not None:
        return enhanced

    return fallback


def _parse_pipe_delimited_block(block: str) -> dict[str, Any] | None:
    """Parse a single line with pipe-delimited key: value | key: value format.

    E.g.: severity: medium | file: src/cache.py | line: 45 | category: Reuse
    Returns a dict of extracted fields, or None if no valid fields found.
    """
    if "|" not in block:
        return None

    result: dict[str, Any] = {}
    parts = [p.strip() for p in block.split("|")]
    valid = 0

    for part in parts:
        match = re.match(
            r'(?:\*\*)?["\']?(\w+)["\']?(?:\*\*)?\s*[:=]\s*(?:\*\*)?["\']?(.+?)["\']?(?:\*\*)?\s*$',
            part,
            re.IGNORECASE,
        )
        if match:
            key = match.group(1).lower().strip()
            value = match.group(2).strip()
            # Map known keys
            if key in ("severity", "file", "line", "category", "message", "suggestion", "mentoring"):
                if key == "line":
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        continue
                result[key] = value
                valid += 1

    if valid >= 2:  # At least severity + one other field
        return result
    return None


def _extract_structured_from_markdown(
    text: str, base: dict[str, Any]
) -> dict[str, Any] | None:
    """Extract structured review data from markdown-formatted responses.

    When a model returns nicely-formatted markdown (even without valid JSON),
    try to pull out score, issues, and recommendations using regex patterns.
    Returns None if no meaningful structured data was found beyond the base fallback.
    """
    modified = False
    issues: list[dict[str, Any]] = []

    # Strategy 1: Try parsing each line as a pipe-delimited issue block.
    # This handles list items like:
    #   - severity: medium | file: src/cache.py | line: 45 | ...
    for line in text.splitlines():
        line = line.strip()
        if not line or _HEADING_RE.match(line) or line.startswith("- [") or "##" in line:
            continue
        # Strip leading list marker if present
        line_content = re.sub(r'^\s*[-*]\s+', '', line).strip()
        if not line_content or "severity" not in line_content.lower():
            continue

        field_data = _parse_pipe_delimited_block(line_content)
        if field_data is not None:
            issues.append(_build_issue_from_field_data(field_data, line_content))

    # Strategy 2: Also try multi-line blocks for YAML-like indented format.
    # This handles:
    #   issues:
    #   - severity: medium
    #     file: auth/login.py
    #     line: 42
    #     ...
    blocks = re.split(r"\n\s*\n", text)

    for block in blocks:
        stripped = block.strip()
        if not stripped or _HEADING_RE.match(stripped):
            continue

        severity_match = _SEVERITY_RE.search(stripped)
        if not severity_match:
            continue

        field_data: dict[str, Any] | None = {
            "severity": severity_match.group(1).lower(),
        }

        for regex_key, regex in [
            ("file", _FILE_RE),
            ("line", _LINE_RE),
            ("category", _CATEGORY_RE),
            ("message", _MESSAGE_RE),
            ("suggestion", _SUGGESTION_RE),
        ]:
            m = regex.search(stripped)
            if m:
                val = m.group(1).strip() if regex_key != "line" else int(m.group(1))
                field_data[regex_key] = val

        issue = _build_issue_from_field_data(field_data, stripped)

        # Deduplicate: only add if not already captured by pipe-delimited parsing
        if issue not in issues:
            issues.append(issue)

    if issues:
        base["issues"] = issues
        modified = True

    # Try to find a score
    score_match = _SCORE_RE.search(text)
    if score_match:
        try:
            base["score"] = float(score_match.group(1))
            modified = True
        except (TypeError, ValueError):
            pass

    # Try to find recommendations (only under a ## Recommendations or ## Issues Found section)
    # Look for a "Recommendations" section in the markdown
    rec_section = re.split(
        r'(?:#{1,3}\s*)?[Rr]ecommendations?\s*:?\s*\n',
        text,
        maxsplit=1,
    )
    if len(rec_section) > 1:
        rec_body = rec_section[1]
        # Split by blank lines to separate from next section
        rec_body = rec_body.split("\n\n")[0] if "\n\n" in rec_body else rec_body
        recommendations = []
        # Extract all lines starting with - or 1.
        for line in rec_body.splitlines():
            line = line.strip()
            if not line:
                continue
            # Match list items
            list_match = re.match(r'^\s*[-*\d]+\.?\s+(.+)$', line)
            if list_match:
                rec = list_match.group(1).strip()
                if rec and len(rec) > 10:
                    recommendations.append(rec)
        if recommendations:
            base["recommendations"] = recommendations
            modified = True

    # Try to find a better summary from structured markdown
    summary_match = _SUMMARY_RE.search(text)
    if summary_match:
        candidate = summary_match.group(1).strip()
        if candidate and len(candidate) > 20:
            base["summary"] = candidate
            modified = True

    if not modified:
        return None

    return base


def normalize_review_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Map template fields (metrics.score, best_practice_notes) to runner schema."""
    data = dict(raw)

    score = data.get("score")
    metrics = data.get("metrics")
    if score is None and isinstance(metrics, dict):
        score = metrics.get("score")
    try:
        data["score"] = float(score if score is not None else 7.0)
    except (TypeError, ValueError):
        data["score"] = 7.0

    recommendations = data.get("recommendations")
    if not recommendations:
        notes = data.get("best_practice_notes") or []
        if isinstance(notes, list):
            data["recommendations"] = [str(n) for n in notes if n]

    issues: list[dict[str, Any]] = []
    for item in data.get("issues") or []:
        if not isinstance(item, dict):
            continue
        issue = dict(item)
        if not issue.get("rule") and issue.get("category"):
            issue["rule"] = issue["category"]
        mentoring = issue.pop("mentoring", None)
        suggestion = issue.get("suggestion")
        if mentoring and suggestion:
            issue["suggestion"] = f"{suggestion} — {mentoring}"
        elif mentoring and not suggestion:
            issue["suggestion"] = str(mentoring)
        if issue.get("file") in ("null", "None", ""):
            issue["file"] = None
        issues.append(issue)
    data["issues"] = issues

    summary = data.get("summary")
    data["summary"] = str(summary).strip() if summary else "Review complete"
    data["recommendations"] = [
        str(r).strip() for r in (data.get("recommendations") or []) if str(r).strip()
    ]
    return data


def _extract_json_object(text: str) -> dict[str, Any] | None:
    # Strip leading/trailing code fences
    stripped = _CODE_FENCE_RE.sub("", text).strip()
    # Also strip any inline code fence markers
    stripped = _INLINE_FENCE_RE.sub("", stripped).strip()
    # Strip leading "json" label
    stripped = re.sub(r'^json\s+', '', stripped, flags=re.IGNORECASE).strip()

    for candidate in (stripped, text.strip()):
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                chunk = stripped[start : index + 1]
                try:
                    loaded = json.loads(chunk)
                    if isinstance(loaded, dict):
                        return loaded
                except json.JSONDecodeError:
                    repaired = _repair_truncated_json(chunk)
                    if repaired is not None:
                        return repaired
                return None

    repaired = _repair_truncated_json(stripped[start:])
    return repaired


def _repair_truncated_json(chunk: str) -> dict[str, Any] | None:
    """Best-effort repair when the model hits max_tokens mid-JSON."""
    trimmed = chunk.rstrip().rstrip(",")
    for suffix in ("", "]", "}", "]}", "\"}", "\"]}", "null]}", "null]}"):
        try:
            loaded = json.loads(trimmed + suffix)
            if isinstance(loaded, dict):
                logger.warning("Repaired truncated JSON response from model")
                return loaded
        except json.JSONDecodeError:
            continue
    return None


def _fallback_payload(text: str, reason: str) -> dict[str, Any]:
    summary = _summary_from_markdown(text)
    return {
        "summary": summary,
        "issues": [],
        "recommendations": [],
        "score": 5.0,
        "_parse_warning": reason,
    }


def _summary_from_markdown(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return "Review could not be generated — the model returned an empty response."

    collected: list[str] = []
    in_summary = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("### summary") or lower == "## summary":
            in_summary = True
            continue
        if in_summary and stripped.startswith("#"):
            break
        if in_summary and stripped:
            collected.append(stripped)
            if len("\n".join(collected)) > 2000:
                break

    if collected:
        body = "\n".join(collected).strip()
    else:
        body = "\n".join(lines[:20]).strip()

    if len(body) > 2500:
        body = body[:2500].rstrip() + "\n\n… _(truncated — model did not return valid JSON)_"

    if not body:
        return (
            "Review could not be fully parsed — the model did not return valid JSON. "
            "Check workflow logs for the raw response."
        )

    return body + "\n\n⚠️ _Partial review: model output was not valid JSON._"
