"""Parse and normalize LLM review responses into a consistent dict shape."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


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

    logger.warning("Model response was not valid JSON; using markdown fallback")
    return _fallback_payload(text, "non_json_response")


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
    stripped = _CODE_FENCE_RE.sub("", text).strip()

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
