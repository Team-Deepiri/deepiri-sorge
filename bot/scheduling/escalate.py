"""Escalate tickets + Gemini multiplex (N tickets → 1 request)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

from bot.schemas import ReviewIssue, ReviewResult, issues_from_parsed
from bot.utils.response_parser import normalize_review_payload


@dataclass
class EscalateTicket:
    """Compact escalate unit — same-PR multiplex or cross-PR ledger."""

    ticket_id: str
    reason: str
    files: list[str]
    estimated_tokens: int
    complexity: float
    priority: int
    groq_summary: str
    groq_score: float
    groq_issues: list[dict[str, Any]]
    contested_diff: str
    repo: str = ""
    pr_number: int = 0
    installation_id: int | None = None
    head_sha: str = ""
    status: str = "pending"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EscalateTicket":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def new_ticket_id() -> str:
    return uuid.uuid4().hex[:12]


def truncate_diff(raw: str, max_chars: int = 12_000) -> str:
    if len(raw) <= max_chars:
        return raw
    half = max_chars // 2
    return raw[:half] + "\n\n…[diff truncated for escalate multiplex]…\n\n" + raw[-half:]


def build_multiplex_prompt(tickets: list[EscalateTicket]) -> str:
    parts = [
        "You are finishing PR review escalations. Groq already triaged each ticket; "
        "produce a deeper review per ticket. Return JSON only:",
        '{"reviews":[{"ticket_id":"...","summary":"...","score":0-10,'
        '"issues":[{"severity":"low|medium|high|critical","file":"...","line":0,'
        '"message":"...","rule":"...","suggestion":"..."}],'
        '"recommendations":["..."]}]}',
        "One entry per ticket_id. Be concrete; do not invent files not in the hunks.",
        "",
    ]
    for t in tickets:
        parts.append(f"=== TICKET {t.ticket_id} reason={t.reason} ===")
        parts.append(f"files: {', '.join(t.files) or '(unknown)'}")
        parts.append(f"groq_score={t.groq_score} complexity={t.complexity:.2f}")
        parts.append(f"groq_summary: {t.groq_summary}")
        if t.groq_issues:
            parts.append(f"groq_issues: {json.dumps(t.groq_issues)[:2000]}")
        parts.append("diff hunks:")
        parts.append(t.contested_diff[:12_000])
        parts.append("")
    return "\n".join(parts)


def parse_multiplex_response(
    content: str,
    tickets: list[EscalateTicket],
    *,
    model: str,
    latency_ms: float,
    tokens_used: int | None = None,
) -> dict[str, ReviewResult]:
    """Map ticket_id → ReviewResult. Missing tickets are omitted."""
    out: dict[str, ReviewResult] = {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # try extract JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            logger.warning("Multiplex escalate: no JSON in Gemini response")
            return out
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("Multiplex escalate: failed to parse Gemini JSON")
            return out

    reviews = data.get("reviews") or data.get("tickets") or []
    if isinstance(data, list):
        reviews = data
    by_id = {t.ticket_id: t for t in tickets}

    for item in reviews:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("ticket_id") or item.get("id") or "")
        if tid not in by_id:
            # single-ticket fallback: only one pending
            if len(tickets) == 1 and not tid:
                tid = tickets[0].ticket_id
            else:
                continue
        normalized = normalize_review_payload(item)
        issues = issues_from_parsed(normalized)
        score = float(normalized.get("score") or 0.0)
        out[tid] = ReviewResult(
            summary=str(normalized.get("summary") or ""),
            issues=issues,
            recommendations=list(normalized.get("recommendations") or []),
            score=score,
            latency_ms=latency_ms,
            model=model,
            tokens_used=tokens_used,
            review_type="gemini",
            parse_warning=normalized.get("parse_warning"),
        )
    return out
