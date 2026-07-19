"""Post-review finding adjudicator — LLM judgment, not keyword filters.

After the primary review, classify each finding given deploy facts from the
repo (Redis present, compose topology, etc.). Drop speculative theater; keep
actionable defects and present-tense architecture gaps (e.g. process-local
state when Redis already exists).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from bot.schemas import ReviewIssue, ReviewResult, compute_score_from_issues
from bot.utils.http_retry import post_with_retry
from bot.utils.response_parser import _extract_json_object

CompleteFn = Callable[[str], dict[str, Any] | None]

_ADJUDICATE_SYSTEM = """You adjudicate code-review findings. Return ONE raw JSON object only.
For each finding decide:
- keep: real defect, security, broken contract, race, or missing test
- keep_architecture: present-tense gap vs how this repo already deploys
  (multi-worker, shared Redis/cache, reconnect). NOT "someday if N is huge".
- demote: soft polish; keep but severity must become low
- drop: speculative micro-opt, lock theater when I/O is outside the lock,
  re-litigating a fix already in the diff, or hypothetical scale with no
  deploy evidence

Prefer keep_architecture when deploy_facts show Redis (or similar) and the
finding is about process-local membership/session/auth that should be shared.
Prefer drop for finer locks / profiling-gated advice / "if peers grow enormously".
"""


@dataclass
class AdjudicationRecord:
    index: int
    action: str
    reason: str


@dataclass
class AdjudicationReport:
    kept: list[ReviewIssue] = field(default_factory=list)
    dropped: list[AdjudicationRecord] = field(default_factory=list)
    demoted: list[AdjudicationRecord] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.dropped or self.demoted)


def collect_deploy_facts(repo_root: Path | None) -> dict[str, Any]:
    """Inventory how the target repo actually runs — evidence for adjudication."""
    facts: dict[str, Any] = {
        "has_redis_dep": False,
        "has_redis_url_config": False,
        "compose_mentions_redis": False,
        "compose_files": [],
        "notes": [],
    }
    if repo_root is None or not repo_root.is_dir():
        return facts

    root = repo_root.resolve()

    for name in ("pyproject.toml", "requirements.txt", "package.json", "Cargo.toml", "go.mod"):
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if "redis" in text:
            facts["has_redis_dep"] = True
            facts["notes"].append(f"redis referenced in {name}")

    for rel in (
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
        "docker/docker-compose.yml",
        "docker/docker-compose.yaml",
    ):
        path = root / rel
        if not path.is_file():
            continue
        facts["compose_files"].append(rel)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if "redis" in text:
            facts["compose_mentions_redis"] = True
            facts["notes"].append(f"redis service/config in {rel}")

    for rel in ("config.py", "settings.py", ".env.example"):
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "REDIS" in text or "redis" in text.lower():
            facts["has_redis_url_config"] = True
            facts["notes"].append(f"REDIS/redis config in {rel}")

    # Shallow scan for config modules mentioning Redis without hardcoding one repo.
    for path in root.rglob("config.py"):
        try:
            if path.relative_to(root).parts[0] in {".venv", "node_modules", "venv", ".git"}:
                continue
        except ValueError:
            continue
        if path.stat().st_size > 200_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "REDIS" in text or "Redis" in text:
            facts["has_redis_url_config"] = True
            facts["notes"].append(f"Redis config in {path.relative_to(root)}")
            break

    return facts


def _default_complete(prompt: str) -> dict[str, Any] | None:
    """Best-effort Groq then OpenRouter JSON completion."""
    attempts: list[tuple[str, str, str]] = []
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        attempts.append(
            (
                groq_key,
                os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"),
                os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            )
        )
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        attempts.append(
            (
                or_key,
                os.getenv(
                    "OPENROUTER_ENDPOINT",
                    "https://openrouter.ai/api/v1/chat/completions",
                ),
                os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
            )
        )

    for api_key, endpoint, model in attempts:
        try:
            resp = post_with_retry(
                endpoint,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _ADJUDICATE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1500,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
                max_retries=1,
            )
            if resp.status_code >= 400:
                logger.debug(f"Adjudicator HTTP {resp.status_code} from {endpoint}")
                continue
            data = resp.json()
            content = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            )
            parsed = _extract_json_object(content)
            if isinstance(parsed, dict):
                return parsed
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.debug(f"Adjudicator complete failed: {exc}")
            continue
    return None


def _build_prompt(
    issues: list[ReviewIssue],
    deploy_facts: dict[str, Any],
) -> str:
    payload = {
        "deploy_facts": deploy_facts,
        "findings": [
            {
                "index": i,
                "severity": issue.severity,
                "file": issue.file,
                "line": issue.line,
                "rule": issue.rule,
                "message": issue.message,
                "suggestion": issue.suggestion,
            }
            for i, issue in enumerate(issues)
        ],
        "response_schema": {
            "decisions": [
                {
                    "index": "int matching findings[].index",
                    "action": "keep|keep_architecture|demote|drop",
                    "reason": "short string",
                }
            ]
        },
    }
    return (
        "Adjudicate these review findings using deploy_facts as evidence.\n"
        + json.dumps(payload, indent=2)
    )


def apply_decisions(
    issues: list[ReviewIssue],
    decisions: list[dict[str, Any]],
) -> AdjudicationReport:
    """Apply adjudicator decisions; unknown/missing actions default to keep."""
    by_index: dict[int, dict[str, Any]] = {}
    for raw in decisions:
        if not isinstance(raw, dict):
            continue
        try:
            idx = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[idx] = raw

    report = AdjudicationReport()
    for i, issue in enumerate(issues):
        decision = by_index.get(i) or {}
        action = str(decision.get("action") or "keep").strip().lower()
        reason = str(decision.get("reason") or "")[:200]

        if action == "drop":
            report.dropped.append(AdjudicationRecord(i, action, reason))
            continue
        if action == "demote":
            report.demoted.append(AdjudicationRecord(i, action, reason))
            report.kept.append(
                ReviewIssue(
                    severity="low",
                    file=issue.file,
                    line=issue.line,
                    message=issue.message,
                    rule=issue.rule,
                    suggestion=issue.suggestion,
                )
            )
            continue
        if action == "keep_architecture" and issue.severity == "low":
            # Architecture gaps against current deploy deserve at least medium.
            report.kept.append(
                ReviewIssue(
                    severity="medium",
                    file=issue.file,
                    line=issue.line,
                    message=issue.message,
                    rule=issue.rule or "Architecture",
                    suggestion=issue.suggestion,
                )
            )
            continue
        report.kept.append(issue)
    return report


class FindingAdjudicator:
    """LLM post-filter for review findings. Fails open on errors."""

    def __init__(self, complete: CompleteFn | None = None):
        self._complete = complete or _default_complete

    def adjudicate_result(
        self,
        result: ReviewResult,
        *,
        repo_root: Path | None = None,
        deploy_facts: dict[str, Any] | None = None,
    ) -> ReviewResult:
        if not result.issues:
            return result

        facts = deploy_facts if deploy_facts is not None else collect_deploy_facts(repo_root)
        prompt = _build_prompt(result.issues, facts)
        try:
            parsed = self._complete(prompt)
        except Exception as exc:
            logger.warning(f"FindingAdjudicator failed open: {exc}")
            return result

        if not isinstance(parsed, dict):
            logger.info("FindingAdjudicator: no usable JSON; keeping all findings")
            return result

        decisions = parsed.get("decisions")
        if not isinstance(decisions, list):
            logger.info("FindingAdjudicator: missing decisions[]; keeping all findings")
            return result

        report = apply_decisions(result.issues, decisions)
        if not report.changed and len(report.kept) == len(result.issues):
            return result

        logger.info(
            f"FindingAdjudicator: {len(result.issues)} in → {len(report.kept)} out "
            f"({len(report.dropped)} dropped, {len(report.demoted)} demoted)"
        )
        for rec in report.dropped:
            logger.info(f"  dropped[{rec.index}]: {rec.reason}")
        for rec in report.demoted:
            logger.info(f"  demoted[{rec.index}]: {rec.reason}")

        meta = dict(result.routing_meta or {})
        meta["finding_adjudicator"] = {
            "dropped": len(report.dropped),
            "demoted": len(report.demoted),
            "details": [
                {"index": r.index, "action": r.action, "reason": r.reason}
                for r in [*report.dropped, *report.demoted]
            ],
        }

        return ReviewResult(
            summary=result.summary,
            issues=report.kept,
            recommendations=result.recommendations,
            score=compute_score_from_issues(report.kept),
            latency_ms=result.latency_ms,
            model=result.model,
            tokens_used=result.tokens_used,
            review_type=result.review_type,
            parse_warning=result.parse_warning,
            routing_meta=meta,
        )
