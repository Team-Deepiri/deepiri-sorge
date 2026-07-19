"""Tests for LLM finding adjudicator (no keyword regex filter)."""

from __future__ import annotations

from pathlib import Path

from bot.finding_adjudicator import (
    FindingAdjudicator,
    apply_decisions,
    collect_deploy_facts,
)
from bot.schemas import ReviewIssue, ReviewResult


def _issue(**kwargs) -> ReviewIssue:
    base = dict(
        severity="medium",
        file="a.py",
        line=1,
        message="something",
        rule="Architecture",
        suggestion="fix it",
    )
    base.update(kwargs)
    return ReviewIssue(**base)


def test_apply_drop_demote_keep_architecture():
    issues = [
        _issue(message="global lock contention someday"),
        _issue(message="naming nit", severity="medium"),
        _issue(message="process-local membership with Redis in repo", severity="low"),
        _issue(message="auth bypass", severity="high"),
    ]
    report = apply_decisions(
        issues,
        [
            {"index": 0, "action": "drop", "reason": "speculative"},
            {"index": 1, "action": "demote", "reason": "polish"},
            {"index": 2, "action": "keep_architecture", "reason": "deploy gap"},
            {"index": 3, "action": "keep", "reason": "real bug"},
        ],
    )
    assert len(report.dropped) == 1
    assert len(report.demoted) == 1
    assert len(report.kept) == 3
    assert report.kept[0].severity == "low"
    assert report.kept[1].severity == "medium"  # promoted from low
    assert report.kept[2].severity == "high"


def test_adjudicator_fail_open_on_bad_json():
    result = ReviewResult(
        summary="ok",
        issues=[_issue()],
        recommendations=[],
        score=9.0,
        latency_ms=1.0,
        model="test",
    )
    out = FindingAdjudicator(complete=lambda _p: None).adjudicate_result(result)
    assert len(out.issues) == 1


def test_adjudicator_applies_llm_decisions():
    result = ReviewResult(
        summary="ok",
        issues=[
            _issue(message="finer locks until profiling"),
            _issue(message="use shared Redis for membership"),
        ],
        recommendations=[],
        score=8.0,
        latency_ms=1.0,
        model="test",
    )

    def fake_complete(_prompt: str):
        return {
            "decisions": [
                {"index": 0, "action": "drop", "reason": "hypothetical"},
                {"index": 1, "action": "keep_architecture", "reason": "redis exists"},
            ]
        }

    out = FindingAdjudicator(complete=fake_complete).adjudicate_result(result)
    assert len(out.issues) == 1
    assert "Redis" in out.issues[0].message
    assert out.routing_meta["finding_adjudicator"]["dropped"] == 1


def test_collect_deploy_facts_sees_redis(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('dependencies = ["redis>=5"]\n')
    (tmp_path / "docker-compose.yml").write_text("services:\n  redis:\n    image: redis\n")
    facts = collect_deploy_facts(tmp_path)
    assert facts["has_redis_dep"] is True
    assert facts["compose_mentions_redis"] is True
