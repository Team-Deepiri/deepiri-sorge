"""Tests for Gemini escalate multiplex + ledger."""

from __future__ import annotations

from pathlib import Path

from bot.escalate_ledger import EscalateLedger
from bot.scheduling.escalate import (
    EscalateTicket,
    build_multiplex_prompt,
    new_ticket_id,
    parse_multiplex_response,
    truncate_diff,
)


def test_truncate_diff_keeps_ends():
    raw = "A" * 100 + "MID" + "B" * 100
    out = truncate_diff(raw, max_chars=50)
    assert "truncated" in out
    assert out.startswith("A")
    assert out.endswith("B")


def test_parse_multiplex_response_maps_ticket_ids():
    t1 = EscalateTicket(
        ticket_id="abc123",
        reason="vacuous",
        files=["a.py"],
        estimated_tokens=100,
        complexity=0.4,
        priority=50,
        groq_summary="ok",
        groq_score=10.0,
        groq_issues=[],
        contested_diff="+x",
    )
    content = """
    {"reviews":[{"ticket_id":"abc123","summary":"deeper","score":8.5,
      "issues":[{"severity":"low","file":"a.py","message":"nits"}],
      "recommendations":["ship"]}]}
    """
    out = parse_multiplex_response(content, [t1], model="gemini-test", latency_ms=12.0)
    assert "abc123" in out
    assert out["abc123"].score == 8.5
    assert out["abc123"].review_type == "gemini"
    assert len(out["abc123"].issues) == 1


def test_build_multiplex_prompt_includes_ticket_ids():
    t = EscalateTicket(
        ticket_id="tid99",
        reason="security",
        files=["auth.ts"],
        estimated_tokens=500,
        complexity=0.7,
        priority=100,
        groq_summary="thin",
        groq_score=9.0,
        groq_issues=[],
        contested_diff="+jwt",
    )
    prompt = build_multiplex_prompt([t])
    assert "tid99" in prompt
    assert "security" in prompt
    assert "+jwt" in prompt


def test_file_ledger_enqueue_claim_ack(tmp_path: Path):
    path = tmp_path / "ledger.json"
    ledger = EscalateLedger(file_path=path)
    tid = new_ticket_id()
    ticket = EscalateTicket(
        ticket_id=tid,
        reason="low_score",
        files=["x.py"],
        estimated_tokens=200,
        complexity=0.3,
        priority=50,
        groq_summary="s",
        groq_score=5.0,
        groq_issues=[],
        contested_diff="+1",
        repo="Team-Deepiri/demo",
        pr_number=7,
        installation_id=1,
        head_sha="abc",
    )
    assert ledger.enqueue([ticket]) == 1
    assert ledger.pending_count() == 1
    claimed = ledger.claim(limit=5)
    assert len(claimed) == 1
    assert claimed[0].ticket_id == tid
    assert ledger.pending_count() == 0
    ledger.ack([tid], status="done")
    assert ledger.claim(limit=5) == []


def test_file_ledger_cancel_pr_supersedes(tmp_path: Path):
    path = tmp_path / "ledger.json"
    ledger = EscalateLedger(file_path=path)
    t1 = EscalateTicket(
        ticket_id="one",
        reason="vacuous",
        files=["a.py"],
        estimated_tokens=1,
        complexity=0.1,
        priority=50,
        groq_summary="",
        groq_score=10.0,
        groq_issues=[],
        contested_diff="+",
        repo="org/r",
        pr_number=3,
    )
    ledger.enqueue([t1])
    assert ledger.cancel_pr("org/r", 3) == 1
    assert ledger.pending_count() == 0
