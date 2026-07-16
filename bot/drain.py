"""Gemini escalate drain — claim ledger tickets, multiplex, post upgrades."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

from bot.comment_poster import CommentPoster
from bot.config import Config
from bot.escalate_ledger import EscalateLedger
from bot.github_app import get_installation_token
from bot.quota_tracker import QuotaTracker
from bot.runners.gemini_runner import GeminiRunner
from bot.scheduling.escalate import EscalateTicket


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drain escalate ledger via Gemini multiplex")
    p.add_argument("--config", default="sorge.toml")
    p.add_argument("--limit", type=int, default=8, help="Max tickets per Gemini call")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _token_for_ticket(ticket: EscalateTicket, fallback: str | None) -> str | None:
    if ticket.installation_id:
        tok = get_installation_token(ticket.installation_id)
        if tok:
            return tok
    return fallback or os.getenv("GITHUB_TOKEN")


def run_drain(
    *,
    limit: int = 8,
    dry_run: bool = False,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    ledger = EscalateLedger()
    tickets = ledger.claim(limit=max(1, limit))
    if not tickets:
        logger.info("Drain: no pending escalate tickets")
        return {"claimed": 0, "upgraded": 0, "gemini_calls": 0}

    quota = QuotaTracker.from_config(config.quota)
    if not quota.can_use("gemini"):
        logger.warning("Drain: Gemini quota exhausted — returning tickets to pending")
        ledger.ack([t.ticket_id for t in tickets], status="pending")
        return {
            "claimed": len(tickets),
            "upgraded": 0,
            "gemini_calls": 0,
            "requeued": True,
        }

    runner = GeminiRunner(
        api_key=config.gemini.api_key,
        model=config.gemini.model,
    )
    upgraded = runner.review_escalate_batch(tickets)
    quota.record("gemini")

    posted = 0
    done_ids: list[str] = []
    fail_ids: list[str] = []
    fallback_token = os.getenv("GITHUB_TOKEN")

    for ticket in tickets:
        result = upgraded.get(ticket.ticket_id)
        if result is None:
            fail_ids.append(ticket.ticket_id)
            continue
        note = (
            f"_Upgraded review (Gemini multiplex drain; prior Groq triage "
            f"score={ticket.groq_score}, reason={ticket.reason})._"
        )
        recs = list(result.recommendations or [])
        if note not in recs:
            recs.insert(0, note)
        result.recommendations = recs

        if dry_run:
            logger.info(
                f"Dry-run upgrade {ticket.repo}#{ticket.pr_number} "
                f"ticket={ticket.ticket_id} score={result.score}"
            )
            done_ids.append(ticket.ticket_id)
            posted += 1
            continue

        token = _token_for_ticket(ticket, fallback_token)
        if not token or not ticket.repo or not ticket.pr_number:
            logger.warning(f"Cannot post upgrade for ticket {ticket.ticket_id}")
            fail_ids.append(ticket.ticket_id)
            continue
        ok = CommentPoster(token).post_review(
            ticket.repo,
            ticket.pr_number,
            result,
            edit_existing=True,
        )
        if ok is not None:
            done_ids.append(ticket.ticket_id)
            posted += 1
        else:
            fail_ids.append(ticket.ticket_id)

    if done_ids:
        ledger.ack(done_ids, status="done")
    if fail_ids:
        ledger.ack(fail_ids, status="pending")

    summary = {
        "claimed": len(tickets),
        "upgraded": posted,
        "gemini_calls": 1,
        "failed": len(fail_ids),
        "ticket_ids": [t.ticket_id for t in tickets],
    }
    logger.info(f"Drain complete: {summary}")
    return summary


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    config = Config()
    if Path(args.config).exists():
        config = Config.from_file(args.config)
    env_config = Config.from_env()
    for key in env_config.model_dump(exclude_unset=True, by_alias=False):
        setattr(config, key, getattr(env_config, key))

    summary = run_drain(limit=args.limit, dry_run=args.dry_run, config=config)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
