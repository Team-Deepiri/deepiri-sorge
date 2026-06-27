"""Main entry point for deepiri-sorge"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from bot.comment_poster import CommentPoster
from bot.config import Config
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser
from bot.runners import GeminiRunner, GroqRunner, OpenRouterRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="deepiri-sorge - Distributed AI PR Review Bot"
    )
    parser.add_argument("--diff", type=str, help="Path to diff file or diff content")
    parser.add_argument("--config", type=str, default="sorge.toml", help="Path to config file")
    parser.add_argument("--pr-number", type=int, help="PR number for commenting")
    parser.add_argument("--repo", type=str, help="Repository in format 'owner/repo'")
    parser.add_argument("--token", type=str, help="GitHub token for API access")
    parser.add_argument("--dry-run", action="store_true", help="Don't post comments, just print output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--mode",
        choices=["auto", "gemini", "openrouter", "groq", "skip"],
        default="auto",
        help="Review mode (default: auto)",
    )
    return parser.parse_args()


def load_diff(diff_arg: str | None) -> str:
    if not diff_arg:
        logger.error("No diff provided")
        sys.exit(1)

    path = Path(diff_arg)
    if path.exists() and path.is_file():
        return path.read_text()
    return diff_arg


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    logger.info(f"deepiri-sorge v{__import__('bot').__version__}")

    config = Config.from_file(args.config) if Path(args.config).exists() else Config()
    logger.debug(f"Config: {config}")

    diff_content = load_diff(args.diff)
    logger.info(f"Loaded diff ({len(diff_content)} bytes)")

    parsed_diff = DiffParser().parse(diff_content)
    logger.info(
        f"Parsed diff: {parsed_diff.files_changed} files, "
        f"+{parsed_diff.lines_added} -{parsed_diff.lines_deleted}"
    )

    decision = DecisionEngine(config).decide(parsed_diff)
    logger.info(f"Decision: {decision.action.value} - {decision.reason}")

    if decision.action == Action.SKIP and args.mode == "auto":
        logger.info("Skipping review")
        print(json.dumps({"action": "skip", "reason": decision.reason}))
        return

    review_result = None
    effective_mode = args.mode if args.mode != "auto" else decision.action.value
    cache_config = config.cache if config.cache.enabled else None

    if effective_mode in ("gemini", Action.GEMINI.value):
        logger.info("Running Gemini review")
        review_result = GeminiRunner(
            api_key=config.gemini.api_key,
            model=config.gemini.model,
            cache_config=cache_config,
        ).review(parsed_diff)

    elif effective_mode in ("openrouter", Action.OPENROUTER.value):
        logger.info("Running OpenRouter review")
        review_result = OpenRouterRunner(
            api_key=config.openrouter.api_key,
            model=config.openrouter.model,
            cache_config=cache_config,
        ).review(parsed_diff)

    elif effective_mode in ("groq", Action.GROQ.value):
        logger.info("Running Groq review")
        review_result = GroqRunner(
            api_key=config.groq.api_key,
            model=config.groq.model,
            cache_config=cache_config,
        ).review(parsed_diff)

    elif effective_mode == "skip":
        logger.info("Skipping review (--mode skip)")
        print(json.dumps({"action": "skip", "reason": "mode=skip"}))
        return

    if review_result:
        logger.info(f"Review complete: {len(review_result.issues)} issues found")

        if args.pr_number and args.repo and not args.dry_run:
            CommentPoster(args.token or "").post_review(
                repo=args.repo,
                pr_number=args.pr_number,
                review=review_result,
            )

        print(json.dumps(review_result.to_dict(), indent=2))
    else:
        logger.warning("No review result generated")


def review() -> None:
    main()


if __name__ == "__main__":
    main()