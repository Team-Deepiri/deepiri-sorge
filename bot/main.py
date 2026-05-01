"""Main entry point for deepiri-sorge"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from bot.comment_poster import CommentPoster
from bot.config import Config
from bot.cpu_reviewer import CPUReviewer
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser
from bot.gpu_runner import GPURunner
from bot.runners import GitHubModelsRunner, GeminiRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="deepiri-sorge - Distributed AI PR Review Bot"
    )
    parser.add_argument(
        "--diff",
        type=str,
        help="Path to diff file or diff content",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sorge.toml",
        help="Path to config file (default: sorge.toml)",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        help="PR number for commenting",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository in format 'owner/repo'",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="GitHub token for API access",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't post comments, just print output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "cpu", "gpu", "github", "gemini", "skip"],
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
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    logger.info(f"deepiri-sorge v{__import__('bot').__version__}")

    config = Config.from_file(args.config) if Path(args.config).exists() else Config()
    logger.debug(f"Config: {config}")

    diff_content = load_diff(args.diff)
    logger.info(f"Loaded diff ({len(diff_content)} bytes)")

    diff_parser = DiffParser()
    parsed_diff = diff_parser.parse(diff_content)
    logger.info(
        f"Parsed diff: {parsed_diff.files_changed} files, "
        f"{parsed_diff.lines_added} additions, "
        f"{parsed_diff.lines_deleted} deletions"
    )

    decision_engine = DecisionEngine(config)
    decision = decision_engine.decide(parsed_diff)

    logger.info(f"Decision: {decision.action.value} - {decision.reason}")

    if decision.action == Action.SKIP and args.mode == "auto":
        logger.info("Skipping review")
        print(json.dumps({"action": "skip", "reason": decision.reason}))
        return

    review_result = None

    effective_mode = args.mode if args.mode != "auto" else decision.action.value

    if effective_mode in ("github", Action.GITHUB_MODELS.value):
        logger.info("Running GitHub Models review")
        runner = GitHubModelsRunner(
            api_key=config.github_models.api_key,
            model=config.github_models.model,
        )
        review_result = runner.review(parsed_diff)

    elif effective_mode in ("gemini", Action.GEMINI.value):
        logger.info("Running Gemini review")
        runner = GeminiRunner(
            api_key=config.gemini.api_key,
            model=config.gemini.model,
        )
        review_result = runner.review(parsed_diff)

    elif effective_mode in ("cpu", Action.CPU_REVIEW.value):
        logger.info("Running CPU review")
        reviewer = CPUReviewer(config)
        review_result = reviewer.review(parsed_diff)

    elif effective_mode in ("gpu", Action.GPU_REVIEW.value):
        logger.info("Running GPU review")
        gpu_runner = GPURunner(config)
        review_result = gpu_runner.review(parsed_diff)

    elif effective_mode == "skip":
        logger.info("Skipping review (--mode skip)")
        print(json.dumps({"action": "skip", "reason": "mode=skip"}))

    if review_result:
        logger.info(f"Review complete: {len(review_result.issues)} issues found")

        if args.pr_number and args.repo and not args.dry_run:
            poster = CommentPoster(args.token or "")
            poster.post_review(
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
