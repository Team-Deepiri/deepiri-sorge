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
from bot.repo_context import RepoContextWeaver
from bot.runners import GeminiRunner, GroqRunner, OpenRouterRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="deepiri-sorge - Distributed AI PR Review Bot"
    )
    parser.add_argument("--diff", type=str, help="Path to diff file or diff content")
    parser.add_argument("--config", type=str, default="sorge.toml", help="Path to config file")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root for system-context weaving (default: cwd)",
    )
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

    engine = DecisionEngine(config)
    decision = engine.decide(parsed_diff)
    logger.info(f"Decision: {decision.action.value} - {decision.reason}")

    if decision.action == Action.SKIP and args.mode == "auto":
        logger.info("Skipping review")
        print(json.dumps({"action": "skip", "reason": decision.reason}))
        return

    context_pack = RepoContextWeaver(config.repo_context).weave(
        Path(args.repo_root),
        parsed_diff,
    )
    repo_context_text = context_pack.text
    context_fingerprint = context_pack.fingerprint

    if args.mode == "auto" and context_pack.text:
        routed = engine.decide(parsed_diff, extra_chars=len(context_pack.text))
        if routed.action != decision.action:
            logger.info(f"Re-routed after repo context: {routed.reason}")
        decision = routed
        logger.info(f"Final decision: {decision.action.value} - {decision.reason}")

    review_result = None
    cache_config = config.cache if config.cache.enabled else None

    review_kwargs = {
        "repo_context": repo_context_text or None,
        "context_fingerprint": context_fingerprint,
    }

    # Build runner lookup
    runners = {
        Action.GEMINI: lambda: GeminiRunner(
            api_key=config.gemini.api_key,
            model=config.gemini.model,
            cache_config=cache_config,
        ).review(parsed_diff, **review_kwargs),
        Action.OPENROUTER: lambda: OpenRouterRunner(
            api_key=config.openrouter.api_key,
            model=config.openrouter.model,
            cache_config=cache_config,
        ).review(parsed_diff, **review_kwargs),
        Action.GROQ: lambda: GroqRunner(
            api_key=config.groq.api_key,
            model=config.groq.model,
            cache_config=cache_config,
        ).review(parsed_diff, **review_kwargs),
    }

    if args.mode == "auto":
        # Use preference chain from decision engine for runtime failover
        # Each provider gets 3 retries (via post_with_retry inside runner)
        chain = engine.get_preference_chain(
            engine._estimate_tokens(parsed_diff, extra_chars=len(repo_context_text or ""))
        )

        for action, enabled in chain:
            if not enabled:
                continue
            logger.info(f"Trying {action.value} review")
            try:
                result = runners[action]()
                if result is not None:
                    review_result = result
                    logger.info(f"Review complete via {action.value}: {len(review_result.issues)} issues found")
                    break
            except Exception as e:
                logger.warning(f"{action.value} failed ({e}), trying next provider...")
                continue
            logger.warning(f"{action.value} returned no result, trying next provider...")

        if not review_result:
            logger.critical(
                "All providers exhausted — review failed "
                "(verify API keys, provider status, and diff size)"
            )
            sys.exit(2)
    else:
        # Explicit mode — run only the requested provider
        effective_mode = args.mode
        runner_fn = runners.get(Action(effective_mode))
        if runner_fn:
            logger.info(f"Running {effective_mode} review (--mode {effective_mode})")
            review_result = runner_fn()
        elif effective_mode == "skip":
            logger.info("Skipping review (--mode skip)")
            print(json.dumps({"action": "skip", "reason": "mode=skip"}))
            return
        else:
            logger.critical(f"Unknown mode: {effective_mode}")
            sys.exit(2)

        if review_result:
            logger.info(f"Review complete: {len(review_result.issues)} issues found")
        else:
            logger.critical(
                f"Review failed: no result from {effective_mode} runner "
                "(verify API keys, provider status, and diff size)"
            )
            sys.exit(2)

    if args.pr_number and args.repo and not args.dry_run:
        CommentPoster(args.token or "").post_review(
            repo=args.repo,
            pr_number=args.pr_number,
            review=review_result,
        )

    print(json.dumps(review_result.to_dict(), indent=2))


def review() -> None:
    main()


if __name__ == "__main__":
    main()