"""Main entry point for deepiri-sorge"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

from bot.claim_verifier import ClaimVerifier
from bot.comment_poster import CommentPoster
from bot.config import Config
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser
from bot.file_splitter import FileSplitter
from bot.github_app import fetch_pr_diff, get_installation_token
from bot.providers import build_providers
from bot.quota_tracker import QuotaTracker
from bot.repo_context import RepoContextWeaver
from bot.review_aggregator import ReviewAggregator
from bot.scheduling.market_score import TEMPLATE_OVERHEAD_TOKENS
from bot.scheduling.scheduler import ReviewScheduler
from bot.schemas import ReviewResult as ReviewResultSchema
from bot.symbol_index import SymbolIndexer, format_symbol_index


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
    parser.add_argument(
        "--installation-id",
        type=int,
        help="GitHub App installation ID (uses App token instead of --token)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't post comments, just print output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--mode",
        choices=["auto", "gemini", "openrouter", "groq", "skip"],
        default="auto",
        help="Review mode (default: auto)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run review even when filters would skip (e.g. /sorge command)",
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


def resolve_github_token(args: argparse.Namespace) -> str:
    if args.installation_id:
        token = get_installation_token(args.installation_id)
        if token:
            return token
        logger.warning("App installation token failed; falling back to --token / GITHUB_TOKEN")
    token = args.token or os.getenv("GITHUB_TOKEN")
    if not token:
        logger.critical("No GitHub token available. Provide --token, --installation-id, or set GITHUB_TOKEN.")
        sys.exit(1)
    return token


def run_auto_review(
    parsed_diff,
    config: Config,
    engine: DecisionEngine,
    *,
    repo_context: str | None,
    context_fingerprint: str,
    extra_chars: int = 0,
    only_provider: str | None = None,
    repo: str = "",
    pr_number: int = 0,
    installation_id: int | None = None,
    head_sha: str = "",
) -> ReviewResultSchema | None:
    """Phase 1: provider-centric scheduler owns dispatch — no worker fallback chain."""
    cache_config = config.cache if config.cache.enabled else None
    quota = QuotaTracker.from_config(config.quota)

    splitter = FileSplitter(
        chunk_budget=config.routing.chunk_budget,
        max_chunk_tokens=config.routing.medium_pr_threshold,
    )
    chunks = splitter.split(parsed_diff)
    unreviewable = [c for c in chunks if c.unreviewable]
    runnable = [c for c in chunks if not c.unreviewable]

    providers = build_providers(config, cache_config=cache_config, only=only_provider)
    if not providers:
        logger.error("No providers enabled")
        return None

    if not runnable:
        return ReviewAggregator.merge(
            [],
            rung="scheduled",
            quota_snapshot=quota.snapshot(),
            unreviewable=unreviewable or None,
        )

    scheduler = ReviewScheduler.from_providers(
        providers,
        quota,
        config,
        repo_context=repo_context,
        context_fingerprint=context_fingerprint,
        prompt_overhead_tokens=TEMPLATE_OVERHEAD_TOKENS + max(0, extra_chars // 4),
        repo=repo,
        pr_number=pr_number,
        installation_id=installation_id,
        head_sha=head_sha,
    )
    logger.info(
        f"Scheduler starting: {len(runnable)} chunk(s), "
        f"providers={[p.name for p in providers]}"
    )
    results, skipped, meta = scheduler.run(runnable)

    return ReviewAggregator.merge(
        results,
        rung=meta.rung,
        quota_snapshot=quota.snapshot(),
        unreviewable=unreviewable or None,
        skipped=skipped or None,
        scheduler_meta=meta,
    )


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    logger.info(f"deepiri-sorge v{__import__('bot').__version__}")

    # Precedence: env vars > file values > pydantic defaults
    config = Config()
    if Path(args.config).exists():
        config = Config.from_file(args.config)
    env_config = Config.from_env()
    # Merge env values into the config object, preserving nested model types
    for key in env_config.model_dump(exclude_unset=True, by_alias=False):
        setattr(config, key, getattr(env_config, key))

    github_token = resolve_github_token(args)

    if args.repo and args.pr_number and not args.diff:
        if not github_token:
            logger.error("Need --token or --installation-id to fetch PR diff")
            sys.exit(1)
        diff_content = fetch_pr_diff(args.repo, args.pr_number, github_token)
    else:
        diff_content = load_diff(args.diff)

    logger.info(f"Loaded diff ({len(diff_content)} bytes)")

    parsed_diff = DiffParser().parse(diff_content)
    logger.info(
        f"Parsed diff: {parsed_diff.files_changed} files, "
        f"+{parsed_diff.lines_added} -{parsed_diff.lines_deleted}"
    )

    engine = DecisionEngine(config)
    decision = engine.decide(parsed_diff)
    if args.mode == "auto":
        logger.info(
            f"Filter decision: {decision.action.value} - {decision.reason} "
            "(auto mode: ReviewScheduler selects providers)"
        )
    else:
        logger.info(f"Decision: {decision.action.value} - {decision.reason}")

    if decision.action == Action.SKIP and args.mode == "auto" and not args.force:
        logger.info("Skipping review")
        print(json.dumps({"action": "skip", "reason": decision.reason}), file=sys.stderr)
        return

    if decision.action == Action.SKIP and args.force:
        logger.info(f"Force review — overriding skip: {decision.reason}")

    repo_root = Path(args.repo_root)
    context_pack = RepoContextWeaver(config.repo_context).weave(
        repo_root,
        parsed_diff,
    )
    repo_context_text = context_pack.text
    context_fingerprint = context_pack.fingerprint

    symbol_indexes = []
    if config.claim_verifier.enabled or config.claim_verifier.include_symbol_index:
        symbol_indexes = SymbolIndexer().index_files(repo_root, parsed_diff.files)
        if config.claim_verifier.include_symbol_index and symbol_indexes:
            index_block = format_symbol_index(
                symbol_indexes,
                max_chars=config.claim_verifier.max_index_chars,
            )
            if index_block:
                repo_context_text = (
                    f"{repo_context_text}\n\n{index_block}"
                    if repo_context_text
                    else index_block
                )
                context_fingerprint = f"{context_fingerprint}:{len(index_block)}"

    extra_chars = len(repo_context_text or "")

    review_result: ReviewResultSchema | None = None

    if args.mode == "auto":
        review_result = run_auto_review(
            parsed_diff,
            config,
            engine,
            repo_context=repo_context_text or None,
            context_fingerprint=context_fingerprint,
            extra_chars=extra_chars,
            repo=args.repo or "",
            pr_number=args.pr_number or 0,
            installation_id=args.installation_id,
            head_sha=os.getenv("SORGE_HEAD_SHA", ""),
        )
        if not review_result:
            logger.critical("All providers exhausted — review failed")
            if not args.dry_run and github_token and args.repo and args.pr_number:
                CommentPoster(github_token).post_comment(
                    args.repo,
                    args.pr_number,
                    "## Sorge AI Code Review\n\n"
                    "> :x: **Review failed** — all AI providers were exhausted or unavailable.\n\n"
                    "Possible causes:\n"
                    "- API key misconfiguration (check `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`)\n"
                    "- Rate limits exceeded (check quota settings in `sorge.toml`)\n"
                    "- Network errors reaching the provider APIs\n\n",
                )
            sys.exit(2)
        logger.info(f"Review complete: {len(review_result.issues)} issues found")
    elif args.mode in ("gemini", "openrouter", "groq"):
        review_result = run_auto_review(
            parsed_diff,
            config,
            engine,
            repo_context=repo_context_text or None,
            context_fingerprint=context_fingerprint,
            extra_chars=extra_chars,
            only_provider=args.mode,
            repo=args.repo or "",
            pr_number=args.pr_number or 0,
            installation_id=args.installation_id,
            head_sha=os.getenv("SORGE_HEAD_SHA", ""),
        )
        if not review_result:
            logger.critical(f"Review failed: no result from {args.mode}")
            if not args.dry_run and github_token and args.repo and args.pr_number:
                CommentPoster(github_token).post_comment(
                    args.repo,
                    args.pr_number,
                    f"## Sorge AI Code Review\n\n"
                    f"> :x: **Review failed** — provider `{args.mode}` returned no result.\n\n"
                    "Possible causes:\n"
                    f"- `{args.mode}` API key missing or invalid\n"
                    "- Rate limits or quota exhaustion\n"
                    "- Network errors reaching the provider\n\n",
                )
            sys.exit(2)
        logger.info(f"Review complete: {len(review_result.issues)} issues found")
    elif args.mode == "skip":
        logger.info("Skipping review (--mode skip)")
        print(json.dumps({"action": "skip", "reason": "mode=skip"}), file=sys.stderr)
        return
    else:
        logger.critical(f"Unknown mode: {args.mode}")
        sys.exit(2)

    if review_result and config.claim_verifier.enabled:
        before = len(review_result.issues)
        review_result = ClaimVerifier().verify_result(
            review_result,
            repo_root=repo_root,
            changed_paths=parsed_diff.files,
            indexes=symbol_indexes or None,
        )
        suppressed = before - len(review_result.issues)
        logger.info(
            f"ClaimVerifier: {before} issues in → {len(review_result.issues)} out "
            f"({suppressed} suppressed)"
        )

    if args.pr_number and args.repo and not args.dry_run:
        CommentPoster(github_token).post_review(
            repo=args.repo,
            pr_number=args.pr_number,
            review=review_result,
        )

    print(json.dumps(review_result.to_dict(), indent=2))


def review() -> None:
    main()


if __name__ == "__main__":
    main()
