"""Main entry point for deepiri-sorge"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from bot.comment_poster import CommentPoster
from bot.config import Config
from bot.context_router import ContextRouter, RoutingPlan
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser
from bot.file_splitter import FileSplitter, ReviewChunk
from bot.github_app import fetch_pr_diff, get_installation_token
from bot.quota_tracker import QuotaTracker
from bot.repo_context import RepoContextWeaver
from bot.review_aggregator import ReviewAggregator
from bot.runners import GeminiRunner, GroqRunner, OpenRouterRunner
from bot.runners.base import ReviewResult
from bot.schemas import ReviewResult as ReviewResultSchema


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


def _make_runner(action: Action, config: Config, cache_config):
    if action == Action.GEMINI:
        return GeminiRunner(
            api_key=config.gemini.api_key,
            model=config.gemini.model,
            cache_config=cache_config,
        )
    if action == Action.OPENROUTER:
        return OpenRouterRunner(
            api_key=config.openrouter.api_key,
            model=config.openrouter.model,
            cache_config=cache_config,
        )
    if action == Action.GROQ:
        return GroqRunner(
            api_key=config.groq.api_key,
            model=config.groq.model,
            cache_config=cache_config,
        )
    return None


def execute_plan(
    plan: RoutingPlan,
    config: Config,
    quota: QuotaTracker,
    *,
    repo_context: str | None,
    context_fingerprint: str,
    cache_config,
    engine: DecisionEngine,
) -> tuple[list[ReviewResult], list[ReviewChunk]]:
    results: list[ReviewResult] = []
    unreviewable = [a.chunk for a in plan.assignments if a.chunk.unreviewable]
    runnable = [a for a in plan.assignments if not a.chunk.unreviewable]

    if not runnable:
        return results, unreviewable

    def run_one(assignment):
        # For OpenRouter, iterate through all configured models
        # Each model gets up to 3 attempts with exponential backoff (1.5×)
        # Fallback models (index > 0) use a shorter timeout to fail fast
        if assignment.action == Action.OPENROUTER:
            models = config.openrouter.models
            for i, model in enumerate(models):
                for attempt in range(1, 4):
                    runner = OpenRouterRunner(
                        api_key=config.openrouter.api_key,
                        model=model,
                        cache_config=cache_config,
                        http_retries=1,
                        http_timeout=120 if i == 0 else 30,
                        use_structured_output=i == 0,
                    )
                    logger.info(
                        f"OpenRouter attempt {attempt}/3 with model {i+1}/{len(models)}: {model}"
                    )
                    result = runner.review(
                        assignment.chunk.parsed_diff,
                        repo_context=repo_context,
                        context_fingerprint=context_fingerprint,
                    )
                    if result and not result.parse_warning:
                        quota.record(assignment.action.value)
                        # Record which model succeeded for observability
                        result.routing_meta = result.routing_meta or {}
                        result.routing_meta["openrouter_model"] = model
                        if i > 0:
                            result.routing_meta["openrouter_rotation"] = True
                        return result
                    if attempt < 3:
                        import time
                        wait = 1.5 * attempt
                        logger.warning(
                            f"OpenRouter model {model} attempt {attempt}/3 failed"
                            + (f" ({result.parse_warning})" if result and result.parse_warning else "")
                            + f"; retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
                logger.warning(
                    f"OpenRouter model {model} exhausted after 3 attempts — trying next model"
                )

            logger.warning("All OpenRouter models exhausted — falling through to fallback chain")
        else:
            # Non-OpenRouter: try the assigned provider first
            runner = _make_runner(assignment.action, config, cache_config)
            if runner:
                result = runner.review(
                    assignment.chunk.parsed_diff,
                    repo_context=repo_context,
                    context_fingerprint=context_fingerprint,
                )
                if result:
                    quota.record(assignment.action.value)
                    return result

        # Fallback: try other providers in preference chain
        chain = engine.get_preference_chain(assignment.chunk.estimated_tokens)
        for action, enabled in chain:
            if action == assignment.action:
                continue  # already tried above (including all OpenRouter models)
            if not enabled or not quota.can_use(action.value):
                continue
            runner = _make_runner(action, config, cache_config)
            if not runner:
                continue
            logger.warning(
                f"Falling back from {assignment.action.value} to {action.value} "
                f"for chunk ({assignment.chunk.estimated_tokens} tokens)"
            )
            result = runner.review(
                assignment.chunk.parsed_diff,
                repo_context=repo_context,
                context_fingerprint=context_fingerprint,
            )
            if result:
                quota.record(action.value)
                return result

        return None

    if len(runnable) == 1:
        r = run_one(runnable[0])
        if r:
            results.append(r)
        return results, unreviewable

    with ThreadPoolExecutor(max_workers=min(4, len(runnable))) as pool:
        futures = {pool.submit(run_one, a): a for a in runnable}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as e:
                logger.warning(f"Chunk review failed: {e}")

    return results, unreviewable


def run_auto_review(
    parsed_diff,
    config: Config,
    engine: DecisionEngine,
    *,
    repo_context: str | None,
    context_fingerprint: str,
    extra_chars: int = 0,
) -> ReviewResult | None:
    cache_config = config.cache if config.cache.enabled else None
    quota = QuotaTracker.from_config(config.quota)

    metrics = engine.compute_metrics(parsed_diff, extra_chars=extra_chars)
    effective = metrics.effective_tokens

    splitter = FileSplitter(
        chunk_budget=config.routing.chunk_budget,
        max_chunk_tokens=config.routing.medium_pr_threshold,
    )
    chunks = splitter.split(parsed_diff)

    router = ContextRouter(
        routing=config.routing,
        groq_enabled=bool(config.groq.enabled),
        openrouter_enabled=bool(config.openrouter.enabled),
        gemini_enabled=bool(config.gemini.enabled),
    )
    plan = router.route(metrics, chunks, quota)

    if not plan.assignments:
        chain = engine.get_preference_chain(effective)
        for action, enabled in chain:
            if not enabled or not quota.can_use(action.value):
                continue
            runner = _make_runner(action, config, cache_config)
            if not runner:
                continue
            result = runner.review(
                parsed_diff,
                repo_context=repo_context,
                context_fingerprint=context_fingerprint,
            )
            if result:
                quota.record(action.value)
                result.routing_meta = {"rung": "fallback", "quota": quota.snapshot()}
                return result
        return None

    logger.info(f"Routing plan: rung={plan.rung}, chunks={len(plan.assignments)}")

    results, unreviewable = execute_plan(
        plan,
        config,
        quota,
        repo_context=repo_context,
        context_fingerprint=context_fingerprint,
        cache_config=cache_config,
        engine=engine,
    )

    all_unreviewable = [c for c in chunks if c.unreviewable] + unreviewable

    return ReviewAggregator.merge(
        results,
        rung=plan.rung,
        quota_snapshot=quota.snapshot(),
        unreviewable=all_unreviewable or None,
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
    logger.info(f"Decision: {decision.action.value} - {decision.reason}")

    if decision.action == Action.SKIP and args.mode == "auto" and not args.force:
        logger.info("Skipping review")
        print(json.dumps({"action": "skip", "reason": decision.reason}), file=sys.stderr)
        return

    if decision.action == Action.SKIP and args.force:
        logger.info(f"Force review — overriding skip: {decision.reason}")

    context_pack = RepoContextWeaver(config.repo_context).weave(
        Path(args.repo_root),
        parsed_diff,
    )
    repo_context_text = context_pack.text
    context_fingerprint = context_pack.fingerprint
    extra_chars = len(repo_context_text or "")

    review_result: ReviewResultSchema | None = None
    cache_config = config.cache if config.cache.enabled else None

    if args.mode == "auto":
        review_result = run_auto_review(
            parsed_diff,
            config,
            engine,
            repo_context=repo_context_text or None,
            context_fingerprint=context_fingerprint,
            extra_chars=extra_chars,
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
    else:
        runners = {
            Action.GEMINI: lambda: GeminiRunner(
                api_key=config.gemini.api_key,
                model=config.gemini.model,
                cache_config=cache_config,
            ).review(
                parsed_diff,
                repo_context=repo_context_text or None,
                context_fingerprint=context_fingerprint,
            ),
            Action.OPENROUTER: lambda: OpenRouterRunner(
                api_key=config.openrouter.api_key,
                model=config.openrouter.model,
                cache_config=cache_config,
            ).review(
                parsed_diff,
                repo_context=repo_context_text or None,
                context_fingerprint=context_fingerprint,
            ),
            Action.GROQ: lambda: GroqRunner(
                api_key=config.groq.api_key,
                model=config.groq.model,
                cache_config=cache_config,
            ).review(
                parsed_diff,
                repo_context=repo_context_text or None,
                context_fingerprint=context_fingerprint,
            ),
        }
        effective_mode = args.mode
        runner_fn = runners.get(Action(effective_mode))
        if runner_fn:
            logger.info(f"Running {effective_mode} review (--mode {effective_mode})")
            review_result = runner_fn()
        elif effective_mode == "skip":
            logger.info("Skipping review (--mode skip)")
            print(json.dumps({"action": "skip", "reason": "mode=skip"}), file=sys.stderr)
            return
        else:
            logger.critical(f"Unknown mode: {effective_mode}")
            sys.exit(2)

        if not review_result:
            logger.critical(f"Review failed: no result from {effective_mode}")
            if not args.dry_run and github_token and args.repo and args.pr_number:
                CommentPoster(github_token).post_comment(
                    args.repo,
                    args.pr_number,
                    f"## Sorge AI Code Review\n\n"
                    f"> :x: **Review failed** — provider `{effective_mode}` returned no result.\n\n"
                    "Possible causes:\n"
                    f"- `{effective_mode}` API key missing or invalid\n"
                    "- Rate limits or quota exhaustion\n"
                    "- Network errors reaching the provider\n\n",
                )
            sys.exit(2)
        logger.info(f"Review complete: {len(review_result.issues)} issues found")

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
