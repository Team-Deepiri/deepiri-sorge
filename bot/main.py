"""Main entry point for deepiri-sorge"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

from bot.build_status import fetch_build_verdict
from bot.claim_verifier import ClaimVerifier
from bot.comment_poster import CommentPoster
from bot.config import Config
from bot.finding_adjudicator import FindingAdjudicator
from bot.context_shaver import ContextShaver, should_engage_context_shave
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser
from bot.file_splitter import FileSplitter
from bot.github_app import fetch_pr_diff, get_installation_token
from bot.manifest_evidence import format_import_manifest_evidence
from bot.providers import build_providers
from bot.quota_tracker import QuotaTracker
from bot.repo_context import RepoContextWeaver
from bot.review_aggregator import ReviewAggregator
from bot.scheduling.history import ProviderHistory
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
    parser.add_argument(
        "--auto-retry",
        action="store_true",
        help="This run is a Worker-scheduled retry; do not enqueue another retry",
    )
    parser.add_argument(
        "--comment-id",
        type=int,
        default=None,
        help="Provisional issue comment id from this run (Starting… → final edit)",
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

    overhead = TEMPLATE_OVERHEAD_TOKENS + max(0, extra_chars // 4)
    history = ProviderHistory()
    engage, engage_reason = should_engage_context_shave(
        enabled=bool(getattr(config.routing, "context_shave_enabled", False)),
        quota=quota,
        providers=providers,
        chunks=runnable,
        prompt_overhead=overhead,
        history=history,
    )
    if engage:
        logger.info(
            f"Context shave fallback engaged (reason={engage_reason}); "
            "Gemini fully dead + oversized for secondary lanes"
        )
        shaver = ContextShaver(
            max_extracts=int(
                getattr(config.routing, "context_shave_max_extracts", 12) or 12
            ),
            shave_slice_budget=int(
                getattr(config.routing, "context_shave_slice_budget", 3500) or 3500
            ),
        )
        shaved, shave_meta = shaver.run(
            parsed_diff,
            config=config,
            quota=quota,
            providers=providers,
            repo_context=repo_context,
            context_fingerprint=context_fingerprint,
            prompt_overhead=overhead,
            repo=repo,
            pr_number=pr_number,
            installation_id=installation_id,
            head_sha=head_sha,
        )
        if shaved is not None:
            logger.info(
                f"Context shave complete: slices={shave_meta.slices} "
                f"llm_extracts={shave_meta.extracts_llm} "
                f"cache_hits={shave_meta.cache_hits} "
                f"mode={shave_meta.synthesize_mode}"
            )
            return shaved
        logger.warning(
            f"Context shave fail-open ({shave_meta.reason}); "
            "continuing with normal scheduler path"
        )
    else:
        logger.debug(f"Context shave not engaged ({engage_reason})")

    scheduler = ReviewScheduler.from_providers(
        providers,
        quota,
        config,
        repo_context=repo_context,
        context_fingerprint=context_fingerprint,
        prompt_overhead_tokens=overhead,
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
    # Nested toggles: pydantic exclude_unset often drops mutations on submodels.
    if os.getenv("SORGE_GEMINI_ENABLED") is not None:
        config.gemini.enabled = os.getenv("SORGE_GEMINI_ENABLED", "").lower() == "true"
    if os.getenv("SORGE_GROQ_ENABLED") is not None:
        config.groq.enabled = os.getenv("SORGE_GROQ_ENABLED", "").lower() == "true"
    if os.getenv("SORGE_OPENROUTER_ENABLED") is not None:
        config.openrouter.enabled = os.getenv("SORGE_OPENROUTER_ENABLED", "").lower() == "true"

    github_token = resolve_github_token(args)

    if getattr(args, "comment_id", None) is None:
        raw_cid = os.getenv("SORGE_COMMENT_ID", "").strip()
        if raw_cid.isdigit():
            args.comment_id = int(raw_cid)
    if getattr(args, "comment_id", None):
        logger.info(f"Using provisional comment_id={args.comment_id} for this run")

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

    # After the skip decision so a lockfile-only PR still skips as deps-only.
    parsed_diff, generated_files = engine.strip_generated_files(parsed_diff)
    if generated_files:
        logger.info(
            f"Excluded {len(generated_files)} generated file(s) from review: "
            f"{', '.join(generated_files[:5])}"
        )

    repo_root = Path(args.repo_root)
    context_pack = RepoContextWeaver(config.repo_context).weave(
        repo_root,
        parsed_diff,
    )
    repo_context_text = context_pack.text
    context_fingerprint = context_pack.fingerprint

    # Compact: only packages newly imported in the DIFF vs HEAD manifests.
    manifest_block = format_import_manifest_evidence(repo_root, parsed_diff)
    if manifest_block:
        repo_context_text = (
            f"{repo_context_text}\n\n{manifest_block}"
            if repo_context_text
            else manifest_block
        )
        context_fingerprint = f"{context_fingerprint}:m{len(manifest_block)}"
        logger.info(
            f"Repo context + IMPORT_VS_MANIFEST → {len(repo_context_text or '')} chars total"
        )

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
                CommentPoster(github_token).upsert_comment(
                    args.repo,
                    args.pr_number,
                    "## Sorge AI Code Review\n\n"
                    "> :x: **Review failed** — all AI providers were exhausted or unavailable.\n\n"
                    "Possible causes:\n"
                    "- API key misconfiguration (check `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`)\n"
                    "- Rate limits exceeded (check quota settings in `sorge.toml`)\n"
                    "- Network errors reaching the provider APIs\n\n",
                    preferred_comment_id=getattr(args, "comment_id", None),
                    reuse_previous=False,
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
                CommentPoster(github_token).upsert_comment(
                    args.repo,
                    args.pr_number,
                    f"## Sorge AI Code Review\n\n"
                    f"> :x: **Review failed** — provider `{args.mode}` returned no result.\n\n"
                    "Possible causes:\n"
                    f"- `{args.mode}` API key missing or invalid\n"
                    "- Rate limits or quota exhaustion\n"
                    "- Network errors reaching the provider\n\n",
                    preferred_comment_id=getattr(args, "comment_id", None),
                    reuse_previous=False,
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
        # CI already compiled this SHA; a green build disproves any predicted
        # compile failure for free. Unknown verdict suppresses nothing.
        head_sha = os.getenv("SORGE_HEAD_SHA", "")
        build_green, build_reason = fetch_build_verdict(
            repo=args.repo or "",
            sha=head_sha,
            token=github_token or "",
        )
        logger.info(f"CI build verdict for claim checks: green={build_green} ({build_reason})")
        review_result = ClaimVerifier(
            build_green=build_green,
            build_sha=head_sha,
        ).verify_result(
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

    if review_result and config.finding_adjudicator.enabled:
        before = len(review_result.issues)
        review_result = FindingAdjudicator().adjudicate_result(
            review_result,
            repo_root=repo_root,
        )
        changed = before - len(review_result.issues)
        if changed:
            logger.info(
                f"FindingAdjudicator: {before} issues in → {len(review_result.issues)} out"
            )

    if args.pr_number and args.repo and not args.dry_run:
        comment_id = CommentPoster(github_token).post_review(
            repo=args.repo,
            pr_number=args.pr_number,
            review=review_result,
            preferred_comment_id=getattr(args, "comment_id", None),
        )
        if comment_id:
            try:
                from bot.escalate_ledger import EscalateLedger

                n = EscalateLedger().attach_comment(
                    args.repo, args.pr_number, comment_id
                )
                if n:
                    logger.info(
                        f"Attached comment_id={comment_id} to {n} escalate ticket(s)"
                    )
            except Exception as e:
                logger.debug(f"Ledger attach_comment skipped: {e}")

        # One automatic delayed retry after capacity defer (not on the retry itself).
        if (
            review_result
            and getattr(review_result, "review_type", "")
            in ("rate_limited", "lock_contention")
            and not getattr(args, "auto_retry", False)
        ):
            import random

            delay = random.randint(
                int(getattr(config.scheduler, "auto_retry_delay_min_sec", 60) or 60),
                int(getattr(config.scheduler, "auto_retry_delay_max_sec", 120) or 120),
            )
            try:
                from bot.escalate_ledger import EscalateLedger

                ok = EscalateLedger().schedule_review_retry(
                    repo=args.repo,
                    pr_number=args.pr_number,
                    installation_id=args.installation_id,
                    delay_sec=delay,
                    comment_id=comment_id,
                )
                if ok and review_result.recommendations is not None:
                    note = (
                        f"Sorge will automatically retry once in ~{delay}s "
                        "(no need to re-comment `/sorge`)."
                    )
                    if note not in review_result.recommendations:
                        review_result.recommendations.insert(0, note)
                    # Refresh comment so the user sees the auto-retry note.
                    CommentPoster(github_token).post_review(
                        repo=args.repo,
                        pr_number=args.pr_number,
                        review=review_result,
                        preferred_comment_id=comment_id,
                    )
            except Exception as e:
                logger.warning(f"Auto-retry enqueue failed: {e}")

    print(json.dumps(review_result.to_dict(), indent=2))


def review() -> None:
    main()


if __name__ == "__main__":
    main()
