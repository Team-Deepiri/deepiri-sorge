#!/usr/bin/env python3
"""Measure the context-budget / review-quality tradeoff.

Answers the question "can we spend fewer context tokens without losing
findings?" with data instead of intuition, by sweeping the repo-context budget
and comparing each run against the full-budget baseline.

Two modes:

  --offline (default)  Builds the real prompts and counts tokens. Costs nothing,
                       calls no provider. Gives you the cost half of the curve:
                       how many tokens each budget actually spends.

  --live               Additionally runs the review at each budget and diffs the
                       findings against the baseline. This is what supports any
                       claim about accuracy — and it burns provider quota, so
                       mind Groq's 200k tokens/day free-tier cap.

Usage:
    gh pr diff 29 --repo Team-Deepiri/deepiri-crankl > /tmp/pr29.diff
    python3 scripts/context_budget_experiment.py \\
        --diff /tmp/pr29.diff --repo-root ~/Desktop/deepiri-crankl

    # add --live to measure findings too (spends quota)
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.config import Config, RepoContextConfig  # noqa: E402
from bot.diff_parser import DiffParser  # noqa: E402
from bot.file_splitter import FileSplitter  # noqa: E402
from bot.repo_context import RepoContextWeaver  # noqa: E402
from bot.runners.groq_runner import GroqRunner  # noqa: E402
from bot.schemas import ReviewIssue  # noqa: E402

# Budgets to sweep, in characters of repo context. 0 disables context entirely,
# which is the floor: what the model sees with nothing but the diff.
DEFAULT_BUDGETS = [0, 700, 1400, 2800, 5600]

# Two findings are "the same" if they sit in the same file within a few lines
# and say substantially the same thing. Line numbers drift between runs, so
# exact matching would understate agreement.
LINE_TOLERANCE = 5
MESSAGE_SIMILARITY = 0.60


@dataclass
class BudgetRun:
    budget: int
    context_chars: int
    context_hits: int
    prompt_tokens: int
    findings: list[ReviewIssue] = field(default_factory=list)
    ran_live: bool = False


def same_finding(a: ReviewIssue, b: ReviewIssue) -> bool:
    if (a.file or "") != (b.file or ""):
        return False
    if a.line is not None and b.line is not None:
        if abs(a.line - b.line) > LINE_TOLERANCE:
            return False
    ratio = difflib.SequenceMatcher(
        None, (a.message or "").lower(), (b.message or "").lower()
    ).ratio()
    return ratio >= MESSAGE_SIMILARITY


def recall_precision(
    baseline: list[ReviewIssue], candidate: list[ReviewIssue]
) -> tuple[float, float]:
    if not baseline:
        return 1.0, 1.0 if not candidate else 0.0
    matched_base = sum(1 for b in baseline if any(same_finding(b, c) for c in candidate))
    matched_cand = sum(1 for c in candidate if any(same_finding(c, b) for b in baseline))
    recall = matched_base / len(baseline)
    precision = matched_cand / len(candidate) if candidate else 0.0
    return recall, precision


def run_budget(
    budget: int,
    *,
    repo_root: Path,
    diff_text: str,
    live: bool,
) -> BudgetRun:
    parsed = DiffParser().parse(diff_text)

    if budget <= 0:
        context_text, hits = "", 0
    else:
        cfg = RepoContextConfig(max_chars=budget)
        pack = RepoContextWeaver(cfg).weave(repo_root, parsed)
        context_text, hits = pack.text, len(pack.hits)

    runner = GroqRunner(api_key="offline" if not live else None)
    runner._repo_context = context_text or None
    runner._prior_partial = None

    # Count the prompt we would actually send, per chunk, summed over chunks.
    chunks = FileSplitter().split(parsed)
    total_tokens = 0
    for chunk in chunks:
        total_tokens += len(runner._build_prompt(chunk.parsed_diff)) // 4

    run = BudgetRun(
        budget=budget,
        context_chars=len(context_text),
        context_hits=hits,
        prompt_tokens=total_tokens,
    )

    if live:
        findings: list[ReviewIssue] = []
        for chunk in chunks:
            result = runner.review(chunk.parsed_diff, repo_context=context_text or None)
            if result and result.issues:
                findings.extend(result.issues)
        run.findings = findings
        run.ran_live = True

    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True, help="Path to a diff file")
    ap.add_argument("--repo-root", required=True, help="Checkout of the reviewed repo")
    ap.add_argument("--live", action="store_true", help="Call providers (spends quota)")
    ap.add_argument(
        "--budgets",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BUDGETS),
        help="Comma-separated repo-context char budgets",
    )
    args = ap.parse_args()

    diff_text = Path(args.diff).read_text()
    repo_root = Path(args.repo_root).resolve()
    budgets = sorted({int(b) for b in args.budgets.split(",") if b.strip()})

    if args.live:
        print("LIVE MODE — this calls real providers and consumes quota.\n")

    runs = [
        run_budget(b, repo_root=repo_root, diff_text=diff_text, live=args.live)
        for b in budgets
    ]

    baseline = max(runs, key=lambda r: r.budget)

    print(f"\ndiff: {args.diff}")
    print(f"repo: {repo_root}")
    print(f"baseline budget: {baseline.budget} chars\n")

    header = f"{'budget':>8} {'ctx chars':>10} {'hits':>5} {'prompt tok':>11} {'vs base':>9}"
    if args.live:
        header += f" {'findings':>9} {'recall':>7} {'precision':>10}"
    print(header)
    print("-" * len(header))

    for run in runs:
        delta = run.prompt_tokens - baseline.prompt_tokens
        pct = (delta / baseline.prompt_tokens * 100) if baseline.prompt_tokens else 0.0
        row = (
            f"{run.budget:>8} {run.context_chars:>10} {run.context_hits:>5} "
            f"{run.prompt_tokens:>11,} {pct:>8.1f}%"
        )
        if args.live:
            recall, precision = recall_precision(baseline.findings, run.findings)
            row += f" {len(run.findings):>9} {recall:>7.2f} {precision:>10.2f}"
        print(row)

    print()
    if args.live:
        print("Read the curve: the cheapest budget whose recall is still ~1.00 is")
        print("the knee. Below it you are paying in findings, not just tokens.")
    else:
        print("Offline mode: token cost only. Re-run with --live to measure whether")
        print("the cheaper budgets actually cost you findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
