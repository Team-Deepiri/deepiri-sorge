#!/usr/bin/env python3
"""Replay the real deepiri-topolsea#21 false positives through the claim verifier.

These are the exact strings Sorge posted on PR #21 (gemini-2.5-flash, scored
5.8/10), run against a real checkout of the repo at the PR head — not fixtures.
All three findings were wrong; two of them are mechanically disprovable:

  1. "HeaderMap ... is not imported"  → it is, on line 5. The first diff hunk
     covers lines 1-4, so the import sat one line outside the window. The old
     include check only understood #include, so a Rust `use` was invisible.
  2. "will break the existing route definition" → the rust CI job was green on
     that exact commit. GitHub already ran the real compiler.
  3. "prefer axum::extract::Headers" → that type does not exist in axum 0.7.
     Not disprovable without resolving dependency symbols, so it must be KEPT.

By default the CI verdict is fetched live from GitHub, which is what production
does. Pass --assume-green to skip the network and exercise the verifier alone.

    python3 scripts/replay_topolsea_findings.py                  # clone + live CI
    python3 scripts/replay_topolsea_findings.py ~/src/topolsea   # existing checkout
    python3 scripts/replay_topolsea_findings.py --assume-green   # no network

Requires: gh (authenticated) when cloning, GITHUB_TOKEN for the live CI check
(export GITHUB_TOKEN=$(gh auth token)).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.build_status import fetch_build_verdict  # noqa: E402
from bot.claim_verifier import ClaimVerifier  # noqa: E402
from bot.schemas import ReviewIssue  # noqa: E402

REPO = "Team-Deepiri/deepiri-topolsea"
PR = 21
HEAD_SHA = "bb88b1c24c7fb77d179498b417a68ece829cf7e1"
ROUTES = "crates/dv-server/src/routes.rs"

MISSING_IMPORT = ReviewIssue(
    severity="low", file=ROUTES, line=1,
    message=(
        "The HeaderMap type is used as an extractor in several handlers "
        "(shard_query, replicate_upsert, replicate_delete, shard_health) but is "
        "not imported, leading to a compilation error. The diff does not include "
        "the necessary use statement."
    ),
    suggestion=(
        "Add use axum::http::HeaderMap; at the top of "
        "crates/dv-server/src/routes.rs to resolve the compilation error."
    ),
)

BROKEN_ROUTES = ReviewIssue(
    severity="critical", file=ROUTES, line=953,
    message=(
        "The shard_health handler's signature has changed significantly, now "
        "accepting a HeaderMap extractor and returning a concrete Response type "
        "instead of impl IntoResponse. This will break the existing route "
        "definition unless the router configuration is updated to match the new "
        "handler signature."
    ),
    suggestion=(
        "Update the route definition for /topolsea/v1/shard_health (and other "
        "affected routes) to reflect the new handler signatures."
    ),
)

HALLUCINATED_TYPE = ReviewIssue(
    severity="info", file=ROUTES, line=628,
    message=(
        "While axum::http::HeaderMap is a valid extractor in Axum, "
        "axum::extract::Headers is often preferred as it provides a Deref "
        "implementation to HeaderMap and is a more idiomatic choice for raw "
        "header access."
    ),
    suggestion="Consider changing the HeaderMap extractor to axum::extract::Headers.",
)

MUST_SUPPRESS = {MISSING_IMPORT.message, BROKEN_ROUTES.message}
MUST_KEEP = {HALLUCINATED_TYPE.message}


def clone_topolsea() -> Path:
    dest = Path(tempfile.mkdtemp(prefix="topolsea-")) / "topolsea"
    print(f"cloning {REPO} → {dest}")
    subprocess.run(["gh", "repo", "clone", REPO, str(dest), "--", "--quiet"], check=True)
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "origin", f"pull/{PR}/head:pr{PR}", "--quiet"],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "checkout", f"pr{PR}", "--quiet"], check=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", nargs="?", help="existing checkout at PR head")
    parser.add_argument(
        "--assume-green",
        action="store_true",
        help="skip the live CI lookup and assert the build was green",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else clone_topolsea()

    if args.assume_green:
        green, reason = True, "assumed (--assume-green)"
    else:
        green, reason = fetch_build_verdict(
            repo=REPO, sha=HEAD_SHA, token=os.getenv("GITHUB_TOKEN", "")
        )
    print(f"\nrepo: {repo_root}")
    print(f"CI verdict for {HEAD_SHA[:7]}: green={green}  ({reason})")
    if green is None:
        print(
            "\nNOTE  no CI verdict, so the build-failure claim cannot be disproven.\n"
            "      export GITHUB_TOKEN=$(gh auth token), or pass --assume-green."
        )

    findings = [MISSING_IMPORT, BROKEN_ROUTES, HALLUCINATED_TYPE]
    report = ClaimVerifier(build_green=green, build_sha=HEAD_SHA).verify_issues(
        findings, repo_root=repo_root, changed_paths=[ROUTES]
    )

    print(f"\nin: {len(findings)}   kept: {len(report.kept)}   "
          f"suppressed: {report.suppressed_count}\n")
    for rec in report.suppressed:
        print(f"  SUPPRESSED  {rec.file}:{rec.line}")
        print(f"              {rec.message[:96]}")
        print(f"              why: {rec.reason}\n")
    for issue in report.kept:
        print(f"  KEPT        {issue.file}:{issue.line}  {issue.message[:80]}")

    got = {r.message for r in report.suppressed}
    kept = {i.message for i in report.kept}
    missed = MUST_SUPPRESS - got
    wrongly_dropped = MUST_KEEP - kept

    print()
    if missed:
        print("FAIL  these known false positives were NOT suppressed:")
        for m in missed:
            print(f"        {m[:96]}")
    if wrongly_dropped:
        print("FAIL  these were suppressed but should have been kept:")
        for m in wrongly_dropped:
            print(f"        {m[:96]}")
    if not missed and not wrongly_dropped:
        print("PASS  the missing-import and build-failure false positives are gone;")
        print("      the hallucinated axum::extract::Headers suggestion is kept,")
        print("      since nothing here can mechanically disprove it.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
