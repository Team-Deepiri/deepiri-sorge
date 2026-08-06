#!/usr/bin/env python3
"""Replay the real deepiri-crankl false positives through the claim verifier.

These are the exact strings Sorge posted on PRs #28 and #29, run against a real
checkout of the repo at the PR head — not fixtures. If the verifier regresses,
this catches it in a way tests/test_claim_verifier.py cannot.

    python3 scripts/replay_crankl_findings.py            # clones to a temp dir
    python3 scripts/replay_crankl_findings.py ~/src/crankl   # use existing checkout

Requires: gh (authenticated) when cloning.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.claim_verifier import ClaimVerifier  # noqa: E402
from bot.schemas import ReviewIssue  # noqa: E402

READER = "src/archive/cran_reader.cpp"

# --- PR #29: openrouter/gpt-oss-20b:free, scored 3.8/10 -----------------------
PR29 = [
    ReviewIssue(
        severity="critical", file=READER, line=86,
        message="Memory leak: mapping not unmapped on early return when size too small.",
        suggestion="Call `map.reset()` before returning -5.",
    ),
    ReviewIssue(
        severity="critical", file=READER, line=98,
        message="Memory leak: mapping not unmapped on magic validation failure.",
        suggestion="Call `map.reset()` before returning -6.",
    ),
    ReviewIssue(
        severity="critical", file=READER, line=112,
        message="Memory leak: mapping not unmapped on layout validation failure.",
        suggestion="Call `map.reset()` before returning layout_rc.",
    ),
    ReviewIssue(
        severity="critical", file=READER, line=124,
        message="Memory leak: mapping not unmapped on checksum mismatch.",
        suggestion="Call `map.reset()` before returning -7.",
    ),
    ReviewIssue(
        severity="low", file=READER, line=7,
        message="Missing include <fcntl.h> for open() declaration.",
        suggestion="Add `#include <fcntl.h>`",
    ),
]

# --- PR #28: the include-path finding the author rebutted with evidence -------
PR28 = ReviewIssue(
    severity="critical", file="tests/ctest/test_simd.cpp", line=2,
    message=(
        'The test includes "internal_headers/simd.hpp" but the CMakeLists.txt only '
        "adds ${CMAKE_SOURCE_DIR}/src to the include directories, so the internal "
        "header cannot be found on non-x86 builds or when the internal_headers "
        "directory is not under src."
    ),
    suggestion=(
        "Add `target_include_directories(test_simd PRIVATE "
        "${CMAKE_SOURCE_DIR}/internal_headers)` (or the appropriate path) in "
        "tests/CMakeLists.txt."
    ),
)

# Only the two context-driven false positives should be suppressed. The four
# "memory leak" claims are weak-model reasoning: the model had ~MappedRegion in
# the diff and still got it wrong. Nothing here can disprove them, and a
# verifier that did would be guessing.
MUST_SUPPRESS = {
    "Missing include <fcntl.h> for open() declaration.",
    PR28.message,
}


def clone_crankl() -> Path:
    dest = Path(tempfile.mkdtemp(prefix="crankl-")) / "crankl"
    print(f"cloning Team-Deepiri/deepiri-crankl → {dest}")
    subprocess.run(
        ["gh", "repo", "clone", "Team-Deepiri/deepiri-crankl", str(dest), "--", "--quiet"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "origin", "pull/29/head:pr29", "--quiet"],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "checkout", "pr29", "--quiet"], check=True)
    return dest


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else clone_crankl()

    findings = [*PR29, PR28]
    report = ClaimVerifier().verify_issues(
        findings,
        repo_root=repo_root,
        changed_paths=[READER, PR28.file],
    )

    print(f"\nrepo: {repo_root}")
    print(f"in: {len(findings)}   kept: {len(report.kept)}   "
          f"suppressed: {report.suppressed_count}\n")

    for rec in report.suppressed:
        print(f"  SUPPRESSED  {rec.file}:{rec.line}")
        print(f"              {rec.message[:96]}")
        print(f"              why: {rec.reason}\n")
    for issue in report.kept:
        print(f"  KEPT        {issue.file}:{issue.line}  {issue.message[:80]}")

    got = {r.message for r in report.suppressed}
    missed = MUST_SUPPRESS - got
    extra = got - MUST_SUPPRESS

    print()
    if missed:
        print("FAIL  these known false positives were NOT suppressed:")
        for m in missed:
            print(f"        {m[:96]}")
    if extra:
        print("FAIL  these were suppressed but should have been kept:")
        for m in extra:
            print(f"        {m[:96]}")
    if not missed and not extra:
        print("PASS  both context-driven false positives suppressed;")
        print("      the 4 weak-model 'memory leak' claims correctly left alone.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
