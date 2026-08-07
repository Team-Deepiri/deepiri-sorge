"""CI verdict for a PR head, used to disprove predicted build failures.

Models routinely predict compile errors that the real compiler disagrees with —
on deepiri-topolsea#21 Gemini claimed a missing import and a broken route
definition on a commit whose Rust job was green. GitHub already ran the actual
toolchain on that SHA, so asking it costs zero prompt tokens and needs no
knowledge of the language involved.

Conservative by construction: anything short of "every completed check
succeeded" returns None, and None never suppresses a finding.
"""

from __future__ import annotations

import requests
from loguru import logger

API_ROOT = "https://api.github.com"
TIMEOUT_SEC = 10.0

# Check runs whose failure says nothing about whether the code compiles. A red
# linter or coverage gate must not be read as "the build is broken", but it also
# must not let us claim green — these are simply ignored in the verdict.
_NON_BUILD_CHECK_HINTS = (
    "codeql",
    "coverage",
    "codecov",
    "license",
    "dependabot",
    "danger",
    "semantic",
    "commitlint",
    "label",
    "size",
    "sonar",
)


def _is_build_relevant(name: str) -> bool:
    lowered = name.lower()
    return not any(hint in lowered for hint in _NON_BUILD_CHECK_HINTS)


def fetch_build_verdict(
    *,
    repo: str,
    sha: str,
    token: str,
) -> tuple[bool | None, str]:
    """Return ``(green, reason)`` for ``sha``.

    ``green`` is True only when at least one build-relevant check completed and
    every completed build-relevant check succeeded. It is None when no verdict
    can be established — no token, no checks, or checks still in flight — and
    False when something build-relevant actually failed.
    """
    if not (repo and sha and token):
        return None, "missing repo, sha, or token"

    url = f"{API_ROOT}/repos/{repo}/commits/{sha}/check-runs"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
        "User-Agent": "deepiri-sorge",
    }
    try:
        response = requests.get(
            url, headers=headers, params={"per_page": 100}, timeout=TIMEOUT_SEC
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.debug(f"Build verdict fetch failed for {repo}@{sha[:7]}: {exc}")
        return None, f"check-runs fetch failed: {exc}"

    runs = payload.get("check_runs") or []
    if not runs:
        return None, "no check runs on this commit"

    relevant = [r for r in runs if _is_build_relevant(str(r.get("name") or ""))]
    if not relevant:
        return None, "no build-relevant checks on this commit"

    pending = [r for r in relevant if r.get("status") != "completed"]
    if pending:
        names = ", ".join(str(r.get("name")) for r in pending[:3])
        return None, f"checks still running ({names})"

    # "neutral" and "skipped" are not evidence either way; require a real pass.
    failed = [
        str(r.get("name"))
        for r in relevant
        if r.get("conclusion") not in ("success", "neutral", "skipped")
    ]
    if failed:
        return False, f"failing checks: {', '.join(failed[:3])}"

    passed = [r for r in relevant if r.get("conclusion") == "success"]
    if not passed:
        return None, "no check actually reported success"

    names = ", ".join(str(r.get("name")) for r in passed[:3])
    return True, f"green: {names}"
