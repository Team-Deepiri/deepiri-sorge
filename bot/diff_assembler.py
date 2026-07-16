"""Assemble PR diffs when GitHub rejects the monolithic diff (406 too_large).

GitHub caps ``Accept: application/vnd.github.v3.diff`` at ~20k lines.
``GET /pulls/{n}/files`` still returns per-file ``patch`` fields (paginated),
so we stitch those into a reviewable unified-diff-like text.
"""

from __future__ import annotations

from typing import Any

import requests
from loguru import logger

GITHUB_API = "https://api.github.com"
DIFF_TOO_LARGE_STATUS = 406


def _auth_headers(token: str, *, accept: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "deepiri-sorge",
    }


def fetch_monolithic_diff(repo: str, pr_number: int, token: str) -> tuple[str | None, requests.Response]:
    """Try the single-shot PR diff. Returns (text, response). text is None on failure."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(
        url,
        headers=_auth_headers(token, accept="application/vnd.github.v3.diff"),
        timeout=120,
    )
    if resp.status_code == 200:
        return resp.text, resp
    return None, resp


def iter_pr_file_pages(
    repo: str,
    pr_number: int,
    token: str,
    *,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """Paginate GET /pulls/{n}/files until exhausted."""
    files: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"
        resp = requests.get(
            url,
            headers=_auth_headers(token, accept="application/vnd.github+json"),
            params={"per_page": per_page, "page": page},
            timeout=120,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        files.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        if page > 50:  # hard safety (~5000 files)
            logger.warning("DiffAssembler: stopped at page cap")
            break
    return files


def stitch_file_patches(files: list[dict[str, Any]]) -> str:
    """Build a unified-diff-like document from per-file patch payloads."""
    parts: list[str] = []
    skipped = 0
    for entry in files:
        path = entry.get("filename") or "unknown"
        status = entry.get("status") or "modified"
        patch = entry.get("patch")
        if not patch:
            # Binary / too-large single file / renamed without patch
            parts.append(
                f"diff --git a/{path} b/{path}\n"
                f"# no patch available (status={status}, "
                f"changes={entry.get('changes', '?')})\n"
            )
            skipped += 1
            continue
        # GitHub patches are already hunk-only; add a minimal header for parsers.
        if not patch.startswith("diff --git"):
            parts.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{patch}\n")
        else:
            parts.append(patch if patch.endswith("\n") else patch + "\n")
    if skipped:
        logger.info(f"DiffAssembler: {skipped} file(s) without patch (binary/truncated)")
    return "".join(parts)


def assemble_pr_diff(repo: str, pr_number: int, token: str) -> str:
    """Prefer monolithic diff; on 406/too_large, assemble from file patches."""
    text, resp = fetch_monolithic_diff(repo, pr_number, token)
    if text is not None:
        return text

    body = ""
    try:
        body = resp.text[:500]
    except Exception:
        body = ""

    if resp.status_code == DIFF_TOO_LARGE_STATUS or "too_large" in body:
        logger.warning(
            f"Monolithic diff unavailable for {repo}#{pr_number} "
            f"(HTTP {resp.status_code}); assembling from /pulls/files"
        )
        files = iter_pr_file_pages(repo, pr_number, token)
        if not files:
            resp.raise_for_status()
            raise RuntimeError(f"Empty file list for {repo}#{pr_number}")
        assembled = stitch_file_patches(files)
        logger.info(
            f"DiffAssembler: stitched {len(files)} file(s), "
            f"{len(assembled)} chars for {repo}#{pr_number}"
        )
        return assembled

    resp.raise_for_status()
    raise RuntimeError(f"Unexpected diff fetch failure: HTTP {resp.status_code}")
